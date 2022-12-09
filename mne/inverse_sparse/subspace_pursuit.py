import numpy as np
from ..utils.numerics import _PCA
from ..utils import verbose, logger
from .. import make_forward_solution, convert_forward_solution
from ..source_space import (_set_source_space_vertices, setup_source_space,
                            setup_volume_source_space)
from ..io.constants import FIFF
from ..minimum_norm import (make_inverse_operator, apply_inverse,
                            apply_inverse_epochs)
from ..epochs import BaseEpochs, EpochsArray, concatenate_epochs
from ..evoked import Evoked
from ..minimum_norm.inverse import (_log_exp_var, _check_or_prepare,
                                    _pick_channels_inverse_operator)
from ..dipole import Dipole
from ..io.pick import pick_types, channel_type, _picks_by_type
from mne_connectivity import spectral_connectivity_epochs
from dPTE import epo_dPTE

def _expl_var(data, resid, inv):
    sel = _pick_channels_inverse_operator(data.ch_names, inv)
    data_arr, resid_arr = data.data[sel], resid.data[sel]
    est_arr = data_arr - resid_arr
    data_w = np.dot(inv["whitener"] , np.dot(inv["proj"], data_arr))
    est_w = np.dot(inv["whitener"] , np.dot(inv["proj"], est_arr))
    expl_var = _log_exp_var(data_w, est_w)
    return expl_var

def make_dipoles_from_stc(stc, fwd):
    times = stc.times
    dipoles = []
    for src_idx in range(fwd["nsource"]):
        pos = fwd["source_rr"][src_idx]
        ori = fwd["source_nn"][src_idx]
        dipole = Dipole(times, pos, stc.data[src_idx,], ori, 0)
        dipoles.append(dipole)
    return dipoles

def _restrict_forward_patch(fwd, inds, copy=True):
    """Restrict a forward model to forward model indices"""
    if any(np.diff(inds) <= 0):
        raise ValueError('Indices must be ordered in increasing order.')
    fwd = fwd.copy() if copy else fwd
    verts = np.concatenate([s["vertno"].astype(int) for s in fwd["src"]])
    src_inds = np.concatenate([np.ones(len(s["vertno"]), dtype=int)*s_idx
                               for s_idx, s in enumerate(fwd["src"])])
    new_verts = [[] for s in fwd["src"]]
    for idx in inds:
        vert, src_idx = verts[idx], src_inds[idx]
        new_verts[src_idx].append(vert)

    fwd["nsource"] = len(inds)
    fwd['source_nn'] = fwd["source_nn"][inds]
    fwd['source_rr'] = fwd["source_rr"][inds]
    fwd["sol"]["data"] = fwd["sol"]["data"][:, inds]
    fwd["sol"]["ncol"] = len(inds)
    _set_source_space_vertices(fwd["src"], new_verts)
    return fwd

def _convert_forward_patch(fwd, new_nns, new_rrs, new_gain, new_verts,
                           copy=True):
    fwd = fwd.copy() if copy else fwd
    # make sure order corresponds to increasing vertices
    start_inds = np.cumsum([0] + [len(nv) for nv in new_verts[:-1]])
    ordered_inds = np.concatenate([np.argsort(nv) + start_idx
                                   for nv, start_idx in zip(new_verts, start_inds)])
    new_verts = [np.sort(nv) for nv in new_verts]
    fwd["nsource"] = len(new_nns)
    fwd['source_nn'] = new_nns[ordered_inds]
    fwd['source_rr'] = new_rrs[ordered_inds]
    fwd["sol"]["data"] = new_gain[:, ordered_inds]
    fwd["sol"]["ncol"] = len(new_nns)
    fwd["source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
    fwd['surf_ori'] = True
    fwd["_orig_sol"] = fwd["sol"]["data"]
    fwd["_orig_source_ori"] = FIFF.FIFFV_MNE_FIXED_ORI
    if fwd["src"].kind == "volume":
        for s in fwd["src"]:
            s["type"] = "discrete"
    _set_source_space_vertices(fwd["src"], new_verts)
    return fwd


def telescope_vertices(subject, coarse_spacing, fine_spacing, vertices,
                       subjects_dir=None, n_jobs=1):
    """
    For all vertices, find a) which patch they inhabit in coarse space, and
    then b) find all the patches in fine space which overlap with the coarse
    patch.
    """
    coarse_src = setup_source_space(subject, spacing=coarse_spacing,
                                    subjects_dir=subjects_dir, n_jobs=n_jobs,
                                    verbose="warning")
    fine_src = setup_source_space(subject, spacing=fine_spacing,
                                  subjects_dir=subjects_dir, n_jobs=n_jobs,
                                  verbose="warning")
    # find patches in coarse space
    pinfos = [s["pinfo"] for s in coarse_src]
    coarse_patch_inds = [[] for s in coarse_src]
    for src_idx, (vertex, pinfo) in enumerate(zip(vertices, pinfos)):
        for pi in pinfo:
            for vtx in vertex:
                if vtx in pi:
                    coarse_patch_inds[src_idx].append(pi)
    # overlapping patches in fine space
    pinfos = [s["pinfo"] for s in fine_src]
    fine_patch_inds = [[] for s in coarse_src]
    for src_idx, pinfo in enumerate(pinfos):
        for pi_idx, pi in enumerate(pinfo):
            for c_pinfo in coarse_patch_inds[src_idx]:
                if len(np.intersect1d(pi, c_pinfo)):
                    fine_patch_inds[src_idx].append(
                                        fine_src[src_idx]["vertno"][pi_idx])
    fine_patch_inds = [np.array(fpi) for fpi in fine_patch_inds]
    return fine_patch_inds

def _chantype_scale_gain_mat(gain, info, scale=True):
    "scale/unscale the gain matrix by channel-type units"
    scalings = {"eeg":1e+6, "mag":1e15, "grad":1e13}
    if not scale:
        # invert the scalings to put them back to original values
        scalings = {k:1/v for k, v in scalings.items()}
    gain = gain.copy()
    ch_types = [channel_type(info, idx)
                for idx in range(len(info["chs"]))]
    ch_types = list(np.unique(ch_types))
    for ch_type in ch_types:
        if ch_type == "eeg":
            sel = pick_types(info, eeg=True, meg=False)
        elif ch_type == "mag":
            sel = pick_types(info, eeg=False, meg="mag")
        elif ch_type == "grad":
            sel = pick_types(info, eeg=False, meg="grad")
        else:
            raise ValueError(f"Channel type {ch_type} not recognised.")
        gain[sel,] *= scalings[ch_type]
    return gain

def _whiten_gain(gain, info):
    pre_whitener = np.empty([len(gain), 1])
    for _, picks_ in _picks_by_type(info, ref_meg=False, exclude=[]):
        pre_whitener[picks_] = np.std(gain[picks_])
    wh_gain = gain / pre_whitener
    return wh_gain, pre_whitener

def _svd_gain(gain, info, n_components, whiten=False):
    gain = gain.copy()
    if whiten:
        gain, pre_whitener = _whiten_gain(gain, info)
    pca = _PCA(n_components=n_components, whiten=whiten)
    u, s, v = pca._fit(gain)
    if whiten:
        svd_gain = u[:, :pca.n_components_] * np.sqrt(gain.shape[0] - 1)
        svd_gain = svd_gain * pre_whitener
    else:
        svd_gain = u[:, :pca.n_components_] * s[:pca.n_components_]
    return svd_gain, u, s, v, pca.n_components_

def mix_patch_forwards(ctx_fwd, sub_fwd, info, trans, bem):
    # created a mixed forward solution out of cortical and volume patch forwards
    mix_src = ctx_fwd["src"] + sub_fwd["src"]
    mix_fwd = make_forward_solution(info, trans, mix_src, bem)
    mix_fwd = convert_forward_solution(mix_fwd, force_fixed=True)
    mix_gain = np.hstack((ctx_fwd["sol"]["data"], sub_fwd["sol"]["data"]))
    mix_fwd["sol"]["data"] = mix_gain

    return mix_fwd

def make_patch_forward(subject, spacing, bem, info, trans, subjects_dir=None,
                       restrict_to=None, volume=False, volume_label=None,
                       patch_comp_n=.9, meg=True, eeg=True,
                       ignore_ref=False, mindist=0., n_jobs=1):
    if volume and not volume_label:
        raise ValueError("If doing a volume source space, volume labels must "
                         "also be specified; volume_label is empty.")
    if volume:
        src = setup_volume_source_space(subject, bem=bem,
                                        subjects_dir=subjects_dir,
                                        mindist=mindist,
                                        volume_label=volume_label)
    else:
        src = setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 n_jobs=n_jobs, verbose="warning")
    if restrict_to:
        _set_source_space_vertices(src, restrict_to)

    # make a forward model for each cortical patch
    new_nns, new_rrs, new_gain, new_verts = [], [], [], [[] for s in src]
    for sr_idx, sr in enumerate(src):
        patch_verts = []
        if sr["pinfo"]:
            for pi in sr["patch_inds"]:
                patch_verts.append(sr["pinfo"][pi])
        else:
            patch_verts = [sr["vertno"]]
        for p_count, pv in enumerate(patch_verts):
            logger.info(f"Source {sr_idx}, Patch {p_count} of {len(patch_verts)}")
            # restrict source space to patch and calc forward
            p_src = src.copy()
            pinfo_list = [[] for s in src]
            pinfo_list[sr_idx] = pv.astype(int)
            _set_source_space_vertices(p_src, pinfo_list)
            fwd = make_forward_solution(info, trans, p_src,
                                        bem, meg=meg, eeg=eeg,
                                        mindist=mindist,
                                        ignore_ref=ignore_ref,
                                        n_jobs=n_jobs,
                                        verbose="warning")
            if not volume:
                fwd = convert_forward_solution(fwd, force_fixed=True)

            # reduce gain matrix dimensionality to n best components
            # or components that explain n variance
            gain = fwd["sol"]["data"].copy()
            reduce_gain, u, s, v, n_comps = _svd_gain(gain, fwd["info"],
                                                      patch_comp_n,
                                                      whiten=True)
            # get prominence of sources in decomposition and use those to derive
            # locations and orientations for components
            nn_weights = v[:n_comps,:].T * s[:n_comps].T
            # reduce this to norm of 3 orientations for location weights
            if not volume:
                rr_weights = nn_weights
            else:
                rr_weights = np.stack([nn_weights[0::3], nn_weights[1::3],
                                       nn_weights[2::3]])
                rr_weights = np.linalg.norm(rr_weights, axis=0)
                nn_weights = nn_weights / sum(nn_weights) # rows sum to 1
                rr_weights = rr_weights / sum(rr_weights) # rows sum to 1
            nns = fwd["source_nn"]
            rrs = fwd["source_rr"]
            # get nn and rr weighted by SVD component contribution
            w_nns = np.dot(nn_weights.T, nns)
            w_rrs = np.dot(rr_weights.T, rrs)
            # find vertex nums which are closest to new_rrs
            for rr, nn in zip(w_rrs, w_nns):
                dists = np.linalg.norm((rrs - rr), axis=1)
                # don't allow repeats of vertices; breaks stuff downstream
                idx = 0
                sort_inds = np.argsort(dists)
                closest_idx = sort_inds[idx]
                while pv[closest_idx] in new_verts[sr_idx]:
                    idx += 1
                    closest_idx = sort_inds[idx]
                new_verts[sr_idx].append(pv[closest_idx])

                new_rrs.append(rrs[closest_idx])
                new_nns.append(nn)
            new_gain.append(reduce_gain)
            logger.info(f"Patch constructed from {fwd['nsource']} of "
                        f"{sum([ps['nuse'] for ps in p_src])} sources, "
                        f"reduced to {n_comps} components.")

    new_nns = np.vstack(new_nns)
    new_rrs = np.vstack(new_rrs)
    new_gain = np.hstack(new_gain)
    new_fwd = _convert_forward_patch(fwd, new_nns, new_rrs, new_gain, new_verts)

    return new_fwd


def _union_s(s1, s2):
    support = np.union1d(s1, s2)
    support = np.sort(support)
    return support


def _select_s(proxy, confl_mat, s, cnx=None):
    """
    Get s strongest sources, with the constraint that mutual coherence
    is below threshold
    """
    if len(proxy.shape) == 3 and cnx is None:
        proxy = proxy.mean(axis=0)
    if cnx is not None:
        ## dpli
        # get only pairs where conflict matrix isn't marked
        # inds = list(zip(*np.where(confl_mat==0)))
        # triu_inds = list(zip(*np.triu_indices(len(confl_mat))))
        # new_inds = tuple(zip(*[idx for idx in inds if idx in triu_inds]))
        # con = spectral_connectivity_epochs(proxy, indices=new_inds,
        #                                    fmin=cnx["fmin"], fmax=cnx["fmax"],
        #                                    faverage=True, sfreq=cnx["sfreq"],
        #                                    method=cnx["method"])
        # con_vec = con.get_data()
        # sort_inds = np.argsort(con_vec[:, 0])[-s:]
        # support = []
        # for s_idx in sort_inds:
        #     support.append(con.indices[0][s_idx])
        #     support.append(con.indices[1][s_idx])
        dpte = epo_dPTE(proxy, cnx["freqs"], cnx["sfreq"],
                        n_cycles=cnx["cycles"], n_jobs=cnx["n_jobs"])
        dpte = dpte.mean(axis=0)
        dpte = dpte*(~confl_mat)
        dpte[dpte>0] = np.abs(dpte[dpte>0] - 0.5)
        sort_inds = tuple(zip(*np.unravel_index(np.argsort(dpte, axis=None),
                                                dpte.shape)))[-s:]
        support = [s for si in sort_inds for s in si]

    else:
        norm_sort_inds = np.argsort(np.linalg.norm(proxy, axis=1))[::-1]
        support = np.array([norm_sort_inds[0]])
        for idx in norm_sort_inds[1:]:
            if len(support) == s:
                break
            if sum(confl_mat[support, idx]) == 0:
                support = np.append(support, idx)

    support = np.unique(support)
    return support


def _proxy_matrix(inst, fwd, cov, lambda2):
    inv_op = make_inverse_operator(inst.info, fwd, cov, loose=0, fixed=True,
                                   depth=None, verbose="warning")
    if isinstance(inst, BaseEpochs):
        stc_and_resid = apply_inverse_epochs(inst, inv_op, lambda2,
                                             method="MNE",
                                             return_residual=True,
                                             verbose="warning")
        stcs, resids = zip(*stc_and_resid)
        resid = EpochsArray(resids, inst.info)
        data = np.stack([stc.data for stc in stcs])
    elif isinstance(inst, Evoked):
        stc, resid = apply_inverse(inst, inv_op, lambda2, method="MNE",
                                   return_residual=True, verbose="warning")
        data = stc.data.copy()

    return data, resid


def subspace_pursuit_level(fwd, inst, cov, s, mu, lambda2, cnx=None,
                           sub_fwd=None, sub_mu=None, max_iter=200):
    # conflict matrix for mutual coherence of gain matrix columns
    gain = fwd["sol"]["data"].copy()
    src_n = gain.shape[1]
    confl_mat = np.zeros((src_n, src_n), dtype=bool)
    wh_gain, _ = _whiten_gain(gain, fwd["info"])
    norms = np.linalg.norm(wh_gain, axis=0)
    inner = np.inner(wh_gain.T, wh_gain.T)
    for src_i in range(src_n):
        for src_j in range(src_i+1, src_n):
            if inner[src_i, src_j] > (mu * norms[src_i] * norms[src_j]):
                confl_mat[src_i, src_j] = 1
    if sub_fwd is not None:
        # mark cortical sources coherent with subcortical as a conflict
        sub_mu = mu if sub_mu is None else sub_mu
        sub_gain = sub_fwd["sol"]["data"].copy()
        wh_sub_gain, _ = _whiten_gain(sub_gain, fwd["info"])
        inner = np.abs(np.inner(wh_gain.T, wh_sub_gain.T))
        wh_gain_norms = np.linalg.norm(wh_gain, axis=0)
        wh_sub_gain_norms = np.linalg.norm(wh_sub_gain, axis=0)
        for idx in range(src_n):
            for s_idx in range(sub_gain.shape[1]):
                if inner[idx, s_idx] > (sub_mu * wh_gain_norms[idx] *
                                        wh_sub_gain_norms[s_idx]):
                    # if any exceed, mark whole row as conflict and go on
                    confl_mat[idx,] = np.ones(confl_mat.shape[1])
                    break
        # set the lower triangle back to 0
        confl_mat[np.tril_indices(src_n, k=-1)] = 0

        confl_sat = confl_mat.sum() / ((src_n**2 + src_n) / 2)
        if confl_sat == 1.:
            raise ValueError("Conflict matrix is saturated. Choose a higher "
                             "sub_mu.")
        elif confl_sat > .8:
            logger.warning("Warning: conflict matrix is >= 80% saturated."
                           "Consider using a higher sub_mu "
                           f"(current value: {sub_mu}.")


    # initial proxy matrix and top s sources
    proxy, _ = _proxy_matrix(inst, fwd, cov, lambda2)
    support = _select_s(proxy, confl_mat, s, cnx=cnx)
    # initial residual
    proxy_fwd = _restrict_forward_patch(fwd, support)
    _, resid = _proxy_matrix(inst, proxy_fwd, cov, lambda2)
    # now iterate
    all_supports = [set(support)]
    for iter_idx in range(max_iter):
        logger.info(f"Iteration {iter_idx}")
        proxy, _ = _proxy_matrix(resid, fwd, cov, lambda2)
        proxy_support = _select_s(proxy, confl_mat, s, cnx=cnx)
        test_support = _union_s(np.array(list(all_supports[-1])),
                                proxy_support)
        test_confl_mat = confl_mat[np.ix_(test_support, test_support)]
        proxy_fwd = _restrict_forward_patch(fwd, test_support)
        est_proxy, _ = _proxy_matrix(inst, proxy_fwd, cov, lambda2)
        est_support_inds = _select_s(est_proxy, test_confl_mat, s, cnx=cnx)
        est_support = test_support[est_support_inds]
        est_fwd = _restrict_forward_patch(fwd, est_support)
        _, resid = _proxy_matrix(inst, est_fwd, cov, lambda2)
        support = set(est_support)
        if support in all_supports:
            break
        else:
            all_supports.append(set(support))

    inv = make_inverse_operator(inst.info, est_fwd, cov, loose=0,
                                fixed=True, depth=None, verbose="warning")
    if isinstance(inst, Evoked):
        # run this because we want the explained variance later
        inv = _check_or_prepare(inv, inst.nave, lambda2, "MNE",
                                None, False)
        stc, resid = apply_inverse(inst, inv, lambda2, method="MNE",
                                   return_residual=True, verbose=None,
                                   prepared=True)
        var_expl = _expl_var(inst, resid, inv)
    else:
        inv = _check_or_prepare(inv, 1, lambda2, "MNE",
                                None, False)
        stc_and_resid = apply_inverse_epochs(inst, inv, lambda2, method="MNE",
                                             return_residual=True, verbose=None,
                                             prepared=True)
        stc, resids = zip(*stc_and_resid)
        resid = EpochsArray(resids, inst.info)
        var_expl = 0

    return stc, est_fwd, var_expl, resid

def subspace_pursuit(subject, spacings, bem, inst, cov, trans, s, lambda2,
                     mu=.5, fwd0=None, return_fwds=False,
                     return_as_dipoles=False, patch_comp_n=.9,
                     subjects_dir=None, max_iter=200, cnx=None, sub_fwd=None,
                     sub_mu=None, n_jobs=1):
    if isinstance(s, int):
        s = [s] * len(spacings)
    elif len(s) != len(spacings):
        raise ValueError("Length of s must be 1 or equal to # of spacings.")
    fwds = []
    if fwd0:
        fwd = fwd0
    else:
        fwd = make_patch_forward(subject, spacings[0], bem, inst.info, trans,
                                 patch_comp_n=patch_comp_n,
                                 subjects_dir=subjects_dir, n_jobs=n_jobs)
    if return_fwds:
        fwds.append(fwd)

    for spacing_idx in range(len(spacings)-1):
        stc, _, var_expl, resid = subspace_pursuit_level(fwd, inst, cov,
                                                         s[spacing_idx],
                                                         mu, lambda2, cnx=cnx,
                                                         sub_fwd=sub_fwd,
                                                         sub_mu=sub_mu,
                                                         max_iter=max_iter)
        logger.info(f"Spacing {spacings[spacing_idx]} explained {var_expl:.1f}"
                     " percent variance.")
        vertices = stc[0].vertices if isinstance(stc, tuple) else stc.vertices
        new_verts = telescope_vertices(subject, spacings[spacing_idx],
                                       spacings[spacing_idx+1],
                                       vertices,
                                       subjects_dir=subjects_dir,
                                       n_jobs=n_jobs)
        fwd = make_patch_forward(subject, spacings[spacing_idx+1], bem,
                                 inst.info, trans, restrict_to=new_verts,
                                 subjects_dir=subjects_dir,
                                 patch_comp_n=patch_comp_n, n_jobs=n_jobs)
        if return_fwds:
            fwds.append(fwd.copy())

    out, est_fwd, var_expl, resid = subspace_pursuit_level(fwd, inst, cov,
                                                           s[spacing_idx+1], mu,
                                                           lambda2, cnx=cnx,
                                                           sub_fwd=sub_fwd,
                                                           sub_mu = sub_mu,
                                                           max_iter=max_iter)
    if return_fwds:
        fwds.append(est_fwd)

    logger.info(f"Spacing {spacings[spacing_idx+1]} explained {var_expl:.1f} "
                 " percent variance.")
    if return_as_dipoles:
        out = make_dipoles_from_stc(out, est_fwd)
    if return_fwds:
        return out, fwds, resid
    else:
        return out, resid
