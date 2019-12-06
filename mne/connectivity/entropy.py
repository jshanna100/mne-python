from ..time_frequency.tfr import tfr_array_morlet
from ..filter import filter_data
from ..parallel import parallel_func
from scipy.signal import hilbert
from scipy.stats import circstd
import numpy as np

def _instant_phase(data, freqs, sfreq, method="wavelet", freq_band=2,
                   cuda=False, n_jobs=1):
    if method == "wavelet":
        phases = tfr_array_morlet(data, sfreq, freqs, output="phase",
                                  n_jobs=n_jobs)
    if method == "hilbert":
        phases = np.empty((data.shape[0], data.shape[1], len(freqs),
                           data.shape[2]))
        for freq_idx,freq in enumerate(list(freqs)):
            if cuda:
                temp_data = filter_data(data, sfreq, l_freq=freq-freq_band/2,
                                        h_freq=freq+freq_band/2, n_jobs="cuda")
            else:
                temp_data = filter_data(data, sfreq, l_freq=freq-freq_band/2,
                                        h_freq=freq+freq_band/2, n_jobs=n_jobs)
            analytic_signal = hilbert(temp_data)
            phases[:,:,freq_idx,:] = np.angle(analytic_signal)

    return phases

def _adjust_binwidth(bw, rad_range=(-np.pi,np.pi), epsilon=1e-8):
    # Scott's choice applied naively would result in one bin width of
    # unequal size to the rest. Presumably it's better to adjust them
    # so that they're equal?
    q,r = divmod(rad_range[1]-rad_range[0], bw)
    new_bw = bw + r / q - epsilon
    bin_nums = np.int((rad_range[1]-rad_range[0]) // new_bw)
    return new_bw, bin_nums

def _scotts_binwidth(phase_series):
    # Determine binwidth by Scott's choice
    N = phase_series.size
    circ_std = circstd(phase_series, low=-np.pi, high=np.pi)
    bin_width = 3.5 * circ_std / N**(1/3)
    return bin_width

def _scotts_bins(phases):
    _, bin_num = _adjust_binwidth(_scotts_binwidth(phases))
    bins = np.linspace(-np.pi, np.pi, num=bin_num)
    return bins

def _histo_1d(phases, epsilon=1e-8):
    if len(phases.shape) > 1:
        raise ValueError("Input must be 1-dimensional.")
    samp_nums = phases.size
    bin_edges = _scotts_bins(phases)
    histo = np.histogram(phases,bins=bin_edges)[0] / samp_nums + epsilon
    return histo

def _diff_transfer_entropy(x_phase, y_phase, x_be, y_be, x_past_be, y_past_be,
                           x_past_histo, y_past_histo, sfreq, delay,
                           epsilon=1e-8):
    # differential transfer entropy of x to y; to get y to x, just * -1

    # we need the delayed and non-delayed phase for x and y
    phases = {}
    samp_delay = np.int(np.round(delay*sfreq))

    if not x_past_histo:
        phases["x_past"] = x_phase[...,:-samp_delay]
    if not y_past_histo:
        phases["y_past"] = y_phase[...,:-samp_delay]
    phases["x"] = x_phase[...,samp_delay:]
    phases["y"] = y_phase[...,samp_delay:]

    # calculate the bin edges
    bin_edges = {}
    for ph_k,ph_v in phases.items():
        bw, bn = _adjust_binwidth(_scotts_binwidth(ph_v))
        bin_edges[ph_k] = np.linspace(-np.pi, np.pi, num=bn)

    # calculate the histograms
    samp_nums = phases["x"].size # x is arbitrary; any other variable would do
    if not x_past_histo:
        x_past_histo = (np.histogram(phases["x_past"],
                        bins=bin_edges["x_past"])[0] / samp_nums + epsilon)
    if not y_past_histo:
        y_past_histo = (np.histogram(phases["y_past"],
                        bins=bin_edges["y_past"])[0] / samp_nums)
    x_past_x_histo = (np.histogramdd(np.array([phases["x_past"].flatten(),
                      phases["x"].flatten()]).T, bins=[bin_edges["x_past"],
                      bin_edges["x"]])[0] / samp_nums + epsilon)
    y_past_y_histo = (np.histogramdd(np.array([phases["y_past"].flatten(),
                      phases["y"].flatten()]).T, bins=[bin_edges["y_past"],
                      bin_edges["y"]])[0] / samp_nums + epsilon)
    x_past_y_past_histo = (np.histogramdd(np.array([phases["x_past"].flatten(),
                           phases["y_past"].flatten()]).T,
                           bins=[bin_edges["x_past"],
                           bin_edges["y_past"]])[0] / samp_nums + epsilon)
    x_x_past_y_past_histo = (np.histogramdd(np.array([phases["x"].flatten(),
                             phases["x_past"].flatten(),
                             phases["y_past"].flatten()]).T,
                             bins=[bin_edges["x"], bin_edges["x_past"],
                             bin_edges["y_past"]])[0] / samp_nums + epsilon)
    y_y_past_x_past_histo = (np.histogramdd(np.array([phases["y"].flatten(),
                             phases["y_past"].flatten(),
                             phases["x_past"].flatten()]).T,
                             bins=[bin_edges["y"], bin_edges["y_past"],
                             bin_edges["x_past"]])[0] / samp_nums + epsilon)

    # entropy
    h_y_past_y = -1 * np.sum(y_past_y_histo * np.log(y_past_y_histo))
    h_y_past_x_past = -1 * np.sum(x_past_y_past_histo
                           * np.log(x_past_y_past_histo))
    h_y_past = -1 * np.sum(y_past_histo * np.log(y_past_histo))
    h_y_y_past_x_past = -1 * np.sum(y_y_past_x_past_histo
                         * np.log(y_y_past_x_past_histo))
    h_x_past_x = -1 * np.sum(x_past_x_histo * np.log(x_past_x_histo))
    h_x_past = -1 * np.sum(x_past_histo * np.log(x_past_histo))
    h_x_x_past_y_past = -1 * np.sum(x_x_past_y_past_histo
                         * np.log(x_x_past_y_past_histo))

    # PTE
    pte_xy = h_y_past_y + h_y_past_x_past - h_y_past - h_y_y_past_x_past
    pte_yx = h_x_past_x + h_y_past_x_past - h_x_past - h_x_x_past_y_past
    d_pte_xy = pte_xy - pte_yx

    return d_pte_xy
