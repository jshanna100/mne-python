from ..time_frequency.tfr import tfr_array_morlet
from ..filter import filter_data
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
    N = phase_series.size
    circ_std = circstd(phase_series, low=-np.pi, high=np.pi)
    bin_width = 3.5 * circ_std / N**(1/3)
    return bin_width

def _estimate_entropy(signals, epsilon=1e-8):
    bin_edges = []
    for sig in signals:
        bw, bn = _adjust_binwidth(_scotts_binwidth(x_past))
        bin_edges.append(np.linspace(-np.pi, np.pi, num=bn))
    histo = np.histogramdd(np.array([sig.flatten() for sig in signals]).T,
                           bins=bin_edges)[0] / signals[0].size + epsilon
    entropy = -1 * sum(histo * np.log(histo))

    return entropy


def _diff_transfer_entropy(x_phase, y_phase, sfreq, delay, epsilon=1e-8):
    # differential transfer entropy of x to y; to get y to x, just * -1
    samp_delay = np.int(np.round(delay*sfreq))
    x_past = x_phase[...,:-samp_delay]
    y_past = y_phase[...,:-samp_delay]
    x = x_phase[...,samp_delay:]
    y = y_phase[...,samp_delay:]

    # calculate the histograms
    samp_nums = x.size # x is arbitrary; any other variable would do
    x_past_histo = (np.histogram(x_past, bins=bin_edges["x_past"])[0] /
                    samp_nums + epsilon)
    y_past_histo = (np.histogram(y_past, bins=bin_edges["y_past"])[0] /
                    samp_nums)
    x_past_x_histo = (np.histogramdd(np.array([x_past.flatten(),
                      x.flatten()]).T, bins=[bin_edges["x_past"],
                      bin_edges["x"]])[0] / samp_nums + epsilon)
    y_past_y_histo = (np.histogramdd(np.array([y_past.flatten(),
                      y.flatten()]).T, bins=[bin_edges["y_past"],
                      bin_edges["y"]])[0] / samp_nums + epsilon)
    x_past_y_past_histo = (np.histogramdd(np.array([x_past.flatten(),
                           y_past.flatten()]).T, bins=[bin_edges["x_past"],
                           bin_edges["y_past"]])[0] / samp_nums + epsilon)
    x_x_past_y_past_histo = (np.histogramdd(np.array([x.flatten(),
                             x_past.flatten(), y_past.flatten()]).T,
                             bins=[bin_edges["x"], bin_edges["x_past"],
                             bin_edges["y_past"]])[0] / samp_nums + epsilon)
    y_y_past_x_past_histo = (np.histogramdd(np.array([y.flatten(),
                             y_past.flatten(), x_past.flatten()]).T,
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
