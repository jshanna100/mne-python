from ..time_frequency.tfr import tfr_array_morlet
from ..filter import filter_data
from scipy.signal import hilbert
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
            ## fix: what's going on here?
            phases[:,:,freq_idx,:] = np.angle(analytic_signal)

    return phases
