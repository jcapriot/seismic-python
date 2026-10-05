"""
Filters that are made from the Fourier transform of each trace: SUMINPHASE and SUTVBAND (``su/main/filters``), with
numpy's FFT on the trace as it is (the SU programs zero-pad each trace for their prime-factor FFT).
"""
import numpy as np

from ..stage import per_trace


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


# ------------------------------------------------------------------------------------------------------- minphase
def _cwp_fft(z, sign):
    """The (unnormalized) transform of the SU library, with exp(sign * i w t) in its kernel. numpy's has sign = -1."""
    if sign == -1:
        return np.fft.fft(z)
    return np.fft.ifft(z) * z.shape[0]


def _kolmogoroff(cx, pnoise, sign1, sign2):
    """Spectral factorization: the spectrum of the minimum phase wavelet that has the amplitudes of cx"""
    n = cx.shape[0]
    r = np.abs(cx)
    rmax = r.max()
    r = np.maximum(r / rmax, pnoise)
    # the log of the power spectrum, as a (real) function of time: the cepstrum
    cepstrum = _cwp_fft(np.log(r * r).astype(np.complex128), sign2) / n
    # fold it, keeping only the causal part (which is what makes the wavelet minimum phase)
    cepstrum[0] *= 0.5
    cepstrum[n // 2] *= 0.5
    cepstrum[n // 2 + 1:] = 0.0
    return np.exp(_cwp_fft(cepstrum, sign1)) * rmax


def _minphase(upstream, *, sign1=1, sign2=-1, pnoise=1.0e-9):
    if sign1 == 0:
        raise ValueError(f"sign1={sign1}")
    if sign2 == 0:
        raise ValueError(f"sign2={sign2}")
    sign1 = 1 if sign1 > 0 else -1
    sign2 = 1 if sign2 > 0 else -1

    def minphase_trace(trace):
        x = np.asarray(trace)
        nt = x.shape[0]
        if nt == 0 or not x.any():
            return trace  # (nothing to find the spectrum of)
        n_fft = nt + nt % 2  # (even, as the factorization needs)
        z = np.zeros(n_fft, dtype=np.complex128)
        z[:nt] = x
        spectrum = _kolmogoroff(_cwp_fft(z, sign1), pnoise, sign1, sign2)
        y = _cwp_fft(spectrum, sign2) / n_fft
        return trace.replace(y.real[:nt].astype(np.float32))

    return per_trace(upstream, minphase_trace)


# SUMINPHASE: convert to the minimum phase equivalent: the wavelet with the same amplitude spectrum that has its energy
# as early as it can (found by the Kolmogoroff spectral factorization).
#
# sign1 : sign of the first transform (1 or -1), default 1
# sign2 : sign of the second transform, default -1
# pnoise : white noise that keeps the log of the spectrum finite, relative to its maximum, default 1e-9
#
# (suminphase pads the trace for its FFT. Here an odd number of samples is made even.)


# --------------------------------------------------------------------------------------------------------- tvband
def _band_filter(corners, n_fft, sample_dt):
    """The amplitude (for the nfft // 2 + 1 frequencies of the transform) of a bandpass filter with sine squared tapers
    between the corner frequencies f1, f2 (up) and f3, f4 (down)"""
    n_freq = n_fft // 2 + 1
    df = 1.0 / (n_fft * sample_dt)
    last = n_freq - 1
    if1 = min(_nint(corners[0] / df), last)
    if2 = min(_nint(corners[1] / df), last)
    if3 = min(_nint(corners[2] / df), last)
    if4 = min(_nint(corners[3] / df), last)
    amplitude = np.zeros(n_freq)
    i = np.arange(if1, if2 + 1)
    amplitude[if1:if2 + 1] = np.sin(0.5 * np.pi / (if2 - if1 + 2) * (i - if1 + 1)) ** 2
    i = np.arange(if3, if4 + 1)
    amplitude[if3:if4 + 1] = np.sin(0.5 * np.pi / (if4 - if3 + 2) * (if4 - i + 1)) ** 2
    amplitude[if2 + 1:if3] = 1.0
    return amplitude


def _tvband(upstream, tf, f, *, dt=None):
    tf = np.atleast_1d(np.asarray(tf, dtype=np.float64))
    f = np.asarray(f, dtype=np.float64)
    if f.ndim == 1 and f.shape[0] == 4:
        f = f[None, :]
    if tf.ndim != 1 or tf.shape[0] == 0:
        raise ValueError("tf must be a 1D array of times")
    if f.shape != (tf.shape[0], 4):
        raise ValueError(f"must give one f 4-tuple of corner frequencies for each ({tf.shape[0]}) tf value")
    if (np.diff(tf) <= 0).any():
        raise ValueError("tf must increase")
    for number, corners in enumerate(f, start=1):
        if corners[0] < 0.0 or corners[0] > corners[1] or corners[1] >= corners[2] or corners[2] > corners[3]:
            raise ValueError(f"Filter #{number} has bad frequencies")

    cache = {}

    def filters_for(n, sample_dt):
        key = (n, sample_dt)
        if key not in cache:
            cache.clear()
            cache[key] = [_band_filter(corners, n, sample_dt) for corners in f]
        return cache[key]

    def tvband_trace(trace):
        x = np.asarray(trace)
        n = x.shape[0]
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        if n == 0:
            return trace
        first_time = trace.header['sample_start']
        # the samples that the filters are centered on, kept on the trace
        centers = [min(max(_nint((t - first_time) / sample_dt), 0), n - 1) for t in tf]
        filters = list(filters_for(n, sample_dt))
        # a filter for the start and for the end of the trace, if the user did not give them
        if centers[0] > 0:
            centers.insert(0, 0)
            filters.insert(0, filters[0])
        if centers[-1] < n - 1:
            centers.append(n - 1)
            filters.append(filters[-1])

        # filter each filter's part of the trace, which goes from the previous center to the next one
        filtered = []
        for j, amplitude in enumerate(filters):
            lo = centers[j - 1] if j > 0 else 0
            hi = centers[j + 1] if j + 1 < len(centers) else n - 1
            part = np.zeros(n, dtype=np.float64)
            part[lo:hi + 1] = x[lo:hi + 1]
            filtered.append(np.fft.irfft(np.fft.rfft(part, n) * amplitude, n))

        # and fade from one into the next between the centers
        out = filtered[0].copy()
        for j in range(len(centers) - 1):
            i = np.arange(centers[j], centers[j + 1] + 1)
            span = centers[j + 1] - centers[j]
            a = (i - centers[j]) / span if span else np.zeros(i.shape[0])
            out[i] = (1 - a) * filtered[j][i] + a * filtered[j + 1][i]
        return trace.replace(out.astype(np.float32))

    return per_trace(upstream, tvband_trace)


# SUTVBAND: a time-variant bandpass filter (sine squared tapers)
#
# tf : times (s) at which the filters are given
# f : four corner frequencies f1, f2, f3, f4 for each time. (or just 4, for one time)
# dt : sample interval (s) for traces that do not have one
#
# The trace is filtered with each filter (the part of it around the time of the filter, up to the times of the
# filters on each side), and the results fade into each other between the times. Before the first time and after
# the last the first and the last filter are used.
#
# Example: `tvband([.2, 1.5], [[10, 12.5, 40, 50], [10, 12.5, 30, 40]])`
