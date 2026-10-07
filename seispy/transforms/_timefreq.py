"""
Time-frequency panels: SUST (``st``, the Stockwell transform), SUGABOR (``gabor``, the multifilter analysis of Dziewonski,
Bloch and Landisman) and SUCWT (``cwt``, a continuous wavelet transform) from ``su/main/transforms``.

Each trace becomes a panel (a gather) of traces, one for each frequency (or scale), that are all as long as the trace: the
time-frequency amplitude. In the headers of the traces of a panel ``ensemble_number`` is the ``trace_id`` of the trace that
they are from, and ``ensemble_trace_number`` is the number of the frequency or scale (from 1). The frequency of each (the
SU programs write it to trace header words that are not here) is found as:

* ``st``: ``(first + k) / (n dt)``, for the traces ``k = 0, 1, ...`` of the panel, where ``n`` is the number of samples of
  the trace, and ``first = int(fmin n dt + 1)``;
* ``gabor``: ``fmin + k band`` for the center of the filter of trace ``k``;
* ``cwt``: the scales are ``base ** (first + k expinc)`` (a smaller scale is a higher frequency).

These use numpy's FFT of the trace as it is (what is done between the transforms is a function of the SU library, see ``_kernels.pyx``), where the SU programs pad the traces to a length that their FFT is fast
for. Not supported: the ``holder`` option of ``sugabor`` and ``sucwt`` (an estimate of the regularity), which index
past the arrays that they have, and the wavelet types of ``sucwt`` that its program does not have (``wtype`` 1 and
2).
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, stage
from . import _hilbert, _kernels
from ..attributes import _attributes
from ..convolution import _convolve

__all__ = ['st', 'gabor', 'cwt']

F32 = np.float32


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _panel(upstream, rows_of):
    """The traces of the panels: for each trace the rows (the 2D array of samples, one row for each trace of the panel)"""
    source = as_trace_iterator(upstream)

    def traces():
        for trace in source:
            if trace.dtype.kind == 'c':
                raise TypeError("This stage works on real traces, but was given a complex one.")
            rows = rows_of(trace)
            number = trace.header['trace_id']
            for k, row in enumerate(rows, start=1):
                yield trace.replace(row.astype(F32), ensemble_number=number, ensemble_trace_number=k)

    return from_iterable(traces())


def _sample_interval(trace, dt):
    sample_dt = dt or trace.d_sample
    if not sample_dt:
        raise ValueError("The trace has no sample interval, and no `dt` was given.")
    return sample_dt


# ------------------------------------------------------------------------------------------------------------- st
def _st(upstream, *, fmin=0.0, fmax=None, dt=None):
    def rows_of(trace):
        x = np.asarray(trace).astype(np.float64)
        n = x.shape[0]
        sample_dt = _sample_interval(trace, dt)
        nyquist = 0.5 / sample_dt
        f_max = nyquist if fmax is None else fmax
        d1 = 1.0 / (n * sample_dt)
        first = int(fmin / d1 + 1)
        last = int(f_max / d1 + 1 + 0.5)
        if n == 0 or last < first:
            return np.zeros((0, n))
        # the transform of the analytic signal (in the convention of the SU transform: the conjugate of numpy's), from the SU library
        h = np.ascontiguousarray(np.conj(np.fft.fft(x)), dtype=np.complex64)
        _kernels.st_analytic(h)
        rows = np.empty((last - first + 1, n))
        for row, k in enumerate(range(first, last + 1)):
            g = _kernels.st_row(h, k)  # (the row, in frequency)
            # (the inverse transform with the sign -1 of the SU transform is numpy's forward)
            rows[row] = np.abs(np.fft.fft(g)) / n
        return rows

    return _panel(upstream, rows_of)


# SUST: the Stockwell transform (S transform): a time-frequency amplitude in which the Gaussian window changes
# with the frequency, which gives better resolution than a fixed window.
#
# fmin, fmax : the least and the greatest frequency (Hz) to output, default 0 and the Nyquist frequency. (The panel
#              starts at the first frequency after 0, since the transform has no 0 frequency.)
# dt : sample interval (s), for traces that do not have one
#
# The frequencies are separated by 1 / (n dt), where n is the number of samples of the trace (sust pads it for its
# FFT, which changes that).
st = stage(_st, parallelism='trace', name='st', validate=True)


# --------------------------------------------------------------------------------------------------------- gabor
def _gabor(upstream, *, fmin=0.0, fmax=None, band=None, beta=3.0, alpha=None, dt=None):
    if beta <= 0.0:
        raise ValueError(f"beta={beta} must be positive")
    if band is not None and band <= 0.0:
        raise ValueError(f"band={band} must be positive")

    def rows_of(trace):
        x = np.asarray(trace).astype(np.float64)
        n = x.shape[0]
        sample_dt = _sample_interval(trace, dt)
        nyquist = 0.5 / sample_dt
        f_max = nyquist if fmax is None else fmax
        width = 0.05 * nyquist if band is None else band
        steep = beta / (width * width) if alpha is None else alpha
        n_filters = _nint((f_max - fmin) / width)
        if n == 0 or n_filters < 1:
            return np.zeros((0, n))
        n_fft = n + n % 2
        spectrum = np.fft.rfft(x, n_fft)
        rows = np.empty((n_filters, n))
        for i in range(n_filters):
            center = fmin + i * width
            filt = _kernels.gabor_filter(center, sample_dt, n_fft, steep, width, 1.0)  # (the function of the SU library)
            filtered = np.fft.irfft(spectrum * filt, n_fft)
            narrow = np.ascontiguousarray(filtered[:n], dtype=F32)
            quadrature = _hilbert.hilbert(narrow)  # (the Hilbert transform of SU)
            rows[i] = _attributes.envelope(narrow, quadrature)  # (and the amplitude of the two of them)
        return rows

    return _panel(upstream, rows_of)


# SUGABOR: a time-frequency amplitude by multifilter analysis: the trace is passed through Gaussian filters, the narrow band
# traces and their quadrature (Hilbert transform) traces give instantaneous amplitudes.
#
# fmin, fmax : the least and the greatest frequency (Hz) of the centers of the filters, default 0 and the Nyquist frequency
# band : the bandwidth of the filters (Hz), default .05 of the Nyquist frequency. Also the spacing of their centers. If it
#        is too big the amplitude is in stripes that are parallel to the frequency axis, if it is too small, to the time axis.
# beta : ln (the amplitude of the peak of the filter over that at its ends), default 3
# alpha : the width parameter of the filters, default beta / band^2
# dt : sample interval (s), for traces that do not have one
#
# There are round((fmax - fmin) / band) filters, so traces, in each panel.
gabor = stage(_gabor, parallelism='trace', name='gabor', validate=True)


# ------------------------------------------------------------------------------------------------------------ cwt
def _cwt(upstream, *, base=10.0, first=-1.0, expinc=0.01, last=1.5, nwavelet=1024, xmin=-20.0, xcenter=0.0, xmax=20.0,
         sigma=1.0):
    if nwavelet <= 1:
        raise ValueError("nwavelet must be greater than 1")
    if base <= 0.0 or base == 1.0:
        raise ValueError("base must be positive, and not 1")
    if expinc <= 0.0:
        raise ValueError("expinc must be positive")
    if xmax <= xmin:
        raise ValueError("xmax must be more than xmin")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    # the integral of the wavelet, and a filter of it for each scale, from the SU library
    wavelet_sum, dx = _kernels.cwt_wavelet(nwavelet, xmin, xcenter, xmax, sigma)
    width = dx * (nwavelet - 1)
    # the scales, from base^first up to base^last (as many as there are in the range)
    n_scales = int(np.floor((last - first) / expinc + 1e-9)) + 1
    if n_scales < 1:
        raise ValueError("There are no scales: last must be at least first.")
    scales = base ** (first + np.arange(n_scales) * expinc)
    filters = [_kernels.cwt_filter(wavelet_sum, float(scale), dx, width) for scale in scales]

    def rows_of(trace):
        x = np.asarray(trace).astype(np.float64)
        n = x.shape[0]
        rows = np.empty((n_scales, n))
        for i, (scale, filt) in enumerate(zip(scales, filters)):
            full = _convolve(x, filt, n)  # (the convolution is kept to the length of the trace)
            narrow = _kernels.cwt_trace(np.ascontiguousarray(full, dtype=F32), filt.shape[0], float(scale))
            quadrature = _hilbert.hilbert(narrow)
            rows[i] = _attributes.envelope(narrow, quadrature)
        return rows

    return _panel(upstream, rows_of)


def cwt(*, base=10.0, first=-1.0, expinc=0.01, last=1.5, nwavelet=1024, xmin=-20.0, xcenter=0.0, xmax=20.0, sigma=1.0):
    """The amplitude of the continuous wavelet transform with the Mexican hat wavelet (SUCWT), a panel of traces of the
    envelope for each scale.

    Parameters
    ----------
    base, first, last, expinc : float
        The scales are ``base ** e`` for the exponents ``e`` from ``first`` to ``last`` in steps of ``expinc``. A scale
        is a length: the smaller it is the higher the frequency that it shows.
    nwavelet, xmin, xcenter, xmax, sigma : float
        The wavelet is the second derivative of a Gaussian with the standard deviation ``sigma`` and its center at
        ``xcenter``, with ``nwavelet`` samples from ``xmin`` to ``xmax``. (sucwt starts it at ``-xcenter`` rather than
        at ``xmin``.)

    The transform is made with the integral of the wavelet, and the difference of that, which is how sucwt gets the
    derivative. As in SU, the convolution is cut to the length of the trace, so what is past it is not seen.
    """
    kwargs = dict(base=base, first=first, expinc=expinc, last=last, nwavelet=nwavelet, xmin=xmin, xcenter=xcenter,
                  xmax=xmax, sigma=sigma)
    _cwt((), **kwargs)  # check the parameters now
    return Stage(_cwt, parallelism='trace', name='cwt', **kwargs)
