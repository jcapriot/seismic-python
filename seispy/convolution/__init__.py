"""
Convolution and correlation: SUACOR, SUCONV and SUXCOR (``su/main/convolution_correlation``).

``acor``, ``conv`` and ``xcor`` are done in the time domain, as in the SU sources with the ``convolve_cwp`` and ``xcor`` of the SU library
(``_cwp.pyx``), by those routines. The ``sufile`` options of SU become a ``Trace`` (or arrays), and suxcor's
``panel`` option, which correlates windows of a whole gather, is not supported.

The traces that come out are along a lag (or, for convolution, time) axis, so their start time is set to match.
sucor, suconv and suxcor leave ``delrt`` alone, except to add the one of the filter to that of the trace in suconv.
"""
import warnings

import numpy as np

from . import _cwp
from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, per_trace, stage

__all__ = ['acor', 'conv', 'xcor', 'acorfrac', 'refcon']


def _real32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def _correlate(x, y, first_lag, n_lags):
    """z[i] = sum_j x[j] y[i+j], for i = first_lag, ..., first_lag + n_lags - 1 (x and y are 0 outside of themselves)"""
    return _cwp.correlate(_real32(x), _real32(y), first_lag, n_lags)


def _convolve(x, y, n_out=None):
    """The first n_out samples (by default all of them) of the convolution of x and y, either of which can be complex"""
    n_out = x.shape[0] + y.shape[0] - 1 if n_out is None else n_out
    if not (np.iscomplexobj(x) or np.iscomplexobj(y)):
        return _cwp.convolve(_real32(x), _real32(y), n_out)
    xr, xi = _real32(np.real(x)), _real32(np.imag(x))
    yr, yi = _real32(np.real(y)), _real32(np.imag(y))
    real = _cwp.convolve(xr, yr, n_out) - _cwp.convolve(xi, yi, n_out)
    imag = _cwp.convolve(xr, yi, n_out) + _cwp.convolve(xi, yr, n_out)
    return (real + 1j * imag).astype(np.complex64)


# ------------------------------------------------------------------------------------------------------------ acor
def _acor(upstream, *, ntout=101, norm=True, sym=True):
    if ntout < 1:
        raise ValueError("ntout must be at least 1")
    # (the first lag, and the sample at lag 0)
    istart = -((ntout - 1) // 2) if sym else 0
    izero = -istart

    def acor_trace(trace):
        x = np.asarray(trace)
        z = _correlate(x, x, istart, ntout)
        if norm:
            z /= z[izero] if z[izero] != 0.0 else np.float32(1.0)
        return trace.replace(z, sample_start=istart * trace.d_sample)

    return per_trace(upstream, acor_trace)


# SUACOR: auto-correlation
#
# ntout : number of time samples on the output, default 101 (odd)
# norm : normalize so that the correlation at lag 0 (its maximum) is 1, default True
# sym : the output is from lag -(ntout - 1) // 2 to + (ntout - 1) // 2, default True. Otherwise it is from lag 0.
#
# The start time of the output is the time of the first lag. (suacor puts a start time of -ntout dt / 2 in a header
# value that is not the one that is used.)
acor = stage(_acor, parallelism='trace', name='acor', validate=True)


# ------------------------------------------------------------------------------------------------------------ conv
def _as_filter(filter):
    """(samples, start time) of a filter that is a trace or an array"""
    if hasattr(filter, 'header'):
        return np.array(filter, dtype=np.float32), filter.header['sample_start']
    samples = np.atleast_1d(np.asarray(filter, dtype=np.float32))
    if samples.ndim != 1 or samples.shape[0] == 0:
        raise ValueError("The filter must be a Trace, or a 1D array with some samples.")
    return samples, 0.0


def _conv(upstream, filter, *, panel=False):
    source = as_trace_iterator(upstream)

    def convolve(trace, samples, start):
        out = _convolve(np.asarray(trace), samples)
        return trace.replace(out, sample_start=trace.header['sample_start'] + start)

    if not panel:
        samples, start = _as_filter(filter)
        return per_trace(source, lambda trace: convolve(trace, samples, start), on_complex='native')

    # a filter for each trace
    def gen():
        filters = iter(filter)
        for trace in source:
            one = next(filters, None)
            if one is None:
                warnings.warn("There are fewer filters than traces, the later traces are left as they are.")
                yield trace
                continue
            yield convolve(trace, *_as_filter(one))

    return from_iterable(gen(), n_traces=source.n_traces)


def conv(filter, *, panel=False):
    """Convolution with a filter (SUCONV).

    Parameters
    ----------
    filter : array or Trace
        The filter. If it is a Trace its start time is added to the start times of the traces.
    panel : bool
        ``filter`` is an iterable of Traces (or arrays), one for each trace, which are matched up in order. This
        needs to know the position of each trace, so it can not be split across workers.

    The traces that come out are ``len(filter) - 1`` samples longer. The sample intervals of the filter and the
    traces are assumed to be the same.
    """
    if not panel:
        _as_filter(filter)  # check it now
    return Stage(_conv, filter, parallelism='serial' if panel else 'trace', name='conv', panel=panel)


# ------------------------------------------------------------------------------------------------------------ xcor
def _xcor(upstream, filter, *, first=True, vibroseis=0):
    samples, filter_start = _as_filter(filter)
    n_filter = samples.shape[0]

    def xcor_trace(trace):
        x = np.asarray(trace)
        n = x.shape[0]
        if vibroseis > 0:
            n_out, first_lag = vibroseis, 0
        else:
            n_out = n + n_filter - 1
            first_lag = -n_filter + 1 if first else -n + 1
        if first:
            z = _correlate(samples, x, first_lag, n_out)
            shift = trace.header['sample_start'] - filter_start
        else:
            z = _correlate(x, samples, first_lag, n_out)
            shift = filter_start - trace.header['sample_start']
        return trace.replace(z, sample_start=first_lag * trace.d_sample + shift)

    return per_trace(upstream, xcor_trace)


def xcor(filter, *, first=True, vibroseis=0):
    """Cross-correlation with a filter (SUXCOR).

    Parameters
    ----------
    filter : array or Trace
        What to correlate every trace with.
    first : bool
        The filter is the first element of the correlation (``z[i] = sum_j filter[j] * trace[i+j]``), otherwise the
        second.
    vibroseis : int
        If more than 0, correlating vibroseis data with a sweep: the output has this many samples and starts at lag 0,
        rather than having all of the lags.
    """
    kwargs = dict(first=first, vibroseis=vibroseis)
    _xcor((), filter, **kwargs)  # check the parameters now
    return Stage(_xcor, filter, parallelism='trace', name='xcor', **kwargs)


# ------------------------------------------------------------------------------------------------------- acorfrac
def _acorfrac(upstream, *, a=0.0, b=0.0, ntout=None, sym=False):
    if ntout is not None and ntout < 1:
        raise ValueError("ntout must be at least 1")

    def acorfrac_trace(trace):
        x = np.asarray(trace)
        nt = x.shape[0]
        if nt == 0:
            return trace
        n_out = nt if ntout is None else ntout
        # padded so that the correlation does not wrap around (sufracacor's is a length that is fast for its FFT)
        n_fft = 2 * nt
        if n_out > n_fft:
            raise ValueError(f"ntout={n_out} is more than the {n_fft} samples that the transform has")
        # the SU transform has the conjugate of numpy's kernel
        spectrum = np.ascontiguousarray(np.conj(np.fft.rfft(x.astype(np.float64), n_fft)), dtype=np.complex64)
        _cwp.acorfrac_spectrum(spectrum, a, b, False)  # (the function of the SU library)
        y = np.fft.irfft(np.conj(spectrum), n_fft)
        if sym:
            # the lags -(n_out - 1) / 2 to (n_out - 1) / 2, around the zero lag
            first = -((n_out - 1) // 2)
            y = y[(first + np.arange(n_out)) % n_fft]
        else:
            y = y[:n_out]
        return trace.replace(y.astype(np.float32))

    return per_trace(upstream, acorfrac_trace)


# SUACORFRAC: general fractional autocorrelation / convolution, in the frequency domain.
#
# a : the exponent of the amplitude, default 0
# b : multiplier of the phase, default 0
# ntout : the number of samples that are output, default the number of samples of the trace
# sym : the output is from lag -(ntout - 1) / 2 to lag +(ntout - 1) / 2 (zero lag in the middle), not from lag 0
#
# Aout exp(-i Pout) = Ain^(1 + a) exp(-i (1 - b) Pin) (suacorfrac writes it with 1 + b, but its (a, b) = (1, 1) is the
# autocorrelation, which has no phase, and (1, -1) the autoconvolution, which has twice the phase). (a, b) = (1, 1) is the autocorrelation, (.5, .5) the half
# autocorrelation, (0, 0) leaves the data as it is, (.5, -.5) is the half autoconvolution and (1, -1) the autoconvolution.
#
# Where suacorfrac says what it does, this does that: its dopow() returns 1 whenever a is 0 (so that b has no effect then),
# it does not divide its inverse FFT by the length of the transform (so that (0, 0) does not change the data), it
# does not use ntout at all, and its sym is a sign change of every other frequency, which is a shift by half of a longer
# padded trace. The transform is padded to twice the length of the trace.
acorfrac = stage(_acorfrac, parallelism='trace', name='acorfrac', validate=True)


# --------------------------------------------------------------------------------------------------------- refcon
def _refcon(upstream, forshot, *, xy=0):
    if xy < 0:
        raise ValueError(f"xy={xy} must not be negative")
    source = as_trace_iterator(upstream)

    def traces():
        forward = iter(forshot)
        current = None
        for _ in range(xy + 1):
            current = next(forward, None)
            if current is None:
                raise ValueError("Can't get the first requested forward trace.")
        for reverse in source:
            out = _convolve(np.asarray(current), np.asarray(reverse))
            # (the sample interval of the output is half that of the input)
            yield reverse.replace(out, d_sample=reverse.d_sample / 2)
            current = next(forward, None)
            if current is None:
                return

    return from_iterable(traces())


def refcon(forshot, *, xy=0):
    """Convolve each trace (the reverse shot) with a trace of a forward shot (SUREFCON), for the refraction convolution
    section method of Palmer and Jones.

    Parameters
    ----------
    forshot : iterable of Trace
        The forward shot traces (SU's ``sufile``). Trace ``n`` of the stream is convolved with trace ``n + xy`` of this
        one, until either of them is out of traces. It is read once, so a one-shot iterator can be used with this
        stage once.
    xy : int
        The number of traces that the forward shot is offset from the first trace, default 0.

    The traces that come out are ``nt + nforshot - 1`` samples long, and have half the sample interval of the input (the
    SU program does that to them too).
    """
    if xy < 0:
        raise ValueError(f"xy={xy} must not be negative")
    return Stage(_refcon, forshot, parallelism='serial', name='refcon', xy=xy)
