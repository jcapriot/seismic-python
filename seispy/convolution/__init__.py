"""
Convolution and correlation: SUACOR, SUCONV and SUXCOR (``su/main/convolution_correlation``).

All three are done in the time domain, as in the SU sources (with the ``convolve_cwp`` and ``xcor`` of the SU library),
by numpy's ``convolve`` and ``correlate``. The ``sufile`` options of SU become a ``Trace`` (or arrays), and suxcor's
``panel`` option, which correlates windows of a whole gather, is not supported.

The traces that come out are along a lag (or, for convolution, time) axis, so their start time is set to match.
sucor, suconv and suxcor leave ``delrt`` alone, except to add the one of the filter to that of the trace in suconv.
"""
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, per_trace, stage

__all__ = ['acor', 'conv', 'xcor']


def _correlate(x, y, first_lag, n_lags):
    """z[i] = sum_j x[j] y[i+j], for i = first_lag, ..., first_lag + n_lags - 1 (x and y are 0 outside of themselves)"""
    z = np.zeros(n_lags, dtype=np.float32)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return z
    full = np.correlate(y, x, mode='full')  # for the lags -(len(x) - 1) to len(y) - 1
    lowest = -(x.shape[0] - 1)
    lo = max(first_lag, lowest)
    hi = min(first_lag + n_lags, lowest + full.shape[0])
    if hi > lo:
        z[lo - first_lag:hi - first_lag] = full[lo - lowest:hi - lowest]
    return z


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
        out = np.convolve(np.asarray(trace), samples)
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
