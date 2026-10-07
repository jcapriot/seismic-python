"""
Mixing and median filtering of a panel about a moveout curve: SUMEDIAN (``su/main/filters``), which suppresses events that
have a given moveout, with ``median`` (its ``median=1``) and ``medmix`` (``median=0``, a weighted moving average).

The panel is all of the traces that the stage is given. Each trace is shifted in time by the time of the curve at the value of
its ``key``, so that an event along the curve is flat; then at each time sample the traces around each trace are mixed or
their median is taken, which keeps the flat events; the result is shifted back and (by default) subtracted from the
traces. The traces at the ends of the panel have the traces beyond them reflected about the end trace.

Not supported: the ``xfile``, ``tfile`` and ``nshift`` parameters (give arrays), ``tmpdir`` and ``verbose``. The
shifts are whole numbers of samples (sumedian rounds them, though its documentation says that it interpolates).
"""
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, header_value
from . import _kernels

__all__ = ['median', 'medmix']

_DEFAULT_MIX = (0.6, 1.0, 1.0, 1.0, 0.6)


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _curve(xshift, tshift):
    xshift = np.atleast_1d(np.asarray(xshift, dtype=np.float64))
    tshift = np.atleast_1d(np.asarray(tshift, dtype=np.float64))
    if xshift.ndim != 1 or xshift.shape != tshift.shape or xshift.shape[0] == 0:
        raise ValueError("xshift and tshift must be 1D arrays of the same length")
    if (np.diff(xshift) <= 0).any():
        raise ValueError("xshift must increase")
    return xshift, tshift


def _filter_panel(upstream, xshift, tshift, key, sign, subtract, combine, half_width):
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        if any(t.dtype.kind == 'c' for t in panel):
            raise TypeError("This stage works on real traces, but was given a complex one.")
        arrays = [np.asarray(t, dtype=np.float64) for t in panel]
        n = arrays[0].shape[0]
        if any(a.shape[0] != n for a in arrays):
            raise ValueError("The traces must all have the same number of samples.")
        dt = panel[0].d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")

        # the shift of each trace, in samples: the time of the curve at its key, relative to its first sample
        shifts = []
        for trace in panel:
            first = trace.header['sample_start']
            time = np.interp(header_value(trace, key), xshift, tshift, left=first, right=tshift[-1])
            shifts.append(_nint((time - first) / dt))
        shift_max = max(shifts)
        offsets = np.array([shift_max + sign * s for s in shifts])
        offsets = offsets - min(0, offsets.min())  # (all of them in the panel)
        width = n + int(offsets.max())
        shifted = np.zeros((len(panel), width))
        for i, (a, offset) in enumerate(zip(arrays, offsets)):
            shifted[i, offset:offset + n] = a

        # the panel reflected about its end traces, for the window around each trace
        if half_width > 0:
            mode = 'reflect' if len(panel) > 1 else 'edge'
            padded = np.pad(shifted, ((half_width, half_width), (0, 0)), mode=mode)
        else:
            padded = shifted
        combined = combine(padded, len(panel))

        for i, trace in enumerate(panel):
            offset = int(offsets[i])
            row = shifted[i] - combined[i] if subtract else combined[i]
            yield trace.replace(row[offset:offset + n].astype(np.float32))

    return from_iterable(traces(), n_traces=source.n_traces)


def _median(upstream, xshift, tshift, *, key='trace_id', nmed=5, sign=-1, subtract=True):
    xshift, tshift = _curve(xshift, tshift)
    if nmed % 2 == 0:
        warnings.warn("increased nmed by 1 to ensure it is odd")
        nmed += 1

    def combine(padded, n_traces):
        # (the median across the traces of the window about each trace, by the function of the SU library)
        padded = np.ascontiguousarray(padded, dtype=np.float32)
        return np.array([_kernels.median_across(padded[i:i + nmed]) for i in range(n_traces)])

    return _filter_panel(upstream, xshift, tshift, key, sign, subtract, combine, (nmed - 1) // 2)


def _medmix(upstream, xshift, tshift, *, key='trace_id', mix=None, sign=-1, subtract=True):
    xshift, tshift = _curve(xshift, tshift)
    weights = np.asarray(_DEFAULT_MIX if mix is None else mix, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] % 2 == 0:
        raise ValueError("number of mixing coefficients must be odd")
    n_mix = weights.shape[0]
    weights = weights / n_mix

    # (the weight of the trace k places before the center is that of the one k after it, as in sumedian: the first weight is the
    # last trace of the window)
    window_weights = np.ascontiguousarray(weights[::-1], dtype=np.float32)

    def combine(padded, n_traces):
        # (the weighted sum across the traces of the window about each trace, by the function of the SU library)
        padded = np.ascontiguousarray(padded, dtype=np.float32)
        return np.array([_kernels.mix_across(padded[i:i + n_mix], window_weights) for i in range(n_traces)])

    return _filter_panel(upstream, xshift, tshift, key, sign, subtract, combine, (n_mix - 1) // 2)


def _check_sign(sign):
    if sign not in (1, -1):
        raise ValueError("sign must be -1 (an upward shift) or 1 (downward)")


def median(xshift, tshift, *, key='trace_id', nmed=5, sign=-1, subtract=True):
    """Median filter about a moveout curve (SUMEDIAN with median=1): suppresses the events that have that moveout.

    Parameters
    ----------
    xshift, tshift : arrays
        The curve: values of ``key`` that increase, and the times (s) of the moveout at them. Linearly interpolated, with
        the first time of the trace to the left, and the last of ``tshift`` to the right.
    key : header name or function
        Gives the position along the curve of each trace, default ``'trace_id'`` (SU's tracl). It can be ``'offset'``.
    nmed : int
        The (odd) number of traces to take the median of, default 5.
    sign : int
        -1 (the default) to shift the traces up, 1 to shift them down (for the upgoing events of a VSP).
    subtract : bool
        Subtract the filtered panel from the traces (the default, which removes the events along the curve), or output it.
    """
    _check_sign(sign)
    _curve(xshift, tshift)  # check the parameters now
    return Stage(_median, xshift, tshift, parallelism='ensemble', name='median', key=key, nmed=nmed, sign=sign,
                 subtract=subtract)


def medmix(xshift, tshift, *, key='trace_id', mix=None, sign=-1, subtract=True):
    """Weighted moving average about a moveout curve (SUMEDIAN with median=0): suppresses the events that have that
    moveout. The parameters are those of ``median``, with ``mix`` (the odd number of weights of the traces around each
    one, default ``[.6, 1, 1, 1, .6]``, divided by their number) in place of ``nmed``."""
    _check_sign(sign)
    _medmix((), xshift, tshift, key=key, mix=mix)  # check the parameters now
    return Stage(_medmix, xshift, tshift, parallelism='ensemble', name='medmix', key=key, mix=mix, sign=sign,
                 subtract=subtract)
