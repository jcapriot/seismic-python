"""
Operations on a whole data set, or on two: SUFLIP and SUVCAT (``su/main/operations``).

``flip`` turns the data set, which is a matrix of traces by samples, over (the whole data set is kept in memory). ``vcat`` joins
the traces of a second data set onto the ends of the traces of the stream, with an overlap.
"""
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage

__all__ = ['flip', 'vcat']

_FLIPS = (-1, 0, 1, 2, 3)


# ----------------------------------------------------------------------------------------------------------- flip
def _flipped(data, how):
    """The data set (traces by samples) flipped as SUFLIP does"""
    if how == -1:  # 90 degrees counter-clockwise
        return data.T[:, ::-1]
    if how == 0:  # transposed
        return data.T
    if how == 1:  # 90 degrees clockwise
        return data.T[::-1, :]
    if how == 2:  # right to left: the order of the traces
        return data[::-1, :]
    return data[:, ::-1]  # 3, top to bottom: the order of the samples


def _flip(upstream, *, flip=1):
    if flip not in _FLIPS:
        raise ValueError(f"flip = {flip!r}, it must be -1, 0, 1, 2, or 3")
    source = as_trace_iterator(upstream)

    def traces():
        traces = list(source)
        if not traces:
            return
        if len({t.n_sample for t in traces}) != 1:
            raise ValueError("The traces must all have the same number of samples to flip them.")
        flipped = np.ascontiguousarray(_flipped(np.stack([np.asarray(t) for t in traces]), flip))
        for i, row in enumerate(flipped):
            # (the headers are taken in order, not flipped with the data; a data set with more samples than traces
            # that is turned over has more traces to make than headers to give them, which reuse the last)
            yield traces[min(i, len(traces) - 1)].replace(row, trace_id=i + 1)

    return from_iterable(traces())


def flip(flip=1):
    """Flip a data set (a matrix of traces by samples) in various ways (SUFLIP).

    Parameters
    ----------
    flip : {-1, 0, 1, 2, 3}
        1: 90 degrees clockwise, -1: 90 degrees counter-clockwise, 0: transpose (the traces become the samples),
        2: flip right to left (reverse the order of the traces), 3: flip top to bottom (reverse every trace).

    The whole data set is held in memory. The headers are those of the traces in order, with the trace number from 1 (the
    sample interval is not changed by the turns, though it is meaningless there: set it again if you need it). The traces
    must all have the same number of samples.
    """
    kwargs = dict(flip=flip)
    _flip((), **kwargs)  # check the parameters now
    return Stage(_flip, parallelism='serial', name='flip', **kwargs)


# ----------------------------------------------------------------------------------------------------------- vcat
_TAPTYPES = (0, 1, 2, 3)


def _overlap(top, bottom, taptype):
    """The samples where the end of `top` and the start of `bottom` overlap, combined"""
    n = top.shape[0]
    if taptype == 0:
        return (top + bottom) * np.float32(0.5)
    if taptype == 1:
        return np.where(np.abs(top) > np.abs(bottom), top, bottom)
    if taptype == 2:
        # (cos(x) of the top and 1 - cos(x) of the bottom, with x from 0 to pi / 2 across the overlap)
        x = np.arange(n) / (n - 1) if n > 1 else np.zeros(1)
        s1 = np.cos(0.5 * np.pi * x).astype(np.float32)
        return s1 * top + (np.float32(1.0) - s1) * bottom
    return top + bottom


def _vcat(upstream, other, *, taplen=0, taptype=0):
    if taptype not in _TAPTYPES:
        raise ValueError("taptype must be 0, 1, 2, or 3")
    if taplen < 0:
        raise ValueError("taplen must not be negative")
    source = as_trace_iterator(upstream)
    second = as_trace_iterator(other)

    def traces():
        pairs = 0
        iterator = iter(second)
        for first in source:
            last = next(iterator, None)
            if last is None:
                warnings.warn(f"The stream still had traces when the second data set was exhausted ({pairs} pairs of traces)")
                return
            pairs += 1
            x, y = np.asarray(first), np.asarray(last)
            if taplen > x.shape[0] or taplen > y.shape[0]:
                raise ValueError(f"taplen={taplen} is more than the number of samples of the traces "
                                 f"({x.shape[0]} and {y.shape[0]})")
            head = x[:x.shape[0] - taplen]
            tail = y[taplen:]
            if taplen:
                middle = _overlap(x[x.shape[0] - taplen:], y[:taplen], taptype)
                data = np.concatenate([head, middle.astype(x.dtype, copy=False), tail])
            else:
                data = np.concatenate([head, tail])
            yield first.replace(data.astype(np.result_type(x, y), copy=False))
        if next(iterator, None) is not None:
            warnings.warn(f"The second data set still had traces when the stream was exhausted ({pairs} pairs of traces)")

    return from_iterable(traces())


def vcat(other, *, taplen=0, taptype=0):
    """Append the traces of another data set to the ends of the traces of the stream, with an overlap (SUVCAT).

    The traces of the stream and of ``other`` are matched up in order (and it stops when either runs out); each output trace
    is the trace of the stream, then that of ``other``, with ``taplen`` samples where they overlap.

    Parameters
    ----------
    other : iterable of traces
        The data set to append to (the second file of suvcat).
    taplen : int
        Number of samples of the overlap.
    taptype : {0, 1, 2, 3}
        How the overlap is made: 0 the average, 1 the sample of the larger magnitude, 2 a cosine scaled average (from the
        trace of the stream at the start of the overlap to the other at its end, as the documentation of SUVCAT says: the
        program has an integer division in its weights that makes them 1 until the last sample), 3 the sum.
    """
    kwargs = dict(taplen=taplen, taptype=taptype)
    _vcat((), (), **kwargs)  # check the parameters now
    return Stage(_vcat, other, parallelism='serial', name='vcat', **kwargs)
