"""
SUTAUP: forward and inverse slant stacks (tau-p transforms), in the t-x and F-K domains (``su/main/transforms``).

The traces of the stream are one panel, held in memory. The transforms are the routines of the SU library (``par/lib/taup.c``,
``_taup.pyx``). The F-X routines of that file, with Beylkin's approach, are not part of SUTAUP.

Where the program plainly does not do what its documentation says this does what the documentation says: the program has room
for ``np`` output traces whichever way it transforms, so an inverse transform to more traces than there are slopes writes beyond
what it allocated. The output has the traces that are asked for (``nx`` for the inverse, ``np`` for the forward transform).
"""
import numpy

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage
from . import _taup

__all__ = ['taup']


def _taup_stage(upstream, *, option=1, dt=None, nx=None, dx=1.0, npoints=71, pmin=0.0, pmax=0.006, np=None, fmin=3.0,
                xmin=0.0):
    if option not in (1, 2, 3, 4):
        raise ValueError("option flag has to be between 1 and 4")
    if np is not None and np < 2:
        raise ValueError("np must be at least 2")
    if npoints < 1:
        raise ValueError("npoints must be at least 1")
    source = as_trace_iterator(upstream)

    def traces():
        traces = list(source)
        if not traces:
            return
        ntr = len(traces)
        nt = traces[0].n_sample
        if any(t.n_sample != nt for t in traces):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in traces):
            raise TypeError("taup works on real traces, but was given a complex one.")
        sample_dt = dt if dt is not None else traces[0].d_sample
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        n_slopes = ntr if np is None else np
        n_x = ntr if nx is None else nx
        if n_slopes < 2:
            raise ValueError("np must be at least 2 (the number of traces is the default)")
        dp = (pmax - pmin) / (n_slopes - 1)
        data = numpy.ascontiguousarray(numpy.stack([numpy.asarray(t) for t in traces]), dtype=numpy.float32)
        out = _taup.slant_stack(option, data, sample_dt, n_x, xmin, dx, n_slopes, pmin, dp, fmin, npoints)
        for i, row in enumerate(out):
            yield traces[min(i, ntr - 1)].replace(row, trace_id=i + 1, d_sample=sample_dt)

    return from_iterable(traces())


def taup(option=1, *, dt=None, nx=None, dx=1.0, npoints=71, pmin=0.0, pmax=0.006, np=None, fmin=3.0, xmin=0.0):
    """Forward and inverse t-x and F-K slant stacks (tau-p transforms) of the panel of traces of the stream (SUTAUP).

    The forward transforms take ``nx`` traces and make ``np``, one for each slope from ``pmin`` to ``pmax``. The inverse
    ones take ``np`` traces and make ``nx``. The slope sampling interval is ``(pmax - pmin) / (np - 1)``, and the
    slopes of the output (forward) are not in the trace headers.

    Parameters
    ----------
    option : {1, 2, 3, 4}
        1: forward F-K domain computation, 2: forward t-x domain, 3: inverse F-K domain, 4: inverse t-x domain.
    dt : float, optional
        The time sampling interval (s), by default that of the traces.
    nx : int, optional
        The number of horizontal samples (traces), by default the number of traces of the stream.
    dx : float
        The horizontal sampling interval (m).
    npoints : int
        Number of points of the rho filter (the inverse t-x transform).
    pmin, pmax : float
        The minimum and maximum slope (s/m).
    np : int, optional
        The number of slopes, by default the number of traces of the stream.
    fmin : float
        Minimum frequency of interest. (The F-K routines ignore it.)
    xmin : float
        The offset of the first trace.

    The cascade of a forward and an inverse transform keeps the relative amplitudes of a panel, but not the absolute
    amplitudes: a scale factor has to be applied to compare to the original.
    """
    kwargs = dict(dt=dt, nx=nx, dx=dx, npoints=npoints, pmin=pmin, pmax=pmax, np=np, fmin=fmin, xmin=xmin)
    _taup_stage((), option=option, **kwargs)  # check the parameters now
    return Stage(_taup_stage, option=option, parallelism='serial', name='taup', **kwargs)
