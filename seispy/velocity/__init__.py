"""
Velocity analysis: SUVELAN and SURELAN (``su/main/velocity_analysis``).

``velan`` is the stacking velocity semblance of CDP gathers, and ``relan`` the residual moveout semblance of migrated gathers
(Liu's velocity analysis). A gather is the run of traces that have the same value of ``key`` (a name from ``trace.header``, or
a function of a trace, like the keys of ``seispy.parallel.group_by``): by default ``ensemble_number``, the CDP of SU. Each
gather makes ``nv`` (or ``nr``) traces, one for each velocity (or r parameter): the semblance as a function of time (or depth),
with ``ensemble_trace_number`` the number of the velocity from 1, so that its velocity is ``fv + (number - 1) * dv``. They have
the header of the first trace of their gather, at zero offset (the source and receiver at the midpoint), and ``d_sample`` is
that of the traces, times the ratio.

A stage can be split between gathers across workers: ``pmap(velan(), key='ensemble_number')``. What is done to the traces
of a gather is done by functions of the SU library (``_kernels.pyx``).

Where the program plainly does not do what its documentation says this does what the documentation says: the smoothing
window of SUVELAN does not have its last sample (SURELAN has it), and the samples that its NMO could not compute are left out.
The output traces of the programs take the header of the first trace of the next gather; these take that of their own.
"""
import collections
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stacking import _at_midpoint
from ..stage import stage
from . import _kernels

__all__ = ['velan', 'relan', 'intvel', 'stkvel', 'IntervalModel', 'StackingModel']


def _gather_samples(group, name):
    arrays = []
    for trace in group:
        x = np.asarray(trace)
        if x.dtype.kind == 'c':
            raise TypeError(f"{name} works on real traces, but was given a complex one.")
        arrays.append(np.ascontiguousarray(x, dtype=np.float32))
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("The traces of a gather must have the same number of samples.")
    return arrays


def _semblance_traces(group, sums, n_rows, ratio, nsmooth, pwr):
    """The semblance traces of a gather, from the sums"""
    sem = _kernels.semblance(*sums, ratio, nsmooth, pwr)
    first = group[0]
    sample_dt = first.d_sample * ratio
    for i in range(n_rows):
        yield first.replace(sem[i], d_sample=sample_dt, ensemble_trace_number=i + 1, **_at_midpoint(first))


# ----------------------------------------------------------------------------------------------------------- velan
def _velan(upstream, *, nv=50, dv=50.0, fv=1500.0, anis1=0.0, anis2=0.0, smute=1.5, dtratio=5, nsmooth=None, pwr=1.0,
           key='ensemble_number'):
    if nv < 1:
        raise ValueError("nv must be at least 1")
    if smute <= 1.0:
        raise ValueError("smute must be greater than 1.0")
    if dtratio < 1:
        raise ValueError("dtratio must be at least 1")
    if pwr < 0.0:
        raise ValueError("we are not looking for noise: pwr < 0")
    if pwr == 0.0:
        raise ValueError("we are creating an all-white semblance: pwr = 0")
    if nsmooth is None:
        nsmooth = dtratio * 2 + 1
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _gather_samples(group, 'velan')
            nt = arrays[0].shape[0]
            dt = group[0].d_sample
            if not dt:
                raise ValueError("The trace has no sample interval.")
            ft = group[0].header['sample_start']
            sums = _kernels.sums(nv, nt)
            no_moveout = False
            for trace, x in zip(group, arrays):
                status = _kernels.velan_accumulate(x, dt, ft, trace.header['offset'], nv, dv, fv, anis1, anis2, smute,
                                                   *sums)
                if status < 0:
                    raise ValueError("anis2 too small")
                no_moveout = no_moveout or status > 0
            if no_moveout:
                warnings.warn("no moveout; check anis1 and anis2")
            yield from _semblance_traces(group, sums, nv, dtratio, nsmooth, pwr)

    return from_iterable(traces())


# SUVELAN: stacking velocity semblance of CDP gathers
#
# nv, dv, fv : the number of velocities, the interval between them, and the first one, default 50, 50 and 1500
# anis1, anis2 : the quartic term of the traveltime, t^2 = t0^2 + x^2 / v^2 + anis1 x^4 / (1 + anis2 x^2), default 0
# smute : samples with NMO stretch exceeding smute are zeroed, default 1.5
# dtratio : the ratio of the output to the input time sampling intervals, default 5
# nsmooth : the length of the smoothing window of the numerators and denominators, default 2 dtratio + 1
# pwr : the semblance to this power, default 1
# key : the header value that the traces of a gather share, default 'ensemble_number'
velan = stage(_velan, parallelism='ensemble', name='velan', validate=True)


# ----------------------------------------------------------------------------------------------------------- relan
def _relan(upstream, *, nr=51, dr=0.01, fr=-0.25, smute=1.5, dzratio=5, nsmooth=None, key='ensemble_number'):
    if nr < 1:
        raise ValueError("nr must be at least 1")
    if smute <= 1.0:
        raise ValueError("smute must be greater than 1.0")
    if dzratio < 1:
        raise ValueError("dzratio must be at least 1")
    if nsmooth is None:
        nsmooth = dzratio * 2 + 1
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _gather_samples(group, 'relan')
            nz = arrays[0].shape[0]
            dz = group[0].d_sample
            if not dz:
                raise ValueError("The trace has no sample interval.")
            fz = group[0].header['sample_start']
            sums = _kernels.sums(nr, nz)
            for trace, x in zip(group, arrays):
                _kernels.relan_accumulate(x, dz, fz, trace.header['offset'], nr, dr, fr, smute, *sums)
            yield from _semblance_traces(group, sums, nr, dzratio, nsmooth, 1.0)

    return from_iterable(traces())


# SURELAN: residual moveout semblance of migrated CDP gathers, z(h)^2 = z(0)^2 + r h^2 for the depth z and the offset h
# (Z. Liu's velocity analysis)
#
# nr, dr, fr : the number of r parameters, the interval between them, and the first one, default 51, .01 and -.25
# smute : samples with residual moveout stretch exceeding smute are zeroed, default 1.5
# dzratio : the ratio of the output to the input depth sampling intervals, default 5
# nsmooth : the length of the smoothing window of the numerators and denominators, default 2 dzratio + 1
# key : the header value that the traces of a gather share, default 'ensemble_number'
#
# The traces are in depth: their sample interval and first sample are the depth ones.
relan = stage(_relan, parallelism='ensemble', name='relan', validate=True)


# ------------------------------------------------------------------------------------------------- intvel and stkvel
IntervalModel = collections.namedtuple('IntervalModel', 'thickness velocity times')
StackingModel = collections.namedtuple('StackingModel', 'times velocity')


def intvel(vs, t0):
    """The interval velocity model of a stacking velocity model (SUINTVEL), by the formulas of Schneider (1984).

    Parameters
    ----------
    vs : array
        The stacking velocities, positive.
    t0 : array
        The normal incidence (two-way) times, increasing.

    Returns
    -------
    ``IntervalModel(thickness, velocity, times)``: the thickness and the velocity of each layer (``times`` are the ``t0``).
    """
    vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))
    t0 = np.atleast_1d(np.asarray(t0, dtype=np.float64))
    if vs.ndim != 1 or vs.shape != t0.shape or vs.shape[0] == 0:
        raise ValueError("vs and t0 must be 1D arrays of the same length")
    if (vs <= 0).any():
        raise ValueError(f"vs's must be positive: vs[{int(np.argmax(vs <= 0))}] = {vs[vs <= 0][0]}")
    if (t0 < 0).any():
        raise ValueError(f"t0's must be positive: t0[{int(np.argmax(t0 < 0))}] = {t0[t0 < 0][0]}")
    if (np.diff(t0) <= 0).any():
        raise ValueError("the t0's must increase")
    h = np.empty_like(vs)
    v = np.empty_like(vs)
    h[0] = 0.5 * vs[0] * t0[0]
    v[0] = vs[0]
    half = 0.5 * t0
    dt = np.diff(half)
    dvh = np.diff(vs * vs * half)
    if (dvh <= 0).any():
        raise ValueError("The stacking velocities do not make a model of interval velocities (the layer needs v h > 0)")
    h[1:] = np.sqrt(dvh * dt)
    v[1:] = np.sqrt(dvh / dt)
    return IntervalModel(h, v, t0)


def stkvel(v, h, dip=0.0):
    """The stacking velocity model of a model of layers of constant dip (SUSTKVEL), by the formulas of Schneider (1984).

    Parameters
    ----------
    v, h : arrays
        The interval velocities and the thicknesses of the layers (at the cmp), positive.
    dip : float
        The (constant) dip of the layers, degrees.

    Returns
    -------
    ``StackingModel(times, velocity)``: the zero incidence times and the stacking velocities that ``sunmo`` needs (``tnmo``
    and ``vnmo``).
    """
    v = np.atleast_1d(np.asarray(v, dtype=np.float64))
    h = np.atleast_1d(np.asarray(h, dtype=np.float64))
    if v.ndim != 1 or v.shape != h.shape or v.shape[0] == 0:
        raise ValueError("v and h must be 1D arrays of the same length")
    if (v <= 0).any():
        raise ValueError("v's must be positive")
    if (h <= 0).any():
        raise ValueError("h's must be positive")
    cosdip = np.cos(np.radians(dip))
    if cosdip <= 0:
        raise ValueError("dip must be under 90 degrees")
    t0by2 = np.cumsum(h / v)
    vh = np.cumsum(v * h)
    return StackingModel(2.0 * t0by2 * cosdip, np.sqrt(vh / t0by2) / cosdip)
