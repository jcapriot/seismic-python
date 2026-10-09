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
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stacking import _at_midpoint
from ..stage import stage
from . import _kernels

__all__ = ['velan', 'relan']


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
