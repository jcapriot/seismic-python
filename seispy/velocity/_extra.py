"""
More velocity analysis, from ``su/main/velocity_analysis``: SUTIVEL (``tivel``), SUVEL2DF (``vel2df``), SURELANAN
(``relanan``) and the coherence measures of SUVELAN_UCCS, SUVELAN_NCCS, SUVELAN_USEL and SUVELAN_NSEL (``velan_uccs``,
``velan_nccs``, ``velan_usel`` and ``velan_nsel``).

A gather is the run of traces that have the same value of ``key`` (by default ``ensemble_number``, the CDP), as for
``velan``; each makes traces with ``ensemble_trace_number`` the number of the velocity from 1, at zero offset.

Where a program plainly does not do what its documentation says this does what the documentation says; the differences
are with each stage.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stacking import _at_midpoint
from ..stage import stage
from . import _more

__all__ = ['tivel', 'vel2df', 'relanan', 'velan_uccs', 'velan_nccs', 'velan_usel', 'velan_nsel']

F32 = np.float32


def _real_gather(group, name):
    arrays = []
    for trace in group:
        x = np.asarray(trace)
        if x.dtype.kind == 'c':
            raise TypeError(f"{name} works on real traces, but was given a complex one.")
        arrays.append(np.ascontiguousarray(x, dtype=np.float32))
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("The traces of a gather must have the same number of samples.")
    return arrays


def _sample_interval(trace, dt):
    sample_dt = trace.d_sample or dt
    if not sample_dt:
        raise ValueError("The trace has no sample interval, and no `dt` was given.")
    return sample_dt


# ------------------------------------------------------------------------------------------------------------ tivel
_P_OPTIONS = {1: 'vnmo', 4: 'v', 7: 'dv', 10: 'd2v'}
_DEGREE_OPTIONS = {2: 'vnmo', 5: 'v', 8: 'dv', 11: 'd2v', 13: 'test'}
_RADIAN_OPTIONS = {3: 'vnmo', 6: 'v', 9: 'dv', 12: 'd2v'}


def tivel(a=2500.0, b=1250.0, e=0.20, d=0.10, *, maxangle=90.0, nangle=9001, np_values=8001, option=1, normalize=False):
    """Velocity tables of a transversely isotropic medium, from Thomsen's parameters (SUTIVEL).

    Parameters
    ----------
    a, b : float
        Alpha and beta: the vertical P and SV velocities (m/s), default 2500 and 1250.
    e, d : float
        Epsilon and delta, default .2 and .1.
    maxangle : float
        The greatest angle (degrees), default 90.
    nangle : int
        The number of angles, default 9001.
    np_values : int
        The number of ray parameters of the tables as a function of ``p`` (``np`` in the program), default 8001.
    option : int
        1 vnmo(p) (what TI DMO uses), 2 vnmo(theta) in degrees, 3 vnmo(theta) in radians, 4 vphase(p), 5 and 6 vphase(theta),
        7 first derivative of vphase(p), 8 and 9 of vphase(theta), 10 second derivative of vphase(p), 11 and 12 of
        vphase(theta), 13 ``p^2 + 1 / vnmo(theta)^2 - 1 / vnmo(0)^2`` against theta (zero for all theta when d = e), 14
        vnmo(p) for weak anisotropy.
    normalize : bool
        Scale vnmo by the cosine, and vphase by ``1 / sqrt(1 + 2 e sin^2(theta))``.

    Returns
    -------
    (increment, values) : the increment of the abscissa (p, or the angle: degrees for options 2, 5, 8, 11 and 13,
        radians for 3, 6, 9 and 12, always starting at 0), and the values (float64).

    The program does not output options 7 and 10 (it leaves them out of the places that tables as a function of p are
    written): they are here.
    """
    if option not in range(1, 15):
        raise ValueError(f"option={option} must be from 1 to 14")
    if option == 14:
        p_max = 1.0 / (a * np.sqrt(1 + 2 * e))
        dp = p_max / (np_values - 1)
        for ip in range(np_values):
            if _more.tivel_vnmoweak(a, b, e, d, ip * dp) < 0.0:
                p_max = (ip - 1) * dp
                break
        dp = p_max / (np_values - 1)
        return dp, np.array([_more.tivel_vnmoweak(a, b, e, d, ip * dp) for ip in range(np_values)])
    n_angle = max(int(nangle), 2)
    p, v, dv, d2v, vnmo = _more.tivel_table(a, b, e, d, maxangle, n_angle, bool(normalize))
    d_angle = maxangle / (n_angle - 1)
    if option in _P_OPTIONS:
        column = {'vnmo': vnmo, 'v': v, 'dv': dv, 'd2v': d2v}[_P_OPTIONS[option]]
        p_top = float(p.max())
        dp = p_top / (np_values - 4)
        grid = np.arange(np_values) * np.float32(dp)
        # (linear interpolation, with the end values held)
        return dp, np.interp(grid, p, column, left=column[0], right=column[-1])
    if option == 13:
        values = p.astype(np.float64) ** 2 + 1.0 / vnmo.astype(np.float64) ** 2 - 1.0 / float(vnmo[0]) ** 2
    else:
        name = (_DEGREE_OPTIONS if option in _DEGREE_OPTIONS else _RADIAN_OPTIONS)[option]
        values = {'vnmo': vnmo, 'v': v, 'dv': dv, 'd2v': d2v}[name].astype(np.float64)
    increment = d_angle if option in _DEGREE_OPTIONS else d_angle * np.pi / 180.0
    return increment, values


# ----------------------------------------------------------------------------------------------------------- vel2df
def _vel2df(upstream, *, tn, offsetm=None, nv=50, dv=50.0, fv=1500.0, nvh=50, dvh=50.0, fvh=1500.0, xod=1.5, dtratio=5,
            nsmooth=None, vavg=None, dt=None, key='ensemble_number'):
    if nv < 1 or nvh < 1:
        raise ValueError("nv and nvh must be at least 1")
    if dtratio < 1:
        raise ValueError("dtratio must be at least 1")
    nsmooth = dtratio * 2 + 1 if nsmooth is None else nsmooth
    average = fv + 0.5 * (nv - 1) * dv if vavg is None else vavg
    # (the documentation has the maximum offset as a required parameter, offsetm, which the program asks for and does
    # not use: it works one out from the offset to depth ratio)
    offset_max = offsetm if offsetm is not None else 0.5 * tn * xod * average
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _real_gather(group, 'vel2df')
            first = group[0]
            sample_dt = _sample_interval(first, dt)
            ft = first.header['sample_start']
            shape = (nv, nvh, nsmooth)
            num, den, nnz = (np.zeros(shape, dtype=np.float32) for _ in range(3))
            for trace, x in zip(group, arrays):
                offset = trace.header['offset']
                if abs(offset) > offset_max:
                    continue
                _more.vel2df_accumulate(x, F32(sample_dt), F32(ft), F32(offset), F32(tn), nv, F32(dv), F32(fv), nvh,
                                        F32(dvh), F32(fvh), nsmooth, num, den, nnz)
            numerator = (num.astype(np.float64) ** 2).sum(axis=2)
            denominator = (nnz.astype(np.float64) * den).sum(axis=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                panel = np.where(denominator != 0.0, numerator / np.where(denominator != 0.0, denominator, 1.0), 0.0)
            for iv in range(nv):
                yield first.replace(panel[iv].astype(np.float32), d_sample=dvh, sample_start=fvh,
                                    ensemble_trace_number=iv + 1, **_at_midpoint(first))

    return from_iterable(traces())


# SUVEL2DF: stacking velocity semblance for one time (tn), over the NMO velocity and the horizontal velocity (eta
# analysis), for CDP gathers
#
# tn : the zero offset time of the reflection (s)
# offsetm : the greatest offset that is used, by default 0.5 tn xod vavg
# nv, dv, fv : the number, interval and first of the NMO velocities, default 50, 50 and 1500
# nvh, dvh, fvh : the same for the horizontal velocities
# xod : the maximum offset to depth ratio to resolve (for the default of offsetm), default 1.5
# dtratio, nsmooth : the window of the semblance sums is nsmooth samples (default 2 dtratio + 1) around tn
# vavg : the average velocity of the search, by default fv + (nv - 1) dv / 2
# dt : the sample interval for traces that do not have one
# key : the header value that the traces of a gather share, default 'ensemble_number'
#
# Each gather makes nv traces of nvh samples (the semblance over the horizontal velocities, from fvh in steps of
# dvh: the sample interval of the trace and its start time): trace k is for the NMO velocity fv + (k - 1) dv.
vel2df = stage(_vel2df, parallelism='ensemble', name='vel2df', validate=True)


# --------------------------------------------------------------------------------------------------------- relanan
def _relanan(upstream, *, x, z, nr1=51, dr1=0.01, fr1=-0.25, nr2=51, dr2=0.01, fr2=-0.25, dzratio=5, nsmooth=None,
             method='linear', key='ensemble_number', position=None):
    if nr1 < 1 or nr2 < 1:
        raise ValueError("nr1 and nr2 must be at least 1")
    if dzratio < 1:
        raise ValueError("dzratio must be at least 1")
    nsmooth = dzratio * 2 + 1 if nsmooth is None else nsmooth
    x = np.ascontiguousarray(x, dtype=np.float32)
    z = np.ascontiguousarray(z, dtype=np.float32)
    if x.ndim != 1 or x.shape != z.shape or x.shape[0] < 1:
        raise ValueError("x and z are the picked points of the reflector: arrays of the same length")
    if method not in ('linear', 'mono', 'akima', 'spline'):
        raise ValueError(f"{method!r} is an unknown interpolation method (linear, mono, akima or spline)")
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _real_gather(group, 'relanan')
            first = group[0]
            dz = first.d_sample
            if not dz:
                raise ValueError("The trace has no sample interval.")
            fz = first.header['sample_start']
            where = position(first) if position is not None else first.header['ensemble_number']
            # the depth of the reflector at the position of this gather (the program has the one of the first gather
            # for all of them)
            depth = _more.interpolate(x, z, method, F32(where))
            shape = (nr1, nr2, nsmooth)
            num, den, nnz = (np.zeros(shape, dtype=np.float32) for _ in range(3))
            for trace, samples in zip(group, arrays):
                half_offset = trace.header['offset'] / 2.0  # (the program's is an integer division)
                _more.relanan_accumulate(samples, F32(dz), F32(fz), F32(half_offset), F32(depth), nr1, F32(dr1),
                                         F32(fr1), nr2, F32(dr2), F32(fr2), nsmooth, num, den, nnz)
            numerator = (num.astype(np.float64) ** 2).sum(axis=2)
            denominator = (nnz.astype(np.float64) * den).sum(axis=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                panel = np.where(denominator != 0.0, numerator / np.where(denominator != 0.0, denominator, 1.0), 0.0)
            for i in range(nr1):
                yield first.replace(panel[i].astype(np.float32), d_sample=dr2, sample_start=fr2,
                                    ensemble_trace_number=i + 1, **_at_midpoint(first))

    return from_iterable(traces())


# SURELANAN: residual moveout semblance analysis for anisotropic media, of migrated gathers (depth), for the depth of a
# reflector: z(h)^2 = z(0)^2 + r1 h^2 + r2 h^4 / (h^2 + z(0)^2), h the half offset
#
# x, z : the points of the picked reflector: the CDP and the depth of each (the program reads the file ``refl``, with
#        the pairs z, x)
# method : the interpolation of the reflector to the CDP of a gather: 'linear' (default), 'mono', 'akima' or 'spline'
# nr1, dr1, fr1 and nr2, dr2, fr2 : the number, interval and first of the r1 and r2 parameters, default 51, .01 and -.25
# dzratio, nsmooth : the semblance window is nsmooth samples (default 2 dzratio + 1) around the reflector
# key : the header value that the traces of a gather share, default 'ensemble_number'
# position : the position that x is for (a header name or function of a trace), default the ensemble number
#
# Each gather makes nr1 traces of nr2 samples (the semblance as a function of r2 from fr2 in steps of dr2): trace k is
# for r1 = fr1 + (k - 1) dr1.
relanan = stage(_relanan, parallelism='ensemble', name='relanan', validate=True)


# ------------------------------------------------------------------------------------------------- velan cc sums
def _velan_cc_stage(upstream, *, normalize, selective, name, nv=50, dv=100.0, fv=1500.0, smute=1.5, dtratio=5,
                    nsmooth=None, pwr=1.0, tau=0.5, dt=None, key='ensemble_number'):
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
    if selective and not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be from 0 to 1")
    nsmooth = dtratio * 2 + 1 if nsmooth is None else nsmooth
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _real_gather(group, name)
            first = group[0]
            sample_dt = _sample_interval(first, dt)
            ft = first.header['sample_start']
            offsets = np.array([t.header['offset'] for t in group], dtype=np.float32)
            panel = _more.velan_cc(np.stack(arrays), offsets, F32(sample_dt), F32(ft), nv, F32(dv), F32(fv), F32(smute),
                                   dtratio, nsmooth, normalize, selective, F32(tau if selective else 0.0), F32(pwr))
            for iv in range(nv):
                yield first.replace(panel[iv], d_sample=sample_dt * dtratio, ensemble_trace_number=iv + 1,
                                    **_at_midpoint(first))

    return from_iterable(traces())


def _velan_uccs(upstream, **kwargs):
    return _velan_cc_stage(upstream, normalize=False, selective=False, name='velan_uccs', **kwargs)


def _velan_nccs(upstream, **kwargs):
    return _velan_cc_stage(upstream, normalize=True, selective=False, name='velan_nccs', **kwargs)


def _velan_usel(upstream, **kwargs):
    return _velan_cc_stage(upstream, normalize=False, selective=True, name='velan_usel', **kwargs)


def _velan_nsel(upstream, **kwargs):
    return _velan_cc_stage(upstream, normalize=True, selective=True, name='velan_nsel', **kwargs)


# The coherence of the traces of a gather (after NMO) from the sum of the cross-correlations of their pairs, in windows
# of nsmooth samples:
#
# velan_uccs : the unnormalized sum of all of the pairs
# velan_nccs : each pair divided by the geometric mean of the energies of its traces in the window, and the sum by the
#              number of pairs
# velan_usel, velan_nsel : the sums of only the pairs that have a large differential moveout (the same, unnormalized and
#              normalized): those where the difference of the squares of the offsets is at least tau times that of the
#              gather (tau 0 is all of the pairs, 1 only the shortest and longest offsets), default .5
#
# nv, dv, fv : the number, interval and first of the velocities, default 50, 100 and 1500
# smute, dtratio, nsmooth, pwr : as for velan; nsmooth is the window of the sums (default 2 dtratio + 1), pwr a power of the
#                                values (with their sign)
# key : the header value that the traces of a gather share, default 'ensemble_number'
#
# The programs only make one trace for a gather and, with the sums of the gathers that came before it left in them, do
# not use ``pwr`` or ``nx`` (the number of the traces of a gather is counted), and add the squares of the traces (not
# pairs) to the sums, which their documentation does not have: here a gather makes nv traces of the sum of the pairs.
velan_uccs = stage(_velan_uccs, parallelism='ensemble', name='velan_uccs', validate=True)
velan_nccs = stage(_velan_nccs, parallelism='ensemble', name='velan_nccs', validate=True)
velan_usel = stage(_velan_usel, parallelism='ensemble', name='velan_usel', validate=True)
velan_nsel = stage(_velan_nsel, parallelism='ensemble', name='velan_nsel', validate=True)
