"""
Kirchhoff datuming (``su/main/datuming``): SUDATUMK2DR and SUDATUMK2DS (``datumk2dr``, ``datumk2ds``: receivers of shot
gathers, sources of receiver gathers, for irregular surfaces given by splines) and SUKDMDCR and SUKDMDCS (``kdmdcr``,
``kdmdcs``: the 2.5-D data mapping formula for a recording surface and a datum given by depths).

All four take traveltime tables as the array ``ttab`` (ns, nzt, nxt), as ``seispy.migration.kdmig2d`` does, with the
traces of the gathers in the stream, and return the datumed traces with the headers of the input traces.
"""
import math
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage
from . import _kernels

F32 = np.float32

_K2D_ERRORS = {1: "Topography definition is out of range!", 2: "Datuming surface is out of output range!",
               6: "Padded nt is too big for the datuming filter!"}
_DC_ERRORS = {1: "Topography definition is out of range!", 2: "Recording surface is out of range!",
              3: "Datuming surface is out of travel time range!", 4: "Datum and recording surface overlap!",
              5: "A stationary point is above the datum (negative zst)!",
              6: "Padded nt is too big for the datuming filter!"}


def _tables(ttab):
    tables = np.asarray(ttab)
    if tables.ndim != 3:
        raise ValueError("The traveltime tables must be a 3-D array (ns, nzt, nxt).")
    ns, nzt, nxt = tables.shape
    if ns < 3 or nzt < 2 or nxt < 2:
        raise ValueError("The traveltime tables need at least three sources and two samples in x and in z.")
    # [ns][nxt][nzt], z fast (a copy: the reference times are subtracted in place)
    return np.ascontiguousarray(np.transpose(tables, (0, 2, 1)), dtype=F32).copy(), ns, nzt, nxt


def _surface(points, sgn, name):
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 2:
        raise ValueError(f"{name} is two or more (x, z) pairs.")
    if np.any(np.diff(arr[:, 0]) <= 0):
        raise ValueError(f"The x of {name} must increase.")
    return np.ascontiguousarray(arr[:, 0], dtype=F32), np.ascontiguousarray(arr[:, 1] * sgn, dtype=F32)


def _nint_trace(trace):
    return np.ascontiguousarray(np.asarray(trace), dtype=F32)


def _datumk2d(upstream, *, receivers, ttab, fxt, dxt, fzt, dzt, fs, ds, fxi, dxi, nxi, sgn, surf=None, dt=None, ft=None,
              fzo=None, dzo=None, nzo=None, fxso=None, dxso=None, nxso=None, fxgo=None, dxgo=None, nxgo=None, exgo=None,
              fmax=None, offmax=99999.0, aperx=None, angmax=60.0, v0=1500.0, dvz=0.0, antiali=True, ntr=100000):
    tab, ns, nzt, nxt = _tables(ttab)
    if sgn not in (-1, 1):
        raise ValueError("sgn must be 1 (down) or -1 (up).")
    if angmax < 0.00001:
        raise ValueError("angmax must be positive!")
    if nxi < 3:
        raise ValueError("nxi must be at least 3.")
    fxt, dxt, fzt, dzt, fs, ds = (float(v) for v in (fxt, dxt, fzt, dzt, fs, ds))
    ext = fxt + (nxt - 1) * dxt
    ezt = fzt + (nzt - 1) * dzt
    es = fs + (ns - 1) * ds
    nxso = (nxt - 1) * 2 + 1 if nxso is None else int(nxso)
    fxso = fxt if fxso is None else float(fxso)
    dxso = dxt * 0.5 if dxso is None else float(dxso)
    nxgo = (nxt - 1) * 2 + 1 if nxgo is None else int(nxgo)
    fxgo = fxt if fxgo is None else float(fxgo)
    dxgo = dxt * 0.5 if dxgo is None else float(dxgo)
    nzo = (nzt - 1) * 5 + 1 if nzo is None else int(nzo)
    fzo = (fzt if fzo is None else float(fzo)) * sgn
    dzo = (dzt * 0.2 if dzo is None else float(dzo)) * sgn
    if nxso < 1 or nxgo < 1 or nzo < 1 or dxso <= 0 or dxgo <= 0 or dzo == 0:
        raise ValueError("The output grid is not allowed.")
    exso = fxso + (nxso - 1) * dxso
    exgo = fxgo + (nxgo - 1) * dxgo if exgo is None or receivers else float(exgo)
    ezo = fzo + (nzo - 1) * dzo
    if fxt > fxso or fxt > fxgo or ext < exso or ext < exgo or fzt > fzo or ezt < ezo:
        raise ValueError("The datuming output range is out of the traveltime table!")
    if surf is None:
        surf = [[(0.0, 0.0), (99999.0, 0.0)], [(0.0, 0.0), (99999.0, 0.0)]]
    if len(surf) != 2:
        raise ValueError("surf is two surfaces: the recording surface and the new datum.")
    x0, z0 = _surface(surf[0], sgn, "the recording surface")
    x1, z1 = _surface(surf[1], sgn, "the datum")
    lateral = 0.5 * nxt * dxt if aperx is None else float(aperx)
    source = as_trace_iterator(upstream)
    dr = dxgo if receivers else dxso
    nxgt = int((exgo - fxgo) / dxgo + 1) if not receivers else 0
    n_gathers = nxso if receivers else nxgt
    n_out = nxgo if receivers else nxso

    def traces():
        first = True
        state = None
        dats = {}
        headers = {}
        used = 0
        s = fxso if receivers else 0.0
        fxin = fxgo if receivers else fxso
        n_t = 0
        for jtr, trace in enumerate(source, start=1):
            if jtr > ntr:
                break
            if first:
                if trace.dtype.kind == 'c':
                    raise TypeError("This stage works on real traces, but was given a complex one.")
                n_t = trace.n_sample
                sample_dt = float(dt) if dt is not None else float(trace.d_sample)
                if sample_dt < 1e-7:
                    raise ValueError("dt must be positive!")
                first_t = float(ft) if ft is not None else float(trace.header['sample_start'])
                high = float(fmax) if fmax is not None else 0.25 / sample_dt
                szif, sz, nangl, status = _kernels.datumk2d_geometry(x0, z0, x1, z1, F32(0.025 * dr), int(nxi), F32(fxi),
                                                                     F32(dxi), ns, F32(fs), F32(ds))
                if status:
                    raise ValueError("The surfaces do not cover the locations (fxi, dxi, nxi) and the sources of the "
                                     "tables.")
                rmax = min(max(es - fxt, ext - fs), 0.5 * offmax + lateral)
                nr = 2 + int(rmax / dr)
                tb, pb, sigb, angb = _kernels.datumk2d_tables(tab, nr, F32(dr), F32(dzt), F32(fzt), F32(v0), F32(dvz), sz,
                                                              F32(fxt), F32(dxt), F32(fs), F32(ds))
                mtmax = max(int(2 * dr * math.sin(angmax * math.pi / 180.0) / (v0 * sample_dt)), 1)
                state = _kernels.Datumk2d(tab, tb, pb, angb, sigb, szif, nangl, F32(fs), F32(ds), F32(fxt), F32(dxt),
                                          F32(fzt), F32(dzt), F32(fxi), F32(dxi), nzo, F32(fzo), F32(dzo), F32(lateral),
                                          F32(high), F32(angmax), mtmax, int(bool(antiali)), int(sgn), F32(sample_dt),
                                          F32(first_t))
                first = False
            elif trace.n_sample != n_t:
                raise ValueError("The traces must all have the same number of samples.")
            sx, gx = float(trace.header['tx_loc'][0]), float(trace.header['rx_loc'][0])
            if receivers:
                if sx != s:
                    s = sx
                    if gx > fxgo:
                        fxin = gx
            elif gx != s:
                fxin = sx
                s = gx
            if min(sx, gx) >= fs and max(sx, gx) <= es and abs(gx - sx) <= abs(offmax):
                io = int((sx - fxso) / dxso) if receivers else int((gx - fxgo) / dxgo)
                if not 0 <= io < n_gathers:
                    raise ValueError(f"The trace {jtr} (sx={sx}, gx={gx}) is not in one of the output gathers.")
                gather = headers.setdefault(io, [])
                if len(gather) >= n_out:
                    raise ValueError(f"The gather {io} has more traces than the {n_out} output positions.")
                gather.append(trace)
                dat = dats.get(io)
                if dat is None:
                    dat = dats[io] = np.zeros((n_out, n_t), dtype=F32)
                status = state.add(_nint_trace(trace), dat, F32(fxin), F32(dr), F32(gx if receivers else sx))
                if status:
                    raise ValueError(_K2D_ERRORS.get(status, f"datuming failed ({status})"))
                used += 1
        if state is None:
            return
        scal = F32(1.0 / math.sqrt(2.0 * math.pi) * dr)
        for io in sorted(headers):
            for ixo, trace in enumerate(headers[io]):
                yield trace.replace(np.ascontiguousarray(dats[io][ixo] * scal))

    return from_iterable(traces())


def _kdmdc(upstream, *, sources, ttab, fxt, dxt, fzt, dzt, fs, ds, nxso, dxso, nxgo, dxgo, fxso=0.0, fxgo=0.0, dt=None,
           ft=None, dzo=None, offmax=99999.0, zrec=None, zdat=None, aperx=None, v0=1500.0, freq=50.0, scale=1.0, dc=0,
           ntr=100000):
    tab, ns, nzt, nxt = _tables(ttab)
    if dc not in (0, 1):
        raise ValueError("dc must be 0 or 1!")
    if dc and not sources:
        raise ValueError("dc is for the datuming of sources.")
    nxso, nxgo = int(nxso), int(nxgo)
    fxt, dxt, fzt, dzt, fs, ds = (float(v) for v in (fxt, dxt, fzt, dzt, fs, ds))
    fxso, dxso, fxgo, dxgo = float(fxso), float(dxso), float(fxgo), float(dxgo)
    if nxso < 1 or nxgo < 1:
        raise ValueError("nxso and nxgo must be at least 1.")
    ext = fxt + (nxt - 1) * dxt
    ezt = fzt + (nzt - 1) * dzt
    es = fs + (ns - 1) * ds
    if (nxgo if sources else nxso) > 1 and dxgo != dxso:
        raise ValueError("in this implementation, dxso must = dxgo!")
    if dxgo <= 0.0 or dxso <= 0.0:
        raise ValueError("in this implementation, dxgo and dxso must be >0!")
    dxi, fxi = (dxso, fxso) if sources else (dxgo, fxgo)
    nxi = int((fxgo + (nxgo + nxso - 2) * dxgo) / dxi) + 1
    exso = fxso + (nxso - 1) * dxso
    exgo = fxgo + (nxi - 1) * dxgo
    dzo = dzt * 0.2 if dzo is None else float(dzo)
    if dzo <= 0.0:
        raise ValueError("dzo must be positive.")
    fzo = fzt
    nzo = 1 + int(((nzt - 1) * dzt) / dzo)
    ezo = fzo + (nzo - 1) * dzo
    if fxt > fxso or fxt > fxgo or ext < exso or ext < exgo or fzt > fzo or ezt < ezo:
        raise ValueError("The output range is out of the traveltime table!")
    if freq <= 0.0:
        raise ValueError("freq must be positive and non-zero!")
    lateral = 0.5 * nxt * dxt if aperx is None else float(aperx)
    fgl = fxso + fxgo
    ngl = nxgo + (nxso - 1)

    def surface(value, name):
        if value is None:
            raise ValueError(f"{name} must be specified.")
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(nxi, float(arr), dtype=F32)
        if arr.ndim != 1 or arr.shape[0] < nxi:
            raise ValueError(f"not enough points in {name}: {nxi} are needed.")
        return np.ascontiguousarray(arr[:nxi], dtype=F32)

    szif = np.ascontiguousarray(np.stack([surface(zrec, 'zrec'), surface(zdat, 'zdat')], axis=1), dtype=F32)
    n_gathers = ngl if sources else nxso
    n_out = nxso if sources else nxgo
    dx = dxso if sources else dxgo
    source = as_trace_iterator(upstream)

    def traces():
        dats = {}
        headers = {}
        first = True
        n_t = 0
        s = 0.0 if sources else fxso
        fxin = fxso if sources else fxgo
        sample_dt = first_t = 0.0
        for jtr, trace in enumerate(source, start=1):
            if jtr > ntr:
                break
            if first:
                if trace.dtype.kind == 'c':
                    raise TypeError("This stage works on real traces, but was given a complex one.")
                n_t = trace.n_sample
                sample_dt = float(dt) if dt is not None else float(trace.d_sample)
                if sample_dt < 1e-7:
                    raise ValueError("dt must be positive!")
                first_t = float(ft) if ft is not None else float(trace.header['sample_start'])
                if 1.0 / (2.0 * freq) < sample_dt:
                    warnings.warn("freq set too high - you may have numerical singularities !")
                first = False
            elif trace.n_sample != n_t:
                raise ValueError("The traces must all have the same number of samples.")
            sx, gx = float(trace.header['tx_loc'][0]), float(trace.header['rx_loc'][0])
            if sources:
                if gx != s:
                    fxin = sx
                    s = gx
            elif sx != s:
                s = sx
                if gx > fxgo:
                    fxin = gx
            if min(sx, gx) >= fs and max(sx, gx) <= es and abs(gx - sx) <= offmax:
                io = int((gx - fgl) / dxgo) if sources else int((sx - fxso) / dxso)
                if not 0 <= io < n_gathers:
                    raise ValueError(f"The trace {jtr} (sx={sx}, gx={gx}) is not in one of the output gathers.")
                gather = headers.setdefault(io, [])
                if len(gather) >= n_out:
                    raise ValueError(f"The gather {io} has more traces than the {n_out} output positions.")
                gather.append(trace)
                dat = dats.get(io)
                if dat is None:
                    dat = dats[io] = np.zeros((n_out, n_t), dtype=F32)
                status = _kernels.kdmdc_add(bool(sources), _nint_trace(trace), dat, F32(first_t), F32(sample_dt), F32(sx),
                                            F32(gx), F32(lateral), F32(fxin), F32(dx), nzo, F32(fzo), F32(dzo), nxi,
                                            F32(fxi), F32(dxi), F32(fs), F32(ds), tab, F32(fzt), F32(dzt), F32(fxt),
                                            F32(dxt), szif, F32(v0), int(dc), ngl, F32(freq))
                if status:
                    raise ValueError(_DC_ERRORS.get(status, f"datuming failed ({status})"))
        scal = F32(dx * (1.0 / math.sqrt(2.0 * math.pi * v0)) * scale)
        for io in sorted(headers):
            for ixo, trace in enumerate(headers[io]):
                yield trace.replace(np.ascontiguousarray(dats[io][ixo] * scal))

    return from_iterable(traces())


# ----------------------------------------------------------------------------------------------------------- public
def datumk2dr(ttab, *, fxt, dxt, fzt, dzt, fs, ds, fxi, dxi, nxi, sgn, **options):
    """Kirchhoff datuming of the receivers of 2-D prestack shot gathers (SUDATUMK2DR).

    The stream is the shot gathers (``tx_loc[0]`` is the shot x, ``rx_loc[0]`` the receiver x). The receivers move from
    the first surface of ``surf`` to the second, by the far field approximation of Berryhill's extrapolation operator
    with the traveltimes of the tables ``ttab`` (ns, nzt, nxt) of the sources at ``fs + is * ds`` (SU's rayt2d, as for
    ``migration.kdmig2d``: ``fxt``, ``dxt``, ``fzt``, ``dzt`` are their grid). The traces come out in the order of the
    output gathers (the shots from ``fxso`` by ``dxso``), the i-th trace of a gather being the one at the i-th output
    receiver ``fxin + i * dxgo`` of the program, with the header of the i-th input trace of the gather.

    Parameters
    ----------
    fxi, dxi, nxi : float, float, int
        The x of the first surface location, the interval and the number of locations at which the surfaces are
        sampled (they must cover the shots and receivers).
    sgn : int
        1 to move the receivers down, -1 up (the z of ``surf`` and the output depths are multiplied by it, and the
        program's range checks are as it has them).
    surf : pair of arrays, optional
        The recording surface and the new datum, each (n, 2) points (x, z), z downward, joined by cubic splines;
        by default both are the surface ``(0, 0), (99999, 0)``.
    dt, ft : float, optional
        The sample interval and first time of the traces (default from the traces).
    fzo, dzo, nzo, fxso, dxso, nxso, fxgo, dxgo, nxgo :
        The output grid: the depths that must lie within the tables, the shots, and the receivers per shot; by
        default ``fzt``, ``0.2 dzt``, ``5 (nzt - 1) + 1``, ``fxt``, ``0.5 dxt``, ``2 (nxt - 1) + 1``.
    fmax : float
        The high cut of the input (default ``0.25 / dt``).
    offmax, aperx, angmax : float
        The largest absolute offset (99999), the lateral aperture (``nxt dxt / 2``) and the angle from vertical (60).
    v0, dvz : float
        The reference velocity at the surface and its vertical gradient, for the amplitudes.
    antiali : bool
        The anti-aliasing filter (default True).
    ntr : int
        The largest number of input traces (100000).

    Where the program does not do what it documents: its output would write outside of its arrays for a shot that is
    not on the output grid (that is an error here); the datuming surface from a single ``surf`` was undefined (two are
    needed here); indices that read outside of the tables are kept inside them.
    """
    kwargs = dict(ttab=ttab, fxt=fxt, dxt=dxt, fzt=fzt, dzt=dzt, fs=fs, ds=ds, fxi=fxi, dxi=dxi, nxi=nxi, sgn=sgn,
                  **options)
    return Stage(_datumk2d, parallelism='serial', name='datumk2dr', receivers=True, **kwargs)


def datumk2ds(ttab, *, fxt, dxt, fzt, dzt, fs, ds, fxi, dxi, nxi, sgn, **options):
    """Kirchhoff datuming of the sources of 2-D prestack receiver gathers (SUDATUMK2DS).

    As :func:`datumk2dr` with the roles of the sources and receivers exchanged: the stream is the receiver gathers,
    the output sources are at ``fxin + i * dxso`` with the first the x of the source of the first trace of the gather
    (the program takes ``dxso`` for the splines and the reference table, and ``exgo`` for the last receiver gather).
    The output gathers are indexed by ``(gx - fxgo) / dxgo`` (up to ``exgo``, default ``fxgo + (nxgo - 1) dxgo``).

    Where the program does not do what it documents: the default ``offmax`` is 99999 (it used 999999), the number of
    lateral samples of the reference quantities uses the interval of the sources (it used the receivers'), and the
    output is ``nxso`` traces per gather (it was ``nxgo`` when that was smaller, overrunning its array).
    """
    kwargs = dict(ttab=ttab, fxt=fxt, dxt=dxt, fzt=fzt, dzt=dzt, fs=fs, ds=ds, fxi=fxi, dxi=dxi, nxi=nxi, sgn=sgn,
                  **options)
    return Stage(_datumk2d, parallelism='serial', name='datumk2ds', receivers=False, **kwargs)


def kdmdcr(ttab, *, fxt, dxt, fzt, dzt, fs, ds, nxso, dxso, nxgo, dxgo, **options):
    """2.5-D datuming of the receivers of common shot gathers by the constant background data mapping (SUKDMDCR).

    The stream is the shot gathers, the datuming downward from the recording surface ``zrec`` to the datum ``zdat``
    (numbers, or one depth for each of the ``nxi = (fxgo + (nxgo + nxso - 2) dxgo) / dxgo + 1`` locations from ``fxgo``
    by ``dxgo``). The shots are at ``fxso + i dxso`` (default 0; ``dxso`` must equal ``dxgo``), with ``nxgo`` output
    receivers each starting at the first receiver after the shot. The tables ``ttab`` are as for :func:`datumk2dr`.

    Other options: ``dt``, ``ft``, ``dzo`` (the depth interval of the surfaces, ``0.2 dzt``), ``offmax`` (99999),
    ``aperx`` (``nxt dxt / 2``), ``v0`` (1500), ``freq`` (50, the dominant frequency: the stationary point must be
    ``v0 / (4 freq)`` below the datum), ``scale`` (1) and ``ntr``.
    """
    return Stage(_kdmdc, parallelism='serial', name='kdmdcr', sources=False, ttab=ttab, fxt=fxt, dxt=dxt, fzt=fzt,
                 dzt=dzt, fs=fs, ds=ds, nxso=nxso, dxso=dxso, nxgo=nxgo, dxgo=dxgo, **options)


def kdmdcs(ttab, *, fxt, dxt, fzt, dzt, fs, ds, nxso, dxso, nxgo, dxgo, **options):
    """2.5-D datuming of the sources of common receiver gathers by the constant background data mapping (SUKDMDCS).

    As :func:`kdmdcr` for the sources of receiver gathers: ``dc=0`` for receivers on the recording surface, ``dc=1``
    for receivers that were already datumed (they are on the datum). The output gathers (``nxgo + nxso - 1``) are
    indexed from ``fxso + fxgo`` by ``dxgo``.
    """
    return Stage(_kdmdc, parallelism='serial', name='kdmdcs', sources=True, ttab=ttab, fxt=fxt, dxt=dxt, fzt=fzt,
                 dzt=dzt, fs=fs, ds=ds, nxso=nxso, dxso=dxso, nxgo=nxgo, dxgo=dxgo, **options)
