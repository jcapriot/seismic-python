"""
Migration: SUSTOLT, SUMIGFD, SUMIGFFD, SUMIGPS, SUMIGPSPI, SUKDMIG2D, SUKTMIG2D, SUMIGSPLIT, SUGAZMIG, SUMIGTK, SUMIGPSTI,
SUMIGPREFD, SUMIGPREFFD, SUMIGPRESP, SUMIGPREPSPI (prestack shots), SUMIGTOPO2D, SUINVVXZCO, SUMIGGBZO and SUMIGGBZOAN
(``su/main/migration_inversion``).

The programs that migrate a section (``migfd``, ``migffd``, ``migps``, ``migpspi``) read all of the traces of their input, which have
to be in the order of the midpoints (SU: "sorted by increasing or decreasing cdp"), and make the migrated traces. ``stolt`` migrates
common-offset gathers (or a stacked section, which is one gather) of the traces of the cdp bins ``cdpmin`` .. ``cdpmax``, with the
bin number the ``ensemble_number`` of the traces, and one gather ends and another begins where the offset changes.

The sample interval is that of the traces. The spacing of the midpoints ``dx`` is given, or worked out from the midpoints of the
first two traces (SU used 1.0 where the header had none). Depth migrations make traces of ``nz`` samples ``dz`` apart (the
``sampling_unit`` of these is meters), with the velocities of a depth (nz, nx) array, whose x direction is that of the traces.
These are stages that cannot be split between gathers (``parallelism='serial'``).
"""
import math
import warnings

import numpy as np

from ..container import Trace, as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stage import stage
from . import _kernels

__all__ = ['stolt', 'migfd', 'migffd', 'migps', 'migpspi', 'kdmig2d', 'ktmig2d', 'migsplit', 'gazmig', 'migtk', 'migpsti',
           'migprefd', 'migpreffd', 'migpresp', 'migprepspi', 'migtopo2d', 'invvxzco', 'miggbzo', 'miggbzoan']

F32 = np.float32
_METERS = 2


def _sample_interval(traces, dt):
    if dt is not None:
        return float(dt)
    if traces[0].d_sample:
        return float(traces[0].d_sample)
    warnings.warn("The traces have no sample interval, assuming dt=0.004")
    return 0.004


def _midpoint_spacing(traces, dx):
    if dx is not None:
        return float(dx)
    if len(traces) > 1:
        def mid(t):
            tx, rx = t.header['tx_loc'], t.header['rx_loc']
            return 0.5 * (tx[0] + rx[0]), 0.5 * (tx[1] + rx[1])

        (x0, y0), (x1, y1) = mid(traces[0]), mid(traces[1])
        spacing = float(np.hypot(x1 - x0, y1 - y0))
        if spacing > 0.0:
            return spacing
    warnings.warn("The midpoint spacing is not known (give dx), assuming dx=1.0")
    return 1.0


def _section(upstream):
    traces = list(as_trace_iterator(upstream))
    if not traces:
        return traces, None
    n = traces[0].n_sample
    if any(t.n_sample != n for t in traces):
        raise ValueError("The traces must all have the same number of samples.")
    if any(t.dtype.kind == 'c' for t in traces):
        raise TypeError("This stage works on real traces, but was given a complex one.")
    return traces, np.ascontiguousarray([np.asarray(t, dtype=F32) for t in traces], dtype=F32)


def _velocity(vel, nz, nx):
    v = np.ascontiguousarray(vel, dtype=F32)
    if v.shape != (nz, nx):
        raise ValueError(f"The velocity must have the shape (nz, nx) = {(nz, nx)}, with nx the number of traces.")
    return v


# --------------------------------------------------------------------------------------------------------------- stolt
def _stolt(upstream, *, cdpmin, cdpmax, dxcdp, noffmix=1, tmig=(0.0,), vmig=(1500.0,), smig=1.0, vscale=1.0, fmax=None, lstaper=0,
           lbtaper=0):
    if cdpmin > cdpmax:
        raise ValueError("cdpmin must not be greater than cdpmax")
    if noffmix < 1:
        raise ValueError("noffmix must be at least 1")
    tmig = [float(t) for t in np.atleast_1d(tmig)]
    vmig = [float(v) for v in np.atleast_1d(vmig)]
    if len(tmig) != len(vmig):
        raise ValueError("number of tmig and vmig must be equal")
    if any(b <= a for a, b in zip(tmig, tmig[1:])):
        raise ValueError("tmig must increase monotonically")
    nx = cdpmax - cdpmin + 1
    source = as_trace_iterator(upstream)

    def traces():
        nt = dt = None
        times = values = None
        gather = None            # the traces of the offset that is being collected, p(t, x)
        mix = None               # the sum of the migrated gathers of the mix, q(t, x)
        headers = []             # the traces of the mix, in order
        n_offsets = 0
        old_offset = None

        def migrate():
            nonlocal gather, mix, n_offsets
            out = _kernels.stolt(gather, dt, float(dxcdp), times, values, float(vscale), float(smig), fmax_used, int(lstaper),
                                 int(lbtaper))
            mix += out
            gather = np.zeros((nx, nt), dtype=F32)
            n_offsets += 1

        def emit():
            for t in headers:
                yield t.replace(mix[int(t.header['ensemble_number']) - cdpmin].copy())

        for trace in source:
            if nt is None:
                if float(trace.header['sample_start']) != 0.0:
                    raise ValueError("cannot handle non-zero time of first sample")
                if trace.dtype.kind == 'c':
                    raise TypeError("This stage works on real traces, but was given a complex one.")
                nt = trace.n_sample
                dt = float(trace.d_sample)
                fmax_used = min(float(fmax) if fmax is not None else 0.5 / dt, 0.5 / dt)
                # (the times end at the end of the data, and the last velocity is repeated there)
                times = np.array(tmig + [nt * dt], dtype=F32)
                values = np.array(vmig + [vmig[-1]], dtype=F32)
                gather = np.zeros((nx, nt), dtype=F32)
                mix = np.zeros((nx, nt), dtype=F32)
                old_offset = float(trace.header['offset'])
            elif trace.n_sample != nt:
                raise ValueError("The traces must all have the same number of samples.")
            offset = float(trace.header['offset'])

            # if an offset is complete, migrate it
            if offset != old_offset:
                migrate()
            # if a mix of offsets is complete
            if n_offsets == noffmix:
                yield from emit()
                headers, n_offsets = [], 0
                mix = np.zeros((nx, nt), dtype=F32)
            cdp = int(trace.header['ensemble_number'])
            if cdpmin <= cdp <= cdpmax:
                headers.append(trace)
                gather[cdp - cdpmin] = np.asarray(trace, dtype=F32)
            old_offset = offset

        if nt is not None:
            migrate()
            yield from emit()

    return from_iterable(traces())


# SUSTOLT: Stolt migration for stacked data or common-offset gathers
#
# cdpmin, cdpmax : the first and last cdp bin number (the ensemble_number) of the data; traces outside are dropped
# dxcdp : the distance between the cdp bins
# noffmix : the number of offsets to mix (unstacked data), the ratio of shot to cdp spacing; the migrated traces of the offsets
#           of a mix are summed
# tmig, vmig : the times and rms velocities of the velocity function (constant extrapolation); smig the stretch factor ("W"),
#           0.6 typical if the velocity increases; vscale a factor for the velocities
# fmax : the largest frequency of the data, default the Nyquist
# lstaper, lbtaper : the lengths of the side tapers (traces) and the bottom taper (samples)
#
# The traces of an offset have to be next to each other, and are NMO corrected. The sampling of the time of the first sample
# has to be 0. Differences from SU: the offset of a trace outside of the cdp range also counts for the end of a gather (SU
# repeated the migration of an empty gather).
stolt = stage(_stolt, parallelism='serial', name='stolt', validate=True)


# ------------------------------------------------------------------------------------------------------------ the rest
def _depth_traces(traces, data, nz, dz):
    return from_iterable((t.replace(np.ascontiguousarray(data[i]), d_sample=float(dz), sample_start=0.0, sampling_unit=_METERS)
                          for i, t in enumerate(traces)), n_traces=len(traces))


def _migfd(upstream, *, vel, nz, dz, dt=None, dx=None, dip=65):
    if nz < 1 or dz <= 0:
        raise ValueError("nz and dz must be positive")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nx = len(traces)
    out = _kernels.migfd(section, _velocity(vel, nz, nx), int(nz), float(dz), _sample_interval(traces, dt),
                         _midpoint_spacing(traces, dx), int(dip))
    return _depth_traces(traces, out, nz, dz)


# SUMIGFD: 45 to 90 degree finite-difference depth migration of zero-offset data
#
# vel : the velocities, an array (nz, nx), nx the number of traces; nz, dz : the number of depth samples and their interval
# dip : the maximum dip, 45, 65, 79, 80, 87, 89 or 90 (the cost is 45 = 65 = 79 < 80 < 87 < 89 < 90); another value is 79
migfd = stage(_migfd, parallelism='serial', name='migfd', validate=True)


def _migffd(upstream, *, vel, nz, dz, dt=None, dx=None):
    if nz < 1 or dz <= 0:
        raise ValueError("nz and dz must be positive")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nx = len(traces)
    out = _kernels.migffd(section, _velocity(vel, nz, nx), int(nz), float(dz), _sample_interval(traces, dt),
                          _midpoint_spacing(traces, dx))
    return _depth_traces(traces, out, nz, dz)


# SUMIGFFD: Fourier finite-difference depth migration of zero-offset data (a hybrid of phase shift and finite-difference
# migration, for velocities that change laterally); the parameters are those of migfd (without dip)
migffd = stage(_migffd, parallelism='serial', name='migffd', validate=True)


def _migps(upstream, *, dt=None, dx=None, ffil=None, tmig=(0.0,), vmig=(1500.0,), vt=None, nxpad=0, ltaper=0, np_slopes=200, ntflag=1):
    tmig = [float(t) for t in np.atleast_1d(tmig)]
    vmig = [float(v) for v in np.atleast_1d(vmig)]
    if vt is None:
        if len(tmig) != len(vmig):
            raise ValueError("number of tmig and vmig must be equal")
        if any(b <= a for a, b in zip(tmig, tmig[1:])):
            raise ValueError("tmig must increase monotonically")
    if ntflag not in (1, 2, 3):
        raise ValueError("ntflag is 1 (normal), 2 (turned) or 3 (both).")
    if ffil is not None and len(ffil) != 4:
        raise ValueError("if ffil is specified, exactly 4 values must be provided")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nt = section.shape[1]
    sample_dt = _sample_interval(traces, dt)
    if vt is None:
        times = sample_dt * np.arange(nt)
        velocity = np.interp(times, tmig, vmig)  # (linear interpolation and constant extrapolation)
    else:
        velocity = np.asarray(vt, dtype=float)
        if velocity.shape != (nt,):
            raise ValueError("vt must have a velocity for each time sample.")
    filt = np.array([0.0, 0.0, 0.5 / sample_dt, 0.5 / sample_dt] if ffil is None else ffil, dtype=F32)
    out = _kernels.migps(section, sample_dt, _midpoint_spacing(traces, dx), filt, int(nxpad), int(ltaper), int(np_slopes),
                         int(ntflag), np.ascontiguousarray(velocity, dtype=F32))
    return from_iterable((t.replace(np.ascontiguousarray(out[i])) for i, t in enumerate(traces)), n_traces=len(traces))


# SUMIGPS: migration by phase shift with turning rays, for v(t) (interval velocities)
#
# tmig, vmig : the times and interval velocities (linear interpolation, constant extrapolation), or vt, a velocity for each
#           time sample
# ffil : the trapezoidal window of the frequencies to migrate (Hz), default 0, 0, 0.5 / dt, 0.5 / dt
# nxpad : the number of cdps to pad with zeros before the Fourier transform; ltaper the length of the linear taper of the edges
# np_slopes : the number of slopes of the table (SU's np); ntflag 1 normal, 2 turned, 3 both rays
migps = stage(_migps, parallelism='serial', name='migps', validate=True)


def _migpspi(upstream, *, vel, nz, dz, dt=None, dx=None):
    if nz < 1 or dz <= 0:
        raise ValueError("nz and dz must be positive")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nx = len(traces)
    out = _kernels.migpspi(section, _velocity(vel, nz, nx), int(nz), float(dz), _sample_interval(traces, dt),
                           _midpoint_spacing(traces, dx))
    return _depth_traces(traces, out, nz, dz)


# SUMIGPSPI: Gazdag's phase shift plus interpolation depth migration of zero-offset data, for lateral velocity variation; the
# parameters are those of migfd (without dip)
migpspi = stage(_migpspi, parallelism='serial', name='migpspi', validate=True)


# ------------------------------------------------------------------------------------------------------------ kdmig2d
def _nint(x):
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _kdmig2d(upstream, *, ttab, fxt, dxt, fzt, dzt, fs, ds, dt=None, ft=None, dxm=None, nxo=None, fxo=None, dxo=None, nzo=None,
             fzo=None, dzo=None, off0=0.0, doff=99999.0, noff=1, absoff=False, limoff=False, fmax=None, offmax=99999.0, aperx=None,
             angmax=60.0, v0=1500.0, dvz=0.0, ls=True, ntr=100000, rscale=1000.0, tv=None, cs=None, output='image'):
    tables = np.asarray(ttab)
    if tables.ndim != 3:
        raise ValueError("The traveltime tables must be a 3-D array (ns, nzt, nxt).")
    ns, nzt, nxt = tables.shape
    if ns < 2 or nzt < 2 or nxt < 2:
        raise ValueError("The traveltime tables need at least two sources and two samples in x and in z.")
    if output not in ('image', 'velan', 'both'):
        raise ValueError("output is 'image', 'velan' or 'both'.")
    if (tv is None) != (cs is None):
        raise ValueError("The velocity analysis needs both tv and cs.")
    npv = tv is not None
    if output != 'image' and not npv:
        raise ValueError("The velocity analysis output needs tv and cs (the traveltime variation and cosine tables).")
    if angmax < 0.00001:
        raise ValueError("angmax must be positive!")
    if noff < 1:
        raise ValueError("noff must be at least 1.")
    ttab_c = np.ascontiguousarray(np.transpose(tables, (0, 2, 1)), dtype=F32).copy()  # [ns][nxt][nzt], z fast
    tv_c = cs_c = None
    if npv:
        tv_c = np.ascontiguousarray(np.transpose(np.asarray(tv), (0, 2, 1)), dtype=F32)
        cs_c = np.ascontiguousarray(np.transpose(np.asarray(cs), (0, 2, 1)), dtype=F32)
        if tv_c.shape != ttab_c.shape or cs_c.shape != ttab_c.shape:
            raise ValueError("tv and cs must have the shape of the traveltime tables.")

    fxt, dxt, fzt, dzt, fs, ds = float(fxt), float(dxt), float(fzt), float(dzt), float(fs), float(ds)
    ext = _nint(rscale * (fxt + (nxt - 1) * dxt)) / rscale
    ezt = _nint(rscale * (fzt + (nzt - 1) * dzt)) / rscale
    es = _nint(rscale * (fs + (ns - 1) * ds)) / rscale
    nxo = (nxt - 1) * 2 + 1 if nxo is None else int(nxo)
    fxo = fxt if fxo is None else float(fxo)
    dxo = dxt * 0.5 if dxo is None else float(dxo)
    nzo = (nzt - 1) * 5 + 1 if nzo is None else int(nzo)
    fzo = fzt if fzo is None else float(fzo)
    dzo = dzt * 0.2 if dzo is None else float(dzo)
    exo = _nint(rscale * (fxo + (nxo - 1) * dxo)) / rscale
    ezo = _nint(rscale * (fzo + (nzo - 1) * dzo)) / rscale
    if fxt > fxo or ext < exo or fzt > fzo or ezt < ezo:
        raise ValueError("The migration output range is out of the traveltime table!")
    if nxo < 1 or nzo < 1 or dxo <= 0 or dzo <= 0:
        raise ValueError("The output grid is not allowed.")
    source = as_trace_iterator(upstream)

    def traces():
        state = None
        used = 0
        spacing = None
        for trace in source:
            if state is None:
                if trace.dtype.kind == 'c':
                    raise TypeError("This stage works on real traces, but was given a complex one.")
                n_t = trace.n_sample
                sample_dt = float(dt) if dt is not None else float(trace.d_sample)
                if sample_dt < 1e-7:
                    raise ValueError("dt must be positive!")
                first_t = float(ft) if ft is not None else float(trace.header['sample_start'])
                if dxm is None:
                    raise ValueError("dxm (the midpoint spacing of the input data) must be given.")
                spacing = float(dxm)
                if spacing < 1e-7:
                    raise ValueError("dxm must be positive!")
                high = float(fmax) if fmax is not None else 0.25 / sample_dt
                lateral = float(aperx) if aperx is not None else 0.5 * nxt * dxt
                mtmax = max(int(2 * spacing * math.sin(angmax * math.pi / 180.0) / (v0 * sample_dt)), 1)
                rmax = min(max(es - fxt, ext - fs), 0.5 * offmax + lateral)
                nr = 2 + int(rmax / dxo)
                tb, pb, cs0b, angb = _kernels.kdmig2d_reference(nr, nzt, dxo, dzt, fzt, float(dvz), float(v0))
                _kernels.kdmig2d_residual(ttab_c, fs, ds, fxt, dxt, nr, dxo, tb)
                state = _kernels.Kdmig2d(ttab_c, tv_c, cs_c, nr, tb, pb, cs0b, angb, int(noff), nxo, fxo, dxo, nzo, fzo, dzo, fs, ds, es,
                                         fxt, dxt, fzt, dzt, spacing, high, float(angmax), lateral, float(offmax), mtmax, int(bool(ls)),
                                         sample_dt, first_t)
            elif trace.n_sample != n_t:
                raise ValueError("The traces must all have the same number of samples.")
            if used >= ntr:
                break
            sx, gx = float(trace.header['tx_loc'][0]), float(trace.header['rx_loc'][0])
            offset = gx - sx
            if absoff and offset < 0:
                offset = -offset
            io = int((offset - off0) / doff + 0.5)
            if limoff and (io < 0 or io >= noff):
                continue
            io = min(max(io, 0), noff - 1)
            if state.add(np.ascontiguousarray(np.asarray(trace, dtype=F32)), sx, gx, io):
                used += 1
        if state is None:
            return
        image, extra = state.images
        scale = 4 / math.sqrt(math.pi) * spacing / v0
        template = Trace(np.zeros(nzo, dtype=F32), d_sample=float(dzo), sample_start=fzo, sampling_unit='m')
        for ixo in range(nxo):
            x = fxo + ixo * dxo
            for io in range(noff):
                offset = off0 + io * doff
                header = dict(tx_loc=[x - offset / 2.0, 0.0, 0.0], rx_loc=[x + offset / 2.0, 0.0, 0.0], ensemble_number=ixo + 1,
                              ensemble_trace_number=io + 1)
                if output in ('image', 'both'):
                    yield template.replace(np.ascontiguousarray(image[io, ixo] * F32(scale)), **header)
                if output in ('velan', 'both'):
                    yield template.replace(np.ascontiguousarray(extra[io, ixo] * F32(scale)), **header)

    return from_iterable(traces())


# SUKDMIG2D: Kirchhoff depth migration of 2-D poststack or prestack data, from traveltime tables
#
# ttab : the traveltime tables of the sources (SU's rayt2d), an array (ns, nzt, nxt) of the times from the sources fs + is * ds to
#        the grid fzt + izt * dzt, fxt + ixt * dxt
# dt, ft, dxm : the sample interval and first time of the traces (default those of the traces), and the sampling interval of the
#        midpoints (required)
# nxo, fxo, dxo, nzo, fzo, dzo : the output grid (default 2 times finer in x and 5 in z than the tables)
# noff, off0, doff : the offsets of the output (a trace goes to the nearest; absoff uses the absolute offset, limoff drops the
#        traces outside of the range of offsets); offmax the largest offset, aperx the lateral aperture, angmax the largest angle
# v0, dvz : the reference velocity at the surface and its vertical gradient (for the amplitudes); fmax the high cut; ls line source
# tv, cs : the tables of the traveltime variations and of the cosines (ns, nzt, nxt) for the velocity analysis, with
#        output='velan' or 'both' to get the extra amplitudes' image
#
# The traces' sources and receivers are tx_loc[0] and rx_loc[0] (in the units of the tables). The output traces are in
# CDP gathers (ensemble_number the output trace, ensemble_trace_number the offset), sampling_unit meters.
# Differences from SU: the offset of a trace is not truncated to an integer to find its offset bin; ntr counts the migrated
# traces; the default offmax is the documented 99999 (the program had 3000).
kdmig2d = stage(_kdmig2d, parallelism='serial', name='kdmig2d', validate=True)


# ------------------------------------------------------------------------------------------------------------ ktmig2d
def _ktmig2d(upstream, *, dx, vel, angmax=40.0, hoffset=None, nfc=16, fwidth=5, firstcdp=None, fcdpdata=None, dcdp=None):
    if nfc < 1 or fwidth < 1:
        raise ValueError("nfc and fwidth must be at least 1.")
    if not 0.0 <= angmax <= 80.0:
        raise ValueError("The maximum allowed angle is 80 degrees.")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    ntr, nt = section.shape
    velocities = np.asarray(vel, dtype=F32)
    cdps = [int(t.header['ensemble_number']) for t in traces]
    first_data = cdps[0] if fcdpdata is None else int(fcdpdata)
    first_vel = first_data if firstcdp is None else int(firstcdp)
    step = (cdps[1] - cdps[0] if ntr > 1 else 1) if dcdp is None else int(dcdp)
    if velocities.ndim == 1:
        if velocities.shape[0] != nt:
            raise ValueError("A 1-D velocity must have a value for each time sample.")
        per_trace = np.tile(velocities, (ntr, 1))
    else:
        if velocities.ndim != 2 or velocities.shape[1] != nt:
            raise ValueError("The velocities must be an array (ncdp, nt).")
        rows = np.array([first_data + i * step - first_vel for i in range(ntr)])
        if (rows < 0).any() or (rows >= velocities.shape[0]).any():
            raise ValueError("The velocity file does not have the cdps of the traces.")
        per_trace = velocities[rows]
    half = 0.5 * float(traces[0].header['offset']) if hoffset is None else float(hoffset)
    sample_dt = _sample_interval(traces, None)
    out = _kernels.ktmig2d(section, np.ascontiguousarray(per_trace, dtype=F32), sample_dt, float(dx), half, float(angmax), int(nfc),
                           int(fwidth))
    return from_iterable((t.replace(np.ascontiguousarray(out[i])) for i, t in enumerate(traces)), n_traces=len(traces))


# SUKTMIG2D: prestack time migration of a common-offset section with the double square root (DSR) operator, with an
# antialias filter (Gray 1992). The data should have been corrected for the wave shaping factor with sufrac, phasefac=.25.
#
# dx : the distance between consecutive traces; vel : the rms velocities v(t, x), an array (ncdp, nt) (or a 1-D v(t))
# firstcdp : the cdp of the first row of vel; fcdpdata : the cdp of the first trace (default the first trace's ensemble_number);
#        dcdp : the cdps between traces (default from the first two)
# angmax : the maximum aperture angle, 40 degrees (a 10 degree taper); hoffset : the half offset (default half that of the first
#        trace); nfc, fwidth : the Fourier coefficients and the frequency increment (Hz) of the low-pass filters
#
# Differences from SU: the velocity of a trace is the row at its cdp relative to firstcdp (SU forgot firstcdp); on the left of the
# midpoint the antialias filter is chosen as on the right (see the library function).
ktmig2d = stage(_ktmig2d, parallelism='serial', name='ktmig2d', validate=True)


# ------------------------------------------------------------------------------------------------ the zero-offset programs
def _velocity_of_time(tmig, vmig, vt, times, default_v, name):
    """The interval velocity (or another function) of each time: from the file-like array vt, or tmig and vmig by linear
    interpolation with constant extrapolation, as SU's programs do"""
    if vt is not None:
        v = np.asarray(vt, dtype=float)
        if v.shape != (len(times),):
            raise ValueError(f"{name} must have a value for each of the {len(times)} time samples.")
        return v
    tmig = [float(t) for t in np.atleast_1d(0.0 if tmig is None else tmig)]
    vmig = [float(v) for v in np.atleast_1d(default_v if vmig is None else vmig)]
    if len(tmig) != len(vmig):
        raise ValueError(f"number of tmig and {name} must be equal")
    if any(b <= a for a, b in zip(tmig, tmig[1:])):
        raise ValueError("tmig must increase monotonically")
    return np.interp(times, tmig, vmig)


def _migsplit(upstream, *, vel, nz, dz, dt=None, dx=None):
    if nz < 1 or dz <= 0:
        raise ValueError("nz and dz must be positive")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nx = len(traces)
    out = _kernels.migsplit(section, _velocity(vel, nz, nx), int(nz), float(dz), _sample_interval(traces, dt),
                            _midpoint_spacing(traces, dx))
    return _depth_traces(traces, out, nz, dz)


# SUMIGSPLIT: split-step Fourier depth migration of zero-offset data (Stoffa et al., 1990), for lateral velocity variation; the
# parameters are those of migfd (without dip)
migsplit = stage(_migsplit, parallelism='serial', name='migsplit', validate=True)


def _gazmig(upstream, *, tmig=None, vmig=None, vt=None, ntau=None, dtau=None, ftau=None, ft=0.0, Q=1.0e6, ceil=1.0e6, dt=None,
            dx=None):
    if Q == 0.0:
        raise ValueError("Q must not be zero")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nt = section.shape[1]
    sample_dt = _sample_interval(traces, dt)
    n_tau = nt if ntau is None else int(ntau)
    if n_tau < 1:
        raise ValueError("ntau must be positive")
    step = sample_dt if dtau is None else float(dtau)
    if step <= 0.0:
        raise ValueError("dtau must be positive")
    first = float(ft) if ftau is None else float(ftau)
    times = first + step * np.arange(n_tau)  # (SU interpolated tmig and vmig at the times 0, dt, 2 dt ...)
    velocity = _velocity_of_time(tmig, vmig, vt, times, 1500.0, 'vmig')
    out = _kernels.gazmig(section, sample_dt, F32(ft), _midpoint_spacing(traces, dx), n_tau, F32(step), F32(first),
                          np.ascontiguousarray(velocity, dtype=F32), F32(Q), F32(ceil))
    return from_iterable((t.replace(np.ascontiguousarray(out[i]), d_sample=step, sample_start=first)
                          for i, t in enumerate(traces)), n_traces=len(traces))


# SUGAZMIG: Gazdag's phase shift migration of zero-offset data for the interval velocities v(tau), with an attenuation Q
#
# tmig, vmig : the times and interval velocities (linear interpolation, constant extrapolation), or vt, a velocity for each
#        migrated time sample; ft the first time of the data; ntau, dtau, ftau the migrated time samples (default those of the
#        data); Q the quality factor (1e6: no attenuation) and ceil the gain ceiling beyond which migration ceases
#
# Differences from SU: the velocity of a migrated sample is that at its time ftau + i * dtau (the program used i * dt), and vt
# has the ntau values that it needs (the program read nt of them).
gazmig = stage(_gazmig, parallelism='serial', name='gazmig', validate=True)


def _migtk(upstream, *, dxcdp, fmax=None, tmig=None, vmig=None, vt=None, nxpad=0, ltaper=0, speed=1.0, dt=None):
    if nxpad < 0 or ltaper < 0 or speed <= 0.0 or dxcdp <= 0.0:
        raise ValueError("dxcdp and speed must be positive, nxpad and ltaper not negative")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nt = section.shape[1]
    sample_dt = _sample_interval(traces, dt)
    velocity = _velocity_of_time(tmig, vmig, vt, sample_dt * np.arange(nt), 1500.0, 'vmig')
    high = 0.5 / sample_dt if fmax is None else float(fmax)
    out = _kernels.migtk(section, sample_dt, float(dxcdp), high, int(nxpad), int(ltaper), float(speed),
                         np.ascontiguousarray(velocity, dtype=F32))
    return from_iterable((t.replace(np.ascontiguousarray(out[i])) for i, t in enumerate(traces)), n_traces=len(traces))


# SUMIGTK: reverse time finite-difference migration in the (t, k) domain of stacked data (Hale)
#
# dxcdp : the distance between cdps; fmax : the maximum frequency (Nyquist); tmig, vmig (or vt, a velocity for each time sample):
#        the interval velocities of time; nxpad, ltaper : zeros to pad with and the taper of the edges; speed : more than 1
#        for a lot of dispersion
migtk = stage(_migtk, parallelism='serial', name='migtk', validate=True)


def _migpsti(upstream, *, tmig=None, vnmig=None, vmig=None, etamig=None, vnt=None, vt=None, etat=None, ffil=None, nxpad=0,
             ltaper=0, np_slopes=200, ntflag=1, dt=None, dx=None):
    if ntflag not in (1, 2, 3):
        raise ValueError("ntflag is 1 (normal), 2 (turned) or 3 (both).")
    if ffil is not None and len(ffil) != 4:
        raise ValueError("if ffil is specified, exactly 4 values must be provided")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nt = section.shape[1]
    sample_dt = _sample_interval(traces, dt)
    times = sample_dt * np.arange(nt)
    nmo = _velocity_of_time(tmig, vnmig, vnt, times, 1500.0, 'vnmig')
    eta = _velocity_of_time(tmig, etamig, etat, times, 0.0, 'etamig')
    vertical = nmo if vt is None and vmig is None else _velocity_of_time(tmig, vmig, vt, times, 1500.0, 'vmig')
    window = np.array([0.0, 0.0, 0.5 / sample_dt, 0.5 / sample_dt] if ffil is None else ffil, dtype=F32)
    out = _kernels.migpsti(section, sample_dt, _midpoint_spacing(traces, dx), window, int(nxpad), int(ltaper), int(np_slopes),
                           int(ntflag), np.ascontiguousarray(nmo, dtype=F32), np.ascontiguousarray(vertical, dtype=F32),
                           np.ascontiguousarray(eta, dtype=F32))
    return from_iterable((t.replace(np.ascontiguousarray(out[i])) for i, t in enumerate(traces)), n_traces=len(traces))


# SUMIGPSTI: phase shift migration of stacked data with turning rays for a transversely isotropic medium (Alkhalifah)
#
# tmig, vnmig, vmig, etamig : the times, interval NMO velocities, vertical velocities (default the NMO ones) and eta values, or
#        vnt, vt, etat, a value for each time sample; the other parameters are those of migps. Where a slope has no real ray the
#        table is zero (the program left what the allocation had).
migpsti = stage(_migpsti, parallelism='serial', name='migpsti', validate=True)


# ------------------------------------------------------------------------------------------------ the prestack shot programs
def _radial(x, y):
    """The position along the line of a point (x, y) as the programs have it: x for a line along x, and the distance, signed
    by sgn(x - y), for others"""
    return (-1.0 if x - y < 0 else 1.0) * math.hypot(x, y)


def _prestack(program, upstream, *, nxo, nz, dz, vel, dx, nxshot=None, dt=None, fmax=25.0, f1=5.0, f2=10.0, f3=40.0, f4=50.0,
              lpad=9999, rpad=9999, dip=None, nflag=0):
    nxo, nz = int(nxo), int(nz)
    if nxo < 4 or nz < 1 or dz <= 0.0 or dx <= 0.0:
        raise ValueError("nxo must be at least 4, nz at least 1, and dz and dx positive")
    if not 0.0 <= f1 < f2 <= f3 < f4:
        raise ValueError("The frequencies must be 0 <= f1 < f2 <= f3 < f4")
    if lpad < 0 or rpad < 0:
        raise ValueError("lpad and rpad must not be negative")
    if program in ('fd', 'ffd'):
        dip = (79 if program == 'fd' else 45) if dip is None else int(dip)
        if program == 'fd' and dip not in (45, 65, 79, 80, 87, 89, 90):
            raise ValueError(f"dip={dip} must be 45, 65, 79, 80, 87, 89 or 90")
    else:
        dip = 0
    velocity = np.ascontiguousarray(vel, dtype=F32)
    if velocity.shape != (nz, nxo):
        raise ValueError(f"The velocity must have the shape (nz, nxo) = {(nz, nxo)}.")
    to_index = (lambda x: _nint(x / dx)) if program == 'fd' else (lambda x: int(x / dx))
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        n_t = panel[0].n_sample
        if any(t.n_sample != n_t for t in panel):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in panel):
            raise TypeError("This stage works on real traces, but was given a complex one.")
        sample_dt = _sample_interval(panel, dt)
        positions = []
        for t in panel:
            tx, rx = t.header['tx_loc'], t.header['rx_loc']
            positions.append((_radial(tx[0], tx[1]), _radial(rx[0], rx[1])))
        origin = min(positions[0])
        # the shots: traces with the same source position, which must not decrease (the program's checks)
        shots = []
        for trace, (sx, gx) in zip(panel, positions):
            sx, gx = sx - origin, gx - origin
            if shots and sx == shots[-1][0]:
                if gx < shots[-1][2][-1][0]:
                    raise ValueError("gx field must be monotonically increasing within a shot!")
                shots[-1][2].append((gx, trace))
            else:
                if shots and sx < shots[-1][0]:
                    raise ValueError("sx field must be monotonically increasing!")
                shots.append((sx, trace, [(gx, trace)]))
        if nxshot is not None:
            shots = shots[:int(nxshot)]
        image = np.zeros((nxo, nz), dtype=F32)
        for sx, _, members in shots:
            gather = np.zeros((nxo, n_t), dtype=F32)
            previous = None
            for gx, trace in members:
                igx = to_index(gx)
                if not 0 <= igx < nxo:
                    raise ValueError(f"The receiver at {gx} is not within the nxo={nxo} output positions of spacing dx.")
                if igx == previous:
                    warnings.warn("repeated igx!!! check dx or the positions of the traces!!!")
                previous = igx
                gather[igx] = np.asarray(trace, dtype=F32)
            gxs = [g for g, _ in members]
            ixshot = to_index(sx)
            ix2 = min(_nint(min(gxs) / dx) if program == 'fd' else int(min(gxs) / dx), ixshot) - int(lpad)
            ix3 = max(_nint(max(gxs) / dx) if program == 'fd' else int(max(gxs) / dx), ixshot) + int(rpad)
            ix2, ix3 = max(ix2, 0), min(ix3, nxo - 1)
            if not 0 <= ixshot < nxo:
                raise ValueError(f"The shot at {sx} is not within the nxo={nxo} output positions of spacing dx.")
            _kernels.migpre_shot(program, gather, velocity, F32(sample_dt), F32(dx), F32(dz), int(dip), int(nflag), F32(fmax),
                                 F32(f1), F32(f2), F32(f3), F32(f4), int(ix2), int(ix3), int(ixshot), image)
        last = panel[-1]
        for ix in range(nxo):
            x = origin + ix * dx
            yield last.replace(np.ascontiguousarray(image[ix]), d_sample=float(dz), sample_start=0.0, sampling_unit=_METERS,
                               tx_loc=[x, 0.0, 0.0], rx_loc=[x, 0.0, 0.0], ensemble_number=ix)

    return from_iterable(traces())


def _migprefd(upstream, *, nxo, nz, dz, vel, dx, nxshot=None, dt=None, dip=79, fmax=25.0, f1=5.0, f2=10.0, f3=40.0, f4=50.0,
              lpad=9999, rpad=9999):
    return _prestack('fd', upstream, nxo=nxo, nz=nz, dz=dz, vel=vel, dx=dx, nxshot=nxshot, dt=dt, dip=dip, fmax=fmax, f1=f1,
                     f2=f2, f3=f3, f4=f4, lpad=lpad, rpad=rpad)


# SUMIGPREFD: 2-D prestack common-shot 45 to 90 degree finite-difference depth migration (Han)
#
# The stream is the shot gathers in order of the shot (sx of tx_loc, increasing) and of the receivers (gx of rx_loc, increasing
# within a shot). The output is the image of nxo traces of nz samples dz apart, from the position of the first trace.
#
# nxo, nz, dz : the output traces and depth samples (m); vel : the velocities, an array (nz, nxo), x fast; dx : the interval of
#        the output traces (m: receivers are put at the nearest multiple); nxshot : the number of shots to migrate (default all)
# dip : the maximum dip, 45, 65, 79 (default), 80, 87, 89 or 90; fmax : the peak frequency of the Ricker source wavelet;
# f1, f2, f3, f4 : frequencies of the Hamming window; lpad, rpad : zero traces padded on each side of the receivers to make
#        the aperture (default the full one)
#
# Differences from SU: the program put each trace of a gather at the receiver position of the one before it and took the first
# trace of the next shot into the previous shot (its loop read one trace ahead), the sample window was shifted, the aperture used
# the extreme receivers of all the shots that had been read, and f1 and f2 were 10 and 20 where the documentation says 5 and 10.
migprefd = stage(_migprefd, parallelism='serial', name='migprefd', validate=True)


def _migpreffd(upstream, *, nxo, nz, dz, vel, dx, nxshot=None, dt=None, dip=45, fmax=25.0, f1=5.0, f2=10.0, f3=40.0, f4=50.0,
               lpad=9999, rpad=9999):
    return _prestack('ffd', upstream, nxo=nxo, nz=nz, dz=dz, vel=vel, dx=dx, nxshot=nxshot, dt=dt, dip=dip, fmax=fmax, f1=f1,
                     f2=f2, f3=f3, f4=f4, lpad=lpad, rpad=rpad)


# SUMIGPREFFD: 2-D prestack common-shot Fourier finite-difference depth migration (Han); the parameters are those of migprefd,
# with dip 45 (default) or 65
migpreffd = stage(_migpreffd, parallelism='serial', name='migpreffd', validate=True)


def _migpresp(upstream, *, nxo, nz, dz, vel, dx, nxshot=None, dt=None, fmax=25.0, f1=5.0, f2=10.0, f3=40.0, f4=50.0,
              lpad=9999, rpad=9999):
    return _prestack('sp', upstream, nxo=nxo, nz=nz, dz=dz, vel=vel, dx=dx, nxshot=nxshot, dt=dt, fmax=fmax, f1=f1, f2=f2,
                     f3=f3, f4=f4, lpad=lpad, rpad=rpad)


# SUMIGPRESP: 2-D prestack common-shot split-step Fourier depth migration (Han); the parameters are those of migprefd without dip
migpresp = stage(_migpresp, parallelism='serial', name='migpresp', validate=True)


def _migprepspi(upstream, *, nxo, nz, dz, vel, dx, nxshot=None, dt=None, nflag=0, fmax=25.0, f1=5.0, f2=10.0, f3=40.0,
                f4=50.0, lpad=9999, rpad=9999):
    if nflag not in (0, 1):
        raise ValueError("nflag is 0 (no normalization) or 1 (by the source wave field)")
    return _prestack('pspi', upstream, nxo=nxo, nz=nz, dz=dz, vel=vel, dx=dx, nxshot=nxshot, dt=dt, nflag=nflag, fmax=fmax,
                     f1=f1, f2=f2, f3=f3, f4=f4, lpad=lpad, rpad=rpad)


# SUMIGPREPSPI: 2-D prestack common-shot phase shift plus interpolation depth migration (Han); the parameters are those of
# migprefd without dip, and nflag=1 normalizes the cross-correlation by the source wave field
migprepspi = stage(_migprepspi, parallelism='serial', name='migprepspi', validate=True)


# ------------------------------------------------------------------------------------------------------------ migtopo2d
def _migtopo2d(upstream, *, ttab, fxt, dxt, fzt, dzt, fs, ds, fxi, dxi, nxi, surf=None, dt=None, ft=None, dxm=None,
               nxo=None, fxo=None, dxo=None, nzo=None, fzo=None, dzo=None, off0=0.0, doff=99999.0, noff=1, fmax=None,
               offmax=99999.0, aperx=None, angmax=60.0, v0=1500.0, dvz=0.0, ls=True, ntr=100000):
    from ..datuming import _kernels as datuming_kernels
    tables = np.asarray(ttab)
    if tables.ndim != 3:
        raise ValueError("The traveltime tables must be a 3-D array (ns, nzt, nxt).")
    ns, nzt, nxt = tables.shape
    if ns < 3 or nzt < 2 or nxt < 2:
        raise ValueError("The traveltime tables need at least three sources and two samples in x and in z.")
    if angmax < 0.00001:
        raise ValueError("angmax must be positive!")
    if noff < 1 or nxi < 3:
        raise ValueError("noff must be at least 1 and nxi at least 3.")
    tab = np.ascontiguousarray(np.transpose(tables, (0, 2, 1)), dtype=F32).copy()  # [ns][nxt][nzt], z fast
    fxt, dxt, fzt, dzt, fs, ds = (float(v) for v in (fxt, dxt, fzt, dzt, fs, ds))
    ext = fxt + (nxt - 1) * dxt
    ezt = fzt + (nzt - 1) * dzt
    es = fs + (ns - 1) * ds
    nxo = (nxt - 1) * 2 + 1 if nxo is None else int(nxo)
    fxo = fxt if fxo is None else float(fxo)
    dxo = dxt * 0.5 if dxo is None else float(dxo)
    nzo = (nzt - 1) * 5 + 1 if nzo is None else int(nzo)
    fzo = fzt if fzo is None else float(fzo)
    dzo = dzt * 0.2 if dzo is None else float(dzo)
    exo = fxo + (nxo - 1) * dxo
    ezo = fzo + (nzo - 1) * dzo
    if fxt > fxo or ext < exo or fzt > fzo or ezt < ezo:
        raise ValueError("The migration output range is out of the traveltime table!")
    if nxo < 1 or nzo < 1 or dxo <= 0 or dzo <= 0:
        raise ValueError("The output grid is not allowed.")
    if surf is None:
        surf = [(0.0, 0.0), (99999.0, 0.0)]
    points = np.asarray(surf, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2 or np.any(np.diff(points[:, 0]) <= 0):
        raise ValueError("surf is two or more (x, z) points with increasing x.")
    x_surf = np.ascontiguousarray(points[:, 0], dtype=F32)
    z_surf = np.ascontiguousarray(points[:, 1], dtype=F32)
    lateral = 0.5 * nxt * dxt if aperx is None else float(aperx)
    source = as_trace_iterator(upstream)

    def traces():
        state = None
        used = 0
        mig = np.zeros((noff, nxo, nzo), dtype=F32)
        n_t = 0
        for jtr, trace in enumerate(source, start=1):
            if jtr >= ntr + 1 and used:
                break
            if state is None:
                if trace.dtype.kind == 'c':
                    raise TypeError("This stage works on real traces, but was given a complex one.")
                n_t = trace.n_sample
                sample_dt = float(dt) if dt is not None else float(trace.d_sample)
                if sample_dt < 1e-7:
                    raise ValueError("dt must be positive!")
                first_t = float(ft) if ft is not None else float(trace.header['sample_start'])
                if dxm is None:
                    raise ValueError("dxm (the midpoint spacing of the input data) must be given.")
                spacing = float(dxm)
                if spacing < 1e-7:
                    raise ValueError("dxm must be positive!")
                high = float(fmax) if fmax is not None else 0.25 / sample_dt
                mtmax = max(int(2 * spacing * math.sin(angmax * math.pi / 180.0) / (v0 * sample_dt)), 1)
                szif, sz, nangl, status = datuming_kernels.datumk2d_geometry(
                    x_surf, z_surf, x_surf, z_surf, F32(spacing * 0.025), int(nxi), F32(fxi), F32(dxi), ns, F32(fs), F32(ds))
                if status:
                    raise ValueError("The surface does not cover the locations (fxi, dxi, nxi) and the sources of the "
                                     "tables.")
                rmax = min(max(es - fxt, ext - fs), 0.5 * offmax + lateral)
                nr = 2 + int(rmax / dxo)
                tb, pb, sigb, angb = datuming_kernels.datumk2d_tables(tab, nr, F32(dxo), F32(dzt), F32(fzt), F32(v0),
                                                                      F32(dvz), sz, F32(fxt), F32(dxt), F32(fs), F32(ds))
                state = (sample_dt, first_t, spacing, high, mtmax, tb, pb, sigb, angb, np.ascontiguousarray(nangl))
            elif trace.n_sample != n_t:
                raise ValueError("The traces must all have the same number of samples.")
            sample_dt, first_t, spacing, high, mtmax, tb, pb, sigb, angb, nangl = state
            sx, gx = float(trace.header['tx_loc'][0]), float(trace.header['rx_loc'][0])
            io = int((gx - sx - off0) / doff + 0.5)
            io = min(max(io, 0), noff - 1)
            if min(sx, gx) >= fs and max(sx, gx) <= es and abs(gx - sx) <= abs(offmax):
                _kernels.migtopo2d_trace(np.ascontiguousarray(np.asarray(trace, dtype=F32)), mig[io], F32(first_t),
                                         F32(sample_dt), F32(sx), F32(gx), F32(lateral), F32(fxo), F32(dxo), F32(fzo),
                                         F32(dzo), int(bool(ls)), mtmax, F32(spacing), F32(high), F32(angmax), F32(fs),
                                         F32(ds), tb, pb, angb, sigb, tab, F32(fzt), F32(dzt), F32(fxt), F32(dxt),
                                         F32(fxi), F32(dxi), nangl)
                used += 1
        if state is None:
            return
        scale = 4 / math.sqrt(math.pi) * state[2] / v0
        template = Trace(np.zeros(nzo, dtype=F32), d_sample=float(dzo), sample_start=fzo, sampling_unit='m')
        for ixo in range(nxo):
            x = fxo + ixo * dxo
            for io in range(noff):
                offset = off0 + io * doff
                header = dict(tx_loc=[x - offset / 2.0, 0.0, 0.0], rx_loc=[x + offset / 2.0, 0.0, 0.0],
                              ensemble_number=ixo + 1, ensemble_trace_number=io + 1)
                yield template.replace(np.ascontiguousarray(mig[io, ixo] * F32(scale)), **header)

    return from_iterable(traces())


# SUMIGTOPO2D: Kirchhoff depth migration of 2-D poststack or prestack data from a recording surface of variable topography,
# from traveltime tables (rayt2dtopo, or any that considers the topography); the arguments are those of kdmig2d, and
#
# fxi, dxi, nxi : the x of the first location, the interval and the number of the locations of the recording surface at
#        which the depth and the angle of its normal are tabulated (the program asks for them: they must cover the traces)
# surf : the recording surface, an array (n, 2) of (x, z) points joined by a cubic spline (default (0, 0), (99999, 0))
#
# The output traces are in CDP gathers as for kdmig2d. Differences from SU: the default offmax is the documented 99999 (the
# program had 3000); the source position that selects a table when it is outside of them is clamped instead of read from
# outside of the tables.
migtopo2d = stage(_migtopo2d, parallelism='serial', name='migtopo2d', validate=True)


# ------------------------------------------------------------------------------------------------------------ invvxzco
def _invvxzco(upstream, *, vel, dx=50.0, dz=50.0, fx=0.0, vp=None, dt=None, offset=None, dxm=None, fxm=0.0, nxd=5, nxb=None,
              nxc=0, nzc=0, fxo=0.0, dxo=15.0, nxo=101, fzo=0.0, dzo=15.0, nzo=101, fmax=None, ang=180.0, ls=False):
    velocity = np.ascontiguousarray(vel, dtype=F32)
    if velocity.ndim != 2:
        raise ValueError("The velocity is an array (nx, nz), z fast.")
    nx, nz = velocity.shape
    perturbation = np.zeros((0, 0), dtype=F32) if vp is None else np.ascontiguousarray(vp, dtype=F32)
    if vp is not None and perturbation.shape != velocity.shape:
        raise ValueError("The slowness perturbation must have the shape of the velocity.")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nt = section.shape[1]
    sample_dt = _sample_interval(traces, dt)
    spacing = _midpoint_spacing(traces, dxm)
    offs = float(traces[0].header['offset']) if offset is None else float(offset)
    high = 0.25 / sample_dt if fmax is None else float(fmax)
    band = nx // 2 if nxb is None else int(nxb)
    out = _kernels.invvxzco(section, sample_dt, offs, spacing, float(fxm), velocity, float(dx), float(fx), float(dz), int(nxd),
                            band, int(nxc), int(nzc), float(fxo), float(dxo), int(nxo), float(fzo), float(dzo), int(nzo), high,
                            float(ang), int(bool(ls)), perturbation)
    template = traces[0]
    return from_iterable(
        (template.replace(np.ascontiguousarray(out[i]), d_sample=float(dzo), sample_start=float(fzo), sampling_unit=_METERS,
                          tx_loc=[fxo + i * dxo - offs / 2.0, 0.0, 0.0], rx_loc=[fxo + i * dxo + offs / 2.0, 0.0, 0.0],
                          ensemble_number=i) for i in range(int(nxo))), n_traces=int(nxo))


# SUINVVXZCO: 2.5-D Born inversion (migration with amplitudes: Bleistein, Cohen and Hagin) of a common-offset gather, for a
# smooth velocity v(x, z) and an optional slowness perturbation
#
# vel : the velocity, an array (nx, nz) (z fast) on the grid fx + ix dx, iz dz; vp : the slowness perturbation, same shape
# offset : the offset of the gather (the first trace's, signed); dxm, fxm : the midpoint interval and first midpoint of the traces
# nxd : the traveltimes are calculated every nxd midpoints and interpolated between; nxb : the half-width (samples) of the band
#        about the midpoint where they are (nx / 2); nxc, nzc : the corners where the velocity is extrapolated, to avoid turned rays
# fxo, dxo, nxo, fzo, dzo, nzo : the output; fmax : the antialiasing frequency; ang : the largest dip angle; ls : line source
#
# Differences from SU: the program's globals for the failure of the eikonal equation were never set by its library, and are
# gone; indices that it read outside of its arrays are kept inside.
invvxzco = stage(_invvxzco, parallelism='serial', name='invvxzco', validate=True)


# ------------------------------------------------------------------------------------------------------------ miggbzo
def _miggbzo(upstream, *, vel, dz=1.0, dt=None, dx=None, fmin=None, fmax=None, amin=None, amax=60.0, bwh=None):
    velocity = np.ascontiguousarray(vel, dtype=F32)
    if velocity.ndim != 2:
        raise ValueError("The velocity is an array (nx, nz), z fast.")
    if dz <= 0.0:
        raise ValueError("dz must be positive")
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nx = len(traces)
    if velocity.shape[0] != nx:
        raise ValueError(f"The velocity must be (nx, nz) with nx={nx}, the number of traces.")
    nz = velocity.shape[1]
    sample_dt = _sample_interval(traces, dt)
    low = 0.025 / sample_dt if fmin is None else float(fmin)
    high = 10.0 * low if fmax is None else float(fmax)
    upper = float(amax)
    lower = -upper if amin is None else float(amin)
    average = 0.5 * float(velocity.mean())
    half_width = average / low if bwh is None else float(bwh)
    out = _kernels.miggbzo(section, velocity, sample_dt, _midpoint_spacing(traces, dx), float(dz), half_width, low, high,
                           lower, upper)
    return _depth_traces(traces, out, nz, dz)


# SUMIGGBZO: migration of zero-offset data by the accumulation of Gaussian beams (Hill, Cerveny)
#
# vel : the velocities, an array (nx, nz) (z fast, as SU's vfile; nx the number of traces) of the same units as dx; dz : the
#        depth interval; fmin, fmax : the minimum (0.025 / dt) and maximum (10 fmin) frequencies; amin, amax : the minimum and
#        maximum emergence angle (degrees; -amax and 60); bwh : the beam half-width (average velocity / (2 fmin))
#
# Differences from SU: a beam half-width that is too small for dx (the program divided by zero) is an error.
miggbzo = stage(_miggbzo, parallelism='serial', name='miggbzo', validate=True)


# ---------------------------------------------------------------------------------------------------------- miggbzoan
def _miggbzoan(upstream, *, a3333, a1111=None, a1133=None, a1313=None, a1113=None, a3313=None, delta=None, epsilon=None,
               dz=1.0, dt=None, dx=None, fmin=None, fmax=None, amin=None, amax=60.0, bwh=None):
    base = np.ascontiguousarray(a3333, dtype=F32)
    if base.ndim != 2:
        raise ValueError("The elastic coefficients are arrays (nx, nz), z fast.")
    if dz <= 0.0:
        raise ValueError("dz must be positive")

    def coefficient(value, default):
        if value is None:
            return default
        arr = np.ascontiguousarray(value, dtype=F32)
        if arr.shape != base.shape:
            raise ValueError("The elastic coefficients must all have the shape of a3333.")
        return arr

    c3333 = base
    c1313 = coefficient(a1313, F32(0.3) * c3333)
    c1111 = coefficient(a1111, c3333.copy())
    c1133 = coefficient(a1133, F32(0.4) * c3333)
    c1113 = coefficient(a1113, np.zeros_like(c3333))
    c3313 = coefficient(a3313, np.zeros_like(c3333))
    if delta is not None or epsilon is not None:
        # Thomsen's parameters make a1111 and a1133 (a1313 is the shear modulus)
        d = np.zeros_like(c3333) if delta is None else coefficient(delta, None)
        e = np.zeros_like(c3333) if epsilon is None else coefficient(epsilon, None)
        c1111 = (c3333 + 2 * e * c3333).astype(F32)
        c1133 = (np.sqrt(2 * d * c3333 * (c3333 - c1313) + (c3333 - c1313) ** 2) - c1313).astype(F32)
    traces, section = _section(upstream)
    if section is None:
        return from_iterable([])
    nx = len(traces)
    if c3333.shape[0] != nx:
        raise ValueError(f"The elastic coefficients must be (nx, nz) with nx={nx}, the number of traces.")
    nz = c3333.shape[1]
    sample_dt = _sample_interval(traces, dt)
    low = 0.025 / sample_dt if fmin is None else float(fmin)
    high = 10.0 * low if fmax is None else float(fmax)
    upper = float(amax)
    lower = -upper if amin is None else float(amin)
    average = 0.5 * (float(np.sqrt(0.25 * c1111).mean()) + float(np.sqrt(0.25 * c3333).mean()))
    half_width = average / low if bwh is None else float(bwh)
    out = _kernels.miggbzoan(section, c1111, c3333, c1133, c1313, c1113, c3313, sample_dt, _midpoint_spacing(traces, dx),
                             float(dz), half_width, low, high, lower, upper)
    return _depth_traces(traces, out, nz, dz)


# SUMIGGBZOAN: migration of zero-offset data by the accumulation of Gaussian beams in an anisotropic (P-wave) medium
#
# a3333 : the coefficient a3333 (the vertical velocity squared), an array (nx, nz) (z fast; nx the number of traces); a1111,
#        a1133, a1313, a1113, a3313 : the other coefficients, of the same shape (by default a3333, .4 a3333, .3 a3333, 0 and 0:
#        isotropic); delta, epsilon : Thomsen's parameters (they make a1111 and a1133 from a3333 and a1313); dz, fmin, fmax, amin,
#        amax, bwh as for miggbzo (the average velocity of the beam half-width is that of the horizontal and the vertical)
miggbzoan = stage(_miggbzoan, parallelism='serial', name='miggbzoan', validate=True)
