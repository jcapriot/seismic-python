"""
Migration: SUSTOLT, SUMIGFD, SUMIGFFD, SUMIGPS, SUMIGPSPI, SUKDMIG2D and SUKTMIG2D (``su/main/migration_inversion``).

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

__all__ = ['stolt', 'migfd', 'migffd', 'migps', 'migpspi', 'kdmig2d', 'ktmig2d']

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
