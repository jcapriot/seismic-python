"""
Dip moveout (``su/main/dip_moveout``): SUDMOFK, SUDMOTX, SUDMOFKCW, SUDMOVZ, SUDMOTIVZ and SUTIHALEDMO, and SUOCEXT (extrapolation
to a smaller offset by offset continuation, from ``su/main/interp_extrap``, which is the DMO of SUDMOFK with the offset to go to).

The input is NMO corrected, in common-offset gathers: the traces of a gather are the run of traces that have the same
``offset`` (SU's rule: one gather ends and another begins when the offset changes), and the cdp of a trace is its
``ensemble_number``, which must be the cdp bin NUMBER, not its location. The offsets of ``noffmix`` gathers are mixed:
every trace of the mix, with a cdp in ``cdpmin`` to ``cdpmax``, comes out with the sum of the DMO of the traces of the
mix, in the order that the traces came in. A trace out of the range of the cdps is dropped. What is done to the gathers
is done by functions of the SU library (``_kernels.pyx``).

Where the program plainly does not do what its documentation says this does what the documentation says: SU counted a
change of the offset only for the traces of the range of cdps, so that a trace outside of it with another offset made
a new mix of every trace that followed it (here an offset that changes ends a gather wherever the trace is).
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import stage
from . import _kernels

__all__ = ['dmofk', 'dmotx', 'dmofkcw', 'dmovz', 'dmotivz', 'tihaledmo', 'ocext']

F32 = np.float32


def _check_range(cdpmin, cdpmax, dxcdp, noffmix):
    if cdpmin > cdpmax:
        raise ValueError("cdpmin must not be greater than cdpmax")
    if dxcdp == 0:
        raise ValueError("dxcdp must not be 0")
    if noffmix < 1:
        raise ValueError("noffmix must be at least 1")


def _velocity_function(tdmo, vdmo, nt, dt, name, default):
    """The values at the times of the trace from the (tdmo, vdmo) pairs: linear, constant outside of them"""
    tdmo = np.atleast_1d(np.asarray(0.0 if tdmo is None else tdmo, dtype=np.float64))
    vdmo = np.atleast_1d(np.asarray(default if vdmo is None else vdmo, dtype=np.float64))
    if tdmo.shape != vdmo.shape:
        raise ValueError(f"number of tdmo and {name} must be equal")
    if (np.diff(tdmo) <= 0).any():
        raise ValueError("tdmo must increase monotonically")
    return np.interp(np.arange(nt) * dt, tdmo, vdmo)


def _trace_values(trace, dt=None):
    """samples, sample interval and first time of a trace, for DMO"""
    if trace.dtype.kind == 'c':
        raise TypeError("DMO works on real traces, but was given a complex one.")
    sample_dt = float(trace.d_sample) if dt is None else float(dt)
    if not sample_dt > 0:
        raise ValueError("The trace has no sample interval.")
    return np.ascontiguousarray(np.asarray(trace), dtype=F32), sample_dt, float(trace.header['sample_start'])


class _Mix:
    """The traces of the mix that is being made: their headers and the sum of the DMO of the gathers of its offsets"""

    def __init__(self, nx, nt):
        self.q = np.zeros((nx, nt), dtype=F32)
        self.headers = []
        self.offsets = 0

    def emit(self, cdpmin):
        for trace in self.headers:
            yield trace.replace(np.ascontiguousarray(self.q[int(trace.header['ensemble_number']) - cdpmin]))


def _fk_family(upstream, name, cdpmin, cdpmax, dxcdp, noffmix, make_state):
    """SUDMOFK and its kin (SUDMOFKCW, SUDMOVZ, SUDMOTIVZ): the gather of an offset is a zero padded array of
    traces (nxfft + 2, nt), which is transformed in place and added to the mix"""
    _check_range(cdpmin, cdpmax, dxcdp, noffmix)
    source = as_trace_iterator(upstream)
    nx = cdpmax - cdpmin + 1

    def traces():
        state = None
        gather = None
        mix = None
        old = None
        for trace in source:
            offset = float(trace.header['offset'])
            if state is None:
                data, dt, ft = _trace_values(trace)
                nt = data.shape[0]
                state = make_state(trace, nt, dt, ft)
                mix = _Mix(nx, nt)
                old = offset
                gather = np.zeros((_kernels.nxfft(nx, dxcdp, offset) + 2, nt), dtype=F32)
            elif trace.n_sample != nt:
                raise ValueError("The traces must all have the same number of samples.")
            if offset != old:
                # an offset is complete
                state(gather, old)
                mix.q += gather[:nx]
                mix.offsets += 1
                gather = np.zeros((_kernels.nxfft(nx, dxcdp, offset) + 2, nt), dtype=F32)
                old = offset
            if mix.offsets == noffmix:
                yield from mix.emit(cdpmin)
                mix = _Mix(nx, nt)
            cdp = int(trace.header['ensemble_number'])
            if cdpmin <= cdp <= cdpmax:
                mix.headers.append(trace)
                gather[cdp - cdpmin] = np.asarray(trace, dtype=F32)
        if state is not None:
            state(gather, old)
            mix.q += gather[:nx]
            yield from mix.emit(cdpmin)

    return from_iterable(traces())


# ------------------------------------------------------------------------------------------------------------- dmofk
def _dmofk(upstream, *, cdpmin, cdpmax, dxcdp, noffmix, tdmo=None, vdmo=None, sdmo=1.0, fmax=None):
    def make_state(trace, nt, dt, ft):
        velocities = _velocity_function(tdmo, vdmo, nt, dt, 'vdmo', 1500.0)
        vrms = _kernels.vrms(np.atleast_1d(np.asarray(0.0 if tdmo is None else tdmo, dtype=F32)),
                             np.atleast_1d(np.asarray(1500.0 if vdmo is None else vdmo, dtype=F32)), nt, dt, ft)
        top = float(fmax) if fmax is not None else 0.5 / dt

        def apply(gather, offset):
            _kernels.fk_offset(gather, offset, top, float(sdmo), cdpmax - cdpmin + 1, float(dxcdp), dt, ft, vrms)

        return apply

    return _fk_family(upstream, 'dmofk', cdpmin, cdpmax, dxcdp, noffmix, make_state)


# SUDMOFK: DMO of common-offset gathers via the F-K domain (log stretch) method (Hale)
#
# cdpmin, cdpmax : the range of cdps (their numbers) to which DMO is applied; dxcdp : the distance between cdps
# noffmix : the number of offsets to mix: typically no smaller than the ratio of the shotpoint spacing to the cdp spacing
#     (2 or 3 times it often gives better results)
# tdmo, vdmo : the times (s) and rms velocities (m/s) of a velocity function of time, for a first order correction of
#     depth-variable velocity; sdmo : the DMO stretch factor, 1 (0.6 for typical v(z)); fmax : the maximum frequency
dmofk = stage(_dmofk, parallelism='serial', name='dmofk', validate=True)


# ------------------------------------------------------------------------------------------------------------- dmotx
def _dmotx(upstream, *, cdpmin, cdpmax, dxcdp, noffmix, offmax=3000.0, tmute=2.0, vrms=1500.0, nsmax=400):
    _check_range(cdpmin, cdpmax, dxcdp, noffmix)
    source = as_trace_iterator(upstream)
    ncdp = cdpmax - cdpmin + 1
    lds, ifds = 125, -100

    def traces():
        tables = None
        q = None
        headers = []
        noff = 0
        old = None
        for trace in source:
            offset = float(trace.header['offset'])
            data, dt, ft = _trace_values(trace)
            if tables is None:
                nt = data.shape[0]
                ts, amps, ds = _kernels.tx_tables(float(dxcdp), dt, float(offmax), float(tmute), float(vrms), int(nsmax), lds,
                                                  ifds)
                q = np.zeros((ncdp, nt), dtype=F32)
                tables = True
                old = offset
            elif data.shape[0] != nt:
                raise ValueError("The traces must all have the same number of samples.")
            if offset != old:
                noff += 1
                old = offset
            if noff == noffmix:
                _kernels.tx_shape(ds, ifds, q)
                for header in headers:
                    yield header.replace(np.ascontiguousarray(q[int(header.header['ensemble_number']) - cdpmin]))
                headers = []
                noff = 0
                q = np.zeros((ncdp, nt), dtype=F32)
            cdp = int(trace.header['ensemble_number'])
            if cdpmin <= cdp <= cdpmax:
                headers.append(trace)
                _kernels.tx_trace(ts, amps, offset, cdp - cdpmin, float(dxcdp), dt, ft, data, q)
        if tables is not None:
            _kernels.tx_shape(ds, ifds, q)
            for header in headers:
                yield header.replace(np.ascontiguousarray(q[int(header.header['ensemble_number']) - cdpmin]))

    return from_iterable(traces())


# SUDMOTX: DMO of common-offset gathers via the T-X domain (Kirchhoff, non-aliased integral) method (Hale)
#
# cdpmin, cdpmax, dxcdp, noffmix : as for dmofk (noffmix should typically equal the ratio of the shotpoint spacing to the
#     cdp spacing)
# offmax, tmute, vrms : the largest offset, the mute time at it and the rms velocity there (defaults for metric units); they
#     need not be precise: overestimate offmax, underestimate tmute and vrms. No mute is done: they only make the operator.
dmotx = stage(_dmotx, parallelism='serial', name='dmotx', validate=True)


# ------------------------------------------------------------------------------------------------------------- dmofkcw
def _dmofkcw(upstream, *, cdpmin, cdpmax, dxcdp, noffmix, tdmo=None, vdmo=None, gamma=0.5, ntable=1000, sdmo=1.0, flip=False,
             fmax=None):
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    def make_state(trace, nt, dt, ft):
        vrms = _kernels.vrms(np.atleast_1d(np.asarray(0.0 if tdmo is None else tdmo, dtype=F32)),
                             np.atleast_1d(np.asarray(1500.0 if vdmo is None else vdmo, dtype=F32)), nt, dt, ft)
        _velocity_function(tdmo, vdmo, nt, dt, 'vdmo', 1500.0)  # (the checks of the pairs)
        if gamma != 1.0:
            zoh, boh = _kernels.kcw_table(int(ntable), float(gamma))
        else:
            zoh = boh = None
        s1, s2 = _kernels.kcw_stretch(float(sdmo), float(gamma))
        if flip:
            s1, s2 = s2, s1
        sign = -1.0 if flip else 1.0
        top = float(fmax) if fmax is not None else 0.5 / dt

        def apply(gather, offset):
            _kernels.kcw_offset(gather, offset, top, cdpmax - cdpmin + 1, float(dxcdp), dt, ft, vrms, float(gamma), boh, zoh,
                                int(ntable), s1, s2, sign)

        return apply

    return _fk_family(upstream, 'dmofkcw', cdpmin, cdpmax, dxcdp, noffmix, make_state)


# SUDMOFKCW: DMO of converted waves via the F-K domain (log stretch) method (Alfaraj), for common-offset gathers
#
# as dmofk, and gamma : the velocity ratio, upgoing / downgoing (0.5); ntable : the number of tabulated z/h and b/h (1000);
# flip : negative shifts, with s1 and s2 exchanged, for the other side of a split spread (the offsets are not signed here:
# end-on data is recommended).
dmofkcw = stage(_dmofkcw, parallelism='serial', name='dmofkcw', validate=True)


# ------------------------------------------------------------------------------------------------------------- dmovz
def _vz_model_function(upstream, cdpmin, cdpmax, dxcdp, noffmix, fmax, smute, speed, nptab, build):
    def make_state(trace, nt, dt, ft):
        if ft != 0.0:
            raise ValueError(f"ft={ft} --- nonzero ft not yet implemented!")
        model = build(nt, dt)
        top = float(fmax) if fmax is not None else 0.5 / dt

        def apply(gather, offset):
            model.offset(gather, cdpmax - cdpmin + 1, float(dxcdp), offset, top, float(smute), float(speed))

        return apply

    return _fk_family(upstream, 'dmovz', cdpmin, cdpmax, dxcdp, noffmix, make_state)


def _dmovz(upstream, *, cdpmin, cdpmax, dxcdp, noffmix, v=None, tdmo=None, vdmo=None, fmax=None, smute=1.5, speed=1.0,
           nptab=401):
    if smute <= 1.0 or speed <= 0:
        raise ValueError("smute must be over 1 and speed positive.")

    def build(nt, dt):
        if v is not None:
            values = np.asarray(v, dtype=F32)
            if values.shape != (nt,):
                raise ValueError(f"cannot read {nt} velocities: v has {values.size}")
        else:
            values = _velocity_function(tdmo, vdmo, nt, dt, 'vdmo', 1500.0).astype(F32)
        return _kernels.VzModel(np.ascontiguousarray(values), float(dt), int(nptab))

    return _vz_model_function(upstream, cdpmin, cdpmax, dxcdp, noffmix, fmax, smute, speed, nptab, build)


# SUDMOVZ: DMO of common-offset gathers for a V(z) medium (Artley), with rays traced through the interval velocities
#
# cdpmin, cdpmax, dxcdp, noffmix : as for dmofk
# v : the interval velocities (m/s) at the times of the traces (an array of nt values), or tdmo and vdmo, the times and
#     the interval velocities that they are interpolated from (linearly), 1500 m/s by default
# fmax : the largest frequency of the data; smute : the NMO stretch mute (1.5): the DMO is not computed for samples that
#     have been stretched more; speed : a knob for speed (and aliasing), 1; nptab : the number of ray parameters of the tables
# (the first sample of the traces must be at time 0)
dmovz = stage(_dmovz, parallelism='serial', name='dmovz', validate=True)


# ----------------------------------------------------------------------------------------------------------- dmotivz
def _dmotivz(upstream, *, cdpmin, cdpmax, dxcdp, noffmix, vn=None, v=None, eta=None, tdmo=None, vndmo=None, vdmo=None,
             etadmo=None, fmax=None, smute=1.5, speed=1.0, nptab=401):
    if smute <= 1.0 or speed <= 0:
        raise ValueError("smute must be over 1 and speed positive.")

    def build(nt, dt):
        if vn is not None:
            vnt = np.asarray(vn, dtype=F32)
            if vnt.shape != (nt,):
                raise ValueError(f"cannot read {nt} velocities: vn has {vnt.size}")
        else:
            vnt = _velocity_function(tdmo, vndmo, nt, dt, 'vndmo', 1500.0).astype(F32)
        if eta is not None:
            etat = np.asarray(eta, dtype=F32)
            if etat.shape != (nt,):
                raise ValueError(f"cannot read {nt} values: eta has {etat.size}")
        else:
            etat = _velocity_function(tdmo, etadmo, nt, dt, 'etadmo', 0.0).astype(F32)
        if v is not None:
            vt = np.asarray(v, dtype=F32)
            if vt.shape != (nt,):
                raise ValueError(f"cannot read {nt} velocities: v has {vt.size}")
        elif vdmo is not None:
            vt = _velocity_function(tdmo, vdmo, nt, dt, 'vdmo', 1500.0).astype(F32)
        else:
            vt = vnt  # (as the documentation says: the velocity is the NMO velocity if it is not given)
        return _kernels.VzModel(np.ascontiguousarray(vt), float(dt), int(nptab), np.ascontiguousarray(vnt),
                                np.ascontiguousarray(etat))

    return _vz_model_function(upstream, cdpmin, cdpmax, dxcdp, noffmix, fmax, smute, speed, nptab, build)


# SUDMOTIVZ: DMO of common-offset gathers for a transversely isotropic V(z) medium (Alkhalifah and Artley)
#
# as dmovz, with the NMO interval velocities vn (or tdmo and vndmo), the vertical interval velocities v (or vdmo; by default
# those of vn) and the interval values of eta (or etadmo, 0 by default)
dmotivz = stage(_dmotivz, parallelism='serial', name='dmotivz', validate=True)


# --------------------------------------------------------------------------------------------------------- tihaledmo
def _tihaledmo(upstream, *, nxmax, option=1, dx=50.0, v=1500.0, h=200.0, ntpad=0, nxpad=None, vnmo=None, dp=None, e=0.0,
               d=0.0, dt=None):
    if nxmax < 1:
        raise ValueError("nxmax must be at least 1.")
    if option not in (1, 2, 3, 4, 5, 6):
        raise ValueError("option is 1 to 6.")
    if v <= 0 and option in (3, 6):
        raise ValueError("must enter positive value for v when option=3 or 6")
    if option == 5 and (vnmo is None or dp is None):
        raise ValueError("option 5 needs vnmo(p) and its increment dp.")
    source = as_trace_iterator(upstream)

    def traces():
        panel = []
        for trace in source:
            panel.append(trace)
            if len(panel) == nxmax:
                break
        if not panel:
            return
        data, sample_dt, _ = _trace_values(panel[0], dt)
        rows = np.empty((len(panel), data.shape[0]), dtype=F32)
        for i, trace in enumerate(panel):
            if trace.n_sample != data.shape[0]:
                raise ValueError("The traces must all have the same number of samples.")
            rows[i] = np.asarray(trace, dtype=F32)
        pad = int(h / dx) if nxpad is None else int(nxpad)
        table = np.zeros(0, dtype=F32) if vnmo is None else np.ascontiguousarray(np.asarray(vnmo, dtype=F32))
        out = _kernels.tihaledmo(rows, sample_dt, float(dx), float(h), float(v), int(option), int(ntpad), pad, table,
                                 0.0 if dp is None else float(dp), float(e), float(d))
        for trace, row in zip(panel, out):
            yield trace.replace(np.ascontiguousarray(row))

    return from_iterable(traces())


# SUTIHALEDMO: DMO of one common-offset gather (the traces of the stream, at most nxmax), in the f-k domain, with the
# operator of option: 1 Hale (thesis), 2 Bleistein true amplitude, 3 Bleistein cos*cos weighted (needs v), 4 Zhang,
# 5 Tsvankin anisotropic (needs vnmo, the NMO velocity as a function of the ray parameter at intervals dp), 6 Tsvankin VTI
# weak anisotropy (e and d are Thomsen's epsilon and delta, and needs v)
#
# dx : the midpoint interval; v : the velocity, which limits the aperture; h : the half offset; ntpad, nxpad : the samples
# and traces of padding (h / dx traces by default)
tihaledmo = stage(_tihaledmo, parallelism='serial', name='tihaledmo', validate=True)


# ------------------------------------------------------------------------------------------------------------ ocext
def _ocext(upstream, *, cdpmin, cdpmax, dxcdp, noffmix, offextr, tdmo=None, vdmo=None, sdmo=1.0, fmax=None):
    def make_state(trace, nt, dt, ft):
        vrms = _kernels.vrms(np.atleast_1d(np.asarray(0.0 if tdmo is None else tdmo, dtype=F32)),
                             np.atleast_1d(np.asarray(1500.0 if vdmo is None else vdmo, dtype=F32)), nt, dt, ft)
        top = float(fmax) if fmax is not None else 0.5 / dt
        # (velocities that are not increasing in time are checked like those of dmofk)
        _velocity_function(tdmo, vdmo, nt, dt, 'vdmo', 1500.0)

        def apply(gather, offset):
            _kernels.ocext_offset(gather, offset, float(offextr), top, float(sdmo), cdpmax - cdpmin + 1, float(dxcdp), dt,
                                  ft, vrms)

        return apply

    stream = _fk_family(upstream, 'ocext', cdpmin, cdpmax, dxcdp, noffmix, make_state)
    # the traces of the output are at the offset that is extrapolated to
    return from_iterable((t.replace(offset=float(offextr)) for t in stream))


# SUOCEXT: extrapolation of common-offset gathers to a smaller offset (Theodoro and Larner), by offset continuation in the
# f-k domain: the DMO of dmofk, to the offset offextr instead of to zero offset, with a term for the amplitudes
#
# cdpmin, cdpmax, dxcdp, noffmix, tdmo, vdmo, sdmo, fmax : as for dmofk
# offextr : the offset (m) to extrapolate to
#
# The output traces have the offset ``offextr``.
ocext = stage(_ocext, parallelism='serial', name='ocext', validate=True)
