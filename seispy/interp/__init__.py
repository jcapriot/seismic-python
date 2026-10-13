"""
Interpolation of traces (``su/main/interp_extrap``): SUINTERP (``interp``: synthetic traces between traces, along picked
dips, from the SU library) and SUINTERPFOWLER (``interpfowler``: an output image from constant velocity panels). The offset
extrapolation of SUOCEXT is ``seispy.dmo.ocext``.

Where a program plainly does not do what its documentation says this does what the documentation says; the differences
are with each stage.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage
from . import _kernels

__all__ = ['interp', 'interpfowler']

F32 = np.float32


# ----------------------------------------------------------------------------------------------------------- interp
def _interp(upstream, *, ninterp=1, freq1=4.0, freq2=20.0, deriv=False, linear=False, lent=5, lenx=1, lagc=400,
            xopt=0, iopt=0, dt=None):
    if ninterp < 0:
        raise ValueError("ninterp must not be negative")
    if iopt not in (0, 1, 2):
        raise ValueError(f"iopt={iopt} must be 0, 1 or 2")
    if xopt not in (0, 1):
        raise ValueError(f"xopt={xopt} must be 0 or 1")
    if freq2 <= 0.0:
        raise ValueError("freq2 must be positive")
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        first = panel[0]
        nt = first.n_sample
        if any(t.n_sample != nt for t in panel):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in panel):
            raise TypeError("interp works on real traces, but was given a complex one.")
        sample_dt = first.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        data = np.ascontiguousarray(np.stack([np.asarray(t) for t in panel]), dtype=F32)
        out = _kernels.interpolate(data, ninterp, F32(freq1), F32(freq2), lagc, lent, lenx, xopt, F32(sample_dt), iopt,
                                   bool(deriv), bool(linear))
        if iopt != 0 or len(panel) < 2 or ninterp == 0:
            for trace, row in zip(panel, out):
                yield trace.replace(np.ascontiguousarray(row))
            return
        step = ninterp + 1
        for ix, trace in enumerate(panel):
            yield trace.replace(np.ascontiguousarray(out[ix * step]))
            if ix == len(panel) - 1:
                break
            nxt = panel[ix + 1]
            for k in range(ninterp):
                w = (k + 1.0) / step
                # (the program gives them the header of the trace before: the positions are moved between the two)
                tx = [(1 - w) * a + w * b for a, b in zip(trace.header['tx_loc'], nxt.header['tx_loc'])]
                rx = [(1 - w) * a + w * b for a, b in zip(trace.header['rx_loc'], nxt.header['rx_loc'])]
                yield trace.replace(np.ascontiguousarray(out[ix * step + k + 1]), tx_loc=tx, rx_loc=rx)

    return from_iterable(traces())


def interp(*, ninterp=1, freq1=4.0, freq2=20.0, deriv=False, linear=False, lent=5, lenx=1, lagc=400, xopt=0, iopt=0,
           dt=None):
    """Interpolate traces by automatic event picking (SUINTERP).

    The stream is a gather in memory. ``ninterp`` traces are made between each pair of traces, by moving the events of
    the two along dips that are picked from the low frequencies, which are not aliased (the key assumption), so that the
    full bandwidth data is interpolated. The place where it is most likely to fail is the first breaks.

    Parameters
    ----------
    ninterp : int
        The number of traces to make between each pair, default 1.
    freq1, freq2 : float
        The corner frequencies (Hz) of the range that is not aliased and is used to pick, default 4 and 20.
    deriv : bool
        Take the vertical derivative for the picking section (for interpolating velocities, not seismic).
    linear : bool
        Linear interpolation in time (not the 8 point sinc), also for velocities.
    lent, lenx : int
        The length of the smoothers of the dip estimate in time (5) and in space (1), in samples.
    lagc : float
        The length (ms) of the AGC for the dip estimate, default 400.
    xopt : {0, 1}
        The spatial derivative by an FFT (0, for regular and not noisy traces) or by differences (1, works for irregular
        spacing; used for less than 16 traces).
    iopt : {0, 1, 2}
        0 interpolate, 1 the low-pass model (to check that the picking is not aliased), 2 the dips in samples per trace.
    dt : float
        The sample interval for traces that do not have one.

    The interpolated traces have the header of the trace before them, with the positions between those of the
    neighbours (the program leaves them as the preceding ones). The program only takes the first ``nxmax`` traces, 500
    by default, and ignores the rest: all of them are used here.
    """
    kwargs = dict(ninterp=ninterp, freq1=freq1, freq2=freq2, deriv=deriv, linear=linear, lent=lent, lenx=lenx,
                  lagc=lagc, xopt=xopt, iopt=iopt, dt=dt)
    _interp((), **kwargs)  # check the parameters now
    return Stage(_interp, parallelism='serial', name='interp', **kwargs)


# ------------------------------------------------------------------------------------------------------ interpfowler
def _interpfowler(upstream, *, cdps, times, velocities, etas=None, vmin=1500.0, vmax=2500.0, nv=21, etamin=0.10,
                  etamax=0.25, neta=11, ncdps=1130, nvstack=0, vminstack=1450.0, vscale=1.0, dt=None):
    nv, neta, ncdps = int(nv), int(neta), int(ncdps)
    if nv < 1 or neta < 1 or ncdps < 1:
        raise ValueError("nv, neta and ncdps must be at least 1")
    n_control = len(cdps)
    if n_control < 1 or len(times) != n_control or len(velocities) != n_control:
        raise ValueError("cdps, times and velocities (and etas) need an entry for each control point")
    cvs = nvstack > 5
    if not cvs and (etas is None or len(etas) != n_control):
        raise ValueError("etas must be given for each control point")
    # the grid of the constant velocity panels (the program changes it for the panels of CVS: slowness squared)
    dv = (vmax - vmin) / max(nv - 1, 1)
    deta = (etamax - etamin) / max(neta - 1, 1)
    if deta == 0.0:
        deta, neta = 1.0, 1
    if cvs:
        etamin = etamax = 0.0
        deta = 1.0
        vmin = 0.0
        nv = int(nvstack)
        neta = 1
        dv = 1.0 / (vminstack * vminstack * (nvstack - 5))
    cdp_values = np.asarray(cdps, dtype=np.float64)
    source = as_trace_iterator(upstream)
    per_cdp = nv * neta

    def control_functions(nt, sample_dt):
        time = np.arange(nt) * sample_dt
        vel = np.empty((n_control, nt))
        eta = np.zeros((n_control, nt))
        for i in range(n_control):
            t = np.asarray(times[i], dtype=np.float64)
            v = np.asarray(velocities[i], dtype=np.float64) * vscale
            if t.shape != v.shape or t.ndim != 1 or t.shape[0] < 1:
                raise ValueError("For each cdp control point, t and v must have the same number of values")
            vel[i] = np.interp(time, t, v)
            if cvs:
                vel[i] = 1.0 / (vel[i] * vel[i])
            else:
                e = np.asarray(etas[i], dtype=np.float64)
                if e.shape != t.shape:
                    raise ValueError("For each cdp control point, t, v and eta must have the same number of values")
                eta[i] = np.interp(time, t, e)
        return vel, eta

    def traces():
        iterator = iter(source)
        for icmp in range(ncdps):
            panels = []
            for _ in range(per_cdp):
                trace = next(iterator, None)
                if trace is None:
                    raise ValueError("can't get input trace: the stream has fewer than ncdps * nv * neta traces")
                panels.append(trace)
            if icmp == 0:
                nt = panels[0].n_sample
                sample_dt = panels[0].d_sample or dt
                if not sample_dt:
                    raise ValueError("The trace has no sample interval, and no `dt` was given.")
                vel, eta = control_functions(nt, sample_dt)
            p = np.stack([np.asarray(t, dtype=np.float64) for t in panels])  # (neta * nv, nt)
            # the control points that this cdp is between (the first and last are held outside of them)
            if icmp <= cdp_values[0] or n_control == 1:
                ic1 = ic2 = 0
                wc = 0.0
            elif icmp >= cdp_values[-1]:
                ic1 = ic2 = n_control - 1
                wc = 0.0
            else:
                for ic in range(1, n_control):
                    ic1, ic2 = ic - 1, ic
                    wc = (cdp_values[ic2] - icmp) / (cdp_values[ic2] - cdp_values[ic1])
                    if cdp_values[ic1] < icmp <= cdp_values[ic2]:
                        break
            ee = (wc * eta[ic1] + (1.0 - wc) * eta[ic2] - etamin) / deta
            vv = (wc * vel[ic1] + (1.0 - wc) * vel[ic2] - vmin) / dv
            # (the program truncates, which takes a position just before the grid to be on it)
            ie, iv = np.floor(ee).astype(int), np.floor(vv).astype(int)
            we, wv = ee - ie, vv - iv
            out = np.zeros(nt)
            samples = np.arange(nt)
            if neta == 1:
                ok = (iv >= 0) & (iv < nv - 1)
                i = samples[ok]
                out[i] = (1 - wv[i]) * p[iv[i], i] + wv[i] * p[iv[i] + 1, i]
            else:
                ok = (iv >= 0) & (iv < nv - 1) & (ie >= 0) & (ie < neta - 1)
                i = samples[ok]
                j, k = iv[i], ie[i]
                out[i] = ((1 - we[i]) * (1 - wv[i]) * p[k * nv + j, i] + (1 - we[i]) * wv[i] * p[k * nv + j + 1, i]
                          + we[i] * (1 - wv[i]) * p[(k + 1) * nv + j, i] + we[i] * wv[i] * p[(k + 1) * nv + j + 1, i])
            last = panels[-1]
            tx, rx = last.header['tx_loc'], last.header['rx_loc']
            middle = [0.5 * (tx[0] + rx[0]), 0.5 * (tx[1] + rx[1])]
            yield last.replace(out.astype(np.float32), d_sample=sample_dt, sample_start=0.0, ensemble_number=icmp,
                               ensemble_trace_number=0, tx_loc=middle + [tx[2]], rx_loc=middle + [rx[2]])

    return from_iterable(traces(), n_traces=ncdps)


def interpfowler(cdps, times, velocities, etas=None, *, vmin=1500.0, vmax=2500.0, nv=21, etamin=0.10, etamax=0.25,
                 neta=11, ncdps=1130, nvstack=0, vminstack=1450.0, vscale=1.0, dt=None):
    """Interpolate an output image from constant velocity panels (SUINTERPFOWLER), of ``sutifowler`` or of CVS.

    The stream is ``ncdps`` groups of ``nv * neta`` traces (the panels of each CDP, for the velocities ``vmin`` to ``vmax``
    and the etas ``etamin`` to ``etamax``, the velocity changing fastest): for each CDP one trace comes out, from a bilinear
    interpolation along the (time, velocity, eta) function of that CDP, which is itself interpolated linearly between the
    control points, and held outside of the first and the last.

    Parameters
    ----------
    cdps : list of float
        The CDP number of each control point, in increasing order.
    times, velocities, etas : list of arrays
        For each control point the times (s) of the (t, v, eta) triplets, and their velocities and etas (the program takes
        them from parameters that are repeated for each control point). ``etas`` is not needed for the panels of CVS.
    vmin, vmax, nv, etamin, etamax, neta, ncdps :
        As given to ``sutifowler`` (defaults 1500, 2500, 21, .10, .25, 11, 1130).
    nvstack, vminstack : int, float
        With ``nvstack > 5`` the input is that of CVS (``nvstack`` panels of constant slowness squared from ``vminstack``,
        which is 1450 as the documentation has it, and the program 0.25), and the others are ignored.
    vscale : float
        A factor for the velocity functions, default 1.
    dt : float
        The sample interval for traces that do not have one.

    The program takes a value just before the first velocity as on the grid, extrapolating a little (it truncates):
    that is zero here.
    """
    kwargs = dict(cdps=cdps, times=times, velocities=velocities, etas=etas, vmin=vmin, vmax=vmax, nv=nv, etamin=etamin,
                  etamax=etamax, neta=neta, ncdps=ncdps, nvstack=nvstack, vminstack=vminstack, vscale=vscale, dt=dt)
    _interpfowler((), **kwargs)  # check the parameters now
    return Stage(_interpfowler, parallelism='serial', name='interpfowler', **kwargs)
