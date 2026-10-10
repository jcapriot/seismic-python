"""
Finite-difference modelling of the acoustic wave equation: SUFDMOD1 (1-D, first order) and SUFDMOD2 (2-D, second order).

The models are arrays and nothing touches the disk: the seismograms are streams of traces, and the wavefield snapshots are
`Snapshots`, an iterable of float32 arrays.
"""
import numpy as np

from ..container import Trace, from_iterable
from . import _fdmod

__all__ = ['fdmod1', 'fdmod1_snapshots', 'fdmod2', 'Snapshots']

abs_ = abs


class Snapshots:
    """The frames of a wavefield, each a float32 array, one for each recorded time step.

    Iterating gives the frames (once, if they are being made as they are asked for). ``dt`` is the time between the frames, the first
    is at time 0, ``spacing`` and ``origin`` are those of the axes of a frame. ``n_frames`` is the number of frames.
    """

    def __init__(self, frames, *, n_frames, dt, spacing, origin):
        self._frames = frames
        self.n_frames = n_frames
        self.dt = dt
        self.spacing = spacing
        self.origin = origin

    @property
    def times(self):
        return self.dt * np.arange(self.n_frames)

    def __iter__(self):
        return iter(self._frames)

    def __len__(self):
        return self.n_frames


def _as_profile(values, name):
    v = np.ascontiguousarray(values, dtype=np.float32)
    if v.ndim != 1 or v.shape[0] < 2:
        raise ValueError(f"{name} must be a 1-D array of at least 2 samples.")
    return v


def _fdmod1_setup(vel, density, dz, fz, sz, rz, tmax, nt, styp, freq, abs, td, zd, press):
    rv = _as_profile(vel, "The velocity")
    nz = rv.shape[0]
    if density is None:
        rd = np.full(nz, 2500.0, dtype=np.float32)
    else:
        rd = _as_profile(density, "The density")
        if rd.shape != rv.shape:
            raise ValueError("The density must have the shape of the velocity.")
    if len(abs) != 2:
        raise ValueError("abs gives the absorbing conditions of the top and the bottom: two values.")
    if styp not in (0, 1, 2):
        raise ValueError("styp is the source type: 0 (gauss), 1 (ricker 1) or 2 (ricker 2).")
    if press not in (0, 1, False, True):
        raise ValueError("press must equal 0 or 1")
    if td < 1 or zd < 1 or zd > nz:
        raise ValueError("td and zd are undersampling factors: at least 1, and zd at most nz.")
    dt, nt_out, t0, ies = _fdmod.fdmod1_plan(rv, dz, tmax, 0 if nt is None else nt, styp, freq)
    isz = int(round((sz - fz) / dz))
    irz = int(round((rz - fz) / dz))
    if not 0 <= isz < nz:
        raise ValueError("The source is outside of the model.")
    if not 0 <= irz < nz:
        raise ValueError("The receiver is outside of the model.")
    return rv, rd, dt, nt_out, t0, ies, isz, irz


def fdmod1(vel, *, sz, tmax, dz=1.0, fz=0.0, rz=1.0, density=None, nt=None, styp=0, freq=15.0, abs=(0, 1), td=1, zd=1, press=True):
    """Finite-difference modelling in one dimension (first order, explicit, velocity/pressure) for the acoustic wave equation (SUFDMOD1).

    ``vel`` is the velocity at the samples ``fz + iz * dz`` (a 1-D array), ``density`` the density (2500 where it is not given). The
    source, at ``sz``, is added to the pressure, with ``styp`` 0 for a gaussian, 1 for ricker 1 or 2 for ricker 2, of approximate
    centre frequency ``freq``. The seismogram, a stream of one trace, is that of the pressure (the particle velocity if ``press`` is
    false) at ``rz``; the trace is shifted by a time delay to make the source zero phase (``sample_start`` is minus the delay). The
    time step is worked out for stability and ``nt`` is 1 + ``tmax`` / dt unless it is given. ``abs`` are the absorbing conditions
    at the top and bottom (1 to absorb); ``td`` undersamples the trace in time. ``fdmod1_snapshots`` gives the wavefield.

    The documentation of SU has ``rz=1`` where the program has 0, and reads only the first of the two values of ``abs``;
    the documented default and both values are used here.
    """
    rv, rd, dt, nt_out, t0, ies, isz, irz = _fdmod1_setup(vel, density, dz, fz, sz, rz, tmax, nt, styp, freq, abs, td, zd, press)
    seis, _ = _fdmod.fdmod1_run(rv, rd, dz, dt, nt_out, t0, ies, isz, irz, int(abs[0]), int(abs[1]), styp, freq, td, zd, int(press),
                                False)
    trace = Trace(seis, d_sample=td * dt, sample_start=-t0, trace_type=1).replace(
        trace_id=1, tx_loc=[0.0, 0.0, -float(sz)], rx_loc=[0.0, 0.0, -float(rz)])
    return from_iterable([trace], n_traces=1)


def fdmod1_snapshots(vel, *, sz, tmax, dz=1.0, fz=0.0, rz=1.0, density=None, nt=None, styp=0, freq=15.0, abs=(0, 1), td=1, zd=1,
                     press=True):
    """The wavefield of `fdmod1` (SUFDMOD1's wfile): `Snapshots`, one frame for each ``td`` time steps, each the ``nz // zd`` samples
    ``fz + iz * zd * dz`` of the pressure (or particle velocity). The parameters are those of `fdmod1`."""
    rv, rd, dt, nt_out, t0, ies, isz, irz = _fdmod1_setup(vel, density, dz, fz, sz, rz, tmax, nt, styp, freq, abs, td, zd, press)
    _, snaps = _fdmod.fdmod1_run(rv, rd, dz, dt, nt_out, t0, ies, isz, irz, int(abs[0]), int(abs[1]), styp, freq, td, zd, int(press),
                                 True)
    return Snapshots(snaps, n_frames=snaps.shape[0], dt=td * dt, spacing=(zd * dz,), origin=(fz,))


F32 = np.float32


def _nint(x):
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


class fdmod2:
    """Finite-difference modelling in two dimensions (second order, explicit) for the acoustic wave equation (SUFDMOD2).

    ``vel`` is the velocity, an array (nz, nx) on the grid ``fz + iz * dz``, ``fx + ix * dx``; ``density`` (the same shape) is
    optional, and a constant density counts as 1. The source is a ricker wavelet (the single frequency 2 ``fpeak`` if ``mono``) added
    to the pressure: for one point ``(xs, zs)`` a point source of strength ``sstrength``; for several points (``xs`` and ``zs`` of
    the same length) an extended source along a spline through them, tapered at its ends over ``pwt`` grid points; and with ``pw``
    a horizontal plane wave source at the depth ``zs`` (the first of ``zs``). The time step is that for stability, ``nt`` is 1 +
    ``tmax`` / dt unless it is given, ``fmax`` (the default is vmin / (10 h), h the smaller of the spacings) and ``fpeak`` (the
    default is 0.5 fmax) the maximum and peak frequencies. ``abs`` are the absorbing boundary conditions of the top, left, bottom
    and right sides (0 on the top is the free surface).

    Nothing is written to files. The seismograms are traces, as streams: ``horizontal_line`` (at the depth ``hsz``, a trace for each
    x sample), ``vertical_line`` (at ``vsx``, a trace for each z sample) and ``source_points`` (at the sources, if ``source_points``
    is true); they run the model, if it has not been, to its end. ``snapshots(mt)`` gives the wavefield, one frame (nz, nx) for every
    ``mt`` time steps, as the model is run. The traces start at minus the delay of the source, ``1 / fpeak``, so that time 0 is the
    centre of the wavelet.

    Differences from SU: the plane wave source is at the positions ``fx + ix * dx`` and the depth ``zs`` (the program ignored ``fx`` and
    ``fz`` for them); the delay of the source is in the headers of the traces (the program wrote 0); the extended source takes
    ``fpeak`` as it is given, as the point source does.
    """

    def __init__(self, vel, *, xs, zs, tmax, density=None, dx=1.0, dz=1.0, fx=0.0, fz=0.0, nt=None, sstrength=1.0, pw=False, pwt=20,
                 mono=False, fmax=None, fpeak=None, hsz=None, vsx=None, source_points=False, abs=(1, 1, 1, 1)):
        v = np.asarray(vel)
        if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] < 3:
            raise ValueError("The velocity must be a 2-D array (nz, nx) of at least 3 samples in each direction.")
        self.nz, self.nx = v.shape
        nx, nz = self.nx, self.nz
        vt = np.ascontiguousarray(v.T, dtype=F32)  # [nx][nz]
        if len(abs) != 4:
            raise ValueError("abs gives the absorbing boundary conditions of the top, left, bottom and right: four values.")
        xs = np.atleast_1d(np.asarray(xs, dtype=F32))
        zs = np.atleast_1d(np.asarray(zs, dtype=F32))
        if xs.shape != zs.shape or xs.ndim != 1:
            raise ValueError("The number of xs must equal the number of zs.")
        if len(xs) == 0:
            raise ValueError("Must specify xs and zs!")
        dx, dz, fx, fz = F32(dx), F32(dz), F32(fx), F32(fz)
        if pw:
            ztmp = zs[0]
            xs = (fx + np.arange(nx) * dx).astype(F32)
            zs = np.full(nx, ztmp, dtype=F32)
        ixs = np.array([_nint((x - fx) / dx) for x in xs], dtype=np.int32)
        izs = np.array([_nint((z - fz) / dz) for z in zs], dtype=np.int32)
        if (ixs < 0).any() or (ixs >= nx).any() or (izs < 0).any() or (izs >= nz).any():
            raise ValueError("A source is outside of the model.")

        vmin, vmax = F32(vt.min()), F32(vt.max())
        if vmax <= 0:
            raise ValueError("The velocities must be positive.")
        h = min(abs_(dx), abs_(dz))
        dt = F32(h / (F32(2.0) * vmax))
        fmax = F32(vmin / (F32(10.0) * h)) if fmax is None else F32(fmax)
        fpeak = F32(F32(0.5) * fmax) if fpeak is None else F32(fpeak)
        if nt is None:
            nt = int(F32(1) + F32(tmax) / dt)
        if nt < 1:
            raise ValueError("There has to be at least one time step.")

        if density is None:
            od, dvv = None, vt * vt
        else:
            rho = np.asarray(density)
            if rho.shape != v.shape:
                raise ValueError("The density must have the shape of the velocity.")
            rt = np.ascontiguousarray(rho.T, dtype=F32)
            if rt.min() == rt.max():  # (as the program does: constant densities are 1)
                od, dvv = None, vt * vt
            else:
                dvv, od = rt * vt * vt, (F32(1.0) / rt).astype(F32)
        self.dt, self.fmax, self.fpeak, self.nt = float(dt), float(fmax), float(fpeak), nt
        self.dx, self.dz, self.fx, self.fz = float(dx), float(dz), float(fx), float(fz)
        self.tdelay = 1.0 / float(fpeak)
        self.xs, self.zs = xs, zs

        self._hs1 = self._vs2 = 0
        if hsz is not None:
            self._hs1 = _nint((F32(hsz) - fz) / dz)
            if not 0 <= self._hs1 < nz:
                raise ValueError("hsz is outside of the model.")
        if vsx is not None:
            self._vs2 = _nint((F32(vsx) - fx) / dx)
            if not 0 <= self._vs2 < nx:
                raise ValueError("vsx is outside of the model.")
        self._want = (hsz is not None, vsx is not None, bool(source_points))
        self._run = _fdmod.Fdmod2Run(np.ascontiguousarray(dvv, dtype=F32), None if od is None else np.ascontiguousarray(od, dtype=F32),
                                     np.ascontiguousarray(xs), np.ascontiguousarray(zs), ixs, izs, float(dx), float(dz), float(fx),
                                     float(fz), float(dt), nt, 1, float(sstrength), int(pwt), int(bool(mono)), float(fmax), float(fpeak),
                                     [int(a) for a in abs], self._hs1, self._vs2, *self._want)

    def _finish(self):
        while not self._run.finished:
            self._run.advance(False)

    def snapshots(self, mt=1):
        """The wavefield: `Snapshots`, a frame (nz, nx) of the pressure for every ``mt`` time steps (as the model is run; once)."""
        if mt < 1:
            raise ValueError("mt must be at least 1.")
        if self._run.step_number != 0:
            raise RuntimeError("The model has been run already.")
        run = self._run
        run_mt = mt

        def frames():
            while not run.finished:
                # (the model records every mt-th step, counting from the first)
                step = run.step_number
                frame = run.advance(step % run_mt == 0)
                if frame is not None:
                    yield frame

        n_frames = len(range(0, self.nt, mt))
        return Snapshots(frames(), n_frames=n_frames, dt=mt * self.dt, spacing=(self.dz, self.dx), origin=(self.fz, self.fx))

    def _traces(self, data, positions, tx_loc, what):
        if data is None:
            raise ValueError(f"The {what} was not asked for.")
        self._finish()
        traces = []
        for i, row in enumerate(np.asarray(data)):
            traces.append(Trace(row, d_sample=self.dt, sample_start=-self.tdelay, trace_type=1).replace(
                trace_id=i + 1, tx_loc=tx_loc(i), rx_loc=positions(i)))
        return from_iterable(traces, n_traces=len(traces))

    @property
    def horizontal_line(self):
        """The seismograms of the horizontal line at ``hsz``: a trace for each x sample"""
        z = self.fz + self._hs1 * self.dz
        x0 = float(self.xs[0])
        z0 = float(self.zs[0])
        return self._traces(self._run.hs, lambda i: [self.fx + i * self.dx, 0.0, -z], lambda i: [x0, 0.0, -z0], "horizontal line")

    @property
    def vertical_line(self):
        """The seismograms of the vertical line at ``vsx``: a trace for each z sample"""
        x = self.fx + self._vs2 * self.dx
        x0 = float(self.xs[0])
        z0 = float(self.zs[0])
        return self._traces(self._run.vs, lambda i: [x, 0.0, -(self.fz + i * self.dz)], lambda i: [x0, 0.0, -z0], "vertical line")

    @property
    def source_points(self):
        """The seismograms at the positions of the sources: a trace for each source"""
        xs, zs = self.xs, self.zs
        pos = lambda i: [float(xs[i]), 0.0, -float(zs[i])]
        return self._traces(self._run.ss, pos, pos, "source point seismograms")
