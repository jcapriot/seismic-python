"""
Finite-difference modelling of the acoustic wave equation: SUFDMOD1 (1-D, first order) and SUFDMOD2 (2-D, second order).

The models are arrays and nothing touches the disk: the seismograms are streams of traces, and the wavefield snapshots are
`Snapshots`, an iterable of float32 arrays.
"""
import math

import numpy as np

from ..container import Trace, from_iterable
from . import _fdmod

__all__ = ['fdmod1', 'fdmod1_snapshots', 'fdmod2', 'fdmod2_pml', 'ea2df', 'remac2d', 'remel2dan', 'fctanismod', 'Snapshots']

abs_ = abs


class Snapshots:
    """The frames of a wavefield, each a float32 array, one for each recorded time step.

    Iterating gives the frames (once, if they are being made as they are asked for). ``dt`` is the time between the frames, the first
    is at time 0, ``spacing`` and ``origin`` are those of the axes of a frame. ``n_frames`` is the number of frames.
    """

    def __init__(self, frames, *, n_frames, dt, spacing, origin, times=None):
        self._frames = frames
        self._times = times
        self.n_frames = n_frames
        self.dt = dt
        self.spacing = spacing
        self.origin = origin

    @property
    def times(self):
        if self._times is not None:
            return np.asarray(self._times)
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
                 mono=False, fmax=None, fpeak=None, hsz=None, vsx=None, source_points=False, abs=(1, 1, 1, 1),
                 _pml_thick=0, _pml_max=1000.0, _pml_program=False):
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
                                     [int(a) for a in abs], self._hs1, self._vs2, *self._want, int(_pml_thick), float(_pml_max),
                                     bool(_pml_program))

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


class fdmod2_pml(fdmod2):
    """Finite-difference modelling in two dimensions (second order) with perfectly matched layer absorbing boundaries (SUFDMOD2_PML).

    This is `fdmod2` (the arguments, the outputs, `snapshots`, and the seismograms are the same) with two more arguments:
    ``pml_thick``, the half-thickness of the layer in grid samples (0 uses the absorbing boundary conditions of `fdmod2` instead), and
    ``pml_max``, the absorption parameter. The layer extends the model on the bottom and right sides, and the sides that are
    absorbing are those given by ``abs``. The program has a point source of strength 1 only (no ``sstrength``, ``mono`` or ``pw``) and
    its extended source has a gaussian derivative wavelet whose frequency is set by ``fmax``. The program is documented as experimental
    and possibly buggy: the arrays of the layer start at 0 here (the program did not set all of them).
    """

    def __init__(self, vel, *, xs, zs, tmax, density=None, dx=1.0, dz=1.0, fx=0.0, fz=0.0, nt=None, fmax=None, fpeak=None, hsz=None,
                 vsx=None, source_points=False, abs=(1, 1, 1, 1), pml_max=1000.0, pml_thick=0):
        if pml_thick < 0:
            raise ValueError("pml_thick cannot be negative.")
        super().__init__(vel, xs=xs, zs=zs, tmax=tmax, density=density, dx=dx, dz=dz, fx=fx, fz=fz, nt=nt, fmax=fmax, fpeak=fpeak,
                         hsz=hsz, vsx=vsx, source_points=source_points, abs=abs, _pml_thick=pml_thick, _pml_max=pml_max,
                         _pml_program=True)


def _grid(a, name, shape=None):
    arr = np.ascontiguousarray(a, dtype=F32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (nz, nx).")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have the shape of c11 {shape}.")
    return arr


class ea2df:
    """Elastic (and anelastic, anisotropic) finite-difference forward modelling in two dimensions, fourth order in space (SUEA2DF).

    The model is given as arrays (nz, nx) on the grid ``fz + iz * dz``, ``fx + ix * dx``: the stiffnesses ``c11`` and ``c55`` (Voigt),
    the density ``rho`` and, for an anisotropic medium (all four together), ``c13``, ``c33``, ``c15`` and ``c35``; ``q``, the Q-factor,
    includes attenuation. The source is at ``sx``, ``sz`` (the middle of the model by default): ``stype`` 'p' is a P source (added to
    the normal stresses), 'v' a velocity source, 'pw' a plane wave at the angle ``sang`` from the vertical; its wavelet ``wtype`` is
    'dg' (derivative of a gaussian), 'ga' (gaussian), 'ri' (ricker), 'sp' or 'sp2' (spikes), of duration ``ts`` and average frequency
    ``favg``. The time runs from ``ft`` to ``lt`` in steps of ``dt`` (the user has to keep it stable: SU suggests dt < 0.6 dx / vmax).
    ``bc`` are the boundary conditions of the top, left, bottom and right sides: 0 none, 1 symmetry, 2 free surface (the top only),
    more than 2 an absorbing layer of that many samples, with the taper ``bc_a * i ** -bc_r``. ``tsw`` uses the shear stress only in
    non-fluid media, which may reduce the dispersion.

    The seismograms are recorded for the whole run: ``horizontal_line`` (a line at the depth ``hsz``) and ``vertical_line`` (at ``vsx``)
    are streams of traces, first those of the horizontal particle velocity (``ensemble_number`` 1) for every sample of the line and
    then those of the vertical particle velocity (``ensemble_number`` 2); they run the model if it has not been. ``snapshots(times)``
    gives the wavefield at the times asked for (the nearest time steps), as the model is run: frames of shape (2, nz, nx), the
    horizontal and vertical particle velocity, or (3, nz, nx), the stresses txx, tzz and txz, if ``stress``.

    Differences from SU: the model is padded for the boundary layers correctly (the program's padding shifted the model and dropped
    its last rows and columns), and the loop does the number of time steps that the seismograms have (the program counted the time in
    single precision, which could add or lose a step).
    """

    def __init__(self, c11, c55, rho, *, c13=None, c33=None, c15=None, c35=None, q=None, dx=10.0, dz=None, fx=-1000.0, fz=0.0,
                 dt=0.001, ft=0.0, lt=1.0, sx=None, sz=None, stype='p', sang=0.0, wtype='ri', ts=0.05, favg=50.0, tsw=False,
                 bc=(10, 10, 10, 10), bc_a=0.95, bc_r=0.0, hsz=0.0, vsx=0.0):
        c11 = _grid(c11, "c11")
        shape = c11.shape
        c55, rho = _grid(c55, "c55", shape), _grid(rho, "rho", shape)
        aniso_parts = (c13, c33, c15, c35)
        aniso = any(a is not None for a in aniso_parts)
        if aniso and any(a is None for a in aniso_parts):
            raise ValueError("An anisotropic medium needs c13, c33, c15 and c35 together.")
        if aniso:
            c13, c33, c15, c35 = (_grid(a, n, shape) for a, n in zip(aniso_parts, ("c13", "c33", "c15", "c35")))
        if q is not None:
            q = _grid(q, "q", shape)
        if len(bc) != 4:
            raise ValueError("bc gives the boundary conditions of the top, left, bottom and right: four values.")
        nz, nx = shape
        dz = dx if dz is None else dz
        xmax, zmax = fx + dx * nx, fz + dz * nz
        sx = (fx + xmax) / 2 if sx is None else sx
        sz = (fz + zmax) / 2 if sz is None else sz
        self.nx, self.nz, self.dt, self.ft, self.dx, self.dz, self.fx, self.fz = nx, nz, float(dt), float(ft), float(dx), float(dz), float(fx), float(fz)
        self.sx, self.sz, self.hsz, self.vsx, self.favg = float(sx), float(sz), float(hsz), float(vsx), float(favg)
        self._run = _fdmod.Ea2dfRun(float(dt), float(ft), float(lt), nx, float(dx), float(fx), nz, float(dz), float(fz), float(sx), float(sz),
                                    stype.encode(), float(sang), wtype.encode(), float(ts), float(favg), int(q is not None), int(aniso),
                                    int(bool(tsw)), [int(b) for b in bc], float(bc_a), float(bc_r), float(hsz), float(vsx),
                                    c11, c55, rho, c13, c33, c15, c35, q)
        self.nt = self._run.nt

    def _finish(self):
        while self._run.advance():
            pass

    def snapshots(self, times, *, stress=False):
        """The wavefield at ``times`` (seconds; the nearest time steps): `Snapshots`, a frame (2, nz, nx) of the particle velocity
        (u, w), or (3, nz, nx) of the stresses (txx, tzz, txz) if ``stress``, for every time asked for that is in the run. The frames
        are made as the model is run (once), in the order of the time steps."""
        if self._run.step_number != 0:
            raise RuntimeError("The model has been run already.")
        steps = [int(_nint(t / self.dt)) for t in np.atleast_1d(times)]
        wanted = [k for k in steps if 0 <= k < self.nt]
        fields = (2, 3, 4) if stress else (0, 1)
        run = self._run

        def frames():
            for step in sorted(set(wanted)):
                while run.step_number <= step:
                    run.advance()
                for _ in range(wanted.count(step)):
                    yield np.stack([run.snapshot(f) for f in fields])

        return Snapshots(frames(), n_frames=len(wanted), dt=self.dt, spacing=(self.dz, self.dx), origin=(self.fz, self.fx),
                         times=[self.dt * k for k in sorted(wanted)])

    def _traces(self, comps, n, b1, b2, positions):
        self._finish()
        hdr = int(round(1000.0 / (2 * self.favg)))
        traces = []
        for c, data in enumerate(comps):
            for k, i in enumerate(range(b1, n - b2)):
                traces.append(Trace(np.ascontiguousarray(data[:, i]), d_sample=self.dt, sample_start=-hdr / 1000.0, trace_type=1).replace(
                    trace_id=len(traces) + 1, ensemble_number=c + 1, ensemble_trace_number=k + 1, tx_loc=[self.sx, 0.0, -self.sz],
                    rx_loc=positions(k)))
        return from_iterable(traces, n_traces=len(traces))

    @property
    def horizontal_line(self):
        """The particle velocity along the line at ``hsz``: the traces of u and then of w for every x sample of the model"""
        self._finish()
        hu, hw, _, _ = self._run.lines()
        wbc = self._run.wbc
        return self._traces((hu, hw), hu.shape[1], wbc[1], wbc[3], lambda k: [self.fx + self.dx * k, 0.0, -self.hsz])

    @property
    def vertical_line(self):
        """The particle velocity along the line at ``vsx``: the traces of u and then of w for every z sample of the model"""
        self._finish()
        _, _, vu, vw = self._run.lines()
        wbc = self._run.wbc
        return self._traces((vu, vw), vu.shape[1], wbc[0], wbc[2], lambda k: [self.vsx, 0.0, -(self.fz + self.dz * k)])


class remac2d:
    """Acoustic 2-D Fourier-method modelling with the Rapid Expansion Method (REM) time integration (SUREMAC2D).

    The method has no restriction on the time step ``dt``, and is free of numerical grid dispersion if there are at least two grid
    points per shortest wavelength. ``vel`` is the velocity, an array (nz, nx), on the grid ``fx + ix * dx``, ``fz + iz * dz``;
    ``density`` (the same shape) is used for ``opflag=0`` (the default if there is a density): the variable density wave equation;
    ``opflag=1`` (the default without a density) is the constant density equation, ``opflag=2`` the non-reflecting equation.
    ``nx`` and ``nz`` have to be lengths of SU's Fourier transforms (``seispy.transforms`` does not need them; see the table ``nctab``
    in pfafft), and odd unless ``opflag=1``. The sources are at the grid points ``isx``, ``isz`` (lists, with ``amps``), spread over
    21 x 21 points with a gaussian of width ``w`` (grid points where it falls to a tenth). The source time function ``wavelet`` is
    'ricker' (with ``fmax``; the maximum is delayed by 3 / fmax from the start), 'spike' (an impulse at t = 0; convolve the
    seismograms with the wavelet later), or an array with its sampling interval ``dt_wavelet``. ``fsflag`` is the free surface, half a
    grid spacing above the first sample, ``vmaxu`` a larger maximum velocity if the run is unstable, and ``iabso`` the sponge boundary
    (``abso``, ``nbwx``, ``nbwz``). Sources and receivers should be about 10 points away from the absorbing boundaries.

    The pressure is recorded along the horizontal lines at the depth samples ``irz`` (``x_sections``, a trace for every x sample)
    and the vertical lines at the x samples ``irx`` (``z_sections``, a trace for every z sample), as streams of traces
    (``ensemble_number`` is the line). The snapshots are every ``dtsnap`` seconds: `Snapshots` of frames (nz, nx) at the times
    ``dtsnap, 2 dtsnap, ...``. The model is run when the first output is asked for.

    Differences from SU: the Bessel coefficients are always computed (no ``prec`` and file ``b_k``); the odd-size check works (the
    program's did not); the dispersion check is only made if ``fmax`` is known.
    """

    def __init__(self, vel, *, nt, dt, dx, dz, isx, isz, density=None, opflag=None, fx=0.0, fz=0.0, irz=(), irx=(), amps=None, w=0.1,
                 wavelet='ricker', fmax=None, dt_wavelet=None, fsflag=False, vmaxu=0.0, dtsnap=0.0, iabso=True, abso=0.1, nbwx=20,
                 nbwz=20):
        v = np.ascontiguousarray(vel, dtype=F32)
        if v.ndim != 2:
            raise ValueError("The velocity must be a 2-D array (nz, nx).")
        self.nz, self.nx = v.shape
        if opflag is None:
            opflag = 0 if density is not None else 1
        if opflag not in (0, 1, 2):
            raise ValueError("opflag is 0 (variable density), 1 (constant density) or 2 (non-reflecting).")
        if opflag == 0:
            if density is None:
                raise ValueError("opflag=0 needs the density.")
            dens = np.ascontiguousarray(density, dtype=F32)
            if dens.shape != v.shape:
                raise ValueError("The density must have the shape of the velocity.")
        else:
            dens = None
        isx = np.atleast_1d(np.ascontiguousarray(isx, dtype=np.int32))
        isz = np.atleast_1d(np.ascontiguousarray(isz, dtype=np.int32))
        if isx.shape != isz.shape or isx.ndim != 1 or len(isx) == 0:
            raise ValueError("isx and isz must have the same number of coordinates.")
        amps = np.ones(len(isx), dtype=F32) if amps is None else np.ascontiguousarray(np.broadcast_to(np.asarray(amps, dtype=F32), len(isx)))
        if isinstance(wavelet, str):
            if wavelet == 'ricker':
                if fmax is None:
                    raise ValueError("fmax required if the Ricker wavelet is selected!")
                sflag, wave, dtw = 2, np.zeros(0, dtype=F32), 0.0
            elif wavelet == 'spike':
                sflag, wave, dtw = 1, np.zeros(0, dtype=F32), 0.0
            else:
                raise ValueError("wavelet is 'ricker', 'spike' or an array.")
        else:
            if dt_wavelet is None or dt_wavelet <= 0:
                raise ValueError("A wavelet given as an array needs its sampling interval dt_wavelet.")
            sflag, wave, dtw = 0, np.ascontiguousarray(wavelet, dtype=F32), float(dt_wavelet)
            if wave.ndim != 1 or len(wave) == 0:
                raise ValueError("The wavelet must be a non-empty 1-D array.")
        if dtsnap < 0:
            raise ValueError("dtsnap must be >= 0!")
        self.dt, self.dx, self.dz, self.fx, self.fz, self.nt = float(dt), float(dx), float(dz), float(fx), float(fz), int(nt)
        self.dtsnap = float(dtsnap)
        self.irz = np.ascontiguousarray(irz, dtype=np.int32).ravel()
        self.irx = np.ascontiguousarray(irx, dtype=np.int32).ravel()
        self.nsnap = _nint(float(F32(F32(self.nt) * F32(self.dt)) / F32(dtsnap))) if dtsnap > 0 else 0
        self.isx, self.isz = isx, isz
        self._args = (int(opflag), self.nx, self.nz, self.nt, self.dx, self.dz, self.dt, isx, isz, amps, float(w), sflag,
                      float(fmax) if fmax is not None else 0.0, wave, dtw, int(bool(fsflag)), float(vmaxu), self.dtsnap, int(bool(iabso)),
                      float(abso), int(nbwx), int(nbwz), v, dens, self.irz, self.irx, self.nsnap)
        self._result = None

    def run(self):
        """Run the model (once); the outputs do this if it has not been"""
        if self._result is None:
            self._result = _fdmod.remac2d_run(*self._args)
        return self

    def _sections(self, data, lines, n, position):
        self.run()
        sx, sz = self.fx + self.isx[0] * self.dx, self.fz + self.isz[0] * self.dz
        traces = []
        for il, line in enumerate(lines):
            for i in range(n):
                traces.append(Trace(np.ascontiguousarray(data[il, :, i]), d_sample=self.dt, trace_type=1).replace(
                    trace_id=len(traces) + 1, ensemble_number=il + 1, ensemble_trace_number=i + 1,
                    tx_loc=[float(sx), 0.0, -float(sz)], rx_loc=position(line, i)))
        return from_iterable(traces, n_traces=len(traces))

    @property
    def x_sections(self):
        """The pressure along the horizontal receiver lines at the depth samples ``irz``: a trace for each x sample"""
        sectx = self.run()._result[0]
        return self._sections(sectx, self.irz, self.nx, lambda line, i: [self.fx + i * self.dx, 0.0, -(self.fz + int(line) * self.dz)])

    @property
    def z_sections(self):
        """The pressure along the vertical receiver lines at the x samples ``irx``: a trace for each z sample"""
        sectz = self.run()._result[1]
        return self._sections(sectz, self.irx, self.nz, lambda line, k: [self.fx + int(line) * self.dx, 0.0, -(self.fz + k * self.dz)])

    @property
    def snapshots(self):
        """The wavefield every ``dtsnap`` seconds (at ``dtsnap``, 2 ``dtsnap``, ...): `Snapshots` of frames (nz, nx)"""
        snap = self.run()._result[2]
        return Snapshots(iter(list(snap)), n_frames=self.nsnap, dt=self.dtsnap, spacing=(self.dz, self.dx), origin=(self.fz, self.fx),
                         times=[(i + 1) * self.dtsnap for i in range(self.nsnap)])


_REM_TYPES = {'p': 0, 's': 1, 'ux': 2, 'uz': 3}
_REM_SOURCES = {'p': 0, 's': 1, 'fx': 2, 'fz': 3}


def _rem_codes(values, table, what):
    codes = []
    for v in np.atleast_1d(values) if not isinstance(values, str) else [values]:
        if isinstance(v, str):
            if v not in table:
                raise ValueError(f"The {what} {v!r} is not one of {sorted(table)}.")
            codes.append(table[v])
        else:
            codes.append(int(v))
    return np.ascontiguousarray(codes, dtype=np.int32)


class remel2dan:
    """Elastic anisotropic 2-D Fourier-method modelling with the Rapid Expansion Method (REM) time integration (SUREMEL2DAN).

    The method has no restriction on the time step ``dt`` and is free of numerical grid dispersion when there are at least two grid
    points per shortest wavelength. The model is arrays (nz, nx) on the grid ``fx + ix * dx``, ``fz + iz * dz``: the density
    ``density`` and either, for an isotropic medium, the velocities ``vp`` and ``vs``, or, for an anisotropic one, the Voigt stiffnesses
    ``c11, c13, c15, c33, c35, c55`` with the global velocities ``vmax`` and ``vmin``. ``nx`` and ``nz`` have to be lengths of SU's
    Fourier transforms and odd. The sources are at the grid points ``isx``, ``isz`` with the types ``styp`` ('p' pressure, 's' shear,
    'fx' and 'fz' single forces; or 0 to 3) and amplitudes ``samp``, spread over 21 x 21 points with a gaussian of width ``w``, with
    the time function ``wavelet``: 'ricker' (with ``fmax``), 'spike', or an array with its sampling interval ``dt_wavelet``.
    ``vmaxu`` is a larger maximum velocity for an unstable run; ``iabso``, ``abso``, ``nbwx`` and ``nbwz`` the sponge boundary.

    ``x_lines`` are the horizontal receiver lines, a list of ``(iz, type)`` with the type 'p', 's', 'ux' or 'uz' (pressure, shear
    strain, horizontal or vertical displacement), and ``z_lines`` the vertical ones, a list of ``(ix, type)``. They are recorded as
    streams of traces (``x_sections``: a trace for every x sample of each line; ``z_sections``), with ``ensemble_number`` the line.
    ``snapshots_of`` is a list of types, and ``dtsnap`` the interval: ``snapshots`` is then a dict of `Snapshots` by the type.
    The model is run when the first output is asked for.

    Differences from SU: the Bessel coefficients are always computed (no ``prec`` and file ``b_k``); the pressure sections along
    vertical lines are made (the program tested for the shear type); the sources are made from cleared arrays and each only once;
    the odd-size check works; the dispersion check is only made if ``fmax`` is known.
    """

    def __init__(self, density, *, nt, dt, dx, dz, isx, isz, styp=0, samp=1.0, vp=None, vs=None, c11=None, c13=None, c15=None,
                 c33=None, c35=None, c55=None, vmax=None, vmin=None, fx=0.0, fz=0.0, x_lines=(), z_lines=(), snapshots_of=(),
                 dtsnap=0.0, w=0.1, wavelet='ricker', fmax=None, dt_wavelet=None, vmaxu=0.0, iabso=True, abso=0.1, nbwx=20, nbwz=20):
        rho = np.ascontiguousarray(density, dtype=F32)
        if rho.ndim != 2:
            raise ValueError("The density must be a 2-D array (nz, nx).")
        self.nz, self.nx = rho.shape
        stiff = (c11, c13, c15, c33, c35, c55)
        aniso = any(c is not None for c in stiff)
        if aniso:
            if any(c is None for c in stiff) or vmax is None or vmin is None:
                raise ValueError("An anisotropic medium needs c11, c13, c15, c33, c35 and c55, and vmax and vmin.")
            arrays = [np.ascontiguousarray(c, dtype=F32) for c in stiff]
            vp = vs = None
        else:
            if vp is None or vs is None:
                raise ValueError("An isotropic medium needs vp and vs.")
            arrays = [None] * 6
            vp, vs = np.ascontiguousarray(vp, dtype=F32), np.ascontiguousarray(vs, dtype=F32)
        for name, a in zip(("vp", "vs", "c11", "c13", "c15", "c33", "c35", "c55"), [vp, vs] + arrays):
            if a is not None and a.shape != rho.shape:
                raise ValueError(f"{name} must have the shape of the density.")
        isx = np.atleast_1d(np.ascontiguousarray(isx, dtype=np.int32))
        isz = np.atleast_1d(np.ascontiguousarray(isz, dtype=np.int32))
        if isx.shape != isz.shape or isx.ndim != 1 or len(isx) == 0:
            raise ValueError("isx and isz must have the same number of coordinates.")
        styp = _rem_codes(styp, _REM_SOURCES, "source type")
        if len(styp) == 1:
            styp = np.repeat(styp, len(isx))
        samp = np.ascontiguousarray(np.broadcast_to(np.asarray(samp, dtype=F32), len(isx)).copy())
        if len(styp) != len(isx):
            raise ValueError("isx and styp arrays must be of same size.")
        if isinstance(wavelet, str):
            if wavelet == 'ricker':
                if fmax is None:
                    raise ValueError("fmax required if the Ricker wavelet is selected!")
                sflag, wave, dtw = 2, np.zeros(0, dtype=F32), 0.0
            elif wavelet == 'spike':
                sflag, wave, dtw = 1, np.zeros(0, dtype=F32), 0.0
            else:
                raise ValueError("wavelet is 'ricker', 'spike' or an array.")
        else:
            if dt_wavelet is None or dt_wavelet <= 0:
                raise ValueError("A wavelet given as an array needs its sampling interval dt_wavelet.")
            sflag, wave, dtw = 0, np.ascontiguousarray(wavelet, dtype=F32), float(dt_wavelet)
        if dtsnap < 0:
            raise ValueError("dtsnap must be >= 0!")
        self.x_lines = [(int(z), t) for z, t in x_lines]
        self.z_lines = [(int(x), t) for x, t in z_lines]
        self.irz = np.ascontiguousarray([z for z, _ in self.x_lines], dtype=np.int32)
        self.rxtyp = _rem_codes([t for _, t in self.x_lines], _REM_TYPES, "receiver type") if self.x_lines else np.zeros(0, np.int32)
        self.irx = np.ascontiguousarray([x for x, _ in self.z_lines], dtype=np.int32)
        self.rztyp = _rem_codes([t for _, t in self.z_lines], _REM_TYPES, "receiver type") if self.z_lines else np.zeros(0, np.int32)
        self.snapshot_types = list(snapshots_of)
        self.sntyp = _rem_codes(self.snapshot_types, _REM_TYPES, "snapshot type") if self.snapshot_types else np.zeros(0, np.int32)
        if dtsnap > 0 and len(self.sntyp) == 0:
            raise ValueError("snapshots_of required if dtsnap is not 0!")
        self.nt, self.dt, self.dx, self.dz, self.fx, self.fz, self.dtsnap = int(nt), float(dt), float(dx), float(dz), float(fx), float(fz), float(dtsnap)
        self.nsnap = _nint(float(F32(F32(self.nt) * F32(self.dt)) / F32(dtsnap))) if dtsnap > 0 else 0
        self.isx, self.isz = isx, isz
        self._args = (int(aniso), self.nx, self.nz, self.nt, self.dx, self.dz, self.dt, isx, isz, styp, samp, float(w), sflag,
                      float(fmax) if fmax is not None else 0.0, wave, dtw, float(vmaxu), float(vmax) if vmax is not None else 0.0,
                      float(vmin) if vmin is not None else 0.0, self.dtsnap, self.sntyp, self.nsnap, int(bool(iabso)), float(abso),
                      int(nbwx), int(nbwz), rho, vp, vs, *arrays, self.irz, self.rxtyp, self.irx, self.rztyp)
        self._result = None

    def run(self):
        """Run the model (once); the outputs do this if it has not been"""
        if self._result is None:
            self._result = _fdmod.remel2dan_run(*self._args)
        return self

    def _sections(self, data, lines, n, position):
        self.run()
        sx, sz = self.fx + self.isx[0] * self.dx, self.fz + self.isz[0] * self.dz
        traces = []
        for il, (line, _) in enumerate(lines):
            for i in range(n):
                traces.append(Trace(np.ascontiguousarray(data[il, :, i]), d_sample=self.dt, trace_type=1).replace(
                    trace_id=len(traces) + 1, ensemble_number=il + 1, ensemble_trace_number=i + 1,
                    tx_loc=[float(sx), 0.0, -float(sz)], rx_loc=position(line, i)))
        return from_iterable(traces, n_traces=len(traces))

    @property
    def x_sections(self):
        """The sections along the horizontal receiver lines ``x_lines``: a trace for each x sample of each line"""
        data = self.run()._result[0]
        return self._sections(data, self.x_lines, self.nx, lambda line, i: [self.fx + i * self.dx, 0.0, -(self.fz + line * self.dz)])

    @property
    def z_sections(self):
        """The sections along the vertical receiver lines ``z_lines``: a trace for each z sample of each line"""
        data = self.run()._result[1]
        return self._sections(data, self.z_lines, self.nz, lambda line, k: [self.fx + line * self.dx, 0.0, -(self.fz + k * self.dz)])

    @property
    def snapshots(self):
        """The wavefield every ``dtsnap`` seconds, a dict of `Snapshots` (frames (nz, nx)) by the type in ``snapshots_of``"""
        snap = self.run()._result[2]
        return {t: Snapshots(iter(list(snap[i])), n_frames=self.nsnap, dt=self.dtsnap, spacing=(self.dz, self.dx),
                             origin=(self.fz, self.fx), times=[(j + 1) * self.dtsnap for j in range(self.nsnap)])
                for i, t in enumerate(self.snapshot_types)}


def _fct_parameter(value, nx, nz, dx, dz, name):
    """An elastic parameter as the array [nx][nz] the solver has: from an array (nz, nx), a constant, or a linear profile
    ``(p00, dpdx, dpdz)`` with the value p00 at the grid point (0, 0)"""
    if isinstance(value, (tuple, list)) and len(value) == 3 and np.ndim(value[0]) == 0:
        p00, dpdx, dpdz = (float(v) for v in value)
        ix = np.arange(nx, dtype=F32)[:, None]
        iz = np.arange(nz, dtype=F32)[None, :]
        return np.ascontiguousarray(p00 + dpdx * ix * F32(dx) + dpdz * iz * F32(dz), dtype=F32)
    arr = np.asarray(value, dtype=F32)
    if arr.ndim == 0:
        return np.full((nx, nz), float(arr), dtype=F32)
    if arr.shape != (nz, nx):
        raise ValueError(f"{name} must be a constant, a linear profile (p00, dpdx, dpdz) or an array of shape (nz, nx) = {(nz, nx)}.")
    return np.ascontiguousarray(arr.T)


class fctanismod:
    """Elastic finite-difference modelling in anisotropic media with a vertical axis of symmetry, with the flux-corrected transport
    (FCT) correction against grid dispersion (SUFCTANISMOD; Fei and Larner, CWP-137).

    The model is the density ``rho`` and the elastic parameters ``aa`` (c11), ``cc`` (c33), ``ff`` (c13), ``ll`` (c44) and ``nn`` (c66),
    each an array (nz, nx), a constant, or a linear profile ``(p00, dpdx, dpdz)`` (the value at the grid point (0, 0) and the
    gradients), on a grid of spacing ``dx``, ``dz``; ``nx`` and ``nz`` are given if there are no arrays. The source is a force at the
    grid point ``(sx, sz)`` (the middle by default), or spread over ``xzsource``, an array (nz, nx), or ``source=2``/``3``: the program's
    reflectors. ``force`` are the components (x, y, z) of the force, ``wavelet`` 1 (AKB), 2 (Ricker), 3 (impulse) or 4 (unity) of
    peak frequency ``fpeak``. The time step ``dt`` has to satisfy vmax dt / (sqrt(2) min(dx, dz)) < 1. ``isurf`` is the surface
    condition: 1 absorbing, 2 free, 3 zero. ``dofct`` applies the FCT correction, in the box ``fctxbeg``..``fctxend``,
    ``fctzbeg``..``fctzend`` (grid points), with the diffusion ``eta0`` and anti-diffusion ``eta`` coefficients and their gradients.
    ``movebc`` confines the computation to a boundary that moves with the wavefield.

    Seismograms are recorded for the whole run: ``reflection(component)`` is the line of receivers at the grid depth
    ``receiverdepth`` (a trace for each x sample) and ``vsp(component)`` the line at the grid position ``vspnx`` (a trace for each z
    sample), as streams of traces, for the component 'x', 'y' or 'z'; they run the model if it has not been. ``snapshots(mt)`` gives
    the wavefield, frames (3, nz, nx) of the x, y and z components for every ``mt`` time steps, as the model is run, and
    ``final_snapshot`` the x component at the last step.

    Differences from SU: the documented defaults are used (nt=200, dt=0.004, isurf=2, dofct=1, source=1, rho 2.0); the linear
    profiles have their parameter at (0, 0) as documented (the program used (-1, -1) for the origin); ``order=4`` is not offered,
    as the program only has the second order solver wired in; the force components can be chosen (the program's way was an
    undocumented option that left the default without a force; the default here is the vertical force).
    """

    def __init__(self, aa=2.0, cc=2.0, ff=2.0, ll=2.0, nn=2.0, rho=2.0, *, nx=None, nz=None, dx=0.02, dz=0.02, nt=200, dt=0.004,
                 sx=None, sz=None, receiverdepth=None, vspnx=None, xzsource=None, source=1, impulse=False, isurf=2, dofct=True,
                 fctxbeg=0, fctzbeg=0, fctxend=None, fctzend=None, force=(False, False, True), wavelet=1, movebc=False, fpeak=20.0,
                 eta0=0.03, eta=0.04, deta0dx=0.0, deta0dz=0.0, detadx=0.0, detadz=0.0):
        arrays = [np.asarray(v) for v in (aa, cc, ff, ll, nn, rho) if not (isinstance(v, (tuple, list)) and len(v) == 3) and np.ndim(v) == 2]
        if arrays:
            nz_a, nx_a = arrays[0].shape
            if (nx is not None and nx != nx_a) or (nz is not None and nz != nz_a):
                raise ValueError("nx and nz do not match the arrays.")
            nx, nz = nx_a, nz_a
        elif nx is None or nz is None:
            nx = 100 if nx is None else nx
            nz = 100 if nz is None else nz
        grids = [_fct_parameter(v, nx, nz, dx, dz, name) for v, name in zip((aa, cc, ff, ll, nn, rho), ("aa", "cc", "ff", "ll", "nn", "rho"))]
        if (grids[5] <= 0).any():
            raise ValueError("The density must be positive.")
        sx = nx // 2 if sx is None else int(sx)
        sz = nz // 2 if sz is None else int(sz)
        receiverdepth = sz if receiverdepth is None else int(receiverdepth)
        vspnx = sx if vspnx is None else int(vspnx)
        xz = None
        if xzsource is not None:
            xz = np.ascontiguousarray(np.asarray(xzsource, dtype=F32).T)
            if xz.shape != (nx, nz):
                raise ValueError("xzsource must have the shape (nz, nx).")
        if len(force) != 3:
            raise ValueError("force gives the x, y and z components: three flags.")
        self.nx, self.nz, self.nt, self.dx, self.dz, self.dt = int(nx), int(nz), int(nt), float(dx), float(dz), float(dt)
        self.sx, self.sz, self.receiverdepth, self.vspnx = sx, sz, receiverdepth, vspnx
        self._run = _fdmod.FctRun(self.nx, self.nz, self.nt, self.dx, self.dz, self.dt, sx, sz, receiverdepth, vspnx, int(bool(impulse)),
                                  int(source), int(isurf), int(bool(dofct)), int(fctxbeg), int(fctzbeg),
                                  self.nx if fctxend is None else int(fctxend), self.nz if fctzend is None else int(fctzend),
                                  *(int(bool(f)) for f in force), int(wavelet), int(bool(movebc)), float(fpeak), float(eta0), float(eta),
                                  float(deta0dx), float(deta0dz), float(detadx), float(detadz), *grids, xz)
        self.vmax = float(self._run.vmax)
        self.stability = self.vmax * self.dt / (math.sqrt(2.0) * min(self.dx, self.dz))  # (should be < 1)

    def _finish(self):
        while self._run.advance():
            pass

    def snapshots(self, mt=1):
        """The wavefield: `Snapshots`, a frame (3, nz, nx) of the x, y and z components of the particle velocity for every ``mt``
        time steps (made as the model is run, once)"""
        if mt < 1:
            raise ValueError("mt must be at least 1.")
        if self._run.step_number != 0:
            raise RuntimeError("The model has been run already.")
        run = self._run

        def frames():
            while run.step_number < self.nt:
                step = run.step_number
                run.advance()
                if step % mt == 0:
                    yield np.stack([run.snapshot(c) for c in range(3)])

        return Snapshots(frames(), n_frames=len(range(0, self.nt, mt)), dt=mt * self.dt, spacing=(self.dz, self.dx), origin=(0.0, 0.0))

    @property
    def final_snapshot(self):
        """The x component of the wavefield at the last time step, (nz, nx)"""
        self._finish()
        return self._run.snapshot(0)

    def _lines(self, data, n, position):
        traces = []
        for i in range(n):
            traces.append(Trace(np.ascontiguousarray(data[i]), d_sample=self.dt, trace_type=1).replace(
                trace_id=i + 1, ensemble_trace_number=i + 1, tx_loc=[self.sx * self.dx, 0.0, -self.sz * self.dz], rx_loc=position(i)))
        return from_iterable(traces, n_traces=len(traces))

    def _component(self, component):
        try:
            return {'x': 0, 'y': 1, 'z': 2}[component]
        except KeyError:
            raise ValueError("component is 'x', 'y' or 'z'.") from None

    def reflection(self, component='z'):
        """The seismograms of the receivers along the horizontal line at the grid depth ``receiverdepth``: a trace for each x sample"""
        c = self._component(component)
        self._finish()
        refl, _ = self._run.records()
        return self._lines(refl[c], self.nx, lambda i: [i * self.dx, 0.0, -self.receiverdepth * self.dz])

    def vsp(self, component='z'):
        """The seismograms of the receivers along the vertical line at the grid position ``vspnx``: a trace for each z sample"""
        c = self._component(component)
        self._finish()
        _, vsp = self._run.records()
        return self._lines(vsp[c], self.nz, lambda i: [self.vspnx * self.dx, 0.0, -i * self.dz])
