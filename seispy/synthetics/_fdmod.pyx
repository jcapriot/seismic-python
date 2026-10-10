# cython: embedsignature=True, language_level=3
# cython: linetrace=True
"""
Finite-difference modelling of the acoustic wave equation, with the library functions of the SU fork (``su_fdmod1_*``,
``su_fdmod2_*``, in ``su/main/synthetics_waveforms_testpatterns``).
"""
from .. cimport su
import numpy as np
cimport cython


def fdmod1_plan(const float[::1] rv, float dz, float tmax, int nt, int styp, float freq):
    """The time step, the number of time steps, the time shift of the source and the number of steps of the source: SUFDMOD1"""
    cdef:
        float dt, t0
        int nt_out, ies, status
    status = su.su_fdmod1_plan(&rv[0], rv.shape[0], dz, tmax, nt, styp, freq, &dt, &nt_out, &t0, &ies)
    if status != 0:
        raise ValueError("The parameters are not allowed.")
    return dt, nt_out, t0, ies


def fdmod1_run(const float[::1] rv, const float[::1] rd, float dz, float dt, int nt, float t0, int ies, int isz, int irz,
               int abs_top, int abs_bottom, int styp, float freq, int td, int zd, int press, bint snapshots):
    """The seismogram, and the snapshots if asked for, of SUFDMOD1: (seismogram, snapshots or None)"""
    cdef:
        int nz = rv.shape[0]
        int[2] absc
        int n_out, nsamples = nt // td + 1
        float[::1] sismo = np.zeros(nsamples, dtype=np.float32)
        float[:, ::1] snaps
        float *snaps_ptr = NULL
    absc[0] = abs_top
    absc[1] = abs_bottom
    if snapshots:
        snaps = np.zeros((nsamples, nz // zd), dtype=np.float32)
        snaps_ptr = &snaps[0, 0]
    with nogil:
        n_out = su.su_fdmod1_run(&rv[0], &rd[0], nz, dz, dt, nt, t0, ies, isz, irz, absc, styp, freq, td, zd, press,
                                 &sismo[0], snaps_ptr)
    if n_out < 0:
        raise ValueError("The parameters are not allowed.")
    return np.asarray(sismo)[:n_out], (np.asarray(snaps)[:n_out] if snapshots else None)


cdef class Fdmod2Run:
    """The time stepping of SUFDMOD2: the fields, the source and the seismograms. Made and driven by `seispy.synthetics.fdmod2`."""
    cdef:
        int nx, nz, ns, nt, mt, pwt, mono, it, hs1, vs2, abs_flags[4]
        float dx, dz, fx, fz, dt, t, fmax, fpeak, sstrength
        bint has_od, has_hs, has_vs, has_ss
        float[:, ::1] dvv, od, s, f0, f1, f2
        float[::1] xs, zs, vs_spline, xsd, zsd
        int[::1] ixs, izs
        float[:, ::1] _hs, _vs, _ss
        public object hs, vs, ss

    def __init__(self, float[:, ::1] dvv, od, float[::1] xs, float[::1] zs, int[::1] ixs, int[::1] izs,
                 float dx, float dz, float fx, float fz, float dt, int nt, int mt, float sstrength, int pwt, int mono,
                 float fmax, float fpeak, abs_flags, int hs1, int vs2, bint want_hs, bint want_vs, bint want_ss):
        cdef int i
        self.dvv = dvv
        self.nx = dvv.shape[0]
        self.nz = dvv.shape[1]
        self.has_od = od is not None
        if self.has_od:
            self.od = od
        self.xs = xs
        self.zs = zs
        self.ixs = ixs
        self.izs = izs
        self.ns = xs.shape[0]
        self.dx = dx
        self.dz = dz
        self.fx = fx
        self.fz = fz
        self.dt = dt
        self.nt = nt
        self.mt = mt
        self.sstrength = sstrength
        self.pwt = pwt
        self.mono = mono
        self.fmax = fmax
        self.fpeak = fpeak
        for i in range(4):
            self.abs_flags[i] = abs_flags[i]
        self.hs1 = hs1
        self.vs2 = vs2
        self.s = np.zeros((self.nx, self.nz), dtype=np.float32)
        self.f0 = np.zeros((self.nx, self.nz), dtype=np.float32)  # t - dt
        self.f1 = np.zeros((self.nx, self.nz), dtype=np.float32)  # t
        self.f2 = np.zeros((self.nx, self.nz), dtype=np.float32)  # t + dt
        if self.ns > 1:
            self.vs_spline = np.zeros(self.ns, dtype=np.float32)
            self.xsd = np.zeros(self.ns * 4, dtype=np.float32)
            self.zsd = np.zeros(self.ns * 4, dtype=np.float32)
            su.su_fdmod2_exsrc_setup(self.ns, &self.xs[0], &self.zs[0], &self.vs_spline[0], &self.xsd[0], &self.zsd[0])
        self.has_hs = want_hs
        self.has_vs = want_vs
        self.has_ss = want_ss
        self.hs = np.zeros((self.nx, nt), dtype=np.float32) if want_hs else None
        self.vs = np.zeros((self.nz, nt), dtype=np.float32) if want_vs else None
        self.ss = np.zeros((self.ns, nt), dtype=np.float32) if want_ss else None
        if want_hs:
            self._hs = self.hs
        if want_vs:
            self._vs = self.vs
        if want_ss:
            self._ss = self.ss
        self.it = 0
        self.t = 0.0

    @property
    def finished(self):
        return self.it >= self.nt

    @property
    def step_number(self):
        return self.it

    def advance(self, bint want_frame):
        """Do one time step. Returns a copy of the new pressure field (nx, nz) if this step is recorded (every mt-th) and
        ``want_frame``, otherwise None."""
        cdef:
            int ix, iz, is_, it = self.it
            const float *od = NULL
            float[:, ::1] tmp
            bint recorded = (it % self.mt) == 0
        if it >= self.nt:
            raise StopIteration()
        if self.has_od:
            od = &self.od[0, 0]
        with nogil:
            # update the source function
            if self.ns == 1:
                su.su_fdmod2_ptsrc(self.sstrength, self.xs[0], self.zs[0], self.nx, self.dx, self.fx, self.nz, self.dz, self.fz,
                                   self.dt, self.t, self.fmax, self.fpeak, self.mono, &self.s[0, 0])
            else:
                su.su_fdmod2_exsrc(self.ns, &self.xs[0], &self.zs[0], &self.vs_spline[0], &self.xsd[0], &self.zsd[0],
                                   self.nx, self.dx, self.fx, self.nz, self.dz, self.fz,
                                   self.dt, self.t, self.fpeak, self.pwt, self.mono, &self.s[0, 0])
            # one time step
            su.su_fdmod2_tstep(self.nx, self.dx, self.nz, self.dz, self.dt, &self.dvv[0, 0], od, &self.s[0, 0],
                               &self.f0[0, 0], &self.f1[0, 0], &self.f2[0, 0], self.abs_flags)
        frame = np.array(self.f2, dtype=np.float32).T.copy() if (recorded and want_frame) else None
        if self.has_hs:
            for ix in range(self.nx):
                self._hs[ix, it] = self.f2[ix, self.hs1]
        if self.has_vs:
            for iz in range(self.nz):
                self._vs[iz, it] = self.f2[self.vs2, iz]
        if self.has_ss:
            for is_ in range(self.ns):
                self._ss[is_, it] = self.f2[self.ixs[is_], self.izs[is_]]
        # roll the time slices
        tmp = self.f0
        self.f0 = self.f1
        self.f1 = self.f2
        self.f2 = tmp
        self.t += self.dt
        self.it += 1
        return frame
