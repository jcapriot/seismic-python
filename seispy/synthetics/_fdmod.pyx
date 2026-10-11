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
        bint has_od, has_hs, has_vs, has_ss, pml_program
        su.FdmodPml *pml
        float[:, ::1] dvv, od, s, f0, f1, f2
        float[::1] xs, zs, vs_spline, xsd, zsd
        int[::1] ixs, izs
        float[:, ::1] _hs, _vs, _ss
        public object hs, vs, ss

    def __init__(self, float[:, ::1] dvv, od, float[::1] xs, float[::1] zs, int[::1] ixs, int[::1] izs,
                 float dx, float dz, float fx, float fz, float dt, int nt, int mt, float sstrength, int pwt, int mono,
                 float fmax, float fpeak, abs_flags, int hs1, int vs2, bint want_hs, bint want_vs, bint want_ss,
                 int pml_thick=0, float pml_max=1000.0, bint pml_program=False):
        cdef int i
        self.pml = NULL
        self.pml_program = pml_program
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
        if pml_thick > 0:
            self.pml = su.su_fdmod2_pml_init(self.nx, self.nz, dx, dz, dt, pml_thick, pml_max, &self.dvv[0, 0],
                                             &self.od[0, 0] if self.has_od else NULL)
        self.it = 0
        self.t = 0.0

    def __dealloc__(self):
        if self.pml is not NULL:
            su.su_fdmod2_pml_free(self.pml)
            self.pml = NULL

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
            elif self.pml_program:
                su.su_fdmod2pml_exsrc(self.ns, &self.xs[0], &self.zs[0], &self.vs_spline[0], &self.xsd[0], &self.zsd[0],
                                      self.nx, self.dx, self.fx, self.nz, self.dz, self.fz,
                                      self.dt, self.t, self.fmax, &self.s[0, 0])
            else:
                su.su_fdmod2_exsrc(self.ns, &self.xs[0], &self.zs[0], &self.vs_spline[0], &self.xsd[0], &self.zsd[0],
                                   self.nx, self.dx, self.fx, self.nz, self.dz, self.fz,
                                   self.dt, self.t, self.fpeak, self.pwt, self.mono, &self.s[0, 0])
            # one time step
            if self.pml is NULL:
                su.su_fdmod2_tstep(self.nx, self.dx, self.nz, self.dz, self.dt, &self.dvv[0, 0], od, &self.s[0, 0],
                                   &self.f0[0, 0], &self.f1[0, 0], &self.f2[0, 0], self.abs_flags)
            else:
                su.su_fdmod2_pml_tstep(self.pml, self.nx, self.dx, self.nz, self.dz, self.dt, &self.dvv[0, 0], od,
                                       &self.s[0, 0], &self.f0[0, 0], &self.f1[0, 0], &self.f2[0, 0], self.abs_flags)
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


cdef const float *_ptr(object a):
    # (a is None, or a C-contiguous float32 2-D array that outlives the call)
    cdef float[:, ::1] v
    if a is None:
        return NULL
    v = a
    return &v[0, 0]


cdef class Ea2dfRun:
    """The model and the time stepping of SUEA2DF. Made and driven by `seispy.synthetics.ea2df`."""
    cdef:
        su.Ea2dfModel *M
        public int nx, nz, nt, nxpadded, nzpadded
        public object wbc

    def __cinit__(self):
        self.M = NULL

    def __dealloc__(self):
        if self.M is not NULL:
            su.su_ea2df_free(self.M)
            self.M = NULL

    def __init__(self, float dt, float ft, float lt, int nx, float dx, float fx, int nz, float dz, float fz, float sx, float sz,
                 bytes stype, float sang, bytes wtype, float ts, float favg, int qsw, int aniso, int tsw, bc, float bc_a,
                 float bc_r, float hsz, float vsx, c11, c55, rho, c13, c33, c15, c35, q):
        cdef:
            int bcc[4]
            int err = 0, i, done
            int wbcc[4]
        for i in range(4):
            bcc[i] = bc[i]
        self.M = su.su_ea2df_create(dt, ft, lt, nx, dx, fx, nz, dz, fz, sx, sz, stype, sang, wtype, ts, favg, qsw, aniso, tsw, bcc,
                                    bc_a, bc_r, hsz, vsx, _ptr(c11), _ptr(c55), _ptr(rho), _ptr(c13), _ptr(c33), _ptr(c15),
                                    _ptr(c35), _ptr(q), &err)
        if self.M is NULL:
            messages = {
                -1: "sx: the source must lie within the model.",
                -2: "sz: the source must lie within the model.",
                -3: "The parameters are not allowed.",
                -4: "Unknown source type (stype) or wave type (wtype).",
                -5: "hsz or vsx is outside of the model.",
                -6: "The densities must be positive.",
            }
            raise ValueError(messages.get(err, "The model could not be made."))
        su.su_ea2df_info(self.M, &self.nxpadded, &self.nzpadded, &self.nt, wbcc, &done)
        self.nx = nx
        self.nz = nz
        self.wbc = [wbcc[0], wbcc[1], wbcc[2], wbcc[3]]

    @property
    def step_number(self):
        cdef int nxp, nzp, nt, done
        cdef int wbcc[4]
        su.su_ea2df_info(self.M, &nxp, &nzp, &nt, wbcc, &done)
        return done

    def advance(self):
        """Do the next time step; False when the model has done nt steps."""
        cdef int r
        with nogil:
            r = su.su_ea2df_step(self.M)
        return bool(r)

    def snapshot(self, int which):
        """A field of the model at this time without the boundary layers (nz, nx): 0 u, 1 w, 2 txx, 3 tzz, 4 txz"""
        out = np.empty((self.nz, self.nx), dtype=np.float32)
        cdef float[:, ::1] v = out
        su.su_ea2df_snapshot(self.M, which, &v[0, 0])
        return out

    def lines(self):
        """The recorded lines, copies: (hu, hw) [nt, nxpadded] and (vu, vw) [nt, nzpadded] of the particle velocity"""
        cdef const float *hu
        cdef const float *hw
        cdef const float *vu
        cdef const float *vw
        su.su_ea2df_lines(self.M, &hu, &hw, &vu, &vw)
        cdef float[:, ::1] a = <float[:self.nt, :self.nxpadded]> hu
        cdef float[:, ::1] b = <float[:self.nt, :self.nxpadded]> hw
        cdef float[:, ::1] c = <float[:self.nt, :self.nzpadded]> vu
        cdef float[:, ::1] d = <float[:self.nt, :self.nzpadded]> vw
        return np.array(a), np.array(b), np.array(c), np.array(d)


def remac2d_run(int opflag, int nx, int nz, int nt, float dx, float dz, float dt, const int[::1] isx, const int[::1] isz, const float[::1] amps, float w,
                int sflag, float fmax, const float[::1] wave, float dtwave, int fsflag, float vmaxu, float dtsnap, int iabso, float abso,
                int nbwx, int nbwz, const float[:, ::1] vel, dens, const int[::1] irz, const int[::1] irx, int nsnap):
    """The pressure sections along the horizontal (irz) and vertical (irx) receiver lines, and the snapshots, of SUREMAC2D"""
    cdef:
        int nsectx = irz.shape[0], nsectz = irx.shape[0], status
        float[:, :, ::1] sectx = np.zeros((max(nsectx, 1), nt, nx), dtype=np.float32)
        float[:, :, ::1] sectz = np.zeros((max(nsectz, 1), nt, nz), dtype=np.float32)
        float[:, :, ::1] snap = np.zeros((max(nsnap, 1), nz, nx), dtype=np.float32)
        float[:, ::1] d
        const float *dp = NULL
        const float *wp = NULL
        const int *rzp = NULL
        const int *rxp = NULL
    if dens is not None:
        d = dens
        dp = &d[0, 0]
    if wave.shape[0] > 0:
        wp = &wave[0]
    if nsectx:
        rzp = &irz[0]
    if nsectz:
        rxp = &irx[0]
    with nogil:
        status = su.su_remac2d(opflag, nx, nz, nt, dx, dz, dt, isx.shape[0], &isx[0], &isz[0], &amps[0], w, sflag, fmax,
                               wave.shape[0], dtwave, wp, fsflag, vmaxu, dtsnap, iabso, abso, nbwx, nbwz, &vel[0, 0], dp,
                               nsectx, rzp, nsectz, rxp, &sectx[0, 0, 0], &sectz[0, 0, 0], nsnap, &snap[0, 0, 0])
    if status != 0:
        messages = {
            -1: "nx and nz have to be lengths of the Fourier transform (see the table nctab of pfafft), and odd unless opflag=1.",
            -2: "A source box is beyond the model's boundary.",
            -3: "The parameters are not allowed (or too few expansion terms: increase nt*dt or the velocity).",
            -4: "vmax is zero.",
            -5: "fmax is too high for the spatial sampling.",
            -6: "vmaxu is smaller than the maximum velocity.",
            -7: "A receiver line is outside of the model.",
        }
        raise ValueError(messages.get(status, "SUREMAC2D failed."))
    return np.asarray(sectx)[:nsectx], np.asarray(sectz)[:nsectz], np.asarray(snap)[:nsnap]


cdef const float *_cptr(object a):
    cdef const float[:, ::1] v
    if a is None:
        return NULL
    v = a
    return &v[0, 0]


def remel2dan_run(int amode, int nx, int nz, int nt, float dx, float dz, float dt, const int[::1] isx, const int[::1] isz,
                  const int[::1] styp, const float[::1] samp, float w, int sflag, float fmax, const float[::1] wave, float dtwave,
                  float vmaxu, float vmax0, float vmin0, float dtsnap, const int[::1] sntyp, int nsnap, int iabso, float abso,
                  int nbwx, int nbwz, dens, vp, vs, c11, c13, c15, c33, c35, c55, const int[::1] irz, const int[::1] rxtyp,
                  const int[::1] irx, const int[::1] rztyp):
    """The sections along the horizontal (irz) and vertical (irx) receiver lines, and the snapshots, of SUREMEL2DAN"""
    cdef:
        int nsectx = irz.shape[0], nsectz = irx.shape[0], nsntyp = sntyp.shape[0], status
        float[:, :, ::1] xsect = np.zeros((max(nsectx, 1), nt, nx), dtype=np.float32)
        float[:, :, ::1] zsect = np.zeros((max(nsectz, 1), nt, nz), dtype=np.float32)
        float[:, :, :, ::1] snap = np.zeros((max(nsntyp, 1), max(nsnap, 1), nz, nx), dtype=np.float32)
        const float *wp = NULL
        const int *p_irz = NULL
        const int *p_rxtyp = NULL
        const int *p_irx = NULL
        const int *p_rztyp = NULL
        const int *p_sntyp = NULL
        const float *p_dens = _cptr(dens)
        const float *p_vp = _cptr(vp)
        const float *p_vs = _cptr(vs)
        const float *p_c11 = _cptr(c11)
        const float *p_c13 = _cptr(c13)
        const float *p_c15 = _cptr(c15)
        const float *p_c33 = _cptr(c33)
        const float *p_c35 = _cptr(c35)
        const float *p_c55 = _cptr(c55)
    if wave.shape[0] > 0:
        wp = &wave[0]
    if nsectx:
        p_irz = &irz[0]
        p_rxtyp = &rxtyp[0]
    if nsectz:
        p_irx = &irx[0]
        p_rztyp = &rztyp[0]
    if nsntyp:
        p_sntyp = &sntyp[0]
    with nogil:
        status = su.su_remel2dan(amode, nx, nz, nt, dx, dz, dt, isx.shape[0], &isx[0], &isz[0], &styp[0], &samp[0], w, sflag, fmax,
                                 wave.shape[0], dtwave, wp, vmaxu, vmax0, vmin0, dtsnap, nsntyp, p_sntyp, nsnap, iabso, abso, nbwx, nbwz,
                                 p_dens, p_vp, p_vs, p_c11, p_c13, p_c15, p_c33, p_c35,
                                 p_c55, nsectx, p_irz, p_rxtyp, nsectz, p_irx, p_rztyp,
                                 &xsect[0, 0, 0], &zsect[0, 0, 0], &snap[0, 0, 0, 0])
    if status != 0:
        messages = {
            -1: "nx and nz have to be lengths of the Fourier transform (see the table nctab of pfafft), and odd.",
            -2: "A source box is beyond the model's boundary.",
            -3: "The parameters are not allowed (or too few expansion terms: increase nt*dt or the velocity).",
            -4: "vmax is zero.",
            -5: "fmax is too high for the spatial sampling.",
            -6: "vmaxu is smaller than the maximum velocity.",
            -7: "A receiver line is outside of the model.",
        }
        raise ValueError(messages.get(status, "SUREMEL2DAN failed."))
    return np.asarray(xsect)[:nsectx], np.asarray(zsect)[:nsectz], np.asarray(snap)[:nsntyp, :nsnap]


cdef class FctRun:
    """The model and the time stepping of SUFCTANISMOD. Made and driven by `seispy.synthetics.fctanismod`."""
    cdef:
        su.FctModel *M
        public int nx, nz, nt

    def __cinit__(self):
        self.M = NULL

    def __dealloc__(self):
        if self.M is not NULL:
            su.su_fctanismod_free(self.M)
            self.M = NULL

    def __init__(self, int nx, int nz, int nt, float dx, float dz, float dt, int sx, int sz, int receiverdepth, int vspnx,
                 int impulse, int source, int isurf, int dofct, int fctxbeg, int fctzbeg, int fctxend, int fctzend,
                 int forcex, int forcey, int forcez, int wavelet, int movebc, float fpeak, float eta0, float eta,
                 float deta0dx, float deta0dz, float detadx, float detadz,
                 const float[:, ::1] aa, const float[:, ::1] cc, const float[:, ::1] ff, const float[:, ::1] ll,
                 const float[:, ::1] nn, const float[:, ::1] rho, xzsource):
        cdef:
            int err = 0
            const float[:, ::1] xz
            const float *xzp = NULL
        if xzsource is not None:
            xz = xzsource
            xzp = &xz[0, 0]
        self.M = su.su_fctanismod_create(nx, nz, nt, dx, dz, dt, sx, sz, receiverdepth, vspnx, impulse, source, isurf, dofct,
                                         fctxbeg, fctzbeg, fctxend, fctzend, forcex, forcey, forcez, wavelet, movebc, fpeak,
                                         eta0, eta, deta0dx, deta0dz, detadx, detadz, &aa[0, 0], &cc[0, 0], &ff[0, 0],
                                         &ll[0, 0], &nn[0, 0], &rho[0, 0], xzp, &err)
        if self.M is NULL:
            messages = {
                -3: "The parameters are not allowed.",
                -4: "wavelet is 1 (AKB), 2 (Ricker), 3 (impulse) or 4 (unity).",
                -5: "The source, the receiver depth or the VSP position is outside of the grid.",
            }
            raise ValueError(messages.get(err, "The model could not be made."))
        self.nx = nx
        self.nz = nz
        self.nt = nt

    @property
    def vmax(self):
        return su.su_fctanismod_vmax(self.M)

    @property
    def step_number(self):
        return su.su_fctanismod_done(self.M)

    def advance(self):
        """Do the next time step; False when the model has done nt steps."""
        cdef int r
        with nogil:
            r = su.su_fctanismod_step(self.M)
        return bool(r)

    def snapshot(self, int which):
        """A component of the wavefield (nz, nx) at this time: 0 u (x), 1 v (y), 2 w (z)"""
        out = np.empty((self.nx, self.nz), dtype=np.float32)
        cdef float[:, ::1] v = out
        su.su_fctanismod_snapshot(self.M, which, &v[0, 0])
        return np.ascontiguousarray(out.T)

    def records(self):
        """Copies of the seismograms: refl (3, nx, nt), the horizontal line at the receiver depth, and vsp (3, nz, nt)"""
        cdef const float *refl
        cdef const float *vsp
        su.su_fctanismod_records(self.M, &refl, &vsp)
        cdef float[:, :, ::1] a = <float[:3, :self.nx, :self.nt]> <float *> refl
        cdef float[:, :, ::1] b = <float[:3, :self.nz, :self.nt]> <float *> vsp
        return np.array(a), np.array(b)
