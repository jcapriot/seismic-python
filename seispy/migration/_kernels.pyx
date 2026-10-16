# cython: embedsignature=True, language_level=3
"""
Migration, with the library functions of the SU fork (``su_stolt``, ``su_migfd``, ``su_migffd``, ``su_migps``,
``su_migpspi``, in ``su/main/migration_inversion``). The sections are float32 arrays (traces, samples).
"""
from .. cimport su
import numpy as np
from ..stretching import _resample  # (this fills the sinc table that ints8r and ints8c use, before any thread can need it)


_MESSAGES = {-3: "The parameters are not allowed."}


def _check(int status):
    if status != 0:
        raise ValueError(_MESSAGES.get(status, "The migration failed."))


def stolt(const float[:, ::1] gather, float dt, float dx, const float[::1] tmig, const float[::1] vmig, float vscale,
          float smig, float fmax, int lstaper, int lbtaper):
    """Stolt migration of one common-offset gather (nx, nt) of the cdps in order; the velocity pairs are those of the program"""
    cdef int nx = gather.shape[0], nt = gather.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nt), dtype=np.float32)
    with nogil:
        status = su.su_stolt(nx, nt, dt, dx, tmig.shape[0], &tmig[0], &vmig[0], vscale, smig, fmax, lstaper, lbtaper,
                             &gather[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migfd(const float[:, ::1] section, const float[:, ::1] vel, int nz, float dz, float dt, float dx, int dip):
    """45 to 90 degree finite-difference depth migration of the section (nx, nt) with the velocity (nz, nx): (nx, nz)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nz), dtype=np.float32)
    if vel.shape[0] != nz or vel.shape[1] != nx:
        raise ValueError("The velocity must be (nz, nx).")
    with nogil:
        status = su.su_migfd(nx, nt, nz, dz, dt, dx, dip, &section[0, 0], &vel[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migffd(const float[:, ::1] section, const float[:, ::1] vel, int nz, float dz, float dt, float dx):
    """Fourier finite-difference depth migration of the section (nx, nt) with the velocity (nz, nx): (nx, nz)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nz), dtype=np.float32)
    if vel.shape[0] != nz or vel.shape[1] != nx:
        raise ValueError("The velocity must be (nz, nx).")
    with nogil:
        status = su.su_migffd(nx, nt, nz, dz, dt, dx, &section[0, 0], &vel[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migps(const float[:, ::1] section, float dt, float dx, const float[::1] ffil, int nxpad, int ltaper, int np_, int ntflag,
          const float[::1] vt):
    """Phase shift migration with turning rays of the section (nx, nt), with the interval velocities vt(t): (nx, nt)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nt), dtype=np.float32)
    if ffil.shape[0] != 4 or vt.shape[0] != nt:
        raise ValueError("ffil needs 4 values, and vt a velocity for each time sample.")
    with nogil:
        status = su.su_migps(nx, nt, dt, dx, &ffil[0], nxpad, ltaper, np_, ntflag, &vt[0], &section[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migpspi(const float[:, ::1] section, const float[:, ::1] vel, int nz, float dz, float dt, float dx):
    """Phase shift plus interpolation depth migration of the section (nx, nt), velocity (nz, nx): (nx, nz)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nz), dtype=np.float32)
    if vel.shape[0] != nz or vel.shape[1] != nx:
        raise ValueError("The velocity must be (nz, nx).")
    with nogil:
        status = su.su_migpspi(nx, nt, nz, dz, dt, dx, &section[0, 0], &vel[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


# the tables of the sinc interpolators are made when this module is imported (ints8r and ints8c do it at their first call)
su.su_ints8_tables()


def kdmig2d_reference(int nr, int nzt, float dxo, float dzt, float fzt, float dvz, float v0):
    """The reference traveltime, lateral slowness, cosine and angle tables (nr, nzt) of SUKDMIG2D"""
    cdef float[:, ::1] tb = np.zeros((nr, nzt), dtype=np.float32)
    cdef float[:, ::1] pb = np.zeros((nr, nzt), dtype=np.float32)
    cdef float[:, ::1] cs0b = np.zeros((nr, nzt), dtype=np.float32)
    cdef float[:, ::1] angb = np.zeros((nr, nzt), dtype=np.float32)
    su.su_kdmig2d_reference(nr, nzt, dxo, dzt, fzt, dvz, v0, &tb[0, 0], &pb[0, 0], &cs0b[0, 0], &angb[0, 0])
    return np.asarray(tb), np.asarray(pb), np.asarray(cs0b), np.asarray(angb)


def kdmig2d_residual(float[:, :, ::1] ttab, float fs, float ds, float fxt, float dxt, int nr, float dxo,
                     const float[:, ::1] tb):
    """Subtracts the reference traveltime from the tables ttab (ns, nxt, nzt), in place"""
    cdef int ns = ttab.shape[0], nxt = ttab.shape[1], nzt = ttab.shape[2]
    su.su_kdmig2d_residual(ns, fs, ds, nxt, fxt, dxt, nzt, nr, dxo, &tb[0, 0], &ttab[0, 0, 0])


cdef class Kdmig2d:
    """The state of a SUKDMIG2D migration: the tables, and the images that the traces are added to"""
    cdef:
        object tb, pb, cs0b, angb, ttab, tv, cs, mig, mig1
        float[:, :, ::1] ttab_v, mig_v, mig1_v
        const float[:, :, ::1] tv_v, cs_v
        float[:, ::1] tsum, tt, tvsum, cssum
        const float[:, ::1] tb_v, pb_v, cs0b_v, angb_v
        int ns, nxt, nzt, nr, nxo, nzo, noff, mtmax, ls, npv
        float fs, ds, es, fxt, dxt, fzt, dzt, fxo, dxo, fzo, dzo, dxm, fmax, angmax, aperx, offmax, dt, ft

    def __init__(self, ttab, tv, cs, int nr, tb, pb, cs0b, angb, int noff, int nxo, float fxo, float dxo, int nzo, float fzo,
                 float dzo, float fs, float ds, float es, float fxt, float dxt, float fzt, float dzt, float dxm, float fmax,
                 float angmax, float aperx, float offmax, int mtmax, int ls, float dt, float ft):
        self.ttab = ttab
        self.ttab_v = ttab
        self.ns, self.nxt, self.nzt = ttab.shape
        self.npv = tv is not None
        if self.npv:
            self.tv = tv
            self.cs = cs
            self.tv_v = tv
            self.cs_v = cs
            self.tvsum = np.zeros((self.nxt, self.nzt), dtype=np.float32)
            self.cssum = np.zeros((self.nxt, self.nzt), dtype=np.float32)
            self.mig1 = np.zeros((noff, nxo, nzo), dtype=np.float32)
            self.mig1_v = self.mig1
        self.tb, self.pb, self.cs0b, self.angb = tb, pb, cs0b, angb
        self.tb_v, self.pb_v, self.cs0b_v, self.angb_v = tb, pb, cs0b, angb
        self.nr, self.nxo, self.nzo, self.noff, self.mtmax, self.ls = nr, nxo, nzo, noff, mtmax, ls
        self.fs, self.ds, self.es, self.fxt, self.dxt, self.fzt, self.dzt = fs, ds, es, fxt, dxt, fzt, dzt
        self.fxo, self.dxo, self.fzo, self.dzo, self.dxm = fxo, dxo, fzo, dzo, dxm
        self.fmax, self.angmax, self.aperx, self.offmax, self.dt, self.ft = fmax, angmax, aperx, offmax, dt, ft
        self.tsum = np.zeros((self.nxt, self.nzt), dtype=np.float32)
        self.tt = np.zeros((self.nxt, self.nzt), dtype=np.float32)
        self.mig = np.zeros((noff, nxo, nzo), dtype=np.float32)
        self.mig_v = self.mig

    def add(self, const float[::1] data, float sx, float gx, int io):
        """Migrate one trace into the image of the offset io. Returns False if it is outside of the shots or the offset limit."""
        cdef int status
        cdef const float *tvp = NULL
        cdef const float *csp = NULL
        cdef float *tvs = NULL
        cdef float *css = NULL
        cdef float *m1 = NULL
        if self.npv:
            tvp = &self.tv_v[0, 0, 0]
            csp = &self.cs_v[0, 0, 0]
            tvs = &self.tvsum[0, 0]
            css = &self.cssum[0, 0]
            m1 = &self.mig1_v[io, 0, 0]
        with nogil:
            status = su.su_kdmig2d_trace(&data[0], data.shape[0], self.ft, self.dt, sx, gx, &self.mig_v[io, 0, 0], self.aperx,
                                         self.nxo, self.fxo, self.dxo, self.nzo, self.fzo, self.dzo, self.ls, self.mtmax,
                                         self.dxm, self.fmax, self.angmax, &self.tb_v[0, 0], &self.pb_v[0, 0],
                                         &self.cs0b_v[0, 0], &self.angb_v[0, 0], self.nr, &self.ttab_v[0, 0, 0], self.ns,
                                         self.fs, self.ds, self.es, self.offmax, &self.tsum[0, 0], &self.tt[0, 0], self.nzt,
                                         self.fzt, self.dzt, self.nxt, self.fxt, self.dxt, self.npv, tvp, csp, tvs, css, m1)
        if status < 0:
            raise ValueError("The parameters are not allowed.")
        return status == 0

    @property
    def images(self):
        return self.mig, (self.mig1 if self.npv else None)


def ktmig2d(const float[:, ::1] data, const float[:, ::1] vel, float dt, float dx, float h, float angmax, int nfc, int fwidth):
    """Prestack time migration with the double square root operator of a common-offset section (ntr, nt), with the rms
    velocities (ntr, nt) of the cdp of each trace"""
    cdef int ntr = data.shape[0], nt = data.shape[1], status
    cdef float[:, ::1] out = np.zeros((ntr, nt), dtype=np.float32)
    if vel.shape[0] != ntr or vel.shape[1] != nt:
        raise ValueError("The velocities must be (ntr, nt).")
    with nogil:
        status = su.su_ktmig2d(ntr, nt, dt, dx, h, angmax, nfc, fwidth, &data[0, 0], &vel[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migsplit(const float[:, ::1] section, const float[:, ::1] vel, int nz, float dz, float dt, float dx):
    """Split-step Fourier depth migration of the section (nx, nt) with the velocity (nz, nx): (nx, nz)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nz), dtype=np.float32)
    if vel.shape[0] != nz or vel.shape[1] != nx:
        raise ValueError("The velocity must be (nz, nx).")
    with nogil:
        status = su.su_migsplit(nx, nt, nz, dz, dt, dx, &section[0, 0], &vel[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def gazmig(const float[:, ::1] section, float dt, float ft, float dx, int ntau, float dtau, float ftau,
           const float[::1] vt, float qual, float ceil):
    """Gazdag phase shift migration (with Q) of the section (nx, nt) with the velocity vt of the ntau migrated times:
    (nx, ntau)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, ntau), dtype=np.float32)
    if vt.shape[0] != ntau:
        raise ValueError("vt needs a velocity for each migrated time sample.")
    with nogil:
        status = su.su_gazmig(nx, nt, dt, ft, dx, ntau, dtau, ftau, &vt[0], qual, ceil, &section[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migtk(const float[:, ::1] section, float dt, float dx, float fmax, int nxpad, int ltaper, float speed,
          const float[::1] vt):
    """The t-k migration of the section (nx, nt) with the interval velocities vt(t): (nx, nt)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nt), dtype=np.float32)
    if vt.shape[0] != nt:
        raise ValueError("vt needs a velocity for each time sample.")
    with nogil:
        status = su.su_migtk(nx, nt, dt, dx, fmax, nxpad, ltaper, speed, &vt[0], &section[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def migpsti(const float[:, ::1] section, float dt, float dx, const float[::1] ffil, int nxpad, int ltaper, int np_,
            int ntflag, const float[::1] vnt, const float[::1] vt, const float[::1] etat):
    """Phase shift migration with turning rays of the section (nx, nt) for a TI medium, with the functions of time of the
    NMO velocity, the vertical velocity and eta: (nx, nt)"""
    cdef int nx = section.shape[0], nt = section.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nt), dtype=np.float32)
    if ffil.shape[0] != 4 or vnt.shape[0] != nt or vt.shape[0] != nt or etat.shape[0] != nt:
        raise ValueError("ffil needs 4 values, and the velocities and eta a value for each time sample.")
    with nogil:
        status = su.su_migpsti(nx, nt, dt, dx, &ffil[0], nxpad, ltaper, np_, ntflag, &vnt[0], &vt[0], &etat[0],
                               &section[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)


def _shot_status(int status):
    if status == -3:
        raise ValueError("The parameters (or the aperture, the frequencies f1 to f4 or the velocities) are not allowed.")
    if status != 0:
        raise ValueError("The migration failed.")


def migpre_shot(str program, const float[:, ::1] gather, const float[:, ::1] vel, float dt, float dx, float dz, int dip,
                int nflag, float fmax, float f1, float f2, float f3, float f4, int ix2, int ix3, int ixshot,
                float[:, ::1] image):
    """Add the image of one shot gather (nxo, nt) to image (nxo, nz): program is 'fd', 'ffd', 'sp' or 'pspi'; vel is (nz, nxo)"""
    cdef int nxo = gather.shape[0], nt = gather.shape[1], nz = image.shape[1], status = 0
    if vel.shape[0] != nz or vel.shape[1] != nxo or image.shape[0] != nxo:
        raise ValueError("The velocity must be (nz, nxo) and the image (nxo, nz).")
    if program == 'fd':
        with nogil:
            status = su.su_migprefd_shot(nt, nxo, nz, dt, dx, dz, dip, fmax, f1, f2, f3, f4, ix2, ix3, ixshot,
                                         &gather[0, 0], &vel[0, 0], &image[0, 0])
    elif program == 'ffd':
        with nogil:
            status = su.su_migpreffd_shot(nt, nxo, nz, dt, dx, dz, dip, fmax, f1, f2, f3, f4, ix2, ix3, ixshot,
                                          &gather[0, 0], &vel[0, 0], &image[0, 0])
    elif program == 'sp':
        with nogil:
            status = su.su_migpresp_shot(nt, nxo, nz, dt, dx, dz, fmax, f1, f2, f3, f4, ix2, ix3, ixshot,
                                         &gather[0, 0], &vel[0, 0], &image[0, 0])
    elif program == 'pspi':
        with nogil:
            status = su.su_migprepspi_shot(nt, nxo, nz, dt, dx, dz, nflag, fmax, f1, f2, f3, f4, ix2, ix3, ixshot,
                                           &gather[0, 0], &vel[0, 0], &image[0, 0])
    else:
        raise ValueError("unknown program")
    _shot_status(status)


def migtopo2d_trace(const float[::1] trace, float[:, ::1] mig, float ft, float dt, float sx, float gx, float aperx,
                    float fx, float dx, float fz, float dz, int ls, int mtmax, float dxm, float fmax, float angmax,
                    float fs, float ds, const float[:, :, ::1] tb, const float[:, :, ::1] pb,
                    const float[:, :, ::1] angb, const float[:, :, ::1] sigb, const float[:, :, ::1] ttab, float fzt,
                    float dzt, float fxt, float dxt, float fxi, float dxi, const float[::1] nangl):
    """Migrate one trace into the section mig (nxo, nzo); the tables are those of datuming._kernels.datumk2d_tables"""
    cdef:
        int nt = trace.shape[0], nx = mig.shape[0], nz = mig.shape[1], status
        int ns = ttab.shape[0], nxt = ttab.shape[1], nzt = ttab.shape[2], nr = tb.shape[1], nxi = nangl.shape[0]
    with nogil:
        status = su.su_migtopo2d_trace(&trace[0], nt, ft, dt, sx, gx, &mig[0, 0], aperx, nx, fx, dx, nz, fz, dz, ls,
                                       mtmax, dxm, fmax, angmax, ns, fs, ds, nr, &tb[0, 0, 0], &pb[0, 0, 0],
                                       &angb[0, 0, 0], &sigb[0, 0, 0], &ttab[0, 0, 0], nzt, fzt, dzt, nxt, fxt, dxt,
                                       nxi, fxi, dxi, &nangl[0])
    if status:
        raise ValueError("The trace is too long for the filter.")


def invvxzco(const float[:, ::1] data, float dt, float offs, float dxm, float fxm, const float[:, ::1] vel, float dx,
             float fx, float dz, int nxd, int nxb, int nxc, int nzc, float fxo, float dxo, int nxo, float fzo, float dzo,
             int nzo, float fmax, float ang, int ls, const float[:, ::1] vp):
    """The inversion (migration) of a common-offset gather (nxm, nt) with the velocity (nx, nz) (z fast): (nxo, nzo)"""
    cdef int nxm = data.shape[0], nt = data.shape[1], nx = vel.shape[0], nz = vel.shape[1], status
    cdef int pert = vp.shape[0] > 0
    cdef float[:, ::1] out = np.zeros((nxo, nzo), dtype=np.float32)
    cdef const float *vpp = NULL
    if pert:
        if vp.shape[0] != nx or vp.shape[1] != nz:
            raise ValueError("The slowness perturbation must have the shape of the velocity.")
        vpp = &vp[0, 0]
    with nogil:
        status = su.su_invvxzco(nxm, nt, dt, offs, dxm, fxm, nx, nz, dx, fx, dz, nxd, nxb, nxc, nzc, fxo, dxo, nxo, fzo, dzo,
                                nzo, fmax, ang, ls, pert, &vel[0, 0], vpp, &data[0, 0], &out[0, 0])
    if status == -1:
        raise ValueError("The shot or receiver lie outside of the specified (x,z) grid, or the first output trace is out of "
                         "the velocity model.")
    if status == -4:
        raise ValueError("The band nxb is too small for the offset.")
    if status == -5:
        raise ValueError("A raypath has too large a curvature: a smoother velocity is required.")
    if status != 0:
        raise ValueError("The parameters are not allowed.")
    return np.asarray(out)


def miggbzo(const float[:, ::1] section, const float[:, ::1] vel, float dt, float dx, float dz, float bwh, float fmin,
            float fmax, float amin, float amax):
    """Migration of the zero-offset section (nx, nt) by Gaussian beams, with the velocity (nx, nz) (z fast): (nx, nz)"""
    cdef int nx = section.shape[0], nt = section.shape[1], nz = vel.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nz), dtype=np.float32)
    if vel.shape[0] != nx:
        raise ValueError("The velocity must be (nx, nz), with nx the number of traces.")
    with nogil:
        status = su.su_miggbzo(nt, dt, nx, dx, nz, dz, bwh, fmin, fmax, amin, amax, &section[0, 0], &vel[0, 0], &out[0, 0])
    if status == -4:
        raise ValueError("The beam half-width bwh is too small for the trace interval dx.")
    if status != 0:
        raise ValueError("The parameters are not allowed.")
    return np.asarray(out)


def miggbzoan(const float[:, ::1] section, const float[:, ::1] a1111, const float[:, ::1] a3333,
              const float[:, ::1] a1133, const float[:, ::1] a1313, const float[:, ::1] a1113,
              const float[:, ::1] a3313, float dt, float dx, float dz, float bwh, float fmin, float fmax, float amin,
              float amax):
    """Migration of the zero-offset section (nx, nt) by Gaussian beams in an anisotropic medium of the elastic coefficients
    (nx, nz) (z fast): (nx, nz)"""
    cdef int nx = section.shape[0], nt = section.shape[1], nz = a3333.shape[1], status
    cdef float[:, ::1] out = np.zeros((nx, nz), dtype=np.float32)
    for a in (a1111, a3333, a1133, a1313, a1113, a3313):
        if a.shape[0] != nx or a.shape[1] != nz:
            raise ValueError("The elastic coefficients must all be (nx, nz), with nx the number of traces.")
    with nogil:
        status = su.su_miggbzoan(nt, dt, nx, dx, nz, dz, bwh, fmin, fmax, amin, amax, &section[0, 0], &a1111[0, 0],
                                 &a3333[0, 0], &a1133[0, 0], &a1313[0, 0], &a1113[0, 0], &a3313[0, 0], &out[0, 0])
    if status == -4:
        raise ValueError("The beam half-width bwh is too small for the trace interval dx.")
    if status != 0:
        raise ValueError("The parameters are not allowed.")
    return np.asarray(out)
