# cython: embedsignature=True, language_level=3
"""
Migration, with the library functions of the SU fork (``su_stolt``, ``su_migfd``, ``su_migffd``, ``su_migps``,
``su_migpspi``, in ``su/main/migration_inversion``). The sections are float32 arrays (traces, samples).
"""
from .. cimport su
import numpy as np


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
