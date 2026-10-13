# cython: embedsignature=True, language_level=3
"""
The library functions of the datuming programs of SU (``su_datumfd``, in ``su/main/datuming``), on arrays.
"""
from .. cimport su
import numpy as np


def datumfd(const float[:, :, ::1] data, const float[:, ::1] gx, const float[:, ::1] vel1, const float[:, ::1] vel2,
            float dt, float dz, int nz, float dx, float x_0, int buff, int tap_len):
    """The shot gathers (nsx, nx, nt) datumed downward nz steps of dz by the finite difference scheme, with the
    velocities (mx, mz) of the thin lens and of the diffraction term, and the x of each receiver (nsx, nx)"""
    cdef:
        int nsx = data.shape[0]
        int nx = data.shape[1]
        int nt = data.shape[2]
        int mx = vel1.shape[0]
        int mz = vel1.shape[1]
        float[:, :, ::1] out = np.zeros((nsx, nx, nt), dtype=np.float32)
        int status
    if vel2.shape[0] != mx or vel2.shape[1] != mz:
        raise ValueError("The two velocity models must have the same shape.")
    if gx.shape[0] != nsx or gx.shape[1] != nx:
        raise ValueError("gx must have a value for every trace.")
    if nz > mz:
        raise ValueError("nz is more than the number of depth samples of the velocity models.")
    if nsx == 0 or nx < 4 or nt < 2:
        raise ValueError("There must be at least 4 receivers, 2 samples and one shot.")
    with nogil:
        status = su.su_datumfd(nt, dt, nx, nsx, nz, dz, dx, x_0, buff, tap_len, mx, mz, &vel1[0, 0], &vel2[0, 0],
                               &gx[0, 0], &data[0, 0, 0], &out[0, 0, 0])
    return np.asarray(out), status


def datumk2d_geometry(const float[::1] x0, const float[::1] z0, const float[::1] x1, const float[::1] z1, float dsmax,
                      int nxi, float fxi, float dxi, int ns, float fs, float ds):
    """The depths of the two surfaces (recording, datum) at the nxi locations and of the first at the ns sources of
    the tables, and the angles of its normal"""
    cdef:
        float[:, ::1] szif = np.zeros((nxi, 2), dtype=np.float32)
        float[::1] sz = np.zeros(ns, dtype=np.float32)
        float[::1] nangl = np.zeros(nxi, dtype=np.float32)
        int status
    if x0.shape[0] != z0.shape[0] or x1.shape[0] != z1.shape[0]:
        raise ValueError("A surface needs a z for each x.")
    if nxi < 3 or ns < 3:
        raise ValueError("At least 3 surface locations and 3 sources of the tables are needed.")
    if x0.shape[0] < 2 or x1.shape[0] < 2:
        raise ValueError("A surface needs 2 or more (x, z) pairs.")
    status = su.su_datumk2d_geometry(x0.shape[0], &x0[0], &z0[0], x1.shape[0], &x1[0], &z1[0], dsmax, nxi, fxi, dxi,
                                     ns, fs, ds, &szif[0, 0], &sz[0], &nangl[0])
    return np.asarray(szif), np.asarray(sz), np.asarray(nangl), status


def datumk2d_tables(float[:, :, ::1] ttab, int nr, float dr, float dzt, float fzt, float v0, float dvz,
                    const float[::1] sz, float fxt, float dxt, float fs, float ds):
    """The reference quantities (ns, nr, nzt) of the sources; ttab (ns, nxt, nzt) has the reference time subtracted"""
    cdef:
        int ns = ttab.shape[0]
        int nxt = ttab.shape[1]
        int nzt = ttab.shape[2]
        float[:, :, ::1] tb = np.zeros((ns, nr, nzt), dtype=np.float32)
        float[:, :, ::1] pb = np.zeros((ns, nr, nzt), dtype=np.float32)
        float[:, :, ::1] sigb = np.zeros((ns, nr, nzt), dtype=np.float32)
        float[:, :, ::1] angb = np.zeros((ns, nr, nzt), dtype=np.float32)
    if sz.shape[0] != ns:
        raise ValueError("sz needs a value for each source.")
    with nogil:
        su.su_datumk2d_tables(ns, nr, nzt, dr, dzt, fzt, v0, dvz, &sz[0], nxt, fxt, dxt, fs, ds, &tb[0, 0, 0],
                              &pb[0, 0, 0], &sigb[0, 0, 0], &angb[0, 0, 0], &ttab[0, 0, 0])
    return np.asarray(tb), np.asarray(pb), np.asarray(sigb), np.asarray(angb)


cdef class Datumk2d:
    """The tables of a Kirchhoff datuming (SUDATUMK2DR / SUDATUMK2DS), and the datuming of the traces into them"""
    cdef:
        float[:, :, ::1] ttab, tb, pb, angb, sigb
        float[:, ::1] szif
        float[::1] nangl
        int ns, nr, nzt, nxt, nxi, nz, mtmax, antiali, sgn
        float fs, ds, fxt, dxt, fzt, dzt, fxi, dxi, fz, dz, aperx, fmax, angmax, dt, ft

    def __init__(self, float[:, :, ::1] ttab, float[:, :, ::1] tb, float[:, :, ::1] pb, float[:, :, ::1] angb,
                 float[:, :, ::1] sigb, float[:, ::1] szif, float[::1] nangl, float fs, float ds, float fxt, float dxt,
                 float fzt, float dzt, float fxi, float dxi, int nz, float fz, float dz, float aperx, float fmax,
                 float angmax, int mtmax, int antiali, int sgn, float dt, float ft):
        self.ttab, self.tb, self.pb, self.angb, self.sigb = ttab, tb, pb, angb, sigb
        self.szif, self.nangl = szif, nangl
        self.ns, self.nxt, self.nzt = ttab.shape[0], ttab.shape[1], ttab.shape[2]
        self.nr = tb.shape[1]
        self.nxi = szif.shape[0]
        self.fs, self.ds, self.fxt, self.dxt, self.fzt, self.dzt = fs, ds, fxt, dxt, fzt, dzt
        self.fxi, self.dxi, self.nz, self.fz, self.dz = fxi, dxi, nz, fz, dz
        self.aperx, self.fmax, self.angmax, self.mtmax = aperx, fmax, angmax, mtmax
        self.antiali, self.sgn, self.dt, self.ft = antiali, sgn, dt, ft

    def add(self, const float[::1] trace, float[:, ::1] dat, float fx, float dx, float xm):
        """Add the datumed trace into dat (nx, nt); returns the status of the library function"""
        cdef:
            int status
            int nt = trace.shape[0]
            int nx = dat.shape[0]
        if dat.shape[1] != nt:
            raise ValueError("The output traces must have the samples of the input trace.")
        with nogil:
            status = su.su_datumk2d_trace(&trace[0], nt, self.ft, self.dt, &dat[0, 0], self.aperx, nx, fx, dx, self.nz,
                                          self.fz, self.dz, self.mtmax, xm, self.fmax, self.nxi, self.fxi, self.dxi,
                                          self.angmax, self.ns, self.fs, self.ds, self.nr, &self.tb[0, 0, 0],
                                          &self.pb[0, 0, 0], &self.angb[0, 0, 0], &self.sigb[0, 0, 0],
                                          &self.ttab[0, 0, 0], self.nzt, self.fzt, self.dzt, self.nxt, self.fxt,
                                          self.dxt, self.antiali, self.sgn, &self.szif[0, 0], &self.nangl[0])
        return status


def kdmdc_add(bint sources, const float[::1] trace, float[:, ::1] dat, float ft, float dt, float sx, float gx,
              float aperx, float fxin, float dx, int nzo, float fzo, float dzo, int nxi, float fxi, float dxi,
              float fs, float ds, const float[:, :, ::1] ttab, float fzt, float dzt, float fxt, float dxt,
              const float[:, ::1] szif, float v0, int dc, int ngl, float freq):
    """Datum one trace of SUKDMDCR (receivers) or SUKDMDCS (sources) into dat (nx, nt); the status of the library"""
    cdef:
        int status
        int nt = trace.shape[0]
        int nx = dat.shape[0]
        int ns = ttab.shape[0]
        int nxt = ttab.shape[1]
        int nzt = ttab.shape[2]
    if dat.shape[1] != nt:
        raise ValueError("The output traces must have the samples of the input trace.")
    with nogil:
        if sources:
            status = su.su_kdmdcs_trace(&trace[0], nt, ft, dt, sx, gx, &dat[0, 0], aperx, nx, fxin, dx, nzo, fzo, dzo,
                                        nxi, fxi, dxi, ns, fs, ds, &ttab[0, 0, 0], nzt, fzt, dzt, nxt, fxt, dxt,
                                        &szif[0, 0], v0, dc, ngl, freq)
        else:
            status = su.su_kdmdcr_trace(&trace[0], nt, ft, dt, sx, gx, &dat[0, 0], aperx, nx, fxin, dx, nzo, fzo, dzo,
                                        nxi, fxi, dxi, ns, fs, ds, &ttab[0, 0, 0], nzt, fzt, dzt, nxt, fxt, dxt,
                                        &szif[0, 0], v0, freq)
    return status
