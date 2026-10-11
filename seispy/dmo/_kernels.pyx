# cython: embedsignature=True, language_level=3
"""
Dip moveout, with the library functions of the SU fork (``su_dmofk_*``, ``su_dmotx_*``, ``su_dmofkcw_*``, ``su_dmovz_*``,
``su_dmotivz_*`` and ``su_tihaledmo``, in ``su/main/dip_moveout``). The common-offset gathers are float32 arrays.
"""
from .. cimport su
import numpy as np
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)

# ints8c has a sinc table of its own, made at its first call: do it now, while nothing else is running
su.su_ints8_tables()

_MESSAGES = {-3: "The parameters are not allowed.", -4: "No data survive the stretch mute."}


def _check(int status):
    if status < 0:
        raise ValueError(_MESSAGES.get(status, "The DMO failed."))


def vrms(const float[::1] tdmo, const float[::1] vdmo, int nt, float dt, float ft):
    """The rms velocities at the nt times of the trace, by the monotonic cubic interpolation of the (tdmo, vdmo) pairs"""
    cdef float[::1] out = np.zeros(nt, dtype=np.float32)
    if tdmo.shape[0] != vdmo.shape[0] or tdmo.shape[0] < 1:
        raise ValueError("tdmo and vdmo must have the same length.")
    with nogil:
        su.su_dmofk_vrms(tdmo.shape[0], &tdmo[0], &vdmo[0], nt, dt, ft, &out[0])
    return np.asarray(out)


def nxfft(int nx, float dx, float offset):
    """The number of cdps of a common-offset gather of nx cdps after the padding of the Fourier transform"""
    return su.su_dmofk_nxfft(nx, dx, offset)


def fk_offset(float[:, ::1] gather, float offset, float fmax, float sdmo, int nx, float dx, float dt, float ft,
              const float[::1] vrms_):
    """DMO (F-K, log stretch) of the gather (nxfft + 2, nt), in place; its first nx rows are the traces"""
    cdef int nt = gather.shape[1]
    if vrms_.shape[0] != nt or gather.shape[0] < su.su_dmofk_nxfft(nx, dx, offset) + 2:
        raise ValueError("The gather needs nxfft + 2 rows, and the velocities a value for each sample.")
    with nogil:
        su.su_dmofk_offset(offset, fmax, sdmo, nx, dx, nt, dt, ft, &vrms_[0], &gather[0, 0])


def kcw_table(int ntable, float gamma):
    """The tables z/h and b/h of converted waves for the velocity ratio gamma"""
    cdef:
        float[::1] zoh = np.zeros(ntable, dtype=np.float32)
        float[::1] boh = np.zeros(ntable, dtype=np.float32)
        int status
    if ntable < 2:
        raise ValueError("ntable must be at least 2.")
    status = su.su_dmofkcw_table(ntable, gamma, &zoh[0], &boh[0])
    if status == -4:
        raise ValueError("negative sqrt when generating the table of z/h and b/h: gamma is not allowed.")
    _check(status)
    return np.asarray(zoh), np.asarray(boh)


def kcw_stretch(float sdmo, float gamma):
    """The stretch (or squeeze) factors s1 and s2"""
    cdef float s1, s2
    su.su_dmofkcw_stretch(sdmo, gamma, &s1, &s2)
    return s1, s2


def kcw_offset(float[:, ::1] gather, float offset, float fmax, int nx, float dx, float dt, float ft, const float[::1] vrms_,
               float gamma, const float[::1] boh, const float[::1] zoh, int ntable, float s1, float s2, float sign):
    """DMO of converted waves of the gather (nxfft + 2, nt), in place"""
    cdef:
        int nt = gather.shape[1]
        const float *bp = NULL
        const float *zp = NULL
    if vrms_.shape[0] != nt or gather.shape[0] < su.su_dmofk_nxfft(nx, dx, offset) + 2:
        raise ValueError("The gather needs nxfft + 2 rows, and the velocities a value for each sample.")
    if gamma != 1.0:
        if boh is None or zoh is None or boh.shape[0] < ntable or zoh.shape[0] < ntable:
            raise ValueError("gamma not 1 needs the tables of z/h and b/h.")
        bp = &boh[0]
        zp = &zoh[0]
    with nogil:
        su.su_dmofkcw_offset(offset, fmax, nx, dx, nt, dt, ft, &vrms_[0], &gather[0, 0], gamma, bp, zp, ntable, s1, s2, sign)


def tx_tables(float dxcdp, float dt, float offmax, float tmute, float vrms_, int nsmax, int lds, int ifds):
    """The time shifts and amplitudes of the T-X operator, and its shaping filter: (ts, as, ds)"""
    cdef:
        float[::1] ts = np.zeros(nsmax, dtype=np.float32)
        float[::1] as_ = np.zeros(nsmax, dtype=np.float32)
        float[::1] ds = np.zeros(lds, dtype=np.float32)
        int ns
    if nsmax < 2 or lds < 1:
        raise ValueError("nsmax must be at least 2 and lds at least 1.")
    ns = su.su_dmotx_tables(dxcdp, dt, offmax, tmute, vrms_, nsmax, lds, ifds, &ts[0], &as_[0], &ds[0])
    _check(ns)
    return np.asarray(ts)[:ns].copy(), np.asarray(as_)[:ns].copy(), np.asarray(ds)


def tx_trace(const float[::1] ts, const float[::1] as_, float offset, int icdp, float dxcdp, float dt, float ft,
             const float[::1] p, float[:, ::1] q):
    """Add the contribution of the trace p at cdp index icdp to the traces q (ncdp, nt)"""
    cdef:
        int ns = ts.shape[0], ncdp = q.shape[0], nt = q.shape[1]
    if as_.shape[0] != ns or p.shape[0] != nt:
        raise ValueError("The shapes of the tables or of the trace are not those of the section.")
    with nogil:
        su.su_dmotx_trace(ns, &ts[0], &as_[0], offset, icdp, ncdp, dxcdp, nt, dt, ft, &p[0], &q[0, 0])


def tx_shape(const float[::1] ds, int ifds, float[:, ::1] q):
    """Apply the shaping filter to every trace of q (ncdp, nt), in place"""
    cdef int ncdp = q.shape[0], nt = q.shape[1], lds = ds.shape[0]
    with nogil:
        su.su_dmotx_shape(ncdp, nt, lds, ifds, &ds[0], &q[0, 0])


cdef class VzModel:
    """The ray tables of a V(z) medium (SUDMOVZ), or of a transversely isotropic V(z) medium (SUDMOTIVZ)"""
    cdef:
        su.DmoVz *vz
        su.DmoTivz *tivz
        public int nt

    def __cinit__(self):
        self.vz = NULL
        self.tivz = NULL

    def __dealloc__(self):
        if self.vz is not NULL:
            su.su_dmovz_free(self.vz)
        if self.tivz is not NULL:
            su.su_dmotivz_free(self.tivz)

    def __init__(self, const float[::1] v, float dt, int nptab, const float[::1] vn=None, const float[::1] eta=None):
        cdef int err = 0
        self.nt = v.shape[0]
        if vn is None:
            self.vz = su.su_dmovz_create(self.nt, dt, &v[0], nptab, &err)
            if self.vz is NULL:
                _check(err if err < 0 else -3)
        else:
            if vn.shape[0] != self.nt or eta is None or eta.shape[0] != self.nt:
                raise ValueError("v, vn and eta must have the same number of samples.")
            self.tivz = su.su_dmotivz_create(self.nt, dt, &vn[0], &v[0], &eta[0], nptab, &err)
            if self.tivz is NULL:
                _check(err if err < 0 else -3)

    def offset(self, float[:, ::1] gather, int nx, float dx, float offset, float fmax, float smute, float speed):
        """DMO of the gather (nxfft + 2, nt), in place"""
        cdef int status, need
        if gather.shape[1] != self.nt:
            raise ValueError("The gather must have the number of samples of the velocities.")
        need = su.su_dmovz_nxfft(nx, dx, offset) + 2
        if gather.shape[0] < need:
            raise ValueError("The gather needs nxfft + 2 rows.")
        with nogil:
            if self.vz is not NULL:
                status = su.su_dmovz_offset(self.vz, nx, dx, offset, fmax, smute, speed, &gather[0, 0])
            else:
                status = su.su_dmotivz_offset(self.tivz, nx, dx, offset, fmax, smute, speed, &gather[0, 0])
        _check(status)


def tihaledmo(const float[:, ::1] data, float dt, float dx, float h, float v, int option, int ntpad, int nxpad,
              const float[::1] vnmo, float dp, float e, float d):
    """DMO of one common-offset gather (nx, nt) in the f-k domain, with the operator of the option (1 to 6)"""
    cdef:
        int nx = data.shape[0], nt = data.shape[1], status, np_ = 0
        const float *vp = NULL
        float[:, ::1] out = np.zeros((nx, nt), dtype=np.float32)
    if vnmo is not None and vnmo.shape[0] > 0:
        np_ = vnmo.shape[0]
        vp = &vnmo[0]
    with nogil:
        status = su.su_tihaledmo(nx, nt, dt, dx, h, v, option, ntpad, nxpad, np_, dp, vp, e, d, &data[0, 0], &out[0, 0])
    _check(status)
    return np.asarray(out)
