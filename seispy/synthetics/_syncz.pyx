# cython: embedsignature=True, language_level=3
"""
Zero-offset data over dipping interfaces in a medium of constant velocity layers, with the library version of SUSYNCZ
(``su_syncz_tables`` and ``su_syncz_trace``, in ``su/main/synthetics_waveforms_testpatterns``). The tables and the scratch arrays are
made, and kept, here.
"""
from .. cimport su
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)
import numpy as np


_ERRORS = {
    1: "the dip of interface {i} must be from -90 to 90 degrees",
    2: "the dip of interface {i} differs from the one above it by more than 90 degrees",
    3: "the intercept of interface {i} must be below the one above it",
    4: "the velocity of layer {i} must be positive",
    5: "the density of layer {i} must be positive",
    6: "there is no specular ray to interface {i} (the angle is past the critical angle)",
    7: "interface {i} meets one of the others in the region that is observed",
}


cdef class SynczModel:
    """The tables for ninf interfaces, and the traces of ntr positions dx apart. The arrays zint, dip (degrees), v and rho have
    ninf + 1 numbers: the first of zint and dip is the surface (0)."""
    cdef:
        int ninf
        int nt
        float dt
        float tdelay
        float[::1] v, theta, m, b, d, k, w, meet, trcoefs, tout, data
        int[::1] dorow

    def __init__(self, int ninf, int ntr, float dx, const float[::1] zint, const float[::1] dip, const float[::1] v,
                 const float[::1] rho, int nt, float dt, float tdelay):
        cdef:
            int status, info = 0
            int i
            int n1 = ninf + 1
            float[::1] dipr, xl, xr
        if ninf < 0:
            raise ValueError("ninf must not be negative")
        if nt < 1:
            raise ValueError("nt must be at least 1")
        if zint.shape[0] != n1 or dip.shape[0] != n1 or v.shape[0] != n1 or rho.shape[0] != n1:
            raise ValueError(f"zint, dip, v and rho must have {n1} numbers (the first of zint and dip is the surface)")
        self.ninf = ninf
        self.nt = nt
        self.dt = dt
        self.tdelay = tdelay
        self.v = np.array(v, dtype=np.float32)
        self.theta = np.zeros(n1 * n1, dtype=np.float32)
        self.m = np.zeros(n1 * n1, dtype=np.float32)
        self.b = np.zeros(n1 * n1, dtype=np.float32)
        self.d = np.zeros(n1 * n1, dtype=np.float32)
        self.k = np.zeros(n1 * n1, dtype=np.float32)
        self.w = np.zeros(n1 * n1, dtype=np.float32)
        self.meet = np.zeros(n1, dtype=np.float32)
        self.trcoefs = np.zeros(n1, dtype=np.float32)
        self.dorow = np.zeros(n1, dtype=np.intc)
        self.tout = np.empty(nt, dtype=np.float32)
        for i in range(nt):
            self.tout[i] = i * dt
        self.data = np.zeros(nt, dtype=np.float32)
        dipr = np.zeros(n1, dtype=np.float32)
        xl = np.zeros(n1, dtype=np.float32)
        xr = np.zeros(n1, dtype=np.float32)
        with nogil:
            status = su.su_syncz_tables(ninf, ntr, dx, &zint[0], &dip[0], &v[0], &rho[0], &dipr[0], &xl[0], &xr[0],
                                        &self.theta[0], &self.m[0], &self.b[0], &self.d[0], &self.k[0], &self.w[0],
                                        &self.meet[0], &self.trcoefs[0], &self.dorow[0], &info)
        if status != 0:
            raise ValueError(_ERRORS[status].format(i=info))

    def trace(self, float x):
        """The zero-offset trace at the position x"""
        cdef float[::1] out = np.zeros(self.nt, dtype=np.float32)
        with nogil:
            su.su_syncz_trace(self.ninf, x, self.nt, self.dt, self.tdelay, &self.v[0], &self.m[0], &self.b[0], &self.d[0],
                              &self.k[0], &self.w[0], &self.meet[0], &self.trcoefs[0], &self.dorow[0], &self.tout[0],
                              &self.data[0], &out[0])
        return np.asarray(out)
