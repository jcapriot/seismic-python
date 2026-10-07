# cython: embedsignature=True, language_level=3
"""
The coordinate stretches of the SU library: the linear interpolation stretch of SULOG and SUILOG, and the tables of times
and depths of SUTTOZ and SUZTOT (``su_lintrp``, ``su_stretch``, ``su_ttoz_table`` and ``su_ztot_table``).
"""
from .. cimport su
import numpy as np


cdef class Stretch:
    """A stretch of traces of n_in samples to the samples that are at the fractional samples `positions` of them (with
    linear interpolation, and 0 for the ones that are not within the trace). The coefficients are found once."""
    cdef:
        int[::1] it
        float[::1] w
        int n_in
        int n_out

    def __init__(self, const float[::1] positions, int n_in):
        self.n_in = n_in
        self.n_out = positions.shape[0]
        self.it = np.zeros(max(self.n_out, 1), dtype=np.intc)
        self.w = np.zeros(max(2 * self.n_out, 2), dtype=np.float32)
        if self.n_out > 0:
            su.su_lintrp(<float *> &positions[0], &self.w[0], &self.it[0], n_in, self.n_out)

    @property
    def n_out(self):
        return self.n_out

    def apply(self, const float[::1] p):
        if p.shape[0] != self.n_in:
            raise ValueError(f"The stretch is for traces of {self.n_in} samples, not {p.shape[0]}")
        cdef float[::1] q = np.zeros(self.n_out, dtype=np.float32)
        cdef int n_out = self.n_out
        if n_out > 0 and self.n_in > 0:
            with nogil:
                su.su_stretch(&q[0], <float *> &p[0], &self.w[0], &self.it[0], n_out, 2)
        return np.asarray(q)


def ttoz_table(const float[::1] v, float dt, float ft, int nz, float dz, float fz):
    """The time of each of the nz depths fz + i dz, from the velocity at each of the times ft + i dt"""
    cdef int nt = v.shape[0]
    if nt < 1 or nz < 1:
        raise ValueError("There must be some times and some depths")
    cdef float[::1] tz = np.zeros(nz, dtype=np.float32)
    cdef float[::1] z = np.zeros(nt, dtype=np.float32)
    with nogil:
        su.su_ttoz_table(nt, dt, ft, <float *> &v[0], nz, dz, fz, &tz[0], &z[0])
    return np.asarray(tz)


def ztot_table(const float[::1] v, float dz, float fz, int nt, float dt, float ft):
    """The depth of each of the nt times ft + i dt, from the velocity at each of the depths fz + i dz"""
    cdef int nz = v.shape[0]
    if nt < 1 or nz < 1:
        raise ValueError("There must be some times and some depths")
    cdef float[::1] zt = np.zeros(nt, dtype=np.float32)
    cdef float[::1] t = np.zeros(nz, dtype=np.float32)
    with nogil:
        su.su_ztot_table(nz, dz, fz, <float *> &v[0], nt, dt, ft, &zt[0], &t[0])
    return np.asarray(zt)
