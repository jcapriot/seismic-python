# cython: embedsignature=True, language_level=3
"""
The random number generators of the SU library (``franuni`` and ``frannor`` of ``cwp/lib``), with the state of each in an object so
that every stage has its own, which make the numbers that SU makes for the same seed (``su_rng`` of ``suaddnoise.c``).
"""
from .. cimport su
import numpy as np


cdef class Uniform:
    """Uniform random numbers in [0, 1), the sequence of sranuni(seed) and franuni()"""
    cdef su.su_rng rng

    def __init__(self, int seed):
        su.su_rng_seed_uniform(&self.rng, seed)

    def next(self):
        return su.su_rng_uniform(&self.rng)

    def draw(self, size_t n):
        cdef float[::1] out = np.zeros(n, dtype=np.float32)
        cdef size_t i
        for i in range(n):
            out[i] = su.su_rng_uniform(&self.rng)
        return np.asarray(out)


cdef class Normal:
    """Normal random numbers (mean 0, variance 1), the sequence of srannor(seed) and frannor()"""
    cdef su.su_rng rng

    def __init__(self, int seed):
        su.su_rng_seed_normal(&self.rng, seed)

    def next(self):
        return su.su_rng_normal(&self.rng)

    def draw(self, size_t n):
        cdef float[::1] out = np.zeros(n, dtype=np.float32)
        cdef size_t i
        for i in range(n):
            out[i] = su.su_rng_normal(&self.rng)
        return np.asarray(out)


def global_uniform(int seed, size_t n):
    """The numbers of the global generator of the cwp library, for the seed (it is shared by everything that uses it: this is for
    checking Uniform against it)"""
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef size_t i
    su.sranuni(seed)
    for i in range(n):
        out[i] = su.franuni()
    return np.asarray(out)


def global_normal(int seed, size_t n):
    """The numbers of the global normal generator of the cwp library, for the seed (to check Normal against)"""
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef size_t i
    su.srannor(seed)
    for i in range(n):
        out[i] = su.frannor()
    return np.asarray(out)
