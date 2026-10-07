# cython: embedsignature=True, language_level=3
"""
The wavelets of the SU library (``cwp/lib/waveforms.c``), as arrays of samples.
"""
from .. cimport su
import numpy as np


def akb(int nt, float dt, float fpeak):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.akb_wavelet(nt, dt, fpeak, &w[0])
    return np.asarray(w)


def berlage(int nt, float dt, float fpeak, float ampl, float tn, float decay, float ipa):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.berlage_wavelet(nt, dt, fpeak, ampl, tn, decay, ipa, &w[0])
    return np.asarray(w)


def gauss(int nt, float dt, float fpeak):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.gaussian_wavelet(nt, dt, fpeak, &w[0])
    return np.asarray(w)


def gaussd(int nt, float dt, float fpeak):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.gaussderiv_wavelet(nt, dt, fpeak, &w[0])
    return np.asarray(w)


def ricker1(int nt, float dt, float fpeak):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.ricker1_wavelet(nt, dt, fpeak, &w[0])
    return np.asarray(w)


def ricker2(int hlw, float dt, float period, float ampl, float distort):
    """The 2 hlw - 1 samples of the wavelet (its peak is sample hlw - 1)"""
    if hlw < 1:
        raise ValueError("hlw must be at least 1")
    cdef int n = 2 * hlw - 1
    cdef float[::1] w = np.zeros(n, dtype=np.float32)
    with nogil:
        su.ricker2_wavelet(hlw, dt, period, ampl, distort, &w[0])
    return np.asarray(w)


def spike(int nt, int tindex):
    if tindex < 0 or tindex >= nt:
        raise ValueError(f"The spike (sample {tindex}) is not in the {nt} samples of the wavelet.")
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    with nogil:
        su.spike_wavelet(nt, tindex, &w[0])
    return np.asarray(w)


def unit(int nt):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.unit_wavelet(nt, &w[0])
    return np.asarray(w)


def dgauss(double dt, int nt, double t0, float fpeak, int n, int sign):
    """The n-th derivative of a Gaussian (in double precision, and then made float)"""
    cdef double[::1] w = np.zeros(nt, dtype=np.float64)
    if nt > 0:
        with nogil:
            su.deriv_n_gauss(dt, nt, t0, fpeak, n, &w[0], sign, 0)
    return np.asarray(w).astype(np.float32)
