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


def vibro_linear(int nt, float fs, float fe, float T, float dt, float phz):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.su_vibro_linear(&w[0], nt, fs, fe, T, dt, phz)
    return np.asarray(w)


def vibro_segments(int nt, const float[::1] freq, const float[::1] time, float T, float dt, float phz):
    cdef int isegm = time.shape[0]
    if freq.shape[0] != isegm + 1:
        raise ValueError("There is one more frequency than there are segments")
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0 and isegm > 0:
        with nogil:
            su.su_vibro_segments(&w[0], nt, &freq[0], &time[0], isegm, T, dt, phz)
    return np.asarray(w)


def vibro_octave(int nt, float fs, float fe, float T, float dt, float swconst, float phz):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.su_vibro_octave(&w[0], nt, fs, fe, T, dt, swconst, phz)
    return np.asarray(w)


def vibro_hertz(int nt, float fs, float fe, float T, float dt, float swconst, float phz):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.su_vibro_hertz(&w[0], nt, fs, fe, T, dt, swconst, phz)
    return np.asarray(w)


def vibro_tpower(int nt, float fs, float fe, float T, float dt, float swconst, float phz):
    cdef float[::1] w = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.su_vibro_tpower(&w[0], nt, fs, fe, T, dt, swconst, phz)
    return np.asarray(w)
