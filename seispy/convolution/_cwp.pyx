# cython: embedsignature=True, language_level=3
"""
Convolution and cross-correlation with the routines of the SU library (``convolve_cwp`` and ``xcor`` of ``cwp/lib``).
"""
from .. cimport su
import numpy as np


def correlate(const float[::1] x, const float[::1] y, int first_lag, int n_lags):
    """z[i] = sum_j x[j] y[i + j] for the lags i = first_lag, ..., first_lag + n_lags - 1 (x and y are 0 outside of
    themselves)"""
    cdef float[::1] z = np.zeros(n_lags, dtype=np.float32)
    cdef int lx = x.shape[0]
    cdef int ly = y.shape[0]
    if lx == 0 or ly == 0 or n_lags == 0:
        return np.asarray(z)
    with nogil:
        su.xcor(lx, 0, <float *> &x[0], ly, 0, <float *> &y[0], n_lags, first_lag, &z[0])
    return np.asarray(z)


def convolve(const float[::1] x, const float[::1] y, int n_out, int first=0):
    """The samples first, ..., first + n_out - 1 of the convolution of x and y (which are 0 outside of themselves)"""
    cdef float[::1] z = np.zeros(n_out, dtype=np.float32)
    cdef int lx = x.shape[0]
    cdef int ly = y.shape[0]
    if lx == 0 or ly == 0 or n_out == 0:
        return np.asarray(z)
    with nogil:
        su.convolve_cwp(lx, 0, <float *> &x[0], ly, 0, <float *> &y[0], n_out, first, &z[0])
    return np.asarray(z)


def acorfrac_spectrum(float complex[::1] ct, float a, float b, bint sym):
    """Each frequency of the spectrum ct times |S|^a exp(-i b arg S), in place (negating every other one if sym)"""
    cdef int nf = ct.shape[0]
    if nf > 0:
        with nogil:
            su.su_acorfrac_spectrum(nf, <float *> &ct[0], a, b, sym)
