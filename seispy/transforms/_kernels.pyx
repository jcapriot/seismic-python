# cython: embedsignature=True, language_level=3
"""
What SUWFFT, SUCLOGFFT and SUICLOGFFT do to a spectrum, from the SU library (``su_wfft_flatten``, ``su_clogfft_spectrum`` and
``su_iclogfft_spectrum``). The Fourier transforms around them are numpy's.
"""
from .. cimport su
import numpy as np


def wfft_flatten(const float complex[::1] ct, float w0, float w1, float w2):
    """The spectrum flattened by the weighted amplitudes of the frequencies about each one"""
    cdef int nf = ct.shape[0]
    cdef float complex[::1] out = np.zeros(nf, dtype=np.complex64)
    if nf > 0:
        with nogil:
            su.su_wfft_flatten(nf, <float *> &ct[0], w0, w1, w2, <float *> &out[0])
    return np.asarray(out)


def clogfft_spectrum(const float complex[::1] ct):
    """The log amplitude and the (wrapped) phase of each frequency of a spectrum"""
    cdef int nf = ct.shape[0]
    cdef float[::1] log_amp = np.zeros(nf, dtype=np.float32)
    cdef float[::1] phase = np.zeros(nf, dtype=np.float32)
    if nf > 0:
        with nogil:
            su.su_clogfft_spectrum(nf, <float *> &ct[0], &log_amp[0], &phase[0])
    return np.asarray(log_amp), np.asarray(phase)


def iclogfft_spectrum(const float complex[::1] clog, bint sym):
    """exp of a complex log spectrum (0 where the real part is 0), with every other frequency negated if sym"""
    cdef int nf = clog.shape[0]
    cdef float complex[::1] ct = np.zeros(nf, dtype=np.complex64)
    if nf > 0:
        with nogil:
            su.su_iclogfft_spectrum(nf, <float *> &clog[0], sym, <float *> &ct[0])
    return np.asarray(ct)


def st_analytic(float complex[::1] h):
    """The transform of the analytic signal (the positive frequencies twice, the negative ones 0), in place"""
    cdef int n = h.shape[0]
    if n > 0:
        with nogil:
            su.su_st_analytic(n, <float *> &h[0])


def st_row(const float complex[::1] h, int n):
    """The spectrum of the row n of the S transform: the shifted spectrum times the Gaussian of the row"""
    cdef int length = h.shape[0]
    cdef float complex[::1] g = np.zeros(length, dtype=np.complex64)
    if n < 1:
        raise ValueError("n must be at least 1")
    if length > 0:
        with nogil:
            su.su_st_row(length, n, <float *> &h[0], <float *> &g[0])
    return np.asarray(g)


def gabor_filter(float fcent, float dt, int nfft, float alpha, float band, float scale):
    """The Gaussian filter about fcent, for the nfft // 2 + 1 frequencies of the transform"""
    cdef float[::1] filt = np.zeros(nfft // 2 + 1, dtype=np.float32)
    su.su_gabor_filter(fcent, dt, nfft, alpha, band, scale, &filt[0])
    return np.asarray(filt)


def cwt_wavelet(int nwavelet, float xmin, float xcenter, float xmax, float sigma):
    """The integral of the Mexican hat wavelet (times dx) and dx"""
    cdef float[::1] wavelet_sum = np.zeros(nwavelet, dtype=np.float32)
    cdef float dx = su.su_cwt_wavelet(nwavelet, xmin, xcenter, xmax, sigma, &wavelet_sum[0])
    return np.asarray(wavelet_sum), dx


def cwt_filter(const float[::1] wavelet_sum, float scale, float dx, float width):
    """The filter of one scale: the integral of the wavelet stretched by the scale, and flipped"""
    cdef int nwavelet = wavelet_sum.shape[0]
    cdef int room = 2 + <int> (scale * width)
    cdef float[::1] filt = np.zeros(room, dtype=np.float32)
    cdef int nconv
    nconv = su.su_cwt_filter(nwavelet, &wavelet_sum[0], scale, dx, width, &filt[0])
    return np.asarray(filt)[:nconv].copy()


def cwt_trace(const float[::1] convolution, int nconv, float scale):
    """The convolution of a trace with a filter (the first ns samples of it), taken as a scale of the wavelet transform"""
    cdef int ns = convolution.shape[0]
    cdef float[::1] rt = np.zeros(ns, dtype=np.float32)
    if ns > 0:
        su.su_cwt_trace(ns, nconv, &convolution[0], scale, &rt[0])
    return np.asarray(rt)
