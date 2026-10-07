# cython: embedsignature=True, language_level=3
"""
What SUFRAC, SUPHASE, SUMINPHASE and SUTVBAND do to a spectrum, from the SU library (``su_frac_filter``, ``su_phase_spectrum``,
``su_kolmogoroff_*`` and ``su_tvband_*``). The Fourier transforms around them are numpy's.
"""
from .. cimport su
import numpy as np


def frac_filter(int nf, int nfft, float dt, float power, float sign, float phasefac, float scale):
    """The complex filter ((sign) i w)^power with the phase shift phasefac pi, times scale, for nf frequencies"""
    cdef float complex[::1] filt = np.zeros(nf, dtype=np.complex64)
    if nf > 0:
        su.su_frac_filter(nf, nfft, dt, power, sign, phasefac, scale, <float *> &filt[0])
    return np.asarray(filt)


def phase_spectrum(float complex[::1] ct, float a, float b, float c):
    """The new phase a + b phase + c i of each frequency of the spectrum ct, in place"""
    cdef int nf = ct.shape[0]
    if nf > 0:
        with nogil:
            su.su_phase_spectrum(nf, <float *> &ct[0], a, b, c)


def kolmogoroff_log(float complex[::1] cx, float pnoise):
    """The log of the squared amplitude (over the largest, at least pnoise), in place. Gives the largest amplitude."""
    cdef int n = cx.shape[0]
    cdef float rmax = 1.0
    if n > 0:
        with nogil:
            rmax = su.su_kolmogoroff_log(n, pnoise, <float *> &cx[0])
    return rmax


def kolmogoroff_fold(float complex[::1] cx):
    """Scale by 1 / n and keep the causal part, in place"""
    cdef int n = cx.shape[0]
    if n > 0:
        with nogil:
            su.su_kolmogoroff_fold(n, <float *> &cx[0])


def kolmogoroff_exp(float complex[::1] cx, float rmax):
    """exp of the values, times rmax, in place"""
    cdef int n = cx.shape[0]
    if n > 0:
        with nogil:
            su.su_kolmogoroff_exp(n, rmax, <float *> &cx[0])


def tvband_filter(const float[::1] f, int nfft, int nfreq, float dt, float scale):
    """The bandpass filter (sine squared tapers) with the corner frequencies f[4], for nfreq frequencies"""
    if f.shape[0] != 4:
        raise ValueError("a filter has 4 corner frequencies")
    cdef float[::1] filt = np.zeros(nfreq, dtype=np.float32)
    if nfreq > 0:
        su.su_tvband_filter(<float *> &f[0], nfft, nfreq, dt, scale, &filt[0])
    return np.asarray(filt)


def tvband_blend(int i0, int i1, const float[::1] first, const float[::1] second, float[::1] out):
    """out[i] fades from first[i] to second[i] from the sample i0 to i1 (inclusive), in place"""
    if i0 < 0 or i1 >= first.shape[0] or i1 >= second.shape[0] or i1 >= out.shape[0]:
        raise ValueError("The samples are not on the traces")
    if i1 >= i0:
        with nogil:
            su.su_tvband_blend(i0, i1, &first[0], &second[0], &out[0])


def median_across(const float[:, ::1] rows):
    """The median over the rows (an odd number of them) at each of the samples"""
    cdef int nwin = rows.shape[0]
    cdef int n = rows.shape[1]
    if nwin % 2 == 0:
        raise ValueError("The number of rows must be odd")
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef float[::1] scratch = np.zeros(nwin, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_median_across(n, nwin, &rows[0, 0], n, &out[0], &scratch[0])
    return np.asarray(out)


def mix_across(const float[:, ::1] rows, const float[::1] weights):
    """The sum of the rows times their weights, at each of the samples"""
    cdef int nwin = rows.shape[0]
    cdef int n = rows.shape[1]
    if weights.shape[0] != nwin:
        raise ValueError("There is one weight for each row")
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0 and nwin > 0:
        with nogil:
            su.su_mix_across(n, nwin, &rows[0, 0], n, &weights[0], &out[0])
    return np.asarray(out)
