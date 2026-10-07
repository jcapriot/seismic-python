# cython: embedsignature=True, language_level=3
"""
What SUSTACK, SUDIVSTACK, SUPWS and SUSTACKUP do with the traces of a gather, from the SU library (``su_stack_*``, ``su_divstack_*``,
``su_pws_*`` and ``su_stackup_*``). A trace is an array of float32: n samples of `ncomp` floats each (2 for a complex trace,
as numpy lays it out).
"""
from .. cimport su
import numpy as np


def stack_add(const float[::1] x, int ncomp, float[::1] total, int[::1] nnz):
    """Add the trace to the sums, and count the samples that are not 0, in place"""
    cdef int n = nnz.shape[0]
    if x.shape[0] != n * ncomp or total.shape[0] != n * ncomp:
        raise ValueError("The trace and the sums must be the same length")
    if n > 0:
        with nogil:
            su.su_stack_add(n, ncomp, &x[0], &total[0], &nnz[0])


def stack_normalize(float[::1] total, const int[::1] nnz, int ncomp, float normpow):
    """Divide each sample of the sums by the number of values that were not 0, to the power normpow, in place"""
    cdef int n = nnz.shape[0]
    if total.shape[0] != n * ncomp:
        raise ValueError("The sums and the counts must be the same length")
    if n > 0:
        with nogil:
            su.su_stack_normalize(n, ncomp, &total[0], &nnz[0], normpow)


def divstack_power(const float[::1] x, int ncomp, int ntwin, bint peak):
    """The power of the trace in windows of ntwin samples, interpolated between them"""
    cdef int n = x.shape[0] // ncomp
    cdef float[::1] intp = np.zeros(n, dtype=np.float32)
    if ntwin < 1:
        raise ValueError("ntwin must be at least 1")
    if n > 0:
        with nogil:
            su.su_divstack_power(n, ncomp, ntwin, peak, &x[0], &intp[0])
    return np.asarray(intp)


def divstack_add(const float[::1] x, int ncomp, const float[::1] intp, float[::1] sumdata, float[::1] sumscale):
    """Add the trace divided by its power to the sums, and the inverse of its power to the scalers, in place"""
    cdef int n = intp.shape[0]
    if x.shape[0] != n * ncomp or sumdata.shape[0] != n * ncomp or sumscale.shape[0] != n:
        raise ValueError("The trace, the power and the sums must be the same length")
    if n > 0:
        with nogil:
            su.su_divstack_add(n, ncomp, &x[0], &intp[0], &sumdata[0], &sumscale[0])


def divstack_finish(const float[::1] sumdata, const float[::1] sumscale, int ncomp):
    cdef int n = sumscale.shape[0]
    cdef float[::1] out = np.zeros(n * ncomp, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_divstack_finish(n, ncomp, &sumdata[0], &sumscale[0], &out[0])
    return np.asarray(out)


def pws_accumulate(const float[::1] data, const float[::1] hdata, float[::1] stdata, float[::1] psct):
    """Add a trace to the ordinary stack, and its unit phasor to the phase stack (2 floats a sample), in place"""
    cdef int n = data.shape[0]
    if hdata.shape[0] != n or stdata.shape[0] != n or psct.shape[0] != 2 * n:
        raise ValueError("The trace, its Hilbert transform and the stacks must be the right length")
    if n > 0:
        with nogil:
            su.su_pws_accumulate(n, &data[0], &hdata[0], &stdata[0], &psct[0])


def pws_weights(const float[::1] psct, int ntr, float pwr, int isl):
    """The weights of the phase stack: |phase stack| / ntr to the power pwr, smoothed over isl samples"""
    cdef int n = psct.shape[0] // 2
    cdef float[::1] weights = np.zeros(n, dtype=np.float32)
    cdef float[::1] scratch = np.zeros(max(n, 1), dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_pws_weights(n, &psct[0], ntr, pwr, isl, &weights[0], &scratch[0])
    return np.asarray(weights)


def pws_apply(const float[::1] weights, int ntr, float[::1] stdata):
    """The ordinary stack times the weights over ntr, in place"""
    cdef int n = weights.shape[0]
    if stdata.shape[0] != n:
        raise ValueError("The weights and the stack must be the same length")
    if n > 0:
        with nogil:
            su.su_pws_apply(n, &weights[0], ntr, &stdata[0])


def stackup_add(const float[::1] x, double[::1] total, float[::1] sample_fold):
    """Add the trace to the sums, and count the samples that are not 0, in place"""
    cdef int n = x.shape[0]
    if total.shape[0] != n or sample_fold.shape[0] != n:
        raise ValueError("The trace and the sums must be the same length")
    if n > 0:
        with nogil:
            su.su_stackup_add(n, &x[0], &total[0], &sample_fold[0])


def stackup_finish(const double[::1] total, const float[::1] sample_fold):
    cdef int n = total.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_stackup_finish(n, &total[0], &sample_fold[0], &out[0])
    return np.asarray(out)
