# cython: embedsignature=True, language_level=3
"""
The Hilbert transform of the SU library (``hilbert``): the trace is convolved with a 61 point, Hamming windowed
approximation of the ideal Hilbert transformer, in the time domain.
"""
from .. cimport su
import numpy as np


cdef void _make_the_filter() noexcept:
    # hilbert makes its filter (a function level static) the first time that it is called, which is not safe to do
    # from several threads at once. Once made it is only read, so do that once, here, while this module is being
    # imported and nothing else is running.
    cdef float x = 0.0
    cdef float y = 0.0
    su.hilbert(1, &x, &y)


_make_the_filter()


def hilbert(float[::1] x):
    """The Hilbert transform of the samples x"""
    cdef:
        int n = x.shape[0]
        float[::1] y = np.zeros(n, dtype=np.float32)
    if n == 0:
        return np.asarray(y)
    with nogil:
        su.hilbert(n, &x[0], &y[0])
    return np.asarray(y)
