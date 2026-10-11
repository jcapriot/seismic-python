# cython: embedsignature=True, language_level=3
"""
The smoothing of the characteristic function of SUFBPICKW, with the library function of the SU fork (``smooth_segmented_array``
of ``par/lib/smooth.cpp``).
"""
from .. cimport su
import numpy as np


def smooth_triangle(const float[::1] values, int sm):
    """The values smoothed by a running average with a triangular window of half length sm (method 5 of
    ``smooth_segmented_array``, for the values at consecutive indices)"""
    cdef:
        int n = values.shape[0], i
        float[::1] val = np.array(values, dtype=np.float32)
        float[::1] index = np.arange(n, dtype=np.float32)
    if n < 2:
        return np.asarray(val)
    if sm < 1:
        raise ValueError("sm must be at least 1")
    with nogil:
        su.smooth_segmented_array(&index[0], &val[0], n, sm, 1, 5)
    return np.asarray(val)
