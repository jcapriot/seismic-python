# cython: embedsignature=True, language_level=3
"""
The phase unwrapping of the SU library (``cwp/lib/unwrapphase.c``).
"""
from .. cimport su
import numpy as np


def simple(float[::1] phase, int trend, int zeromean, float w):
    """Unwrap a phase by looking for jumps that are more than pi / w (and replace them with the previous change)"""
    cdef int n = phase.shape[0]
    cdef float[::1] out = np.array(phase, dtype=np.float32)
    if w == 0.0:
        raise ValueError("w must not be 0")
    if n > 0:
        with nogil:
            su.simple_unwrap_phase(n, trend, zeromean, w, &out[0])
    return np.asarray(out)


def oppenheim(float[::1] xr, float[::1] xi, float df, int trend, int zeromean):
    """The unwrapped phase of a spectrum with real part xr and imaginary part xi (df apart), found by integrating the
    derivative of the phase (the method of Oppenheim and Schafer)"""
    cdef int n = xr.shape[0]
    if xi.shape[0] != n:
        raise ValueError("xr and xi must be the same length")
    cdef float[::1] phase = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.oppenheim_unwrap_phase(n, trend, zeromean, df, &xr[0], &xi[0], &phase[0])
    return np.asarray(phase)
