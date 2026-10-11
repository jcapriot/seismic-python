# cython: embedsignature=True, language_level=3
"""
A band-limited spike on the samples of a trace, with the 8 point sinc interpolation of the SU library (``ints8r``), as SUADDEVENT
places the events that it adds.
"""
from .. cimport su
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)
import numpy as np


def spike_response(float t, float amplitude, float dt, float t_first, int nt):
    """The nt samples, dt apart from the time t_first, of a spike of the given amplitude at the time t"""
    cdef:
        float[::1] times
        float[::1] out
        float amp = amplitude
        int i
    if nt < 1:
        raise ValueError("nt must be at least 1")
    times = np.empty(nt, dtype=np.float32)
    out = np.zeros(nt, dtype=np.float32)
    for i in range(nt):
        times[i] = t_first + i * dt
    with nogil:
        su.ints8r(1, dt, t, &amp, 0.0, 0.0, nt, &times[0], &out[0])
    return np.asarray(out)
