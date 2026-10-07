# cython: embedsignature=True, language_level=3
"""
The tapers of the SU library (``su_taper_envelope``, ``su_taper_time`` and ``su_ramp``, from SUTAPER and SURAMP).
"""
from .. cimport su
import numpy as np


def envelope(int tap_type, double f, double minimum=0.0, double maximum=1.0):
    """The weight of a taper at f, which goes from 0 to 1 along the taper (types 1 to 5; only the linear one, 1, goes
    between the minimum and the maximum)"""
    cdef float value = su.su_taper_envelope(tap_type, <float> f, <float> minimum, <float> maximum)
    if value == -1.0 and (tap_type < 1 or tap_type > 5):
        raise ValueError(f"taper type {tap_type} must be one of 1 to 5")
    return value


def time_taper(float[::1] x, float t1, float t2, int tap_type, float dt):
    """Taper the start (t1) and the end (t2) of the trace x in place (the times are in the same unit as dt)"""
    cdef int nt = x.shape[0]
    if tap_type < 1 or tap_type > 5:
        raise ValueError(f"taper type {tap_type} must be one of 1 to 5")
    if nt > 0:
        with nogil:
            su.su_taper_time(t1, t2, tap_type, dt, &x[0], nt)


def ramp(float[::1] x, int ntaper1, int ntaper2):
    """Ramp the first ntaper1 and the last ntaper2 samples of the trace x to 0, in place"""
    cdef int nt = x.shape[0]
    if nt > 0:
        with nogil:
            su.su_ramp(&x[0], nt, ntaper1, ntaper2)
