# cython: embedsignature=True, language_level=3
"""
The modes of the mute of the SU library (``su_mute_above``, ... of SUMUTE), each on one trace, in place.
"""
from .. cimport su
import numpy as np


cdef inline float *_pointer(const float[::1] taper) noexcept nogil:
    # (an empty taper is a null pointer: it is not used)
    if taper.shape[0] > 0:
        return <float *> &taper[0]
    return NULL


def taper_weights(int ntaper):
    """The sine squared weights of a taper of ntaper samples"""
    cdef float[::1] taper = np.zeros(max(ntaper, 1), dtype=np.float32)
    if ntaper > 0:
        su.su_mute_taper(ntaper, &taper[0])
    return np.asarray(taper)[:max(ntaper, 0)]


def above(float[::1] x, float t, float tmin, float dt, const float[::1] taper):
    """mode 0: mute above the time t"""
    cdef int nt = x.shape[0]
    cdef int ntaper = taper.shape[0]
    cdef float *tp = _pointer(taper)
    if nt > 0:
        with nogil:
            su.su_mute_above(&x[0], nt, t, tmin, dt, ntaper, tp)


def below(float[::1] x, float t, float tmin, float dt, const float[::1] taper):
    """mode 1: mute below the time t"""
    cdef int nt = x.shape[0]
    cdef int ntaper = taper.shape[0]
    cdef float *tp = _pointer(taper)
    if nt > 0:
        with nogil:
            su.su_mute_below(&x[0], nt, t, tmin, dt, ntaper, tp)


def line(float[::1] x, float t, float tmin, float dt, const float[::1] taper, float fval, float linvel, float tm0):
    """mode 2: mute a zone around a straight line (an air wave)"""
    cdef int nt = x.shape[0]
    cdef int ntaper = taper.shape[0]
    cdef float *tp = _pointer(taper)
    if nt > 0:
        with nogil:
            su.su_mute_line(&x[0], nt, t, tmin, dt, ntaper, tp, fval, linvel, tm0)


def hyperbola(float[::1] x, float t, float tmin, float dt, const float[::1] taper, float fval, float linvel, float tm0):
    """mode 3: mute a zone around a hyperbola"""
    cdef int nt = x.shape[0]
    cdef int ntaper = taper.shape[0]
    cdef float *tp = _pointer(taper)
    if nt > 0:
        with nogil:
            su.su_mute_hyperbola(&x[0], nt, t, tmin, dt, ntaper, tp, fval, linvel, tm0)


def polygon(float[::1] x, float t, float tw, float dt, const float[::1] taper):
    """mode 4: mute a zone of the width tw around the time t"""
    cdef int nt = x.shape[0]
    cdef int ntaper = taper.shape[0]
    cdef float *tp = _pointer(taper)
    if nt > 0:
        with nogil:
            su.su_mute_polygon(&x[0], nt, t, tw, dt, ntaper, tp)
