# cython: embedsignature=True, language_level=3
"""
The instantaneous attributes of the SU library (``su_attr_*`` of SUATTRIBUTES), of the complex trace re + i im.
"""
from .. cimport su
import numpy as np


cdef void _check(const float[::1] re, const float[::1] im) except *:
    if re.shape[0] != im.shape[0]:
        raise ValueError("re and im must be the same length")


def envelope(const float[::1] re, const float[::1] im):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_envelope(n, &re[0], &im[0], &out[0])
    return np.asarray(out)


def phase(const float[::1] re, const float[::1] im, float unwrap):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_phase(n, &re[0], &im[0], unwrap, &out[0])
    return np.asarray(out)


def freq(const float[::1] re, const float[::1] im, float dt, float unwrap):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_freq(n, &re[0], &im[0], dt, unwrap, &out[0])
    return np.asarray(out)


def normamp(const float[::1] re, const float[::1] im):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_normamp(n, &re[0], &im[0], &out[0])
    return np.asarray(out)


def fdenv(const float[::1] re, const float[::1] im, float dt):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_fdenv(n, &re[0], &im[0], dt, &out[0])
    return np.asarray(out)


def sdenv(const float[::1] re, const float[::1] im, float dt):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_sdenv(n, &re[0], &im[0], dt, &out[0])
    return np.asarray(out)


def bandwidth(const float[::1] re, const float[::1] im, float dt):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef float[::1] scratch = np.zeros(max(n, 1), dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_bandwidth(n, &re[0], &im[0], dt, &out[0], &scratch[0])
    return np.asarray(out)


def q(const float[::1] re, const float[::1] im, float dt, float unwrap):
    _check(re, im)
    cdef int n = re.shape[0]
    cdef float[::1] out = np.zeros(n, dtype=np.float32)
    cdef float[::1] scratch = np.zeros(max(2 * n, 1), dtype=np.float32)
    if n > 0:
        with nogil:
            su.su_attr_q(n, &re[0], &im[0], dt, unwrap, &out[0], &scratch[0])
    return np.asarray(out)
