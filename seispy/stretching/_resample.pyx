# cython: embedsignature=True, language_level=3
"""
Sinc resampling of a trace, with the 8 point sinc interpolation of the SU library (``ints8r``), as SURESAMP does.
"""
from .. cimport su
import numpy as np


cdef void _fill_the_table() noexcept:
    # ints8r fills a table of sinc coefficients (a function level static) the first time that it is called, which
    # is not safe to do from several threads at once. Once filled it is only read, so do that once, here, while this
    # module is being imported and nothing else is running.
    cdef float y = 0.0
    cdef float x = 0.0
    cdef float out = 0.0
    su.ints8r(1, 1.0, 0.0, &y, 0.0, 0.0, 1, &x, &out)


_fill_the_table()


def resample(float[::1] x, float dt_in, float tmin_in, int nt, float dt, float tmin):
    """The samples of x, which are dt_in apart and start at tmin_in, at nt times that are dt apart and start at tmin

    Times outside of the input trace are 0.
    """
    cdef:
        int nt_in = x.shape[0]
        int i
        float tvalue = tmin
        float[::1] t = np.empty(nt, dtype=np.float32)
        float[::1] out = np.empty(nt, dtype=np.float32)

    if nt_in == 0 or nt == 0:
        return np.zeros(nt, dtype=np.float32)

    with nogil:
        # (accumulated, as suresamp does)
        for i in range(nt):
            t[i] = tvalue
            tvalue += dt
        su.ints8r(nt_in, dt_in, tmin_in, &x[0], 0.0, 0.0, nt, &t[0], &out[0])
    return np.asarray(out)


def interpolate(float[::1] x, float dt_in, float tmin_in, float[::1] xout):
    """The samples of x, which are dt_in apart and start at tmin_in, at the times xout (which need not be evenly spaced)

    Times outside of the input trace are 0.
    """
    cdef:
        int nt_in = x.shape[0]
        int nout = xout.shape[0]
        float[::1] out = np.zeros(nout, dtype=np.float32)

    if nt_in == 0 or nout == 0:
        return np.asarray(out)

    with nogil:
        su.ints8r(nt_in, dt_in, tmin_in, &x[0], 0.0, 0.0, nout, &xout[0], &out[0])
    return np.asarray(out)
