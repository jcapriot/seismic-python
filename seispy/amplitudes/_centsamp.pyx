# cython: embedsignature=True, language_level=3
"""
SUCENTSAMP: a spike for each lobe of a trace, with the area of the lobe, at its centroid (``su_centsamp``).
"""
from .. cimport su
cimport cython
import numpy as np


@cython.boundscheck(False)  # (time has nt samples, and i is from 0 to nt - 1)
@cython.wraparound(False)
def centsamp(float[::1] x, float dt, int nvals_min):
    """The centroid samples of the trace x, which has dt between its samples"""
    cdef int nt = x.shape[0]
    cdef int i
    cdef float[::1] rt = np.array(x, dtype=np.float32)  # (scratch: it is changed)
    cdef float[::1] ct = np.zeros(nt, dtype=np.float32)
    cdef float[::1] mt = np.zeros(nt, dtype=np.float32)
    cdef float[::1] time = np.empty(nt, dtype=np.float32)
    if nt == 0:
        return np.asarray(ct)
    for i in range(nt):
        time[i] = (i + 1) * dt
    with nogil:
        su.su_centsamp(&rt[0], &ct[0], &mt[0], &time[0], nt, dt, nvals_min)
    return np.asarray(ct)
