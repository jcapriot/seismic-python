# cython: embedsignature=True, language_level=3
"""
The f-x deconvolution of SU (``sufxdecon``) as a function of a gather: the algorithm of the program, in the library.
"""
from .. cimport su
import numpy as np


def fxdecon(const float[:, ::1] data, float dt, float taper, float fmin, float fmax, float twlen, int ntrw, int ntrf):
    """The f-x prediction filtered gather (traces, samples) of the same shape as ``data``"""
    cdef:
        int ntrc = data.shape[0]
        int nsp = data.shape[1]
        float[:, ::1] out = np.zeros((ntrc, nsp), dtype=np.float32)
        int status
    if ntrc == 0 or nsp == 0:
        return np.asarray(out)
    with nogil:
        status = su.su_fxdecon(nsp, ntrc, &data[0, 0], &out[0, 0], dt, taper, fmin, fmax, twlen, ntrw, ntrf)
    if status != 0:
        raise ValueError("The padded length of the transform is too big (the time window is too long).")
    return np.asarray(out)
