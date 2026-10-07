# cython: embedsignature=True, language_level=3
"""
The gain function of SUPGC, from the SU library (``su_pgc_gain``).
"""
from .. cimport su
import numpy as np


def gain_function(const float[::1] total, int icount, int lw):
    """The gain at each sample, from the sum of the absolute values of the traces that were scanned (icount of them), with
    the window of lw samples before and after each sample"""
    cdef int nt = total.shape[0]
    cdef float[::1] g = np.zeros(nt, dtype=np.float32)
    if nt > 0:
        with nogil:
            su.su_pgc_gain(nt, &total[0], icount, lw, &g[0])
    return np.asarray(g)
