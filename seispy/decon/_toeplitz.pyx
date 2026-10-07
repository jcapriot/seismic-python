# cython: embedsignature=True, language_level=3
"""
The Wiener-Levinson solver of the SU library (``stoepf``): Toeplitz matrix equations, as used for deconvolution.
"""
from .. cimport su
import numpy as np


def solve(float[::1] r, float[::1] g):
    """Solve the symmetric Toeplitz system  R f = g  for f, and  R a = (1, 0, ..., 0)  for a (by Levinson recursion)

    r : the first row of the matrix, (autocorrelations)
    g : the right hand side

    Returns f and a (the second is the spiking deconvolution filter).
    """
    cdef int n = r.shape[0]
    if g.shape[0] != n:
        raise ValueError("r and g must have the same length")
    if n == 0:
        raise ValueError("r and g must have some values")
    cdef float[::1] f = np.zeros(n, dtype=np.float32)
    cdef float[::1] a = np.zeros(n, dtype=np.float32)
    with nogil:
        su.stoepf(n, &r[0], &g[0], &f[0], &a[0])
    return np.asarray(f), np.asarray(a)
