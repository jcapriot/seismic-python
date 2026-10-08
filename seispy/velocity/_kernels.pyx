# cython: embedsignature=True, language_level=3
"""
What SUVELAN and SURELAN do with the traces of a gather, from the SU library (``su_velan_accumulate``, ``su_relan_accumulate``
and ``su_velan_semblance``, in ``su/main/velocity_analysis``). The sums for the semblance are float32 arrays with a row for
each velocity (or r parameter), and a column for each sample.
"""
from .. cimport su
import numpy as np


def sums(int n_rows, int n_samples):
    """Zeroed sums for a gather: the numerators, denominators and counts of the semblance"""
    return tuple(np.zeros((n_rows, n_samples), dtype=np.float32) for _ in range(3))


def velan_accumulate(const float[::1] data, float dt, float ft, float offset, int nv, float dv, float fv, float anis1,
                     float anis2, float smute, float[:, ::1] num, float[:, ::1] den, float[:, ::1] nnz):
    """NMO of the trace at each velocity, added to the sums. Returns 1 if a velocity has no moveout, -1 if anis2 is too
    small for the offset, otherwise 0."""
    cdef int nt = data.shape[0]
    cdef int status
    if num.shape[0] != nv or num.shape[1] != nt or den.shape[0] != nv or den.shape[1] != nt \
            or nnz.shape[0] != nv or nnz.shape[1] != nt:
        raise ValueError("The sums must have a row for every velocity, and a column for every sample of the trace.")
    if nt < 2 or nv < 1:
        raise ValueError("There must be at least 2 samples and 1 velocity.")
    with nogil:
        status = su.su_velan_accumulate(nt, dt, ft, offset, nv, dv, fv, anis1, anis2, smute, &data[0],
                                        &num[0, 0], &den[0, 0], &nnz[0, 0])
    return status


def relan_accumulate(const float[::1] data, float dz, float fz, float offset, int nr, float dr, float fr, float smute,
                     float[:, ::1] num, float[:, ::1] den, float[:, ::1] nnz):
    """Residual moveout of the trace at each r parameter, added to the sums"""
    cdef int nz = data.shape[0]
    if num.shape[0] != nr or num.shape[1] != nz or den.shape[0] != nr or den.shape[1] != nz \
            or nnz.shape[0] != nr or nnz.shape[1] != nz:
        raise ValueError("The sums must have a row for every r parameter, and a column for every sample of the trace.")
    if nz < 2 or nr < 1:
        raise ValueError("There must be at least 2 samples and 1 r parameter.")
    with nogil:
        su.su_relan_accumulate(nz, dz, fz, offset, nr, dr, fr, smute, &data[0], &num[0, 0], &den[0, 0], &nnz[0, 0])


def semblance(const float[:, ::1] num, const float[:, ::1] den, const float[:, ::1] nnz, int dtratio, int nsmooth,
              float pwr):
    """The semblance of each row of the sums: (rows, 1 + (nt - 1) // dtratio)"""
    cdef int n_rows = num.shape[0]
    cdef int nt = num.shape[1]
    cdef int ntout = 1 + (nt - 1) // dtratio
    cdef int i
    cdef float[:, ::1] sem = np.zeros((n_rows, ntout), dtype=np.float32)
    if den.shape[0] != n_rows or den.shape[1] != nt or nnz.shape[0] != n_rows or nnz.shape[1] != nt:
        raise ValueError("The sums must be the same size.")
    if dtratio < 1:
        raise ValueError("dtratio must be at least 1")
    with nogil:
        for i in range(n_rows):
            su.su_velan_semblance(nt, ntout, dtratio, nsmooth, pwr, &num[i, 0], &den[i, 0], &nnz[i, 0], &sem[i, 0])
    return np.asarray(sem)
