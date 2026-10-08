# cython: embedsignature=True, language_level=3
"""
The kernels of the dip-dependent divergence correction, with the library version of SUDIPDIVCOR (``su_dipdivcor_table`` and
``su_dipdivcor_filter``, in ``su/main/amplitudes/sudipdivcor.c``). The transforms in x are not here, and the scratch arrays
are made here.
"""
from .. cimport su
import numpy as np


def fft_sizes(int nt, int nx):
    """The numbers of samples of the prime factor FFTs of SU, for the time axis (complex) and for x (real)"""
    return su.npfa(nt), su.npfar(nx)


def divcor_table(int n_slopes, float dt, float[::1] vt, bint trans, bint norm):
    """The table of divergence corrections, (np, nt), for the velocities vt (m/s) at the times it * dt.

    Also returns the first velocity, scaled as SUDIPDIVCOR scales it (it works out the slope sampling and the wavenumbers
    from it).
    """
    cdef:
        int nt = vt.shape[0]
        float[::1] tt
        float[::1] vs
        float[:, ::1] vind
        float[:, ::1] divcor
    if nt < 2:
        raise ValueError("The traces must have at least 2 samples.")
    if n_slopes < 2:
        raise ValueError("np must be at least 2.")
    tt = (np.arange(nt) * dt).astype(np.float32)
    vs = np.zeros(nt, dtype=np.float32)
    vind = np.zeros((nt, 4), dtype=np.float32)
    divcor = np.zeros((n_slopes, nt), dtype=np.float32)
    with nogil:
        su.su_dipdivcor_table(nt, n_slopes, dt, &tt[0], &vt[0], &vs[0], <float (*)[4]> &vind[0, 0],
                              &divcor[0, 0], trans, norm)
    return np.asarray(divcor), vs[0]


def dip_filter(float complex[:, ::1] ptk, float dk, float dpx, float dt, int n_slopes, int nw, int nkmax,
               float[:, ::1] divcor):
    """The dip filter of the divergence correction for the wavenumbers 1 to nkmax - 1 of p(k, t), in place"""
    cdef:
        int ik, nt = ptk.shape[1]
        float k = dk
        float complex[::1] kq = np.zeros(nw, dtype=np.complex64)
        float complex[::1] qq = np.zeros(nw, dtype=np.complex64)
    if divcor.shape[0] != n_slopes or divcor.shape[1] != nt:
        raise ValueError("The table of corrections does not fit the traces.")
    if nw < nt:
        raise ValueError("nw must be at least the number of samples of a trace.")
    if nkmax > ptk.shape[0]:
        raise ValueError("nkmax is more than the number of wavenumbers.")
    with nogil:
        # (k is a float that is added up, as in the program)
        for ik in range(1, nkmax):
            su.su_dipdivcor_filter(k, dpx, dt, n_slopes, nw, nt, &divcor[0, 0],
                                   <su.complex *> &ptk[ik, 0], <su.complex *> &ptk[ik, 0],
                                   <su.complex *> &kq[0], <su.complex *> &qq[0])
            k += dk
