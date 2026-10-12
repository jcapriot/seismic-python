# cython: embedsignature=True, language_level=3
"""
More of the velocity analysis programs of the SU library: SUTIVEL (velocity tables of a TI medium), SUVEL2DF (semblance over
the NMO and the horizontal velocity), SURELANAN (residual moveout semblance of anisotropic media), and SUVELAN_UCCS, _NCCS,
_USEL and _NSEL (the sums of cross-correlations), and the interpolation of the reflector of SURELANAN (``su/main/velocity_analysis``).
"""
from .. cimport su
import numpy as np


def tivel_table(float a, float b, float e, float d, float maxangle, int nangle, bint normalize):
    """p, vphase, dv/dtheta, d2v/dtheta2 and vnmo, for nangle angles from 0 to maxangle (degrees)"""
    if nangle < 2:
        nangle = 2
    cdef:
        float[::1] p = np.zeros(nangle, dtype=np.float32)
        float[::1] v = np.zeros(nangle, dtype=np.float32)
        float[::1] dv = np.zeros(nangle, dtype=np.float32)
        float[::1] d2v = np.zeros(nangle, dtype=np.float32)
        float[::1] vnmo = np.zeros(nangle, dtype=np.float32)
    with nogil:
        su.su_tivel_table(a, b, e, d, maxangle, nangle, normalize, &p[0], &v[0], &dv[0], &d2v[0], &vnmo[0])
    return (np.asarray(p), np.asarray(v), np.asarray(dv), np.asarray(d2v), np.asarray(vnmo))


def tivel_vnmoweak(float a, float b, float e, float d, float p):
    """The NMO velocity of the weak anisotropy approximation for the ray parameter p (-1 when it is evanescent)"""
    return su.su_tivel_vnmoweak(a, b, e, d, p)


def vel2df_accumulate(const float[::1] data, float dt, float ft, float offset, float tn, int nv, float dv, float fv,
                      int nvh, float dvh, float fvh, int nsmooth, float[:, :, ::1] num, float[:, :, ::1] den,
                      float[:, :, ::1] nnz):
    """The trace added to the sums (nv, nvh, nsmooth) of the semblance over two velocities"""
    cdef int nt = data.shape[0]
    if num.shape[0] != nv or num.shape[1] != nvh or num.shape[2] != nsmooth:
        raise ValueError("The sums must be (nv, nvh, nsmooth).")
    if nt < 2 or nsmooth < 1:
        raise ValueError("There must be at least 2 samples and nsmooth must be at least 1.")
    with nogil:
        su.su_vel2df_accumulate(nt, dt, ft, offset, tn, nv, dv, fv, nvh, dvh, fvh, nsmooth, &data[0], &num[0, 0, 0],
                                &den[0, 0, 0], &nnz[0, 0, 0])


def relanan_accumulate(const float[::1] data, float dz, float fz, float half_offset, float Z, int nr1, float dr1,
                       float fr1, int nr2, float dr2, float fr2, int nsmooth, float[:, :, ::1] num,
                       float[:, :, ::1] den, float[:, :, ::1] nnz):
    """The trace added to the sums (nr1, nr2, nsmooth) of the residual moveout semblance"""
    cdef int nz = data.shape[0]
    if num.shape[0] != nr1 or num.shape[1] != nr2 or num.shape[2] != nsmooth:
        raise ValueError("The sums must be (nr1, nr2, nsmooth).")
    if nz < 2 or nsmooth < 1:
        raise ValueError("There must be at least 2 samples and nsmooth must be at least 1.")
    with nogil:
        su.su_relanan_accumulate(nz, dz, fz, half_offset, Z, nr1, dr1, fr1, nr2, dr2, fr2, nsmooth, &data[0],
                                 &num[0, 0, 0], &den[0, 0, 0], &nnz[0, 0, 0])


def velan_cc(const float[:, ::1] data, const float[::1] offsets, float dt, float ft, int nv, float dv, float fv,
             float smute, int dtratio, int nsmooth, bint normalize, bint selective, float tau, float pwr):
    """The panel (nv, ntout) of the sum of the cross-correlations of the traces (nx, nt) of a gather"""
    cdef int nx = data.shape[0]
    cdef int nt = data.shape[1]
    cdef int ntout = 1 + (nt - 1) // dtratio
    cdef float[:, ::1] sem = np.zeros((nv, ntout), dtype=np.float32)
    if offsets.shape[0] != nx:
        raise ValueError("There must be an offset for every trace.")
    if nx == 0 or nt < 2:
        return np.asarray(sem)
    with nogil:
        su.su_velan_cc(nt, dt, ft, nx, &offsets[0], &data[0, 0], nv, dv, fv, smute, dtratio, nsmooth, normalize,
                       selective, tau, pwr, &sem[0, 0])
    return np.asarray(sem)


def interpolate(const float[::1] x, const float[::1] z, str method, float at):
    """The value at ``at`` of the curve through the points (x, z): 'linear', 'mono' (monotonic cubic), 'akima' or 'spline'"""
    cdef int n = x.shape[0]
    cdef float[::1] xo = np.array([at], dtype=np.float32)
    cdef float[::1] zo = np.zeros(1, dtype=np.float32)
    cdef float[:, ::1] yd
    cdef float[::1] xc = np.array(x, dtype=np.float32)
    cdef float[::1] zc = np.array(z, dtype=np.float32)
    if z.shape[0] != n or n < 1:
        raise ValueError("x and z must have the same number (at least 1) of points.")
    if method == 'linear' or n == 1:
        su.intlin(n, &xc[0], &zc[0], zc[0], zc[n - 1], 1, &xo[0], &zo[0])
        return zo[0]
    yd = np.zeros((n, 4), dtype=np.float32)
    if method == 'mono':
        su.cmonot(n, &xc[0], &zc[0], <float (*)[4]> &yd[0, 0])
    elif method == 'akima':
        su.cakima(n, &xc[0], &zc[0], <float (*)[4]> &yd[0, 0])
    elif method == 'spline':
        su.csplin(n, &xc[0], &zc[0], <float (*)[4]> &yd[0, 0])
    else:
        raise ValueError(f"{method!r} is an unknown interpolation method (linear, mono, akima or spline)")
    su.intcub(0, n, &xc[0], <float (*)[4]> &yd[0, 0], 1, &xo[0], &zo[0])
    return zo[0]
