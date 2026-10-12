# cython: embedsignature=True, language_level=3
"""
The generalized Radon transform of SU's ``suradon`` (Anderson): forward and inverse tau-p transforms of a gather in memory
(parabolic, pseudo hyperbolic or linear), the mute that removes multiples in the tau-p domain, and the interpolation of
traces that the program does before it.
"""
from .. cimport su
import numpy as np


cdef void _fill_the_table() noexcept:
    # ints8r (that the interpolation uses) fills a table of sinc coefficients (a function level static) the first time
    # that it is called, which is not safe to do from several threads at once: do that once, here.
    cdef float y = 0.0
    cdef float x = 0.0
    cdef float out = 0.0
    su.ints8r(1, 1.0, 0.0, &y, 0.0, 0.0, 1, &x, &out)


_fill_the_table()


def gofx(int igopt, float offset, float intercept_off, float refdepth):
    """g(x) for the transform options (1 parabolic, 2 pseudo hyperbolic, 3 linear, 4 linear with the absolute value)"""
    return su.su_radon_gofx(igopt, offset, intercept_off, refdepth)


def forward(const float[:, ::1] data, const float[::1] g, float dt, int ntfft, int n_p, float pmin, float dp,
            const float[::1] mutetime, int nk, float f1, float f2, float prewhite):
    """The tau-p panel (np, ntfft) of a gather (nx, nt), g being the function of the offset of each trace"""
    cdef:
        int nx = data.shape[0]
        int nt = data.shape[1]
        float[:, ::1] out = np.zeros((n_p, ntfft), dtype=np.float32)
        int status
    if g.shape[0] != nx or mutetime.shape[0] != nx:
        raise ValueError("g and mutetime need a value for each trace")
    with nogil:
        status = su.su_radon_forward(nx, nt, &data[0, 0], &g[0], dt, ntfft, n_p, pmin, dp, &mutetime[0], nk, f1, f2,
                                     prewhite, &out[0, 0])
    return np.asarray(out)


def inverse(const float[:, ::1] ptau, const float[::1] g, float dt, float pmin, float dp, int ip1, float f1, float f2):
    """The gather (nx, ntfft) of a tau-p panel (np, ntfft), for the traces that g (one value for each) is for. Only the
    ray parameters from ip1 are used (to invert the multiples alone)"""
    cdef:
        int n_p = ptau.shape[0]
        int ntfft = ptau.shape[1]
        int nx = g.shape[0]
        float[:, ::1] out = np.zeros((nx, ntfft), dtype=np.float32)
    with nogil:
        su.su_radon_inverse(n_p, ntfft, nx, &ptau[0, 0], &g[0], dt, pmin, dp, ip1, f1, f2, &out[0, 0])
    return np.asarray(out)


def taupmute(float[::1] rt, int ip, int ipa, int ipb, int ltap):
    """The tau trace of ray parameter ip with the mute that removes the multiples (a new array)"""
    cdef float[::1] out = np.array(rt, dtype=np.float32)
    su.su_radon_taupmute(ip, ipa, ipb, out.shape[0], ltap, &out[0])
    return np.asarray(out)


def xinterpolate(const float[:, ::1] data, int ninterp, float freq1, float freq2, int lagc, int lent, int lenx, int xopt,
                 float dt):
    """The gather (nx, nt) with ninterp traces interpolated (along dips picked from the low frequencies) between each
    pair: 1 + (nx - 1) (1 + ninterp) traces"""
    cdef:
        int nx = data.shape[0]
        int nt = data.shape[1]
        int nout = 1 + (nx - 1) * (1 + ninterp) if nx > 1 else nx
        float[:, ::1] out = np.zeros((nout, nt), dtype=np.float32)
    if nx == 0 or nt == 0:
        return np.asarray(out)
    with nogil:
        su.su_radon_xinterp(nt, nx, ninterp, &data[0, 0], freq1, freq2, lagc, lent, lenx, xopt, dt, &out[0, 0])
    return np.asarray(out)

