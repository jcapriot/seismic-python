# cython: embedsignature=True, language_level=3
"""
The library functions of SU's noise programs (SUSEMBASE and SUSEMGROUP: ``su_sem_scan`` and ``su_sem_wsmooth``; SUHARLAN:
``su_harlan``, in ``su/main/noise``), on arrays.
"""
from .. cimport su
import numpy as np
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)


def sem_scan(const float[:, ::1] data, const int[::1] locs, double scanloc, double locmaxscan, float smin, float sstep,
             int jsmax):
    """The (apparent) dip, power and semblance (samples per location unit) at each sample of the traces data (n, nt) at
    locations locs, by semblance scanning at scanloc over the locations within locmaxscan of it"""
    cdef:
        int n = data.shape[0]
        int nt = data.shape[1]
        float[::1] dip = np.zeros(nt, dtype=np.float32)
        float[::1] power = np.zeros(nt, dtype=np.float32)
        float[::1] sem = np.zeros(nt, dtype=np.float32)
        int numout = 0
    if locs.shape[0] != n:
        raise ValueError("There must be a location for each trace.")
    if n == 0 or nt == 0:
        return np.asarray(dip), np.asarray(power), np.asarray(sem)
    if jsmax < 1:
        raise ValueError("The scan must have at least one dip.")
    with nogil:
        su.su_sem_scan(&data[0, 0], &locs[0], n, scanloc, -locmaxscan, locmaxscan, smin, sstep, jsmax, &dip[0],
                       &power[0], &sem[0], nt, &numout)
    return np.asarray(dip), np.asarray(power), np.asarray(sem)


def sem_smooth(const float[::1] val, const float[::1] wgt, int minsd, int maxsd):
    """The values smoothed over minsd to maxsd samples, weighted"""
    cdef:
        int n = val.shape[0]
        float[::1] out = np.zeros(n, dtype=np.float32)
    if wgt.shape[0] != n:
        raise ValueError("There must be a weight for each value.")
    if n == 0:
        return np.asarray(out)
    with nogil:
        su.su_sem_wsmooth(&val[0], &wgt[0], n, minsd, maxsd, &out[0])
    return np.asarray(out)


def shift(const float[::1] trace, const float[::1] times):
    """The trace at the (fractional) sample indices times, by the 8 point sinc interpolation"""
    cdef:
        int n = trace.shape[0]
        float[::1] out = np.zeros(times.shape[0], dtype=np.float32)
    if n == 0 or times.shape[0] == 0:
        return np.asarray(out)
    with nogil:
        su.ints8r(n, 1.0, 0.0, <float *> &trace[0], 0.0, 0.0, times.shape[0], <float *> &times[0], &out[0])
    return np.asarray(out)


def harlan(const float[:, ::1] data, int rgt, int sts, int norm, int anenv, int scl, int seed, int sditer, int niter,
           int nintlh, float dt, int gopt, float prewhite, int ninterp, float offref, float depthref, float interoff,
           float pmin1, float pmax1, float pmin2, float pmax2, int np_, float dx, float c, float rel1, float rel2,
           float r1, float r2, float fx, float pmula, float pmulb):
    """The extracted signal and noise (nx, nt) of the traces data (nx, nt)"""
    cdef:
        int nx = data.shape[0]
        int nt = data.shape[1]
        float[:, ::1] signal = np.zeros((nx, nt), dtype=np.float32)
        float[:, ::1] noise = np.zeros((nx, nt), dtype=np.float32)
    if nx < 2 or nt < 2:
        raise ValueError("There must be at least two traces of two samples.")
    with nogil:
        su.su_harlan(rgt, sts, norm, anenv, scl, seed, sditer, niter, nintlh, dt, nt, nx, gopt, prewhite, ninterp,
                     offref, depthref, interoff, pmin1, pmax1, pmin2, pmax2, np_, dx, c, rel1, rel2, r1, r2, fx, pmula,
                     pmulb, &data[0, 0], &signal[0, 0], &noise[0, 0])
    return np.asarray(signal), np.asarray(noise)
