# cython: embedsignature=True, language_level=3
"""
The three- and four-component functions of the SU library (``su_polar_station``, ``su_eipofi_station``, ``su_alford_angle``
and ``su_alford_rotate``, in ``su/main/multicomponent``), on the traces of one station as float32 arrays.
"""
from .. cimport su
import numpy as np
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)

WINDOWS = {'boxcar': 0, 'bartlett': 1, 'hanning': 2, 'welsh': 3}

# the rows of the attributes of polar_station
ATTRIBUTES = ('rl', 'theta', 'phi', 'tau', 'e21', 'e31', 'e32', 'pln', 'f1', 'l1', 'er', 'ir', 'qr', 'dir1', 'dir2', 'dir3')


def polar_station(const float[:, ::1] data, int iwl, int iwin, float rlq, int rl, int theta, int phi, int tau, int ellip,
                  int pln, int f1, int l1, int amp, int dir, float fangle):
    """The polarization attributes (16, nt) of a station of the three components (3, nt): see ATTRIBUTES"""
    cdef int nt = data.shape[1]
    cdef float[:, ::1] out = np.zeros((16, nt), dtype=np.float32)
    if data.shape[0] != 3:
        raise ValueError("A station has three components.")
    if iwl < 1 or iwl > nt:
        raise ValueError("The window must have from 1 to nt samples.")
    with nogil:
        su.su_polar_station(nt, iwl, iwin, rlq, rl, theta, phi, tau, ellip, pln, f1, l1, amp, dir, fangle, &data[0, 0],
                            &out[0, 0])
    return np.asarray(out)


def eipofi_station(const float[:, ::1] data, int iwl, float pwr, bint cubic):
    """The filtered station (3, nt) and the weights (rl1, rl2, planarity: (3, nt)) of the SVD polarization filter"""
    cdef int nt = data.shape[1]
    cdef float[:, ::1] out = np.array(data, dtype=np.float32)
    cdef float[:, ::1] weights = np.zeros((3, nt), dtype=np.float32)
    if data.shape[0] != 3:
        raise ValueError("A station has three components.")
    if iwl < 3 or iwl > nt:
        raise ValueError("The window must have from 3 to nt samples.")
    if nt // iwl < (2 if cubic else 1):
        raise ValueError("There must be at least 2 windows for the cubic spline (1 for linear).")
    with nogil:
        su.su_eipofi_station(nt, iwl, pwr, cubic, &out[0, 0], &weights[0, 0])
    return np.asarray(out), np.asarray(weights)


def alford_angle(const float[::1] s11, const float[::1] s12, const float[::1] s21, const float[::1] s22, float dt,
                 int itmin, int itmax, float ang_inc, int taper):
    """(angle, quality, quality of no rotation, lag, cross-correlation) of the best Alford rotation of the window"""
    cdef int nt = s11.shape[0]
    cdef float quality = 0.0, inputquality = 0.0, lag = 0.0, xcorr = 0.0, angle
    if s12.shape[0] != nt or s21.shape[0] != nt or s22.shape[0] != nt:
        raise ValueError("The four traces must have the same number of samples.")
    if itmin < 0 or itmax >= nt or itmax - itmin < 3:
        raise ValueError("The window of analysis must be inside of the trace, with at least 4 samples.")
    with nogil:
        angle = su.su_alford_angle(&s11[0], &s12[0], &s21[0], &s22[0], dt, itmin, itmax, ang_inc, taper, &quality,
                                   &inputquality, &lag, &xcorr)
    return angle, quality, inputquality, lag, xcorr


def alford_rotate(float[::1] s11, float[::1] s12, float[::1] s21, float[::1] s22, float theta, int itmin, int taper):
    """Rotate the four traces by theta degrees, from itmin, in place"""
    cdef int nt = s11.shape[0]
    if s12.shape[0] != nt or s21.shape[0] != nt or s22.shape[0] != nt:
        raise ValueError("The four traces must have the same number of samples.")
    with nogil:
        su.su_alford_rotate(&s11[0], &s12[0], &s21[0], &s22[0], theta, nt, itmin, taper)
