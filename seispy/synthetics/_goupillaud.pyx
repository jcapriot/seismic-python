# cython: embedsignature=True, language_level=3
"""
The impulse response of a lossless Goupillaud medium (layers of the same two-way traveltime), with the library versions of
SUGOUPILLAUDPO (primaries only, ``su_goupillaudpo``) and SUGOUPILLAUD (with the multiples, ``su_goupillaud``), in
``su/main/synthetics_waveforms_testpatterns``. The scratch arrays are made here.
"""
from .. cimport su
import numpy as np


def goupillaudpo_tmax(int n, int l, int k):
    """The number of samples that is long enough for all of the primaries, for n interfaces and source and receiver layers l, k"""
    return su.su_goupillaudpo_tmax(n, l, k)


def goupillaudpo(const float[::1] r, int l, int k, int tmax, int pV):
    """The primaries-only seismogram of the reflection coefficients r (the first is that of the surface).

    Returns (status, seismogram, odd): the status is 0, -1 for a reflection coefficient that is not from -1 to 1, -2 for a
    seismogram that is too short to see any signal, -3 for parameters that are not allowed; ``odd`` says that the seismogram is
    shifted by half a sample.
    """
    cdef:
        int n = r.shape[0] - 1
        int odd = 0
        int status
        float[::1] x
        float[::1] out
    if n < 0:
        return -3, np.zeros(0, dtype=np.float32), 0
    x = np.zeros(max(2 * tmax, 1), dtype=np.float32)
    out = np.zeros(max(tmax, 1), dtype=np.float32)
    with nogil:
        status = su.su_goupillaudpo(n, &r[0], l, k, tmax, pV, &x[0], &out[0], &odd)
    return status, np.asarray(out)[:max(tmax, 0)], odd


def goupillaud_tmax(int n, int l, int k):
    """The default number of samples of the seismogram with the multiples, for n interfaces and source and receiver layers l, k"""
    return su.su_goupillaud_tmax(n, l, k)


def goupillaud(const float[::1] r, int l, int k, int tmax, int pV):
    """The seismogram, with the multiples, of the reflection coefficients r (the first is that of the surface).

    Returns (status, seismogram, odd): the status is 0, -1 for a reflection coefficient that is not from -1 to 1, -3 for
    parameters that are not allowed; ``odd`` says that the seismogram is shifted by half a sample.
    """
    cdef:
        int n = r.shape[0] - 1
        int odd = 0
        int status
        float[::1] out
    if n < 1 or tmax < 1:
        return -3, np.zeros(max(tmax, 0), dtype=np.float32), 0
    out = np.zeros(tmax, dtype=np.float32)
    with nogil:
        status = su.su_goupillaud(n, &r[0], l, k, tmax, pV, &out[0], &odd)
    return status, np.asarray(out), odd
