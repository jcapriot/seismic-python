# cython: embedsignature=True, language_level=3
"""
The filter of SUFILTER: a polygon through (frequency, amplitude) points, with sin^2 tapers between them, calling
``polygonalFilter`` in the SU sources (``su/main/filters/sufilter.c``).
"""
from .. cimport su
import numpy as np


def design(float[::1] f, float[::1] amps, int nfft, float dt):
    """The amplitude of the filter at the nfft // 2 + 1 frequencies of the Fourier transform of nfft samples that
    are dt apart.

    f : frequencies (Hz) that define the filter, increasing
    amps : the amplitudes there
    """
    cdef:
        int npoly = f.shape[0]
        int nf = nfft // 2 + 1
        float[::1] filt = np.empty(nf, dtype=np.float32)
        int[::1] intfr = np.empty(npoly, dtype=np.int32)  # scratch space of polygonalFilter

    if amps.shape[0] != npoly:
        raise ValueError("f and amps must have the same length")
    with nogil:
        su.polygonalFilter(&f[0], &amps[0], npoly, nfft, dt, &filt[0], &intfr[0])
    # (polygonalFilter has the 1/nfft of an unnormalized transform in it, and numpy's transforms are normalized)
    return np.asarray(filt) * np.float32(nfft)
