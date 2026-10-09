# cython: embedsignature=True, language_level=3
"""
The Born responses of a line scatterer (2-D, SUIMP2D) and of a point scatterer (3-D, SUIMP3D) at a receiver, with the library
versions of the programs (``su_imp2d_trace`` and ``su_imp3d_trace``, in ``su/main/synthetics_waveforms_testpatterns``). The scratch
arrays are made, and kept, here.
"""
from .. cimport su
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)
import numpy as np


cdef class BornResponse:
    """What the responses need for traces of nt samples that are dt apart, in a medium of speed c."""
    cdef:
        int nt
        int _nfft
        float dt
        float c
        float[::1] tout
        float[::1] temp
        float[::1] rt
        float complex[::1] ct

    def __init__(self, int nt, float dt, float c):
        cdef int i
        if nt < 1:
            raise ValueError("nt must be at least 1")
        self._nfft = su.su_imp_nfft(nt)
        if self._nfft < 0:
            raise ValueError(f"Padded nt={nt} -- too big")
        self.nt = nt
        self.dt = dt
        self.c = c
        self.tout = np.empty(nt, dtype=np.float32)
        for i in range(nt):
            self.tout[i] = i * dt
        self.temp = np.zeros(nt, dtype=np.float32)
        self.rt = np.zeros(self._nfft, dtype=np.float32)
        self.ct = np.zeros(self._nfft // 2 + 1, dtype=np.complex64)

    @property
    def nfft(self):
        """The length of the FFT that the traces are padded to"""
        return self._nfft

    def response2d(self, float rs, float rg):
        """The response of the line scatterer at a distance rs from the shot and rg from the receiver"""
        cdef float[::1] data = np.zeros(self.nt, dtype=np.float32)
        with nogil:
            su.su_imp2d_trace(self.nt, self.dt, self._nfft, self.c, rs, rg, &self.tout[0], &self.rt[0],
                              <su.complex *> &self.ct[0], &data[0])
        return np.asarray(data)

    def response3d(self, float rs, float rg, bint direct, float rd):
        """The response of the point scatterer, with the direct arrival (rd is its distance) if `direct`"""
        cdef float[::1] data = np.zeros(self.nt, dtype=np.float32)
        with nogil:
            su.su_imp3d_trace(self.nt, self.dt, self._nfft, self.c, rs, rg, direct, rd, &self.tout[0], &self.temp[0],
                              &self.rt[0], <su.complex *> &self.ct[0], &data[0])
        return np.asarray(data)
