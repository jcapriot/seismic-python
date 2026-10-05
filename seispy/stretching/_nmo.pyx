# cython: embedsignature=True, language_level=3
"""
The NMO of a trace, with the library version of SUNMO (``su_nmo_tables`` and ``su_nmo``, in
``su/main/stretching_moveout_resamp/sunmo.c``). The scratch arrays are made, and kept, here.
"""
from .. cimport su
from . import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)
import numpy as np


cdef class NMOTables:
    """What su_nmo needs for traces with one time axis, offset and sloth."""
    cdef:
        int nt
        float dt
        float ft
        int itmute
        bint invert
        bint sscale
        float[::1] ttn, atn, tnt, at, q

    def __init__(self, int nt, float dt, float ft, float offset, float[::1] ovvt, float smute, bint upward,
                 bint invert, bint sscale):
        if nt < 2:
            raise ValueError("NMO needs traces with at least 2 samples.")
        if invert and nt < 4:
            raise ValueError("Inverse NMO needs traces with at least 4 samples.")
        if ovvt.shape[0] != nt:
            raise ValueError("The sloth must have a value for every sample.")
        self.nt = nt
        self.dt = dt
        self.ft = ft
        self.invert = invert
        self.sscale = sscale
        self.ttn = np.zeros(nt, dtype=np.float32)
        self.atn = np.zeros(nt, dtype=np.float32)
        self.tnt = np.zeros(nt, dtype=np.float32)
        self.at = np.zeros(nt, dtype=np.float32)
        self.q = np.zeros(nt, dtype=np.float32)
        cdef int itmute = 0
        with nogil:
            su.su_nmo_tables(
                nt, dt, ft, offset, &ovvt[0], smute, upward, invert, sscale,
                &self.ttn[0], &self.atn[0], &self.tnt[0], &self.at[0], &itmute,
            )
        self.itmute = itmute

    @property
    def itmute(self):
        """Samples before this one are muted"""
        return self.itmute

    def apply(self, float[::1] x, int lmute):
        """The NMO of the samples x, in place"""
        if x.shape[0] != self.nt:
            raise ValueError("The trace has a different number of samples than the tables were made for.")
        with nogil:
            su.su_nmo(
                &x[0], self.nt, self.dt, self.ft, self.itmute, lmute, self.sscale, self.invert,
                &self.ttn[0], &self.atn[0], &self.tnt[0], &self.at[0], &self.q[0],
            )
