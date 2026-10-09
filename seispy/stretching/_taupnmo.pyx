# cython: embedsignature=True, language_level=3
"""
The NMO of a trace of the tau-p domain, with the library version of SUTAUPNMO (``su_taupnmo_tables`` and ``su_taupnmo``, in
``su/main/stretching_moveout_resamp/sutaupnmo.c``). The scratch arrays are made, and kept, here.
"""
from .. cimport su
from . import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)
import numpy as np


cdef class TaupNMOTables:
    """What su_taupnmo needs for traces with one time axis, ray parameter and v^2."""
    cdef:
        int nt
        float dt
        float ft
        int itmute
        bint sscale
        float[::1] ttn, atn, q

    def __init__(self, int nt, float dt, float ft, float p, float[::1] vvt, float smute, bint sscale):
        if nt < 2:
            raise ValueError("NMO needs traces with at least 2 samples.")
        if vvt.shape[0] != nt:
            raise ValueError("v^2 must have a value for every sample.")
        self.nt = nt
        self.dt = dt
        self.ft = ft
        self.sscale = sscale
        self.ttn = np.zeros(nt, dtype=np.float32)
        self.atn = np.zeros(nt, dtype=np.float32)
        self.q = np.zeros(nt, dtype=np.float32)
        cdef int itmute = 0
        with nogil:
            su.su_taupnmo_tables(nt, dt, ft, p, &vvt[0], smute, &self.ttn[0], &self.atn[0], &itmute)
        self.itmute = itmute

    @property
    def itmute(self):
        """Samples from this one on are muted"""
        return self.itmute

    def apply(self, float[::1] x, int lmute):
        """The NMO of the samples x, in place"""
        if x.shape[0] != self.nt:
            raise ValueError("The trace has a different number of samples than the tables were made for.")
        with nogil:
            su.su_taupnmo(
                &x[0], self.nt, self.dt, self.ft, self.itmute, lmute, self.sscale,
                &self.ttn[0], &self.atn[0], &self.q[0],
            )
