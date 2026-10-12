# cython: embedsignature=True, language_level=3
"""
The trace interpolation of SU's ``suinterp`` (``su_interp_traces``, in ``su/main/interp_extrap``): synthetic traces between
the traces of a gather, along dips that are picked from the low frequencies.
"""
from .. cimport su
import numpy as np
from ..stretching import _resample  # (this fills the sinc table that ints8r uses, before any thread can need it)


def interpolate(const float[:, ::1] data, int ninterp, float freq1, float freq2, int lagc, int lent, int lenx, int xopt,
                float dt, int iopt, bint deriv, bint linear):
    """The traces (nout, nt) of the gather (nx, nt) with ninterp traces between each pair (iopt=0), or the low-pass model
    (iopt=1) or the dips in samples per trace (iopt=2) on the traces"""
    cdef int nx = data.shape[0]
    cdef int nt = data.shape[1]
    cdef int capacity = nx if nx < 2 or iopt != 0 else 1 + (nx - 1) * (ninterp + 1)
    cdef float[:, ::1] out
    cdef int nout
    if nx == 0 or nt < 2:
        return np.asarray(data).copy()
    out = np.zeros((max(capacity, nx), nt), dtype=np.float32)
    with nogil:
        nout = su.su_interp_traces(nt, nx, ninterp, &data[0, 0], freq1, freq2, lagc, lent, lenx, xopt, dt, iopt,
                                   deriv, linear, &out[0, 0])
    return np.asarray(out)[:nout].copy()
