# cython: embedsignature=True, language_level=3
"""
The surface consistent residual statics of SU's ``suresstat`` (Ronen and Claerbout), on traces in memory
(``su_resstat`` of ``su/main/statics/suresstat.cpp``).
"""
from .. cimport su
import numpy as np


def resstat(const float[:, ::1] data, const int[::1] icmp, const int[::1] irec, const int[::1] ishot, int ncmp, int nrec,
            int nshot, int ntpick, int niter, int mode, float perc, int sub):
    """The statics (in samples) of the shots and of the receivers, for traces (traces, samples) that have the numbers
    (from 0) of their CMP, receiver and shot"""
    cdef:
        int ntraces = data.shape[0]
        int nt = data.shape[1]
        float[::1] source = np.zeros(max(nshot, 1), dtype=np.float32)
        float[::1] receiver = np.zeros(max(nrec, 1), dtype=np.float32)
    if icmp.shape[0] != ntraces or irec.shape[0] != ntraces or ishot.shape[0] != ntraces:
        raise ValueError("Each trace needs a CMP, a receiver and a shot")
    if ntraces == 0 or nt == 0:
        return np.asarray(source[:nshot]), np.asarray(receiver[:nrec])
    with nogil:
        su.su_resstat(nt, ntraces, &data[0, 0], &icmp[0], &irec[0], &ishot[0], ncmp, nrec, nshot, ntpick, niter, mode,
                      perc, sub, &source[0], &receiver[0])
    return np.asarray(source)[:nshot].copy(), np.asarray(receiver)[:nrec].copy()
