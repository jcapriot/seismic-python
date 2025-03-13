# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from .. cimport container as spyc
from libc.stdlib cimport malloc
import numpy as np

cdef class spike(spyc.BaseTraceIterator):
    cdef:
        size_t nt
        double dt
        double offset
        # spike parameters
        int[:, ::1] spikes

    def __init__(self, size_t nt=64, size_t ntr=32, double dt=0.004, double offset=400, spikes=None):
        self.nt = nt

        self.hdr.n_traces = ntr
        self.hdr.ensemble_type = spyc.EnsembleType.rx_gather
        self.hdr.uniform_traces = True
        self.dt = dt
        self.offset = offset

        if spikes is None:
            spikes = [
                 (ntr // 4, nt // 4 - 1),
                 (ntr // 4, 3 * nt // 4 - 1),
                 (3 * ntr // 4, nt // 4 - 1),
                 (3 * ntr // 4, 3 * nt // 4 - 1),
             ]

        self.spikes = np.require(spikes, dtype=np.int32, requirements='C')

    cdef spyc.Trace next_trace(self):
        if self.i == self.hdr.n_traces:
            raise StopIteration()
        cdef float[::1] data = <float[:self.nt]> malloc(sizeof(float) * self.nt)
        for spike in self.spikes:
            if spike[0] == self.i:
                data[spike[1]] = 1.0
        cdef spyc.spy_trace_header *hdr = spyc.new_hdr()

        hdr.d_sample = self.dt
        hdr.tx_loc[0] = self.i
        hdr.rx_loc[0] = self.i + self.offset
        hdr.offset = self.offset
        hdr.trace_id = self.i + 1

        self.i += 1
        return spyc.Trace.from_trace(hdr, data, True)