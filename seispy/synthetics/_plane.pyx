# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from .. cimport container as spyc
import numpy as np
from libc.math cimport fabs
from libc.stdlib cimport malloc

cdef class plane(spyc.BaseTraceIterator):
    cdef:
        size_t nt
        double dt
        double offset
        bint taper
        double msps
        # plane parameter holders
        float[:] dips
        int[:] cts
        int[:] cxs
        int[:] lens
        int n_planes

        int[:] tfes

    def __init__(self, size_t nt=64, size_t ntr=32, bint taper=False, double dt=0.004, double offset=400.0, planes=None):
        if planes is None:
            planes = [
                # (dip ms/trace, n_trace_extent, i_center_time, i_center_trace)
                (0, 3 * ntr//4, nt//2 - 1, ntr//2 - 1),
                (4, 3 * ntr//4, nt//2 - 1, ntr//2 - 1),
                (8, 3 * ntr//4, nt//2 - 1, ntr//2 - 1),
             ]
        elif planes == "liner":
            nt = 64
            ntr = 64
            planes = [
                # (dip ms/trace, n_trace_extent, i_center_time, i_center_trace)
                (0, ntr//4, nt//2 - 1, 3 * ntr//4 - 1),
                (4, ntr//4, nt//2 - 1, ntr//2 - 1),
                (8, ntr//4, nt//2 - 1, ntr//4 - 1),
             ]
        n_planes = len(planes)
        dips = np.empty(n_planes, dtype=np.float32)
        lens = np.empty(n_planes, dtype=np.int32)
        cts = np.empty(n_planes, dtype=np.int32)
        cxs = np.empty(n_planes, dtype=np.int32)
        for i, plane in enumerate(planes):
            if len(plane) != 4:
                raise TypeError("Must supply four values to describe a plane.")
            dips[i] = plane[0]
            lens[i] = plane[1]
            cts[i] = plane[2]
            cxs[i] = plane[3]
        self.dips = dips
        self.lens = lens
        self.cts = cts
        self.cxs = cxs
        self.tfes = np.zeros(n_planes, dtype=np.int32)

        self.nt = nt
        self.dt = dt
        self.msps = dt * 1_000
        self.offset = offset
        self.taper = taper

        self.hdr.n_traces = ntr
        self.hdr.ensemble_type = spyc.EnsembleType.tx_gather
        self.hdr.uniform_traces = True
        self.n_planes = self.dips.shape[0]

    cdef spyc.Trace next_trace(self):
        if self.i == self.hdr.n_traces:
            raise StopIteration()
        cdef:
            int i, tfe, itr, itless, itmore, cx_i, dt_i, len_i
            float eps, fit, dip_i

        itr = self.i


        cdef float[::1] data = <float[:self.nt]> malloc(sizeof(float) * self.nt)
        data[:] = 0.0

        for i in range(self.n_planes):
            tfe = self.tfes[i]
            dip_i = self.dips[i]
            cx_i = self.cxs[i]
            ct_i = self.cts[i]
            len_i = self.lens[i]
            if (itr >= cx_i - len_i // 2) and (itr <= cx_i + len_i // 2):

                # fit is fractional sample of plane intersection
                fit = ct_i - ( cx_i - itr ) * dip_i / self.msps
                if (fit >= 0) and (fit < self.nt - 1):

                    # linear interpolation
                    itless = <int> fit
                    eps = fit - itless
                    itmore = <int> (fit + 1)
                    data[itless] += 1.0 - eps
                    data[itmore] += eps

                    # taper option
                    if self.taper:
                        # first or last point
                        if tfe == 0 or tfe == len_i:
                            data[itless] /= 6.0
                            data[itmore] /= 6.0
                        # second or next-to-last point
                        if tfe == 1 or tfe == len_i-1:
                            data[itless] /= 3.0
                            data[itmore] /= 3.0

                self.tfes[i] += 1

        cdef spyc.spy_trace_header *hdr = spyc.new_hdr()
        hdr.d_sample = self.dt
        hdr.offset = fabs(self.offset)
        hdr.line_id = 1
        hdr.trace_id = self.i + 1
        hdr.tx_loc[0] = self.i
        hdr.rx_loc[0] = self.i + self.offset

        self.i += 1
        return spyc.Trace.from_trace(hdr, data, True)