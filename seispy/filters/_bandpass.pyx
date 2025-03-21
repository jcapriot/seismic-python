# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from .. cimport container as spyc
from .. cimport su
from libc.math cimport sqrt
from libc.stdlib cimport malloc

cdef class butterworth_bandpass(spyc.BaseTraceIterator):
    cdef:
        int zerophase
        int low_cut, high_cut
        int npoleslo, npoleshi
        float f3dblo, f3dbhi, f3dblo_in, f3dbhi_in
        float last_dt
        spyc.BaseTraceIterator iter_in
        int design_low, design_high
        int none_f3dblo, none_f3dbhi, none_fpasslo, none_fstoplo, none_fpasshi, none_fstophi
        float fstoplo, fpasslo, fstophi, fpasshi
        float astoplo, apasslo, astophi, apasshi
        bint inplace

    def __init__(
        self, trace_iter, bint low_cut=True, bint high_cut=True,
        f_stop_low=None, float a_stop_low=0.05, f_pass_low=None, float a_pass_low=0.95,
        f_stop_high=None, float a_stop_high=0.05, f_pass_high=None, float a_pass_high=0.95,
        int n_poles_low=0, f3db_low=None, int n_poles_high=0, f3db_high=None,
        bint zerophase=True, bint inplace=False
    ):
        self.last_dt = 0
        if isinstance(trace_iter, spyc.TraceCollection):
            trace_iter = trace_iter.__iter__()
        self.iter_in = trace_iter
        self.hdr = self.iter_in.hdr
        self.zerophase = zerophase
        self.low_cut = low_cut
        self.high_cut = high_cut
        self.inplace = inplace

        self.design_low = n_poles_low == 0
        self.design_high = n_poles_high == 0

        self.npoleslo = n_poles_low
        self.apasslo = a_pass_low
        self.astoplo = a_stop_low

        self.npoleshi = n_poles_high
        self.apasshi = a_pass_high
        self.astophi = a_stop_high

        if self.zerophase:
            self.apasslo = sqrt(self.apasslo)
            self.astoplo = sqrt(self.astoplo)
            self.apasshi = sqrt(self.apasshi)
            self.astophi = sqrt(self.astophi)

        if f3db_low is None:
            self.none_f3dblo = True
        else:
            self.f3dblo_in = f3db_low

        if f3db_high is None:
            self.none_f3dbhi = True
        else:
            self.f3dbhi_in = f3db_high

        if f_pass_low is None:
            self.none_fpasslo = True
        else:
            self.fpasslo = f_pass_low

        if f_stop_low is None:
            self.none_fstoplo = True
        else:
            self.fstoplo = f_stop_low

        if f_pass_high is None:
            self.none_fpasshi = True
        else:
            self.fpasshi = f_pass_high

        if f_stop_high is None:
            self.none_fstophi = True
        else:
            self.fstophi = f_stop_high

    cdef void set_filter_params(self, spyc.spy_trace_header *hdr):
        cdef float fstoplo, fstophi, fpasslo, fpasshi
        if hdr.d_sample != self.last_dt:
            self.last_dt = hdr.d_sample

            if self.low_cut:
                if self.design_low:
                    if self.none_fstoplo:
                        fstoplo = .10 * 0.5 # nyq * dt
                    else:
                        fstoplo = self.fstoplo * hdr.d_sample
                    if self.none_fpasslo:
                        fpasslo = .15 * 0.5 # nyq * dt
                    else:
                        fpasslo = self.fpasslo * hdr.d_sample
                    su.bfdesign(fpasslo, self.apasslo, fstoplo, self.astoplo, &self.npoleslo, &self.f3dblo)
                elif self.none_f3dblo:
                    self.f3dblo = .15 * 0.5 # nyq * dt
                else:
                    self.f3dblo = self.f3dblo_in * hdr.d_sample

            if self.high_cut:
                if self.design_high:
                    if self.none_fstophi:
                        fstophi = .55 * 0.5 # nyq * dt
                    else:
                        fstophi = self.fstophi * hdr.d_sample
                    if self.none_fpasshi:
                        fpasshi = .40 * 0.5 # nyq * dt
                    else:
                        fpasshi = self.fpasshi * hdr.d_sample
                    su.bfdesign(fpasshi, self.apasshi, fstophi, self.astophi, &self.npoleshi, &self.f3dbhi)
                elif self.none_f3dbhi:
                    self.f3dbhi = .40 * 0.5 # nyq * dt
                else:
                    self.f3dbhi = self.f3dbhi_in * hdr.d_sample

    cdef spyc.Trace next_trace(self):
        cdef:
            spyc.Trace trace = self.iter_in.next_trace()
            size_t n_sample = trace.data.shape[0]
        self.set_filter_params(trace.hdr)

        cdef:
            float[::1] data
        if self.inplace:
            data = trace.data
        else:
            data = <float[:n_sample]> malloc(n_sample * sizeof(float))
            data[:] = trace.data[:]

        if self.low_cut:
            su.su_bfhighpass(self.zerophase, self.npoleslo, self.f3dblo, n_sample, &data[0], &data[0])

        if self.high_cut:
            su.su_bflowpass(self.zerophase, self.npoleshi, self.f3dbhi, n_sample, &data[0], &data[0])

        cdef spyc.spy_trace_header *hdr = spyc.copy_of_hdr(trace.hdr)
        return spyc.Trace.from_trace(hdr, data, True)