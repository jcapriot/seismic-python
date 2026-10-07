# cython: embedsignature=True, language_level=3
"""
SUGAIN, calling the library version of the program in the SU sources (``su_gain`` in ``su/main/amplitudes/sugain.c``).

The C program keeps its lookup tables and scratch arrays in function level ``static`` variables that are filled in on
the first trace. In the library version they are arguments, so everything that has to persist from trace to trace
lives on the iterator instance here, and the tables are rebuilt whenever the time axis changes.
"""
from .. cimport container as spyc
from .. cimport su
from ..container import join_complex, split_complex
from libc.math cimport fabs
from libc.float cimport FLT_MAX
import numpy as np


cdef inline int nint(double x) noexcept nogil:
    return <int> (x + 0.5 if x > 0.0 else x - 0.5)


cdef struct GainParams:
    float tpow
    float epow
    float etpow
    float gpow
    float vred
    float trap
    float clip
    float pclip
    float nclip
    float qclip
    float scale
    float bias
    int agc
    int gagc
    int qbal
    int pbal
    int mbal
    int maxbal


cdef class _gain(spyc.BaseTraceIterator):
    cdef:
        spyc.BaseTraceIterator iter_in
        GainParams p
        float wagc
        float dt_user
        bint panel
        bint inplace
        # scratch arrays of su_gain, and the tables of t^tpow and exp(epow t^etpow)
        float[::1] absdata, agcdata, d2, w, s
        float[::1] tpowfac, epowfac, tpowfac_tr
        size_t fac_n
        float fac_tmin
        float fac_dt
        list panel_out
        size_t panel_i

    def __init__(
        self, trace_iter, *, tpow=0.0, epow=0.0, etpow=1.0, gpow=1.0, agc=False, gagc=False, wagc=0.5,
        trap=0.0, clip=0.0, pclip=None, nclip=None, qclip=1.0, qbal=False, pbal=False, mbal=False,
        maxbal=False, scale=1.0, norm=0.0, bias=0.0, jon=False, vred=0.0, dt=None, panel=False, inplace=False,
    ):
        self.iter_in = spyc.as_trace_iterator(trace_iter)
        self.hdr = self.iter_in.hdr

        if vred < 0.0:
            raise ValueError(f"vred = {vred}, must be positive")
        if trap < 0.0:
            raise ValueError(f"trap = {trap}, must be positive")
        if clip < 0.0:
            raise ValueError(f"clip = {clip}, must be positive")
        if qclip < 0.0 or qclip > 1.0:
            raise ValueError(f"qclip = {qclip}, must be between 0 and 1")
        if jon:
            # the choices from Claerbout's Imaging the Earth, pp 233-236
            tpow = 2.0
            gpow = 0.5
            qclip = 0.95
        if norm:
            scale /= norm

        self.p.tpow = tpow
        self.p.epow = epow
        self.p.etpow = etpow
        self.p.gpow = gpow
        self.p.vred = vred
        self.p.trap = trap
        self.p.clip = clip
        self.p.pclip = FLT_MAX if pclip is None else pclip
        self.p.nclip = -FLT_MAX if nclip is None else nclip
        self.p.qclip = qclip
        self.p.scale = scale
        self.p.bias = bias
        self.p.agc = agc
        self.p.gagc = gagc
        self.p.qbal = qbal
        self.p.pbal = pbal
        self.p.mbal = mbal
        self.p.maxbal = maxbal
        self.wagc = wagc
        self.dt_user = 0.0 if dt is None else dt
        self.panel = panel
        self.inplace = inplace

        empty = np.empty(0, dtype=np.float32)
        self.absdata = self.agcdata = self.d2 = self.w = self.s = empty
        self.tpowfac = self.epowfac = self.tpowfac_tr = empty
        self.fac_n = 0
        self.panel_out = None
        self.panel_i = 0

    cdef _ensure_size(self, size_t nt):
        if self.tpowfac.shape[0] < nt:
            self.absdata = np.empty(nt, dtype=np.float32)
            self.agcdata = np.empty(nt, dtype=np.float32)
            self.d2 = np.empty(nt, dtype=np.float32)
            self.w = np.empty(nt, dtype=np.float32)  # (only iwagc <= nt/2 of these are used)
            self.s = np.empty(nt, dtype=np.float32)
            self.tpowfac = np.empty(nt, dtype=np.float32)
            self.epowfac = np.empty(nt, dtype=np.float32)
            self.tpowfac_tr = np.empty(nt, dtype=np.float32)
            self.fac_n = 0  # the tables have to be rebuilt

    cdef _apply(self, float *data, size_t nt_in, float tmin, float dt, double offset):
        """Gain one trace, or a whole panel of traces glued together, in place."""
        cdef:
            int nt = <int> nt_in
            int iwagc = 0
            float tred
            float *tfac = NULL
            float *efac = NULL

        self._ensure_size(nt_in)

        if self.p.agc or self.p.gagc:
            iwagc = nint(self.wagc / dt)
            if iwagc < 1:
                raise ValueError(f"wagc={self.wagc:g} must be positive")
            if iwagc > nt:
                raise ValueError(f"wagc={self.wagc:g} too long for trace")
            iwagc >>= 1  # windows are symmetric, so work with the half
            if iwagc < 1:
                raise ValueError(f"wagc={self.wagc:g} is shorter than two samples")

        # rebuild the tables if the time axis changed
        if self.fac_n != nt_in or self.fac_tmin != tmin or self.fac_dt != dt:
            if self.p.tpow != 0.0:
                su.su_gain_tpow_table(&self.tpowfac[0], nt, tmin, dt, self.p.tpow, 0.0)
            if self.p.epow != 0.0:
                su.su_gain_epow_table(&self.epowfac[0], nt, tmin, dt, self.p.epow, self.p.etpow)
            self.fac_n = nt_in
            self.fac_tmin = tmin
            self.fac_dt = dt

        if self.p.tpow != 0.0:
            if self.p.vred > 0.0:
                # the reduced time depends on the offset of each trace
                tred = fabs(<float> offset / self.p.vred)
                su.su_gain_tpow_table(&self.tpowfac_tr[0], nt, tmin, dt, self.p.tpow, tred)
                tfac = &self.tpowfac_tr[0]
            else:
                tfac = &self.tpowfac[0]
        if self.p.epow != 0.0:
            efac = &self.epowfac[0]

        with nogil:
            su.su_gain(
                data, self.p.tpow, self.p.epow, self.p.gpow,
                self.p.agc, self.p.gagc, self.p.qbal, self.p.pbal, self.p.mbal, self.p.scale, self.p.bias,
                self.p.trap, self.p.clip, self.p.qclip, iwagc,
                nt, self.p.maxbal, self.p.pclip, self.p.nclip,
                tfac, efac,
                &self.absdata[0], &self.agcdata[0], &self.d2[0], &self.w[0], &self.s[0],
            )

    cdef float _dt_of(self, spyc.spy_trace_header *hdr) except? -1:
        cdef float dt = <float> hdr.d_sample
        if dt == 0.0:
            dt = self.dt_user
        if dt == 0.0:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        return dt

    cdef spyc.Trace next_trace(self):
        if self.panel:
            return self._next_panel()

        cdef spyc.Trace trace = self.iter_in.next_trace()
        if trace.hdr.data_type == spyc.SPY_DTYPE_COMPLEX64:
            return self._next_complex(trace)
        return self._gain_trace(trace)

    cdef _nonlinear_options(self):
        """The names of the options that are set and have no meaning for complex numbers"""
        names = []
        if not abs(self.p.gpow - 1.0) <= 1e-6:
            names.append('gpow (or jon)')
        if self.p.agc:
            names.append('agc')
        if self.p.gagc:
            names.append('gagc')
        if self.p.trap > 0.0:
            names.append('trap')
        if self.p.clip > 0.0:
            names.append('clip')
        if self.p.pclip < FLT_MAX:
            names.append('pclip')
        if self.p.nclip > -FLT_MAX:
            names.append('nclip')
        if self.p.qclip < 1.0:
            names.append('qclip')
        if self.p.qbal:
            names.append('qbal')
        if self.p.pbal:
            names.append('pbal')
        if self.p.mbal:
            names.append('mbal')
        if self.p.maxbal:
            names.append('maxbal')
        return names

    cdef spyc.Trace _next_complex(self, spyc.Trace trace):
        """The gains that are linear (tpow, epow, vred, scale, norm, bias) are the same for the real and the
        imaginary part, except for the bias, which is a real number that is added to the real part."""
        names = self._nonlinear_options()
        if names:
            raise TypeError(
                f"gain does not define {', '.join(names)} for complex traces (it has tpow, epow, vred, scale, norm and "
                "bias for them)."
            )
        cdef float bias = self.p.bias
        real, imag = split_complex(trace)
        out_real = self._gain_trace(real)
        self.p.bias = 0.0
        try:
            out_imag = self._gain_trace(imag)
        finally:
            self.p.bias = bias
        return join_complex(out_real, out_imag)

    cdef spyc.Trace _gain_trace(self, spyc.Trace trace):
        cdef:
            float[::1] data_in = trace.data
            size_t n_sample = data_in.shape[0]
            float[::1] data
            float dt = self._dt_of(trace.hdr)

        if n_sample == 0:
            return spyc.Trace.from_trace(spyc.copy_of_hdr(trace.hdr), data_in, True)

        if self.inplace:
            data = data_in
        else:
            data = spyc.alloc_data(n_sample)
            data[:] = data_in

        self._apply(&data[0], n_sample, <float> trace.hdr.sample_start, dt, spyc.hdr_offset(trace.hdr))
        return spyc.Trace.from_trace(spyc.copy_of_hdr(trace.hdr), data, True)

    cdef spyc.Trace _next_panel(self):
        """Gain the whole data set at once, which SU does by treating all traces as one long trace."""
        cdef:
            spyc.Trace tr
            list traces
            size_t n, ntr, k
            float[::1] big
            float[::1] data
            float dt

        if self.panel_out is None:
            traces = []
            try:
                while True:
                    traces.append(self.iter_in.next_trace())
            except StopIteration:
                pass
            self.panel_out = []
            self.panel_i = 0
            ntr = len(traces)
            if ntr > 0:
                tr = traces[0]
                n = tr.data.shape[0]
                for k in range(ntr):
                    tr = traces[k]
                    if tr.data.shape[0] != n:
                        raise ValueError("panel gain needs every trace to have the same number of samples.")
                if n > 0:
                    tr = traces[0]
                    for k in range(ntr):
                        spyc.require_real(traces[k])  # (panel gain is for real traces)
                    dt = self._dt_of(tr.hdr)
                    big = spyc.alloc_data(n * ntr)
                    for k in range(ntr):
                        tr = traces[k]
                        big[k * n:(k + 1) * n] = tr.data
                    tr = traces[0]
                    # (SU uses the offset of the last trace read when reducing the time axis)
                    self._apply(&big[0], n * ntr, <float> tr.hdr.sample_start, dt, spyc.hdr_offset((<spyc.Trace> traces[ntr - 1]).hdr))
                    for k in range(ntr):
                        tr = traces[k]
                        data = spyc.alloc_data(n)
                        data[:] = big[k * n:(k + 1) * n]
                        self.panel_out.append(spyc.Trace.from_trace(spyc.copy_of_hdr(tr.hdr), data, True))
                else:
                    self.panel_out = traces

        if self.panel_i == len(self.panel_out):
            raise StopIteration()
        out = self.panel_out[self.panel_i]
        self.panel_i += 1
        return out
