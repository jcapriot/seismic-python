# cython: embedsignature=True, language_level=3
"""
SUFILTER, a zero-phase, sine-squared tapered frequency domain filter, calling the library version of the program in
the SU sources (``polygonalFilter`` and ``su_filter`` in ``su/main/filters/sufilter.c``).

The C program builds its filter once, from the first trace. Here it is rebuilt whenever the length or sample interval
of the traces changes, and SU's limit of ``SU_NFLTS`` samples in a trace does not apply.
"""
import warnings

from .. cimport container as spyc
from .. cimport su
import numpy as np

cdef float FRAC0 = 0.10  # ratio of default f1 to Nyquist
cdef float FRAC1 = 0.15
cdef float FRAC2 = 0.45
cdef float FRAC3 = 0.50
cdef int LOOKFAC = 2  # look ahead factor for npfaro
cdef int PFA_MAX = 720720  # largest allowed nfft


cdef class _filter(spyc.BaseTraceIterator):
    cdef:
        spyc.BaseTraceIterator iter_in
        object f_in
        object amps_in
        float dt_user
        bint inplace
        # the filter, for traces of this length and sample interval
        size_t nt_c
        float dt_c
        int nfft
        int nf
        float[::1] filt, rt, ct
        float[::1] f_use, amps_use
        int[::1] intfr

    def __init__(self, trace_iter, *, f=None, amps=None, dt=None, inplace=False):
        self.iter_in = spyc.as_trace_iterator(trace_iter)
        self.hdr = self.iter_in.hdr
        self.inplace = inplace
        self.dt_user = 0.0 if dt is None else dt
        self.nt_c = 0
        self.dt_c = 0.0

        cdef int i
        if f is not None:
            f = np.atleast_1d(np.asarray(f, dtype=np.float32))
            if f.ndim != 1 or f.shape[0] == 0:
                raise ValueError("f must be a 1D array of frequencies.")
            if f.shape[0] < 2:
                warnings.warn(f"Only {f.shape[0]} value defining filter")
            for i in range(f.shape[0] - 1):
                if f[i] < 0.0 or f[i] > f[i + 1]:
                    raise ValueError("Bad filter parameters, f must be increasing and not negative.")
            if f[f.shape[0] - 1] < 0.0:
                raise ValueError("Bad filter parameters, f must be increasing and not negative.")
        self.f_in = f

        if amps is not None:
            amps = np.atleast_1d(np.asarray(amps, dtype=np.float32))
            if amps.ndim != 1:
                raise ValueError("amps must be a 1D array.")
            if (amps < 0.0).any():
                raise ValueError("amp values must be positive")
            if not (amps > 0.0).any():
                raise ValueError("All amps values are zero")
            if (amps == amps[0]).all():
                warnings.warn("All amps values are the same")
            if f is not None and amps.shape[0] != f.shape[0]:
                raise ValueError("number of f values must = number of amps values")
            if f is None and amps.shape[0] != 4:
                raise ValueError("number of f values must = number of amps values (the default f has 4 values)")
        self.amps_in = amps

    cdef _build(self, size_t nt, float dt):
        """Build the filter for traces with nt samples, and sample interval dt"""
        cdef:
            float nyq = 0.5 / dt
            int nfft = su.npfaro(<int> nt, LOOKFAC * <int> nt)

        if nfft >= PFA_MAX:
            raise ValueError(f"Padded nt={nfft} -- too big")

        if self.f_in is None:
            f = np.array([FRAC0 * nyq, FRAC1 * nyq, FRAC2 * nyq, FRAC3 * nyq], dtype=np.float32)
        else:
            f = self.f_in
        if self.amps_in is None:
            # the default is a trapezoidal bandpass filter
            amps = np.ones(f.shape[0], dtype=np.float32)
            amps[0] = 0.0
            amps[f.shape[0] - 1] = 0.0
        else:
            amps = self.amps_in

        self.f_use = f
        self.amps_use = amps
        self.nfft = nfft
        self.nf = nfft // 2 + 1
        self.filt = np.empty(self.nf, dtype=np.float32)
        self.rt = np.empty(nfft, dtype=np.float32)
        self.ct = np.empty(2 * self.nf, dtype=np.float32)  # nf complex numbers
        self.intfr = np.empty(f.shape[0], dtype=np.int32)  # scratch space of polygonalFilter

        su.polygonalFilter(&self.f_use[0], &self.amps_use[0], f.shape[0], nfft, dt, &self.filt[0], &self.intfr[0])
        self.nt_c = nt
        self.dt_c = dt

    cdef spyc.Trace next_trace(self):
        cdef:
            spyc.Trace trace = self.iter_in.next_trace()
            float[::1] data_in = trace.data
            size_t n_sample = data_in.shape[0]
            float[::1] data
            float dt = <float> trace.hdr.d_sample

        if n_sample == 0:
            return spyc.Trace.from_trace(spyc.copy_of_hdr(trace.hdr), data_in, True)

        if dt == 0.0:
            dt = self.dt_user
        if dt == 0.0:
            # as SU does
            dt = 0.004

        if self.nt_c != n_sample or self.dt_c != dt:
            self._build(n_sample, dt)

        if self.inplace:
            data = data_in
        else:
            data = spyc.alloc_data(n_sample)
            data[:] = data_in

        with nogil:
            su.su_filter(
                &data[0], <int> n_sample, self.nfft, &self.filt[0], &self.rt[0], <su.complex *> &self.ct[0],
            )
        return spyc.Trace.from_trace(spyc.copy_of_hdr(trace.hdr), data, True)
