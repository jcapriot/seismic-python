# cython: embedsignature=True, language_level=3
"""
The iterator behind every SUOP operation, and the operations that are in the SU sources.

SU's program is one long switch on the operation. Here every operation is a function of its own, and an iterator
just applies the function it is given to every trace: ``func(x, dt, nw)`` changes the 1D float32 array ``x`` (a
zero-copy view of the samples of a trace) in place. Most are plain array arithmetic, in ``_numpy_ops.py``. The three
that are more than that, ``saf``, ``freq`` and ``despike``, call the library functions in the SU sources
(``su_op_saf`` and friends, in ``su/main/operations/suop.c``), and are defined here. The stages are made in the package's ``__init__.py``.
"""
from .. cimport container as spyc
from .. cimport su
import numpy as np


def saf(float[::1] x, float dt, int nw):
    """Local extrema to spikes, and fill to the next spike"""
    cdef int nt = x.shape[0]
    cdef float[::1] tmp = np.empty(nt, dtype=np.float32)
    with nogil:
        su.su_op_saf(&x[0], nt, &tmp[0])


def freq(float[::1] x, float dt, int nw):
    """Local dominant frequency"""
    cdef int nt = x.shape[0]
    cdef float[::1] tmp = np.empty(nt, dtype=np.float32)
    cdef float[::1] tmp1 = np.empty(nt, dtype=np.float32)
    with nogil:
        su.su_op_freq(&x[0], nt, dt, &tmp[0], &tmp1[0])


def despike(float[::1] x, float dt, int nw):
    """Despiking with a median filter of nw samples"""
    cdef int nt = x.shape[0]
    cdef float[::1] tmp = np.empty(nt, dtype=np.float32)
    cdef float[::1] tomed = np.empty(nw, dtype=np.float32)
    with nogil:
        su.su_op_despike(&x[0], nt, nw, &tmp[0], &tomed[0])


cdef class _ops(spyc.BaseTraceIterator):
    """Apply ``func(x, dt, nw)`` to the samples of every trace.

    min_samples : the fewest samples a trace can have (operations that look at neighbouring samples)
    needs_dt : the operation uses the sample interval, so the traces must have one
    needs_window : the trace has to have at least nw samples
    complex_ok : the operation is defined for complex samples. (``func`` then gets the complex samples, and can return
        a new array instead of changing them, if the result is not complex.)
    """
    cdef:
        spyc.BaseTraceIterator iter_in
        object func
        object name
        int nw
        bint inplace
        int min_samples
        bint needs_dt
        bint complex_ok

    def __init__(
        self, trace_iter, func, *, name='op', nw=21, inplace=False,
        min_samples=1, needs_dt=False, needs_window=False, complex_ok=False,
    ):
        self.iter_in = spyc.as_trace_iterator(trace_iter)
        self.hdr = self.iter_in.hdr
        self.func = func
        self.name = name
        nw = int(nw)
        if nw < 1:
            raise ValueError("nw must be at least 1.")
        if nw % 2 == 0:
            nw += 1  # make sure nw is odd
        self.nw = nw
        self.inplace = inplace
        self.min_samples = max(min_samples, nw if needs_window else 1)
        self.needs_dt = needs_dt
        self.complex_ok = complex_ok

    cdef spyc.Trace _next_complex(self, spyc.Trace trace):
        if not self.complex_ok:
            raise TypeError(f"{self.name} is not defined for complex traces.")
        n_sample = trace.n_sample
        dt = trace.hdr.d_sample
        if n_sample == 0:
            return trace.replace()
        if n_sample < self.min_samples:
            raise ValueError(f"{self.name} needs traces with at least {self.min_samples} samples.")
        if self.needs_dt and dt == 0.0:
            raise ValueError(f"{self.name} needs the trace to have a sample interval.")
        x = np.array(trace)
        result = self.func(x, dt, self.nw)
        return trace.replace(x if result is None else result)

    cdef spyc.Trace next_trace(self):
        cdef:
            spyc.Trace trace = self.iter_in.next_trace()
            float[::1] data_in = trace.data
            size_t n_sample = data_in.shape[0]
            float[::1] data
            float dt = <float> trace.hdr.d_sample

        if trace.hdr.data_type == spyc.SPY_COMPLEX64:
            return self._next_complex(trace)
        if n_sample == 0:
            return spyc.Trace.from_trace(spyc.copy_of_hdr(trace.hdr), data_in, True)
        if n_sample < self.min_samples:
            raise ValueError(f"{self.name} needs traces with at least {self.min_samples} samples.")
        if self.needs_dt and dt == 0.0:
            raise ValueError(f"{self.name} needs the trace to have a sample interval.")

        if self.inplace:
            data = data_in
        else:
            data = spyc.alloc_data(n_sample)
            data[:] = data_in

        x = np.asarray(data)
        result = self.func(x, dt, self.nw)
        if result is not None:
            x[:] = result  # (an operation can return its result instead of changing the samples)
        return spyc.Trace.from_trace(spyc.copy_of_hdr(trace.hdr), data, True)
