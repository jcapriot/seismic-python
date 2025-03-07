from libc.stdio cimport FILE
from .io cimport spy_off_t
cimport cython

cdef extern from "spy_trace.h" nogil:

    ctypedef struct spy_trace_header:
        size_t n_sample
        double d_sample
        double sample_start
        double offset
        double tx_loc[3]
        double rx_loc[3]
        double mid_point[3]
        int line_id
        int trace_id
        size_t ensemble_number
        size_t ensemble_trace_number    # trace ID within ensemble
        int sampling_unit               # 0 = s, 1  = meters
        int sampling_domain             # 0 (sample unit domain), 1 = sample_unit fourier domain

    ctypedef struct spy_trace:
        spy_trace_header hdr
        float *data

    int SPY_TRC_HDR_SIZE
    int SPY_TRC_SIZE
    int SPY_SMPLNG_UNIT_SEC
    int SPY_SMPLNG_UNIT_METER
    int SPY_SMPLNG_DOM_UNIT
    int SPY_SMPLNG_DOM_FOURIER

    int SPY_UNKNOWN
    int SPY_TX_GATHER
    int SPY_RX_GATHER
    int SPY_COMMON_MIDPOINT
    int SPY_COMMON_OFFSET

cdef spy_trace* new_trace(size_t n_sample, bint zero_fill=?) noexcept nogil
cdef spy_trace* copy_of(spy_trace *tr_in, bint copy_data=?) noexcept nogil
cdef void del_trace(spy_trace *tp, bint del_data) noexcept nogil

@cython.final
cdef class Trace:
    cdef:
        spy_trace* tr
        bint trace_owner
        bint data_owner
        float[::1] trace_data # For holding a reference if it came from python

    @staticmethod
    cdef Trace from_trace(spy_trace *trace, bint trace_owner=?, bint data_owner=?)

    @staticmethod
    cdef Trace from_file_descriptor(FILE *fd)
    cdef to_file_descriptor(self, FILE * fd)

    @staticmethod
    cdef Trace from_bytes(const unsigned char[::1] bys)
    cpdef unsigned char[::1] as_bytes(self)


cdef class CollectionHeader:
    cdef:
        size_t n_traces
        int ensemble_type
        bint uniform_traces

    @staticmethod
    cdef CollectionHeader from_file_descriptor(FILE *fd)
    cdef to_file_descriptor(self, FILE * fd)

    @staticmethod
    cdef CollectionHeader from_bytes(const unsigned char[::1] bys)
    cpdef unsigned char[::1] as_bytes(self)

cdef class TraceCollection:
    cdef:
        CollectionHeader hdr
        # For file based collection
        object file
        bint file_owner
        FILE *fd
        spy_off_t orig_pos

        # For in memory collection
        list traces

        # For an iterator passthrough
        BaseTraceIterator iterator

    @staticmethod
    cdef TraceCollection from_trace_iterator(BaseTraceIterator iterator)


cdef class BaseTraceIterator:
    cdef:
        size_t i
        CollectionHeader hdr

    cdef Trace next_trace(self)