from libc.stdio cimport FILE
from .io cimport spy_off_t
cimport cython

cdef:
    struct spy_trace_header:
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

    size_t SPY_TRC_HDR_SIZE

    enum SamplingUnit:
        seconds
        meters

    enum SamplingDomain:
        unit
        fourier

    enum EnsembleType:
        unknown
        tx_gather
        rx_gather
        common_midpoint
        common_offset

    spy_trace_header* new_hdr(size_t n_sample=?) nogil
    spy_trace_header* copy_of_hdr(spy_trace_header *hdr_in) nogil

@cython.final
cdef class Trace:
    cdef:
        spy_trace_header* hdr
        bint hdr_owner
        float[::1] data

    @staticmethod
    cdef Trace from_trace(spy_trace_header *hdr, float[::1] data, bint hdr_owner=?)

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