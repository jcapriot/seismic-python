from libc.stdio cimport FILE
from .io cimport spy_off_t
cimport cython
from ._cy_enums cimport EnsembleType, SamplingDomain, SamplingUnit

cdef extern from *:
    """
    #define SPY_UNKNOWN 0
    #define SPY_FLOAT32 0
    #define SPY_COMPLEX64 1
    """
    int SPY_UNKNOWN
    int SPY_FLOAT32      # (data_type) real samples, 1 float each
    int SPY_COMPLEX64    # (data_type) complex samples, 2 floats each (real then imaginary)

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
        int data_type                   # SPY_FLOAT32 or SPY_COMPLEX64. n_sample counts samples (not floats)

    size_t SPY_TRC_HDR_SIZE

cdef spy_trace_header* new_hdr(size_t n_sample=?) nogil
cdef spy_trace_header* copy_of_hdr(spy_trace_header *hdr_in) nogil

# the number of floats that make up one sample
cdef inline size_t floats_per_sample(int data_type) noexcept nogil:
    return 2 if data_type == SPY_COMPLEX64 else 1

# Garbage collector managed buffers (the memory is owned by the returned view's base object).
cdef float[::1] alloc_data(size_t n_sample)
cdef unsigned char[::1] alloc_bytes(size_t n_bytes)

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


# raises a TypeError if the trace is complex (for code that only works on real samples)
cdef int require_real(Trace tr) except -1

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

# Coerce a TraceCollection, trace iterator, or any iterable of Trace into a BaseTraceIterator
cpdef BaseTraceIterator as_trace_iterator(object obj)