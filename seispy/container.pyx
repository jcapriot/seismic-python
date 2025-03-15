# cython: embedsignature=True, language_level=3
# cython: linetrace=True
import io

from libc.stdio cimport FILE, fwrite, fread, SEEK_CUR
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
cimport cython
cimport cpython.buffer as pybuf

from .io cimport PyFile_Dup, PyFile_DupClose, spy_off_t, spy_fseek

import os
from contextlib import nullcontext
import numpy as np

cdef size_t SPY_TRC_HDR_SIZE = sizeof(spy_trace_header)

cdef spy_trace_header* new_hdr(size_t n_sample=0) nogil:
    cdef spy_trace_header *hdr = <spy_trace_header *> malloc(SPY_TRC_HDR_SIZE)
    if hdr is NULL:
        with gil:
            raise MemoryError("Unable to allocate a new seispy trace header.")
    memset(hdr, SPY_UNKNOWN, SPY_TRC_HDR_SIZE)
    hdr.n_sample = n_sample
    return hdr

cdef spy_trace_header* copy_of_hdr(spy_trace_header *hdr_in) nogil:
    cdef spy_trace_header *hdr = <spy_trace_header *> malloc(SPY_TRC_HDR_SIZE)
    if hdr is NULL:
        with gil:
            raise MemoryError("Unable to allocate a copy of a seispy trace header.")
    memcpy(hdr, hdr_in, SPY_TRC_HDR_SIZE)
    return hdr

_SAMPLING_MAP = {'s':SamplingUnit.seconds, 'm':SamplingUnit.meters}
_DOMAIN_MAP = {'unit':SamplingDomain.unit, 'fourier':SamplingDomain.fourier}

@cython.final
cdef class Trace:

    def __cinit__(self):
        self.hdr = NULL
        self.hdr_owner = False

    def __dealloc__(self):

        # De-allocate if not null and flag is set
        if self.hdr is not NULL and self.hdr_owner:
            free(self.hdr)

    def __init__(
        self,
        data,
        double d_sample,
        double sample_start=0.0,
        tx_loc=None,
        rx_loc=None,
        sampling_unit='s',
        sampling_domain='unit',

    ):
        self.trace_data = np.require(data, dtype=np.float32, requirements='C')
        cdef spy_trace_header *hdr = new_hdr()
        if hdr is NULL:
            raise MemoryError("Unable to allocate trace.")
        hdr.n_sample = self.trace_data.shape[0]
        hdr.d_sample = d_sample
        hdr.sample_start = sample_start

        if tx_loc is not None:
            hdr.tx_loc = tx_loc

        if rx_loc is not None:
            hdr.rx_loc = rx_loc

        try:
            hdr.sampling_unit = _SAMPLING_MAP[sampling_unit]
        except KeyError:
            raise KeyError("Trace expected `sampling_unit` to be one of 's' (seconds) or 'm' (meters).")

        try:
            hdr.sampling_domain = _DOMAIN_MAP[sampling_domain]
        except KeyError:
            raise KeyError("Trace expected `sampling_domain` to be one of 'unit' (seconds) or 'fourier' (meters).")

        self.hdr = hdr
        self.hdr_owner = True

    def __len__(self):
        return self.n_sample

    @property
    def n_sample(self):
        return self.data.shape[0]

    @property
    def d_sample(self):
        return self.hdr.d_sample

    @property
    def header(self):
        return self.hdr[0]

    @staticmethod
    cdef Trace from_trace(spy_trace_header *hdr, float[::1] data, bint hdr_owner=False):
        cdef Trace tr = Trace.__new__(Trace)
        tr.hdr = hdr
        tr.hdr.n_sample = data.shape[0]
        tr.data = data
        tr.hdr_owner = hdr_owner
        return tr

    @staticmethod
    cdef Trace from_file_descriptor(FILE *fd):

        cdef:
            spy_trace_header *hdr = new_hdr()
            size_t n_read
            size_t n_sample

        n_read = fread(hdr, SPY_TRC_HDR_SIZE, 1, fd)
        if n_read != 1:
            free(hdr)
            raise IOError("Unable to read trace header from file.")

        n_sample = hdr.n_sample
        cdef float[::1] data = <float[:n_sample]> malloc(sizeof(float)*n_sample)

        n_read = fread(&data[0], sizeof(float), n_sample, fd)
        if n_read != n_sample:
            raise IOError("Unable to read expected number of trace samples.")

        return Trace.from_trace(hdr, data, True)

    cdef to_file_descriptor(self, FILE *fd):
        cdef size_t n_write = fwrite(self.hdr, SPY_TRC_HDR_SIZE, 1, fd)
        if n_write != 1:
            raise IOError("Error writing trace header to file.")

        n_write = fwrite(&self.data[0], sizeof(float), self.hdr.n_sample, fd)
        if n_write != self.hdr.n_sample:
            raise IOError("Error writing trace data to file.")

    @staticmethod
    cdef Trace from_bytes(const unsigned char[::1] bys):

        if bys.shape[0] < SPY_TRC_HDR_SIZE:
            raise ValueError(f"Incorrect number of bytes, expected at least {SPY_TRC_HDR_SIZE}, got {bys.shape[0]}.")

        cdef:
            spy_trace_header *hdr = new_hdr()
            size_t n_sample
        # copy header bytes
        memcpy(hdr, &bys[0], SPY_TRC_HDR_SIZE)

        n_sample = hdr.n_sample

        cdef size_t n_total_size = SPY_TRC_HDR_SIZE + sizeof(float)*n_sample
        if bys.shape[0] != n_total_size:
            free(hdr)
            raise ValueError(f"Incorrect number of bytes, expected {n_total_size}, got {bys.shape[0]}.")

        cdef float[::1] data = <float[:n_sample]> malloc(sizeof(float)*n_sample)
        # copy data bytes
        memcpy(&data[0], &bys[SPY_TRC_HDR_SIZE], sizeof(float)*n_sample)

        return Trace.from_trace(hdr, data, True)

    cpdef unsigned char[::1] as_bytes(self):
        cdef size_t data_size = sizeof(float) * self.data.shape[0]
        cdef size_t trace_size = SPY_TRC_HDR_SIZE + data_size
        cdef unsigned char[::1] bys = <unsigned char[:trace_size]> malloc(trace_size)
        memcpy(&bys[0], self.hdr, SPY_TRC_HDR_SIZE)
        memcpy(&bys[SPY_TRC_HDR_SIZE], &self.data[0], data_size)
        return bys

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.obj = self
        buffer.buf = <void *> &self.data[0]
        buffer.len = self.data.shape[0]
        buffer.itemsize = sizeof(float)
        buffer.ndim = 1

        buffer.shape = self.data.shape
        buffer.strides = self.data.strides
        buffer.readonly = 0

        if flags & pybuf.PyBUF_FORMAT:
            buffer.format = 'f'
        else:
            buffer.format = NULL

        buffer.internal = NULL
        buffer.suboffsets = NULL #self.data.suboffsets

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

cdef class CollectionHeader:

    def __cinit__(self):
        self.n_traces = 0
        self.ensemble_type = 0
        self.uniform_traces = False

    @classmethod
    def get_cstruct_byte_size(cls):
        return sizeof(size_t) + sizeof(int) + sizeof(bint)

    @staticmethod
    cdef CollectionHeader from_file_descriptor(FILE *fd):
        cdef CollectionHeader hdr = CollectionHeader.__new__(CollectionHeader)

        cdef size_t n_read = fread(&hdr.n_traces, sizeof(hdr.n_traces), 1, fd)
        if n_read != 1:
            raise IOError("Error reading n_traces from file.")
        n_read = fread(&hdr.ensemble_type, sizeof(hdr.ensemble_type), 1, fd)
        if n_read != 1:
            raise IOError("Error reading ensemble_type from file.")
        n_read = fread(&hdr.uniform_traces, sizeof(hdr.uniform_traces), 1, fd)
        if n_read != 1:
            raise IOError("Error reading uniform_traces from file.")
        return hdr

    cdef to_file_descriptor(self, FILE * fd):

        cdef size_t n_write = fwrite(&self.n_traces, sizeof(self.n_traces), 1, fd)
        if n_write != 1:
            raise IOError("Error writing n_traces to file.")

        fwrite(&self.ensemble_type, sizeof(self.ensemble_type), 1, fd)
        if n_write != 1:
            raise IOError("Error writing ensemble_type to file.")

        fwrite(&self.uniform_traces, sizeof(self.uniform_traces), 1, fd)
        if n_write != 1:
            raise IOError("Error writing uniform_traces to file.")

    @staticmethod
    cdef CollectionHeader from_bytes(const unsigned char[::1] bys):
        cdef CollectionHeader hdr = CollectionHeader.__new__(CollectionHeader)
        cdef size_t offset = 0
        cdef size_t my_size = CollectionHeader.get_cstruct_byte_size()
        if bys.shape[0] != my_size:
            raise ValueError(f"Incorrect number of bytes, expected {my_size}, got {bys.shape[0]}.")

        # memcpy(&hdr.n_traces, &bys[0], my_size)
        memcpy(&hdr.n_traces, &bys[0], sizeof(hdr.n_traces))
        offset += sizeof(hdr.n_traces)

        memcpy(&hdr.ensemble_type, &bys[offset], sizeof(hdr.ensemble_type))
        offset += sizeof(hdr.ensemble_type)

        memcpy(&hdr.uniform_traces, &bys[offset], sizeof(hdr.uniform_traces))
        offset += sizeof(hdr.uniform_traces)
        return hdr

    cpdef unsigned char[::1] as_bytes(self):
        cdef:
            size_t n_bytes = type(self).get_cstruct_byte_size()
            unsigned char[::1] bys = <unsigned char[:n_bytes]> malloc(n_bytes)
            size_t offset = 0

        #memcpy(&bys[0], &self.n_traces, n_bytes)

        memcpy(&bys[0], &self.n_traces, sizeof(self.n_traces))
        offset += sizeof(self.n_traces)

        memcpy(&bys[offset], &self.ensemble_type, sizeof(self.ensemble_type))
        offset += sizeof(self.ensemble_type)

        memcpy(&bys[offset], &self.uniform_traces, sizeof(self.uniform_traces))
        offset += sizeof(self.uniform_traces)

        return bys

cdef class TraceCollection:

    def __cinit__(self):
        self.file = None
        self.traces = None
        self.iterator = None
        self.hdr = CollectionHeader()

    def __init__(self, trace_data, d_sample, **kwargs):
        n_tr = len(trace_data)
        traces = []
        for trace in trace_data:
            traces.append(Trace(trace, d_sample=d_sample, **kwargs))
        self.traces = traces
        self.hdr.n_traces = len(traces)

    def __len__(self):
        return self.hdr.n_traces

    @property
    def n_traces(self):
        return self.hdr.n_traces

    @classmethod
    def from_file(cls, filename):
        cdef:
            CollectionHeader hdr

        if hasattr(filename, 'read'):
            ctx = nullcontext(filename)
        else:
            ctx = open(os.fspath(filename), "rb")

        with ctx as f:
            bts = f.read(n=CollectionHeader.get_cstruct_byte_size())
            hdr = CollectionHeader.from_bytes(bts)

        cdef TraceCollection new_segy = TraceCollection.__new__(TraceCollection)

        new_segy.file = filename
        new_segy.hdr = hdr

        return new_segy

    @staticmethod
    cdef TraceCollection from_trace_iterator(BaseTraceIterator iterator):
        cdef TraceCollection collect = TraceCollection.__new__(TraceCollection)
        collect.iterator = iterator
        collect.hdr = iterator.hdr
        return collect

    @property
    def on_disk(self):
        return self.file is not None

    @property
    def is_iterator(self):
        return self.iterator is not None

    @property
    def in_memory(self):
        return self.traces is not None

    def __iter__(self):
        if self.on_disk:
            return _FileTraceIterator(self.file, self.hdr)
        elif self.in_memory:
            return _MemoryTraceIterator(self.traces, self.hdr)
        elif self.is_iterator:
            return self.iterator
        else:
            raise TypeError('TraceCollection file is not on disk, in memory, nor from an iterator.')

    def to_memory(self):
        if self.in_memory:
            return self
        else:
            self.traces = [trace for trace in self]
            self.iterator = None
            self.file = None
            return self

    def to_file(self, filename):
        if self.on_disk:
            return self
        cdef:
            Trace trace
            FILE *fd
            bint file_owner
            spy_off_t orig_pos = 0


        if hasattr(filename, 'write'):
            ctx = nullcontext(filename)
            try:
                filename = filename.name
            except AttributeError:
                filename = None
            file_owner = False
        else:
            ctx = open(os.fspath(filename), "wb")
            file_owner = True

        with ctx as file:
            fd, orig_pos = PyFile_Dup(file, "wb")
            try:
                self.hdr.to_file_descriptor(fd)
                for trace in self:
                    trace.to_file_descriptor(fd)
            finally:
                PyFile_DupClose(file, fd, orig_pos)
            file.flush()
            if file_owner:
                os.fsync(file.fileno())

        self.file = filename
        self.iterator = None
        self.traces = None
        return self

    def to_stream(self, stream):

        if hasattr(stream, 'write'):
            ctx = nullcontext(stream)
        else:
            ctx = open(stream, 'wb')

        cdef:
            Trace trace

        with ctx as stream_ctx:
            stream_ctx.write(self.hdr.as_bytes())
            for trace in self:
                stream_ctx.write(trace.as_bytes())
            stream_ctx.flush()
        if self.is_iterator:
            # the above will consume the iterator if it came from one.
            self.iterator = None
            # otherwise, don't do anything to the underlying object


cdef class BaseTraceIterator:
    def __cinit__(self):
        self.i = 0
        self.hdr = CollectionHeader()

    cdef Trace next_trace(self):
        raise NotImplementedError(f"cdef next_trace is not implemented on {type(self)}.")

    def __next__(self):
        return self.next_trace()

    def __iter__(self):
        return self

    def to_memory(self):
        return TraceCollection.from_trace_iterator(self).to_memory()

    def to_file(self, filename):
        return TraceCollection.from_trace_iterator(self).to_file(filename)

    def to_stream(self, stream):
        TraceCollection.from_trace_iterator(self).to_stream(stream)


cdef class _MemoryTraceIterator(BaseTraceIterator):
    cdef:
        list traces

    def __init__(self, list traces, CollectionHeader hdr):
        self.traces = traces
        self.hdr = hdr

    cdef Trace next_trace(self):
        if self.i == self.hdr.n_traces:
            raise StopIteration()
        cdef Trace out = self.traces[self.i]
        self.i += 1
        return out


cdef class _FileTraceIterator(BaseTraceIterator):
    cdef:
        FILE *fd
        bint owner
        object file
        spy_off_t orig_pos

    def __cinit__(self):
        self.fd = NULL
        self.owner = False
        self.file = None
        self.hdr.n_traces = 0

    def __dealoc__(self):
        # make sure I get closed up when I'm garbage collected
        self._close_file()

    cdef _close_file(self):
        # first close my duped file
        if self.fd is not NULL and self.file is not None:
            PyFile_DupClose(self.file, self.fd, self.orig_pos)
            self.fd = NULL
        # If I own the original, close it
        if self.owner and self.file is not None:
            self.file.close()
        # and clear my reference to the original
        self.file = None

    def __init__(self, file, CollectionHeader hdr):
        if not hasattr(file, "read"):
            # open the file
            file = open(os.fspath(file), "rb")
            self.owner = True
        else:
            self.owner = False
        self.file = file
        self.hdr = hdr

        try:
            self.fd, self.orig_pos = PyFile_Dup(file, "rb")
            if self.owner:
                # Advance fd to the start of the traces:
                spy_fseek(self.fd, type(hdr).get_cstruct_byte_size(), SEEK_CUR)
        except Exception as err:
            self._close_file()
            raise err

    cdef Trace next_trace(self):
        if self.i == self.hdr.n_traces:
            raise StopIteration()
        cdef Trace out
        try:
            out = Trace.from_file_descriptor(self.fd)
        except Exception as err:
            # if something goes wrong reading in from the file descriptor
            # close myself and re-raise the error.
            self._close_file()
            raise err
        self.i += 1
        if self.i == self.hdr.n_traces:
            # The next request will raise a StopIteration so close myself now.
            self._close_file()
        return out

def _isfileobject(f):
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False
    try:
        f.fileno()
        return True
    except OSError:
        return False