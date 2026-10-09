"""
Files of traces in seispy's own format (the one of ``TraceCollection.to_file``): a header with the number of traces, and then
the traces one after the other. `TraceFileWriter` writes such a file a trace at a time, which can not be done with
``to_file`` (the number of traces is at the start), and can add to a file that is already there.

The format is the raw layout of the C structures of this build, so files are only for the machines and builds that made them.
"""
import os
import struct

from ..container import TraceCollection, collection_header_bytes

__all__ = ['TraceFileWriter', 'read_trace_file']

_HEADER_SIZE = len(collection_header_bytes(0))
_COUNT = struct.Struct('N')  # (the first member of the header is a size_t)


class TraceFileWriter:
    """Writes the traces that it is given to a file, and puts their number in the header of the file when it closes.

    With ``append`` the traces are added to the end of the file, if there is one.
    """

    def __init__(self, path, *, append=False):
        self.path = os.fspath(path)
        self.count = 0
        if append and os.path.exists(self.path) and os.path.getsize(self.path) >= _HEADER_SIZE:
            self._file = open(self.path, 'r+b')
            self.count = _COUNT.unpack(self._file.read(_COUNT.size))[0]
            self._file.seek(0, os.SEEK_END)
        else:
            self._file = open(self.path, 'wb')
            self._file.write(collection_header_bytes(0))

    def write(self, trace):
        self._file.write(trace.as_bytes())
        self.count += 1

    def close(self):
        if self._file is not None:
            self._file.seek(0)
            self._file.write(collection_header_bytes(self.count))
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_trace_file(path):
    """The traces of a file written by `TraceFileWriter` (or ``TraceCollection.to_file``), as a collection"""
    return TraceCollection.from_file(os.fspath(path))
