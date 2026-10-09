"""
Temporary files for the stages that have to read many traces before they can write the first: ``sort``, ``flip``, ... (the
``tmpdir`` of the SU programs).

Such a stage keeps its traces in memory while they fit in a budget (``memory`` bytes), and puts them on disk when they do not.
It says so, loudly, when it does: a message at the WARNING level of the logger ``seispy.spool`` (which python prints to stderr
if logging has not been set up) names the stage and the temporary directory that it is using, and why. The directory is
removed when the stage is done with it, whether it finished, failed or was stopped.

Where the files go, from the first that is set:

1. the ``tmpdir`` given to the stage: a path, or ``True`` for the default directory (``False`` never uses the disk),
2. ``seispy.spool.tmpdir``,
3. the environment variable ``SEISPY_TMPDIR``,
4. ``CWP_TMPDIR`` (the variable that the SU programs look at),
5. the temporary directory of the system.

How much is kept in memory, likewise: the ``memory`` of the stage, ``seispy.spool.memory``, the environment variable
``SEISPY_MEMORY`` (bytes), or 1 GiB. ``memory=0`` puts everything on disk, as the SU programs do.
"""
import gc
import heapq
import logging
import os
import pickle
import shutil
import tempfile

import numpy as np

__all__ = ['tmpdir', 'memory', 'resolve_tmpdir', 'resolve_memory', 'Workspace', 'Panel', 'external_sort']

log = logging.getLogger('seispy.spool')

#: the directory for temporary files, if it is not given to the stage (see the module notes), None for the defaults
tmpdir = None
#: the number of bytes of traces that a stage keeps in memory before using the disk, None for the defaults
memory = None

DEFAULT_MEMORY = 1 << 30
_HEADER_BYTES = 400  # (about what a trace takes besides its samples, for the budget)


def resolve_tmpdir(requested=None):
    """The directory that temporary files go in, or None if the disk is not to be used (``requested=False``)"""
    if requested is False:
        return None
    if requested is not None and requested is not True:
        path = os.fspath(requested)
    else:
        path = tmpdir or os.environ.get('SEISPY_TMPDIR') or os.environ.get('CWP_TMPDIR') or tempfile.gettempdir()
        path = os.fspath(path)
    return path


def check_tmpdir(requested):
    """Raise if the directory is one that can not be written in (a stage calls this when it is made)"""
    path = resolve_tmpdir(requested)
    if path is None:
        return
    if not os.path.isdir(path):
        raise ValueError(f"tmpdir {path!r} is not a directory (or does not exist)")
    if not os.access(path, os.W_OK):
        raise ValueError(f"you can not write in the tmpdir {path!r}")


def resolve_memory(requested=None):
    """The number of bytes of traces to keep in memory"""
    if requested is None:
        requested = memory
    if requested is None and os.environ.get('SEISPY_MEMORY'):
        try:
            requested = int(float(os.environ['SEISPY_MEMORY']))
        except ValueError:
            raise ValueError(f"SEISPY_MEMORY={os.environ['SEISPY_MEMORY']!r} is not a number of bytes") from None
    if requested is None:
        requested = DEFAULT_MEMORY
    if requested < 0:
        raise ValueError("memory must not be negative")
    return int(requested)


def trace_bytes(trace):
    """About how much memory a trace takes"""
    return np.asarray(trace).nbytes + _HEADER_BYTES


class Workspace:
    """A temporary directory for one stage, made when it is first needed and removed by `close` (or leaving ``with``)"""

    def __init__(self, stage, tmpdir=None, reason=''):
        self.stage = stage
        self._requested = tmpdir
        self.reason = reason
        self.path = None
        self._count = 0

    def create(self):
        if self.path is None:
            base = resolve_tmpdir(self._requested)
            if base is None:
                raise MemoryError(f"{self.stage}: the traces do not fit in memory, and tmpdir=False does not let it use the disk")
            self.path = tempfile.mkdtemp(prefix=f'seispy-{self.stage}-', dir=base)
            log.warning("%s: using the temporary directory %s%s", self.stage, self.path,
                        f' ({self.reason})' if self.reason else '')
        return self.path

    def new_file(self, suffix='.tmp'):
        self._count += 1
        return os.path.join(self.create(), f'{self._count:06d}{suffix}')

    def close(self):
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------------------------------------------ sorting
class Descending:
    """A value that sorts the other way round (a key that is sorted from the largest)"""
    __slots__ = ('value',)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        return other.value < self.value

    def __eq__(self, other):
        return self.value == other.value

    def __getstate__(self):
        return self.value

    def __setstate__(self, state):
        self.value = state


def _write_run(path, records):
    with open(path, 'wb') as f:
        for record in records:
            pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)


def _read_run(path):
    with open(path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def external_sort(source, sort_key, *, stage='sort', tmpdir=None, memory=None):
    """The traces of `source` in the order of ``sort_key(trace)`` (a stable sort), using the disk if they do not fit in memory.

    The traces are sorted in chunks that fit in the budget, the chunks go on disk, and the output is a merge of them, so that
    only about one chunk is in memory at a time. Traces with equal keys stay in the order they came in.
    """
    budget = resolve_memory(memory)
    requested = tmpdir
    workspace = Workspace(stage, requested, f"more than {budget} bytes of traces")
    chunk, size, runs, readers = [], 0, [], []

    def flush():
        chunk.sort(key=lambda record: record[0])  # (stable)
        path = workspace.new_file('.run')
        _write_run(path, chunk)
        runs.append(path)
        chunk.clear()

    try:
        for trace in source:
            chunk.append((sort_key(trace), trace))
            size += trace_bytes(trace)
            if size > budget:
                flush()
                size = 0
        if not runs:
            chunk.sort(key=lambda record: record[0])
            for _, trace in chunk:
                yield trace
            return
        chunk.sort(key=lambda record: record[0])  # (the last one stays in memory, and is the last in the merge)
        readers.extend(_read_run(path) for path in runs)
        merged = heapq.merge(*readers, iter(list(chunk)), key=lambda record: record[0])
        chunk.clear()
        for _, trace in merged:
            yield trace
    finally:
        # (the files have to be closed before they can be removed, which windows insists on)
        for reader in readers:
            reader.close()
        workspace.close()


# -------------------------------------------------------------------------------------------------------- panels
class Panel:
    """The traces of a stream as an array (traces by samples) and their headers, for the stages that work on the whole data set.

    The samples are in memory if they fit in the budget, and otherwise in a file that is memory mapped, so that they are read
    as they are used. The headers (traces with no samples) are kept in memory, or in a file. The traces must all have the
    same number of samples and be of the same type. Use it as ``with Panel(...) as panel:``; the file goes at the end.
    """

    def __init__(self, source, *, stage, tmpdir=None, memory=None):
        budget = resolve_memory(memory)
        self.stage = stage
        self._workspace = Workspace(stage, tmpdir, f"more than {budget} bytes of traces")
        self._rows = []
        self._headers = []
        self._offsets = None
        self._header_file = None
        self._data_file = None
        self._data_path = None
        self.n = 0
        self.dtype = None
        self.n_sample = None
        try:
            size = 0
            for trace in source:
                x = np.asarray(trace)
                if self.dtype is None:
                    self.dtype, self.n_sample = x.dtype, x.shape[0]
                elif x.dtype != self.dtype or x.shape[0] != self.n_sample:
                    raise ValueError("The traces must all have the same number of samples (and be of the same type).")
                header_only = trace.replace(np.zeros(0, dtype=self.dtype))
                size += trace_bytes(trace)
                if self._data_file is None and size > budget:
                    self._to_disk()
                if self._data_file is None:
                    self._rows.append(np.array(x))
                    self._headers.append(header_only)
                else:
                    self._append(x, header_only)
                self.n += 1
            self._finish()
        except BaseException:
            self.close()
            raise

    def _to_disk(self):
        self._data_path = self._workspace.new_file('.samples')
        self._data_file = open(self._data_path, 'wb')
        self._header_file = open(self._workspace.new_file('.headers'), 'w+b')
        self._offsets = []
        rows, headers = self._rows, self._headers
        self._rows, self._headers = [], []
        for row, header in zip(rows, headers):
            self._append(row, header)

    def _append(self, row, header):
        self._data_file.write(np.ascontiguousarray(row).tobytes())
        self._offsets.append(self._header_file.tell())
        pickle.dump(header, self._header_file, protocol=pickle.HIGHEST_PROTOCOL)

    def _finish(self):
        if self.n == 0:
            self.array = np.zeros((0, 0), dtype=np.float32)
        elif self._data_file is None:
            self.array = np.stack(self._rows)
            self._rows = []
        else:
            self._data_file.close()
            self._data_file = None
            self.array = np.memmap(self._data_path, dtype=self.dtype, mode='r', shape=(self.n, self.n_sample))

    def header(self, i):
        """The i-th trace, without its samples (to ``replace`` them)"""
        if self._offsets is None:
            return self._headers[i]
        self._header_file.seek(self._offsets[i])
        return pickle.load(self._header_file)

    def __len__(self):
        return self.n

    @property
    def on_disk(self):
        return self._offsets is not None

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
        if self._header_file is not None:
            self._header_file.close()
            self._header_file = None
        array = getattr(self, 'array', None)
        if isinstance(array, np.memmap):
            self.array = np.zeros((0, 0), dtype=np.float32)
            mapping = array._mmap
            del array
            # (windows can not remove a file that is mapped: let go of the mapping, which needs the views of it to be gone)
            for _ in range(3):
                try:
                    mapping.close()
                    break
                except BufferError:
                    gc.collect()
        self._workspace.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
