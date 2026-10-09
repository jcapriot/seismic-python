"""
Programs that split a data set into files, and put it back together, and the data set for looking at sorting: SUSPLIT,
SUCLEAVE, SUPUTGTHR, SUGETGTHR and SUSORTY (``su/main/windowing_sorting_muting``).

``split``, ``cleave`` and ``putgthr`` write the traces that go through them to files, and give them on unchanged: they are
stages, so that what is written is what the stages after them get, and nothing is written until the traces are taken (the
files are closed when the stage is done, or the traces stop being taken). The files are in seispy's own format
(``seispy.io.tracefile``): read them again with ``getgthr`` (a directory of them), or ``seispy.io.tracefile.read_trace_file``.
The files of the SU programs end in ``.su`` or ``.hsu``; these end in ``.spy``, since they are not SU files.

Where the program does something else than what its documentation says, or that loses traces, this does what the
documentation says (see the notes on each).
"""
import glob
import logging
import os
import warnings

from ..container import as_trace_iterator, from_iterable
from ..io.tracefile import TraceFileWriter, read_trace_file
from ..stage import Stage, header_value

__all__ = ['split', 'cleave', 'putgthr', 'getgthr', 'sorty']

log = logging.getLogger('seispy.windowing')

_SUFFIX = '.spy'


def _directory(directory, create=False):
    directory = os.fspath(directory)
    if create:
        os.makedirs(directory, exist_ok=True)
    elif not os.path.isdir(directory):
        raise ValueError(f"the directory {directory!r} does not exist")
    return directory


def _name(key):
    return key if isinstance(key, str) else 'key'


def _as_int(value):
    # (the value of the key as a C integer: floats lose what is after the point)
    return int(value)


class _Writers:
    """The files of a stage: the first time that one is used it is made new (or added to, with `append`), and then added to"""

    def __init__(self, append):
        self.append = append
        self.touched = set()
        self.open = {}

    def get(self, path):
        writer = self.open.get(path)
        if writer is None:
            writer = TraceFileWriter(path, append=self.append or path in self.touched)
            self.touched.add(path)
            self.open[path] = writer
        return writer

    def close(self, path=None):
        for p in ([path] if path else list(self.open)):
            writer = self.open.pop(p, None)
            if writer is not None:
                writer.close()


# ----------------------------------------------------------------------------------------------------------- split
def _split(upstream, *, key='ensemble_number', directory='.', stem='split_', middle=None, suffix=_SUFFIX, numlength=7,
           append=False, close=True):
    source = as_trace_iterator(upstream)
    directory = _directory(directory, create=True)
    middle = _name(key) if middle is None else middle

    def traces():
        writers = _Writers(append)
        last, path = None, None
        try:
            for trace in source:
                value = _as_int(header_value(trace, key))
                if path is None or value != last:
                    if path is not None and close:
                        writers.close(path)
                    path = os.path.join(directory, f"{stem}{middle}{value:0{numlength}d}{suffix}")
                    log.info("split: output file is %s", path)
                last = value
                writers.get(path).write(trace)
                yield trace
        finally:
            writers.close()

    return from_iterable(traces(), n_traces=source.n_traces)


def split(key='ensemble_number', *, directory='.', stem='split_', middle=None, suffix=_SUFFIX, numlength=7, append=False,
          close=True):
    """Split the traces into files by the value of a header word (SUSPLIT), and give them on unchanged.

    The traces with the same value of ``key`` that come one after the other go to the same file, named
    ``{stem}{middle}{value}{suffix}`` with the value (truncated to an integer) padded with zeros to ``numlength``. The most
    efficient way to use it is to sort the traces by ``key`` first: the traces with a value that comes again later are added to
    its file, which is closed in between if ``close`` (as many files as there are values are open otherwise).

    Parameters
    ----------
    key : header name or function
        What to split on, default ``'ensemble_number'`` (the CDP of SU).
    directory : path
        Where the files go (made if it is not there), default the current directory.
    stem, middle, suffix : str
        The parts of the file names; the middle is the name of the key by default.
    append : bool
        Add to the files that are there when it starts (the program does). Otherwise a file that is there is replaced
        the first time that the stage writes to it.
    """
    kwargs = dict(key=key, directory=directory, stem=stem, middle=middle, suffix=suffix, numlength=numlength, append=append,
                  close=close)
    if numlength < 0:
        raise ValueError("numlength must not be negative")
    return Stage(_split, parallelism='serial', name='split', **kwargs)


# --------------------------------------------------------------------------------------------------------- putgthr
def _putgthr(upstream, *, directory, key='ensemble_number', suffix=_SUFFIX, numlength=7, verbose=False):
    source = as_trace_iterator(upstream)
    directory = _directory(directory, create=True)

    def traces():
        writers = _Writers(append=False)
        last, path = None, None
        try:
            for trace in source:
                value = _as_int(header_value(trace, key))
                if path is None or value != last:
                    if path is not None:
                        writers.close(path)
                    # (a gather whose key comes again later is added to its file: the program replaces the file, losing
                    # the first gather)
                    path = os.path.join(directory, f"{_name(key)}={value:0{numlength}d}{suffix}")
                    if verbose:
                        log.info("putgthr: output file is %s", path)
                last = value
                writers.get(path).write(trace)
                yield trace
        finally:
            writers.close()

    return from_iterable(traces(), n_traces=source.n_traces)


def putgthr(directory, key='ensemble_number', *, suffix=_SUFFIX, numlength=7, verbose=False):
    """Put the gathers of the stream in files of their own in a directory (SUPUTGTHR), and give the traces on unchanged.

    A gather is the traces that have the same ``key`` one after the other. Its file is called ``{key}={value}{suffix}`` with
    the value (truncated to an integer) padded with zeros to ``numlength``, e.g. ``ensemble_number=0000042.spy``, so that
    the directory has a file for each gather, to read back with `getgthr`. The program also puts the number of traces of
    the gather in each trace (the header word ntr, which the traces here do not have: the file has their number).

    Parameters
    ----------
    directory : path
        Where the files go (made if it is not there).
    key : header name or function
        What the traces of a gather have in common, default ``'ensemble_number'``.
    """
    if numlength < 0:
        raise ValueError("numlength must not be negative")
    kwargs = dict(key=key, suffix=suffix, numlength=numlength, verbose=verbose)
    return Stage(_putgthr, directory=directory, parallelism='serial', name='putgthr', **kwargs)


# --------------------------------------------------------------------------------------------------------- getgthr
def getgthr(directory, *, pattern='*'):
    """The traces of the files of a directory, one file after the other in the order of their names (SUGETGTHR).

    This is a source (it starts a pipeline: ``getgthr('gathers') | gain(agc=True)``). Every file that matches ``pattern`` is
    taken to be a file of traces, as written by `putgthr` and `split`. (The program's ``vt`` and ``ns``, for traces of
    different lengths, are not needed: the traces of these files have their own.)
    """
    directory = _directory(directory)
    names = sorted(p for p in glob.glob(os.path.join(directory, pattern)) if os.path.isfile(p))
    if not names:
        raise ValueError(f"there are no files in {directory!r} that match {pattern!r}")

    def traces():
        for name in names:
            yield from read_trace_file(name)

    return from_iterable(traces())


# ---------------------------------------------------------------------------------------------------------- cleave
def _cleave(upstream, *, key='offset', abs=True, size=100.0, low=None, high=100000.0, outbase=None, directory='.',
            suffix=_SUFFIX, verbose=False):
    if size <= 0.0:
        raise ValueError("size must be greater than zero")
    low = size / 2.0 if low is None else low
    if low < 0.0:
        raise ValueError("low must not be negative")
    if low > size / 2.0:
        raise ValueError("low must be less than or equal to size/2")
    if low > high:
        raise ValueError("high must be greater than low")
    outbase = f"cleave_{_name(key)}_" if outbase is None else outbase
    source = as_trace_iterator(upstream)
    directory = _directory(directory, create=True)

    # (the ranges, with the same arithmetic as the program, which is careful about round-offs)
    nfile = int(3.50000000001 + (high - low) / size)
    dlow = low
    if abs:
        dlow -= size
    else:
        nfile -= 1
        dlow -= size * nfile
        nfile = nfile * 2

    def file_name(ifile):
        return os.path.join(directory, f"{outbase}{round(ifile * size + dlow + 0.00000000001)}{suffix}")

    def traces():
        writers = _Writers(append=False)
        counts = {}
        sums = {}
        try:
            for trace in source:
                value = header_value(trace, key)
                if value < 0.0 and abs:
                    value = -value
                ifile = int(0.50000000001 + (value - dlow) / size)
                if ifile > nfile - 1:
                    ifile = nfile - 1
                elif ifile < 1:
                    ifile = 0
                writers.get(file_name(ifile)).write(trace)
                counts[ifile] = counts.get(ifile, 0) + 1
                sums[ifile] = sums.get(ifile, 0.0) + value
                yield trace
        finally:
            writers.close()
            for ifile in sorted(counts):
                if verbose:
                    log.info("cleave: %s: %d traces, key average %f", file_name(ifile), counts[ifile],
                             sums[ifile] / counts[ifile])
            if counts and min(counts) == 0:
                warnings.warn(f"{counts[0]} traces have key < low range. Output in extra file {file_name(0)}")
            if counts and max(counts) == nfile - 1:
                warnings.warn(f"{counts[nfile - 1]} traces have key > high range. Output in extra file "
                              f"{file_name(nfile - 1)}")

    return from_iterable(traces(), n_traces=source.n_traces)


def cleave(key='offset', *, abs=True, size=100.0, low=None, high=100000.0, outbase=None, directory='.', suffix=_SUFFIX,
           verbose=False):
    """Cleave the traces into files by ranges of a header word, keeping their order, and give them on unchanged (SUCLEAVE).

    The ranges are ``size`` wide, and the file of a range is called ``{outbase}{n}{suffix}`` with ``n`` the (integer nearest
    to the) middle of the range, ``low + size * N``. The lower boundary of a range is in it, and the higher one in the next.
    No traces are lost: those below the lowest range go to one extra file, and those above the highest to another, with a
    warning.

    Parameters
    ----------
    key : header name or function
        What to cleave on, default ``'offset'``.
    abs : bool
        Use the absolute value of the key; the ranges are then from ``low`` to ``high``, and otherwise from ``-high`` to ``high``.
    size : float
        The size of the ranges (more than 0).
    low : float, optional
        The middle of the lowest range, at most ``size / 2`` (that is the default).
    high : float
        The middle of the highest range, which is adjusted to ``low + size * N``.
    outbase : str, optional
        The beginning of the file names, ``cleave_{key}_`` by default.
    directory : path
        Where the files go (made if it is not there), default the current directory.
    verbose : bool
        Say (to the logger ``seispy.windowing``) how many traces went to each file and the average of their key.
    """
    kwargs = dict(key=key, abs=abs, size=size, low=low, high=high, outbase=outbase, directory=directory, suffix=suffix,
                  verbose=verbose)
    _cleave((), **kwargs)  # check the parameters now
    return Stage(_cleave, parallelism='serial', name='cleave', **kwargs)


# ----------------------------------------------------------------------------------------------------------- sorty
def sorty(nt=100, *, nshot=10, dshot=10.0, noff=20, doff=20.0, dt=0.004):
    """A small common shot data set for looking at how sorting works (SUSORTY): it shows the geometry in the data.

    The data are ``nshot`` shots ``dshot`` apart, each with ``noff`` offsets ``doff`` apart, the first at ``doff``. A quarter of
    the samples of each trace is the source coordinate, then the receiver coordinate, then the offset, and the midpoint, so that
    an image of the traces shows the order that they are in. ``ensemble_number`` is the midpoint, as the CDP of the program.

    (The program says 20 offsets in its documentation and makes 24; this makes 20. It also numbers the traces
    ``shot * nshot + offset + 1``, which is not their number when there are not as many offsets as shots; they are numbered
    from 1 here, in the order they are made.)
    """
    import numpy as np

    from ..container import Trace

    if nt < 4:
        raise ValueError("nt must be at least 4")

    def traces():
        number = 0
        for ishot in range(nshot):
            sx = ishot * dshot
            for ioff in range(noff):
                offset = (ioff + 1) * doff
                gx = sx + offset
                cmp = (sx + gx) / 2.0
                x = np.zeros(nt, dtype=np.float32)
                x[:nt // 4] = sx
                x[nt // 4:nt // 2] = gx
                x[nt // 2:3 * nt // 4] = offset
                x[3 * nt // 4:] = cmp
                number += 1
                yield Trace(x, d_sample=dt).replace(
                    tx_loc=[sx, 0.0, 0.0], rx_loc=[gx, 0.0, 0.0], ensemble_number=int(cmp), trace_id=number,
                )

    return from_iterable(traces(), n_traces=nshot * noff)
