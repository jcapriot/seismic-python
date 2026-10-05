"""
Running seispy pipelines in parallel, using only the standard library.

Unix gets two kinds of parallelism "for free" from ``a | b | c``:

* *pipeline* parallelism: every program in the pipe is its own process, so ``a`` can be working on trace n+1 while
  ``b`` works on trace n. A pipe has a small buffer, which makes a fast producer wait for a slow consumer.
* *data* parallelism (``xargs -P`` / GNU parallel): the same program is run on different pieces of the data.

`prefetch` provides the first, and `pmap` the second. Both are stages, used with ``|`` like any other. Both pull from their source in the calling thread (or
a single helper thread), never concurrently, and both keep only a bounded number of traces in flight, so memory use
stays flat for arbitrarily long streams.

Threads only speed things up when the work releases the GIL (the SU filter kernels do; see the notes in the stages),
or on a free-threaded Python build. ``executor='process'`` works for anything else.
"""
import collections
import concurrent.futures
import itertools
import os
import queue
import threading

from .container import as_trace_iterator, from_iterable
from .stage import Stage

__all__ = ["prefetch", "pmap", "group_by"]

_DONE = object()


class _Raised:
    def __init__(self, exc):
        self.exc = exc


def _key_function(key):
    if key is None or callable(key):
        return key
    # the name of a field in the trace header
    return lambda trace: trace.header[key]


def group_by(source, key):
    """Yield lists of consecutive traces that share the same ``key`` (a header field name or a function of a Trace).

    This is how a stream of traces is cut into gathers/ensembles: SU marks them by a header word changing value,
    e.g. ``group_by(traces, 'ensemble_number')``.
    """
    key = _key_function(key)
    for _, grp in itertools.groupby(as_trace_iterator(source), key=key):
        yield list(grp)


def _batches(source, n, key=None):
    """Lists of at least ``n`` traces. With a key, they are only cut between ensembles."""
    if n < 1:
        raise ValueError("chunk must be at least 1.")
    if key is None:
        while True:
            batch = list(itertools.islice(source, n))
            if not batch:
                return
            yield batch
    else:
        batch = []
        for grp in group_by(source, key):
            batch.extend(grp)
            if len(batch) >= n:
                yield batch
                batch = []
        if batch:
            yield batch


def _threaded(source, buffer=4, batch=16):
    src = as_trace_iterator(source)

    def gen():
        q = queue.Queue(maxsize=buffer)
        stop = threading.Event()

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        def produce():
            try:
                for b in _batches(src, batch):
                    if not put(b):
                        return
                put(_DONE)
            except BaseException as err:  # hand any failure to the consumer
                put(_Raised(err))

        thread = threading.Thread(target=produce, name="seispy-threaded", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _DONE:
                    return
                if isinstance(item, _Raised):
                    raise item.exc
                yield from item
        finally:
            stop.set()
            thread.join()

    return from_iterable(gen(), n_traces=src.n_traces)


def prefetch(buffer=4, batch=16):
    """A stage that produces everything upstream of it in a helper thread.

    Placing it between stages lets them run concurrently, like programs in a unix pipe::

        spike() | bfilt(...) | prefetch() | bfilt(...)

    The helper thread works up to ``buffer`` batches of ``batch`` traces ahead of the consumer. The bounded buffer is
    the pipe: a fast producer blocks once it is that far ahead. Errors raised while producing are re-raised in the
    consumer, and closing/abandoning the result stops the thread.
    """
    if buffer < 1 or batch < 1:
        raise ValueError("buffer and batch must be at least 1.")
    return Stage(_threaded, buffer=buffer, batch=batch, parallelism="serial")


def _apply(stage_, traces):
    """Run a stage over a list of traces, returning a list (runs in a worker)."""
    out = stage_.bind(traces) if isinstance(stage_, Stage) else stage_(traces)
    return list(out)


def pmap(stage_, workers=None, chunk=16, *, key=None, executor="thread", prefetch_chunks=None):
    """A stage that applies ``stage_`` with several workers, preserving trace order::

        source | pmap(bfilt(f_pass_low=10) | bfilt(zerophase=False), workers=8)

    Parameters
    ----------
    stage_ : Stage
        e.g. ``bfilt(f_pass_low=10)`` or a ``|`` pipeline of stages. A fresh copy of the stage is built for every
        chunk, so nothing is shared between workers. The upstream is only read from the calling thread.
    workers : int, optional
        Number of workers, defaults to the number of CPUs.
    chunk : int
        Minimum number of traces handed to a worker at a time.
    key : str or callable, optional
        Header field name, or function of a Trace, that is constant within an ensemble. Required for stages with
        ``parallelism == 'ensemble'``, so that chunks are only cut between ensembles.
    executor : {'thread', 'process'}
        Threads are cheap and share memory, but only run concurrently if the stage releases the GIL (or Python is free
        threaded). Processes need no cooperation from the stage but pay to pickle the traces.
    prefetch_chunks : int, optional
        Maximum number of chunks in flight (default ``2 * workers``). This bounds memory use and how far ahead of the
        consumer the source is read.

    Notes
    -----
    This deliberately does not use ``Executor.map``: that consumes the whole input immediately, which defeats the
    lazy, constant memory behavior of a pipeline.
    """
    level = getattr(stage_, "parallelism", "serial")
    if level == "serial":
        raise ValueError(
            f"{stage_!r} is not marked as parallelizable (parallelism='serial'); use it directly in the pipeline."
        )
    if level == "ensemble" and key is None:
        raise ValueError("This stage works on whole ensembles, give `key=` so that chunks are cut between ensembles.")
    if level != "ensemble":
        key = None
    if executor not in ("thread", "process"):
        raise ValueError("executor must be 'thread' or 'process'.")
    if workers is None:
        workers = os.cpu_count() or 1
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if prefetch_chunks is not None and prefetch_chunks < 1:
        raise ValueError("prefetch_chunks must be at least 1.")
    if chunk < 1:
        raise ValueError("chunk must be at least 1.")

    # trace-wise stages keep the number of traces, anything else could change it
    return Stage(
        _pmap, stage_, workers=workers, chunk=chunk, key=key, executor=executor,
        prefetch_chunks=prefetch_chunks, keeps_count=level == "trace", parallelism="serial",
    )


def _pmap(source, stage_, workers, chunk, key, executor, prefetch_chunks, keeps_count):
    max_in_flight = prefetch_chunks if prefetch_chunks is not None else 2 * workers
    src = as_trace_iterator(source)

    def gen():
        if executor == "thread":
            pool = concurrent.futures.ThreadPoolExecutor(workers)
        else:
            pool = concurrent.futures.ProcessPoolExecutor(workers)
        pending = collections.deque()
        try:
            for batch in _batches(src, chunk, key):
                pending.append(pool.submit(_apply, stage_, batch))
                if len(pending) >= max_in_flight:
                    yield from pending.popleft().result()
            while pending:
                yield from pending.popleft().result()
        finally:
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=True, cancel_futures=True)

    return from_iterable(gen(), n_traces=src.n_traces if keeps_count else None)
