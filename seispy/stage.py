"""
Deferred pipeline stages.

A seispy processing step such as ``butterworth_bandpass`` is a trace iterator that has to be handed its
upstream when it is created. A `Stage` instead records *how* to build a step (its factory and parameters) without an
input yet, which lets us write the equivalent of a unix pipe::

    spike() | bfilt(f_pass_low=10) | bfilt(f_pass_high=60)

and, because a stage is just (factory, parameters), it can be pickled and re-instantiated once per worker
(see `seispy.parallel`).

Every stage also declares how finely its work can be split up (``parallelism``):

``'trace'``
    Each output trace depends only on the matching input trace (filters, gains, mutes, ...). The stream can be cut
    anywhere.
``'ensemble'``
    The stage needs whole gathers (nmo, stack, sort within a gather, ...). The stream can only be cut between gathers.
``'serial'``
    The stage needs the whole stream, or carries state from trace to trace.
"""
import functools

from .container import as_trace_iterator, from_iterable

__all__ = ["Stage", "Pipeline", "stage", "per_trace", "header_value", "PARALLELISM_LEVELS"]

# ordered from least to most restrictive
PARALLELISM_LEVELS = ("trace", "ensemble", "serial")


def _check_level(level):
    if level not in PARALLELISM_LEVELS:
        raise ValueError(f"parallelism must be one of {PARALLELISM_LEVELS}, got {level!r}.")
    return level


class Stage:
    """A processing step with every parameter set except its input."""

    parallelism = "serial"

    def __init__(self, factory, *args, parallelism="serial", name=None, **kwargs):
        self.factory = factory
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.parallelism = _check_level(parallelism)

    def bind(self, upstream):
        """Instantiate this step reading from ``upstream`` (a trace iterator, collection or any iterable of Trace)."""
        return self.factory(upstream, *self.args, **self.kwargs)

    def __ror__(self, upstream):
        # `upstream | stage`
        return self.bind(upstream)

    def __or__(self, other):
        # `stage | stage`
        if isinstance(other, Stage):
            return Pipeline(self, other)
        return NotImplemented

    def __repr__(self):
        params = [repr(a) for a in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()]
        name = self.name or getattr(self.factory, "__name__", repr(self.factory))
        return f"{name}({', '.join(params)})"


class Pipeline(Stage):
    """Several stages applied in order. Its parallelism is that of its most restrictive stage."""

    def __init__(self, *stages):
        flat = []
        for s in stages:
            flat.extend(s.stages if isinstance(s, Pipeline) else [s])
        self.stages = tuple(flat)
        self.parallelism = max((s.parallelism for s in self.stages), key=PARALLELISM_LEVELS.index)

    def bind(self, upstream):
        for s in self.stages:
            upstream = s.bind(upstream)
        return upstream

    def __repr__(self):
        return " | ".join(repr(s) for s in self.stages)


def stage(factory=None, *, parallelism="serial", name=None, validate=False):
    """Turn an iterator class ``factory(upstream, **params)`` into a function returning `Stage` objects.

    Can be used directly, ``bfilt = stage(butterworth_bandpass, parallelism='trace')``, or as a decorator
    (with or without arguments).

    `parallelism` defaults to the safe ``'serial'``, so a stage has to opt in to being split across workers.
    `name` is what the stage is called when printed (e.g. the SU program name), by default the factory's name.
    With `validate`, the parameters are checked as soon as the stage is made (by making the factory's iterator
    around an empty input) instead of when it is first connected to something.
    """
    _check_level(parallelism)

    def decorate(factory):
        @functools.wraps(factory)
        def make(*args, **kwargs):
            if validate:
                factory((), *args, **kwargs)
            return Stage(factory, *args, parallelism=parallelism, name=name, **kwargs)

        if name is not None:
            make.__name__ = name
        return make

    if factory is None:
        return decorate
    return decorate(factory)


def header_value(trace, key):
    """A number from a trace: the header value called `key`, or `key(trace)` if it is a function."""
    if callable(key):
        return float(key(trace))
    return float(trace.header[key])


def per_trace(upstream, func):
    """Apply ``func(trace) -> trace`` to every trace of ``upstream``, for stages that are written in python.

    The number of traces is kept (if it is known).
    """
    source = as_trace_iterator(upstream)
    return from_iterable((func(trace) for trace in source), n_traces=source.n_traces)
