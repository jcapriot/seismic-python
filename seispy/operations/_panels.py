"""
Operations on panels (several traces at a time, or two data sets): SUMIX, and the binary operations of SUOP2
(``su/main/operations``).

``mix`` is a weighted moving average over the traces of a panel. The binary operations take a second data set, an
iterable of traces (the second file of suop2), that is matched up with the traces of the stream in order, and stop when
either runs out. ``sum2``, ``diff2``, ``prod2`` and ``quo2`` combine the traces of two panels, ``ptsum``, ``ptdiff``,
``ptprod`` and ``ptquo`` combine every trace of the panel with one trace (the first of the second set), and
``zipper`` and ``zippol`` make complex traces from two real ones: from the real and imaginary parts, or from the
amplitude and the phase. (The names of the first four have the 2 so that they are not those of the operations of
suop, ``sum`` and ``diff``, that are in ``seispy.operations``.)

The result has the header of the trace of the stream. With ``w1`` and ``w2`` the traces of the stream and of the second set
are multiplied by those first. Where suop2 does arithmetic on complex numbers by hand (and gets division wrong) the
traces are complex numbers, and numpy does it.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..filters import _kernels
from ..stage import Stage

__all__ = [
    'mix', 'sum2', 'diff2', 'prod2', 'quo2', 'ptsum', 'ptdiff', 'ptprod', 'ptquo', 'zipper', 'zippol',
]

_DEFAULT_MIX = (0.6, 1.0, 1.0, 1.0, 0.6)


# ------------------------------------------------------------------------------------------------------------ mix
def _mix(upstream, *, weights=None, key=None):
    weights = np.asarray(_DEFAULT_MIX if weights is None else weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] == 0:
        raise ValueError("weights must be a 1D array with some values")
    n_mix = weights.shape[0]
    weights = weights / n_mix  # (divided by the number of traces, not by the sum of the weights)
    # (the sample of the trace itself is first, the one before it second, ...)
    weights32 = np.ascontiguousarray(weights, dtype=np.float32)
    source = as_trace_iterator(upstream)

    def mixed(traces):
        history = []  # the samples of this trace, and then of those before it
        for count, trace in enumerate(traces, start=1):
            x = np.asarray(trace)
            if history and x.shape != history[0].shape:
                raise ValueError("The traces must all have the same number of samples to be mixed.")
            history.insert(0, x)
            del history[n_mix:]
            if count >= n_mix:
                if x.dtype.kind == 'c':
                    # (a real weighted sum is linear: the real and imaginary parts are each mixed, by the function of the SU library)
                    rows = np.stack(history).view(np.float32).reshape(n_mix, -1)
                    out = _kernels.mix_across(rows, weights32).view(np.complex64)
                else:
                    out = _kernels.mix_across(np.stack(history).astype(np.float32), weights32)
                yield trace.replace(out)
            else:
                yield trace  # (the first traces, that do not have enough before them)

    def traces():
        if key is None:
            yield from mixed(source)
        else:
            for group in group_by(source, key):
                yield from mixed(group)

    return from_iterable(traces(), n_traces=source.n_traces)


def mix(weights=None, *, key=None):
    """A weighted moving average over traces (SUMIX).

    Parameters
    ----------
    weights : array
        The weights of the average, from the trace itself back through the traces before it, default
        ``[.6, 1, 1, 1, .6]``. Their number is the number of traces that are averaged, and the sum is divided by that
        number (not by the sum of the weights).
    key : header name or function, optional
        Start again when this changes, so that each gather is mixed by itself. Without it the whole stream is one panel.

    The average is of the trace and the ones before it, and it starts at the ``len(weights)``-th trace: the first
    ones come out as they are.
    """
    _mix((), weights=weights, key=key)  # check the parameters now
    return Stage(_mix, parallelism='serial' if key is None else 'ensemble', name='mix', weights=weights, key=key)


# ----------------------------------------------------------------------------------------------- binary operations
def _add(x, y):
    return x + y


def _sub(x, y):
    return x - y


def _mul(x, y):
    return x * y


def _div(x, y):
    # (0, not infinity or not a number, where the divisor is 0)
    zero = y == 0
    return np.where(zero, 0.0, x / np.where(zero, 1.0, y))


def _zip(x, y):
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise TypeError("zipper needs real traces.")
    return x.astype(np.float32) + 1j * y.astype(np.float32)


def _zip_polar(x, y):
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise TypeError("zippol needs real traces.")
    return x * (np.cos(y) + 1j * np.sin(y))


def _scaled(trace, weight):
    x = np.asarray(trace)
    return x if weight is None else x * weight


def _panel_op(upstream, other, *, func, w1=None, w2=None):
    source = as_trace_iterator(upstream)
    second = as_trace_iterator(other)

    def traces():
        for first, last in zip(source, second):
            if first.n_sample != last.n_sample:
                raise ValueError(f"The traces have different numbers of samples ({first.n_sample} and {last.n_sample})")
            yield first.replace(func(_scaled(first, w1), _scaled(last, w2)))

    return from_iterable(traces())


def _trace_op(upstream, other, *, func, w1=None, w2=None):
    source = as_trace_iterator(upstream)
    single = other if hasattr(other, 'header') else next(iter(as_trace_iterator(other)), None)
    if single is None:
        raise ValueError("There is no trace to combine the panel with.")
    y = _scaled(single, w2)

    def traces():
        for first in source:
            if first.n_sample != single.n_sample:
                raise ValueError(f"The traces have different numbers of samples ({first.n_sample} and {single.n_sample})")
            yield first.replace(func(_scaled(first, w1), y))

    return from_iterable(traces())


def _define(name, func, factory, doc):
    def make(other, *, w1=None, w2=None):
        return Stage(factory, other, parallelism='serial', name=name, func=func, w1=w1, w2=w2)

    make.__name__ = name
    make.__qualname__ = name
    make.__doc__ = doc
    return make


_PANEL = "{} (the traces of the stream and of ``other``, which are matched up in order), as SUOP2 op={}."
_SINGLE = "{} (every trace of the stream with the first trace of ``other``), as SUOP2 op={}."

sum2 = _define('sum2', _add, _panel_op, _PANEL.format("Sum of two panels", "sum"))
diff2 = _define('diff2', _sub, _panel_op, _PANEL.format("Difference of two panels, stream minus ``other``", "diff"))
prod2 = _define('prod2', _mul, _panel_op, _PANEL.format("Product of two panels", "prod"))
quo2 = _define('quo2', _div, _panel_op, _PANEL.format("Quotient of two panels, 0 where ``other`` is 0", "quo"))
ptsum = _define('ptsum', _add, _trace_op, _SINGLE.format("Sum of a panel and a trace", "ptsum"))
ptdiff = _define('ptdiff', _sub, _trace_op, _SINGLE.format("Difference of a panel and a trace", "ptdiff"))
ptprod = _define('ptprod', _mul, _trace_op, _SINGLE.format("Product of a panel and a trace", "ptprod"))
ptquo = _define('ptquo', _div, _trace_op, _SINGLE.format("Quotient of a panel and a trace, 0 where it is 0", "ptquo"))
zipper = _define(
    'zipper', _zip, _panel_op,
    "Complex traces from two real panels: the stream is the real part, ``other`` the imaginary part (SUOP2 op=zipper).",
)
zippol = _define(
    'zippol', _zip_polar, _panel_op,
    "Complex traces from two real panels: the stream is the amplitude, ``other`` the phase in radians (SUOP2 op=zippol).",
)
