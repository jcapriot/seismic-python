"""
Statistics of a data set, the programs of ``su/main/attributes_parameter_estimation`` that only report on the data: SUMEAN,
SUMAX, SUQUANTILE, SUHISTOGRAM and SUCMP.

The programs print their results (to ``outpar``, or to stdout) and do not make traces, so these are functions that are given the
traces (any iterable of them, such as the end of a pipeline) and return the results instead::

    stats = attributes.max(source | gain(agc=True), mode='abs')
    stats.abs.value, stats.abs.global_value

Trace and sample numbers are python's, counted from 0 (the programs count the traces from 1). The arithmetic is done in double
precision where the programs accumulate in single precision.

Where a program plainly does not do what its documentation says this does what the documentation says: SUMAX remembers the
sample of the maximum of the trace before when the maximum is at sample 0, and counts the first sample of the first trace twice
in the global rms.
"""
import builtins
import itertools
from dataclasses import dataclass

import numpy as np

from ..container import as_trace_iterator

__all__ = ['mean', 'max', 'quantile', 'histogram', 'cmp', 'Mean', 'Extrema', 'Peaks', 'Quantiles']

F32 = np.float32


def _real_samples(trace, name):
    x = np.asarray(trace)
    if x.dtype.kind == 'c':
        raise TypeError(f"{name} works on real traces, but was given a complex one.")
    return x


# ------------------------------------------------------------------------------------------------------------ mean
@dataclass(frozen=True)
class Mean:
    """The mean of each trace (``trace``), and their average (``section``)"""
    trace: np.ndarray
    section: float


def mean(traces, *, power=2.0, abs=False):
    """The L-p mean value of each trace and of the section (SUMEAN).

    Each sample is raised to ``power``, the values are averaged over the trace and the power ``1 / power`` of that is the mean of
    the trace; the mean of the section is the average of the means of the traces. ``power=1`` is the mean amplitude, ``power=2``
    the rms. With ``abs`` the absolute values of the samples are used, otherwise their sign is kept (a negative sample to a
    fractional power is NaN).
    """
    means = []
    for trace in as_trace_iterator(traces):
        x = _real_samples(trace, 'mean').astype(np.float64)
        if abs:
            x = np.abs(x)
        with np.errstate(invalid='ignore'):
            means.append((np.sum(x ** power) / x.shape[0]) ** (1.0 / power))
    if not means:
        raise ValueError("There are no traces.")
    means = np.array(means, dtype=np.float32)
    return Mean(trace=means, section=float(means.astype(np.float64).sum() / means.shape[0]))


# ------------------------------------------------------------------------------------------------------------- max
@dataclass(frozen=True)
class Peaks:
    """One value per trace (``value``) with the sample (``sample``) and the time (``time``, seconds) where it is, and the
    one of the whole data set: ``global_value`` at ``global_trace`` and ``global_sample``"""
    value: np.ndarray
    sample: np.ndarray
    time: np.ndarray
    global_value: float
    global_trace: int
    global_sample: int


@dataclass(frozen=True)
class Extrema:
    """The results of `max`: the `Peaks` of the mode that was asked for (``max`` and ``min``, ``abs`` or ``thd``), or
    ``rms`` (per trace) and ``global_rms``. The others are None."""
    mode: str
    max: Peaks = None
    min: Peaks = None
    abs: Peaks = None
    thd: Peaks = None
    rms: np.ndarray = None
    global_rms: float = None


def _peaks(values, samples, dt, pick):
    values = np.array(values, dtype=np.float32)
    samples = np.array(samples, dtype=np.int64)
    k = int(pick(values))
    return Peaks(value=values, sample=samples, time=samples * float(dt or 0.0), global_value=float(values[k]),
                 global_trace=k, global_sample=int(samples[k]))


def _first_peak_above(x, amp, start):
    """The maximum of the first run of samples above `amp` (from sample `start` on, and not sample 0): value, sample."""
    start = builtins.max(start, 1)
    above = x[start:] > amp
    if not above.any():
        return 0.0, 0
    first = int(np.argmax(above))
    ends = np.flatnonzero(~above[first:])
    stop = first + (int(ends[0]) if ends.size else above.shape[0] - first)
    k = int(np.argmax(x[start + first:start + stop]))
    return float(x[start + first + k]), start + first + k


def max(traces, *, mode='maxmin', threshamp=0.0, threshtime=0.0, dt=None):
    """Trace by trace and global maxima, minima, absolute maxima, rms or threshold maxima (SUMAX).

    Parameters
    ----------
    mode : {'maxmin', 'max', 'min', 'abs', 'rms', 'thd'}
        What to find. ``'thd'`` finds the maximum of the first run of samples that are above ``threshamp``, searching from
        ``threshtime`` (seconds) on (it is for a positive threshold).
    dt : float, optional
        The sample interval (seconds), by default that of the first trace.

    Returns
    -------
    Extrema
        With the `Peaks` of the mode (``result.max`` and ``result.min`` for ``'maxmin'``), or ``rms`` and ``global_rms``.
        When a trace has several samples with its maximum, the first is reported.
    """
    if mode not in ('maxmin', 'max', 'min', 'abs', 'rms', 'thd'):
        raise ValueError(f"{mode!r} is an unknown mode")
    columns = {'max': ([], []), 'min': ([], []), 'abs': ([], []), 'thd': ([], [])}
    rms, sumsq, nsamples, n_traces = [], 0.0, 0, 0
    for trace in as_trace_iterator(traces):
        x = _real_samples(trace, 'max')
        if dt is None:
            dt = trace.d_sample
        n_traces += 1
        if mode in ('maxmin', 'max'):
            k = int(np.argmax(x))
            columns['max'][0].append(x[k])
            columns['max'][1].append(k)
        if mode in ('maxmin', 'min'):
            k = int(np.argmin(x))
            columns['min'][0].append(x[k])
            columns['min'][1].append(k)
        if mode == 'abs':
            k = int(np.argmax(np.abs(x)))
            columns['abs'][0].append(np.abs(x[k]))
            columns['abs'][1].append(k)
        if mode == 'thd':
            if not dt:
                raise ValueError("The trace has no sample interval, and no `dt` was given.")
            value, k = _first_peak_above(x, threshamp, int(threshtime / dt))
            columns['thd'][0].append(value)
            columns['thd'][1].append(k)
        if mode == 'rms':
            sq = float(np.sum(x.astype(np.float64) ** 2))
            rms.append(np.sqrt(sq / x.shape[0]))
            sumsq += sq
            nsamples += x.shape[0]
    if not n_traces:
        raise ValueError("There are no traces.")
    if mode == 'rms':
        return Extrema(mode=mode, rms=np.array(rms, dtype=np.float32), global_rms=float(np.sqrt(sumsq / nsamples)))
    # (the global minimum is the lowest of the minima of the traces, the others are the highest)
    out = {name: _peaks(vals, samples, dt, np.argmin if name == 'min' else np.argmax)
           for name, (vals, samples) in columns.items() if vals}
    return Extrema(mode=mode, **out)


# ------------------------------------------------------------------------------------------------------- quantile
@dataclass(frozen=True)
class Quantiles:
    """The results of `quantile` for a set of samples (the whole data set, or a trace)

    ``percentiles`` has the values at the 1, 5, 25, 50, 75, 95 and 99th percentiles (when quantiles were asked for), ``ranks``
    the values at the ranks that the program shows, the lowest, 5% (of the samples), ... the highest (when ranks were), as
    (rank, value) pairs from 1. ``min`` and ``max`` are the lowest and highest values, with the position of each (a sample of the
    data laid out trace after trace)."""
    percentiles: dict
    ranks: list
    min: float
    min_position: int
    max: float
    max_position: int
    n_samples: int


def _quantiles_of(data, quantiles):
    n = data.shape[0]
    order = np.argsort(data, kind='stable')
    sorted_values = data[order]
    percentiles, ranks = {}, []
    if quantiles:
        for q in (1, 5, 25, 50, 75, 95, 99):
            # round to the qth quantile: q n - 1 (for 0 based) + .5
            percentiles[q] = float(sorted_values[builtins.max(int(0.01 * q * n - 0.5), 0)])
    else:
        picks = [0, n // 20]
        picks.append(n // 2 - picks[-1])
        picks.append(n - 1 - picks[-1])
        picks.extend([n - 1 - n // 20, n - 1])
        ranks = [(r + 1, float(sorted_values[r])) for r in picks]
    return Quantiles(percentiles, ranks, float(sorted_values[0]), int(order[0]), float(sorted_values[-1]),
                     int(order[-1]), n)


def quantile(traces, *, panel=True, quantiles=True):
    """Some quantiles or ranks of the samples of a data set (SUQUANTILE).

    Parameters
    ----------
    panel : bool
        Treat the whole data set as one set of samples (a `Quantiles`), or each trace on its own (a list of them).
    quantiles : bool
        Give the quantiles (the 1, 5, 25, 50, 75, 95 and 99th percentiles), or the ranks instead.
    """
    rows = [_real_samples(trace, 'quantile').astype(np.float32) for trace in as_trace_iterator(traces)]
    if not rows:
        raise ValueError("There are no traces.")
    if panel:
        return _quantiles_of(np.concatenate(rows), quantiles)
    return [_quantiles_of(row, quantiles) for row in rows]


# ------------------------------------------------------------------------------------------------------- histogram
def histogram(traces, *, min, max, bins, trend=0, clip=None, dt=None, datum=None):
    """A histogram of the amplitudes of a data set (SUHISTOGRAM).

    Parameters
    ----------
    min, max, bins
        The range of the amplitudes and the number of bins.
    trend : {0, 1, 2}
        0: a 1D histogram, as ``(left edges of the bins, fraction of the samples in each)``; extreme values are counted in
        the end bins. 1: a 2D histogram (the amplitudes at each time), as an array with a row for every time that has samples
        and the columns time (ms), the amplitudes at the 2.28, 15.87, 50, 84.13 and 97.72 % of the samples (NaN where the
        program does not find one) and the standard deviation. 2: the 2D histogram as a list of traces, one per bin, with the
        fraction of the samples at each time that is in that bin.
    clip : float, optional
        Samples that are not below this are left out (trends 1 and 2).
    dt : float, optional
        The sample interval (ms or feet), by default that of the first trace in milliseconds.
    datum : header name or function, optional
        A shift in time for each trace (the time of the sample is relative to it), trends 1 and 2.
    """
    from ..stage import header_value

    if trend not in (0, 1, 2):
        raise ValueError("trend must be 0, 1 or 2")
    if bins < 1:
        raise ValueError("bins must be at least 1")
    lo = F32(min)
    width = (F32(max) - lo) / F32(bins)
    clip = F32(1e38 if clip is None else clip)
    first = None
    histo = count = total = total_sq = None
    counts = np.zeros(bins, dtype=np.int64)
    n_samples = 0
    for trace in as_trace_iterator(traces):
        x = _real_samples(trace, 'histogram').astype(np.float32)
        if first is None:
            first = trace
            if dt is None:
                dt = trace.d_sample * 1000.0
            if trend:
                histo = np.zeros((x.shape[0], bins), dtype=np.int64)
                count = np.zeros(x.shape[0], dtype=np.float32)
                total = np.zeros(x.shape[0], dtype=np.float32)
                total_sq = np.zeros(x.shape[0], dtype=np.float32)
        with np.errstate(invalid='ignore', over='ignore'):
            j = np.trunc((x - lo) / width)
        if trend == 0:
            j = np.clip(np.nan_to_num(j, nan=0.0), 0, bins - 1).astype(np.int64)
            counts += np.bincount(j, minlength=bins)
            n_samples += x.shape[0]
        else:
            shift = F32(0.0) if datum is None else F32(header_value(trace, datum))
            k = np.trunc((np.arange(x.shape[0], dtype=np.float32) * F32(dt) - shift) / F32(dt))
            ok = (x < clip) & (j >= 0) & (j < bins) & (k >= 0) & (k < histo.shape[0])
            ji, ki, xi = j[ok].astype(np.int64), k[ok].astype(np.int64), x[ok]
            np.add.at(histo, (ki, ji), 1)
            np.add.at(count, ki, 1)
            np.add.at(total_sq, ki, xi * xi)
            np.add.at(total, ki, xi)
    if first is None:
        raise ValueError("There are no traces.")
    if trend == 0:
        edges = lo + np.arange(bins, dtype=np.float32) * width
        return edges, counts.astype(np.float32) / F32(n_samples)
    if trend == 2:
        out = []
        for ibin in range(bins):
            frac = np.where(count > 0, histo[:, ibin] / np.where(count > 0, count, 1), histo[:, ibin])
            out.append(first.replace(frac.astype(np.float32), trace_id=ibin))
        return out
    marks = (2.28, 15.87, 50.00, 84.13, 97.72)
    rows = []
    for i in range(histo.shape[0]):
        if count[i] <= 0:
            continue
        row = [i * first.d_sample * 1000.0] + [np.nan] * 5
        got, running = 0, 0
        for ibin in range(bins):
            running += histo[i, ibin]
            m = F32(100.0 * running) / count[i]
            # (one mark at most per bin: a mark that the same bin also crosses is found by the next bin)
            if got < 5 and m >= marks[got]:
                row[1 + got] = float(lo + ibin * width)
                got += 1
        with np.errstate(divide='ignore', invalid='ignore'):
            sd = np.sqrt((count[i] * total_sq[i] - total[i] * total[i]) / (count[i] * (count[i] - 1)))
        rows.append(row + [float(sd)])
    return np.array(rows, dtype=np.float64).reshape(-1, 7)


# ------------------------------------------------------------------------------------------------------------ cmp
_MISSING = object()


def cmp(a, b, *, limit=1.0e-4):
    """Compare two data sets (SUCMP): None if they are the same, otherwise a message about the first difference.

    The number of traces and of samples have to be equal, the headers equal, and the samples within the fractional difference
    ``limit`` of each other (a sample of ``a`` that is 0 is never different).
    """
    upper, lower = F32(1.0 + limit), F32(1.0 - limit)
    for j, (ta, tb) in enumerate(itertools.zip_longest(as_trace_iterator(a), as_trace_iterator(b), fillvalue=_MISSING)):
        if ta is _MISSING or tb is _MISSING:
            return f"The data sets differ in the number of traces (they differ at trace {j})"
        if ta.header != tb.header:
            return f"The data sets differ in the headers at trace {j}"
        xa, xb = np.asarray(ta), np.asarray(tb)
        if xa.dtype != xb.dtype:
            return f"The data sets differ in the type of the samples at trace {j}"
        if xa.dtype.kind == 'c':
            xa, xb = np.concatenate([xa.real, xa.imag]), np.concatenate([xb.real, xb.imag])
        high, low = xb * upper, xb * lower
        differ = ((xa > 0) & ((xa < low) | (xa > high))) | ((xa < 0) & ((xa > low) | (xa < high)))
        if differ.any():
            i = int(np.argmax(differ))
            return f"The data sets differ at trace {j}, sample {i}: {xa[i]:g} and {xb[i]:g}"
    return None
