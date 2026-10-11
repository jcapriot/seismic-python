"""
Picking (``su/main/picking``): SUFBPICKW (first break picker), SUFNZERO (the time of the first and last non-zero sample) and
SUPICKAMP (amplitudes picked in a window).

The programs print their results and (SUFBPICKW) put them in header words that the traces here do not have, so these are
functions that are given the traces (any iterable of them, such as the end of a pipeline) and return the results::

    picks = picking.fbpickw(source, window=0.03)
    picks.times

Times are in seconds and sample numbers are python's, from 0.
"""
import builtins
from dataclasses import dataclass

import numpy as np

from ..container import as_trace_iterator
from ..stage import header_value
from ..stretching import _resample
from . import _pick

__all__ = ['fbpickw', 'fnzero', 'pickamp', 'FirstBreaks', 'NonZero', 'Amplitudes']

F32 = np.float32


def _nint(x):
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _real(trace, name):
    if trace.dtype.kind == 'c':
        raise TypeError(f"{name} works on real traces, but was given a complex one.")
    return np.asarray(trace, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------------- fbpickw
@dataclass(frozen=True)
class FirstBreaks:
    """The pick of each trace (``times``, seconds; 0 if there is none) and, if asked for, the characteristic function of
    each (``characteristic``: arrays of the length of the traces)"""
    times: np.ndarray
    characteristic: list


def fbpickw(traces, *, window=0.02, offsets=None, times=None, tdv=None, characteristic=False, offset='offset'):
    """First break picks by the ratio of the average amplitude in a window after a sample to that before it (SUFBPICKW).

    Parameters
    ----------
    window : float
        The length (s) of the forward and backward windows.
    offsets, times : arrays, optional
        A template of the first breaks: times of the first break at the offsets, linearly interpolated, held at the ends.
        The pick is the maximum of the characteristic function in a window around the template, of half length ``tdv``.
    tdv : float, optional
        The half length (s) of that window (the whole trace if it is not given).
    characteristic : bool
        Return the characteristic functions too (they are what the program puts in the traces by default, to test the size
        of the window).
    offset : str or function
        The header value of the offset, used with the template.

    Differences from SU: the pick is zero at the edge of the search window of a template only (the program also made zero
    a pick at exactly the middle sample of the trace when there is no template), and the search window is cut at the end
    of the trace (the program searched past it).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if (offsets is None) != (times is None):
        raise ValueError("t and o must be given together")
    if offsets is not None:
        offsets = np.asarray(offsets, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if offsets.shape != times.shape or offsets.ndim != 1 or offsets.shape[0] == 0:
            raise ValueError("t and o has different number of elements")
    picks, functions = [], []
    for trace in as_trace_iterator(traces):
        x = _real(trace, 'fbpickw')
        nt = x.shape[0]
        dt = trace.d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")
        iwindow = _nint(window / dt)
        if iwindow < 1 or 2 * iwindow + 2 > nt:
            raise ValueError("The window is too long for the trace.")
        itdv = nt if tdv is None else _nint(2.0 * tdv / dt)
        if offsets is None:
            ifbt = 0
        else:
            fbt = float(np.interp(header_value(trace, offset), offsets, times))
            ifbt = _nint(fbt / dt)
        absx = np.abs(x.astype(np.float64))
        total = np.concatenate([[0.0], np.cumsum(absx)])
        eps = np.finfo(np.float32).eps
        cf = np.zeros(nt, dtype=np.float32)
        it = np.arange(iwindow, nt - iwindow - 1)
        ampb = total[it] - total[it - iwindow] + eps
        ampf = total[it + iwindow] - total[it] + eps
        cf[it] = ampf / ampb
        # smooth the characteristic function over the samples from iwindow, by a running average with a triangular window
        cf[iwindow:nt - 1] = _pick.smooth_triangle(cf[iwindow:nt - 1], iwindow)
        start = min(max(ifbt - itdv // 2, 0), nt - 1)
        stop = min(start + itdv, nt)
        pick = int(np.argmax(np.abs(cf[start:stop]))) + start
        if offsets is not None and (pick == ifbt - itdv // 2 or pick == ifbt + itdv // 2):
            pick = 0
        picks.append(dt * pick)
        if characteristic:
            functions.append(cf)
    return FirstBreaks(np.array(picks), functions)


# ----------------------------------------------------------------------------------------------------------- fnzero
@dataclass(frozen=True)
class NonZero:
    """The time (s) of the first and of the last sample of each trace that is over the threshold (NaN if there is none)"""
    first: np.ndarray
    last: np.ndarray


def fnzero(traces, *, min=0.0):
    """The time of the first and of the last non-zero sample of each trace (SUFNZERO).

    A sample is zero if its absolute value is not over ``min``. Differences from SU: the times are in seconds (the program's
    were in whole milliseconds), and a trace with no non-zero sample has NaN (the program had -1).
    """
    first, last = [], []
    for trace in as_trace_iterator(traces):
        x = np.abs(_real(trace, 'fnzero'))
        dt = trace.d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")
        nonzero = np.nonzero(x > min)[0]
        if nonzero.shape[0]:
            first.append(nonzero[0] * dt)
            last.append(nonzero[-1] * dt)
        else:
            first.append(np.nan)
            last.append(np.nan)
    return NonZero(np.array(first), np.array(last))


# ---------------------------------------------------------------------------------------------------------- pickamp
@dataclass(frozen=True)
class Amplitudes:
    """What is picked in the window of each trace: ``x`` the lateral position, ``t_above`` and ``t_below`` the window,
    ``max``, ``tmax`` and ``min``, ``tmin`` the extremes and their times, ``abs`` and ``tabs`` the largest absolute value,
    ``energy`` the energy in the window; and the global ones (``global_max`` is ``(value, trace, time)`` and so on)"""
    x: np.ndarray
    t_above: np.ndarray
    t_below: np.ndarray
    max: np.ndarray
    tmax: np.ndarray
    min: np.ndarray
    tmin: np.ndarray
    abs: np.ndarray
    tabs: np.ndarray
    energy: np.ndarray
    global_max: tuple
    global_min: tuple
    global_abs: tuple
    global_energy: tuple


def pickamp(traces, *, d2=None, key=None, x2beg=0.0, x_above=None, t_above=None, x_below=None, t_below=None, wl=0.0,
            dt_resamp=None, tmin=0.0):
    """Pick amplitudes within a window that follows the traces, resampled by 8 point sinc interpolation (SUPICKAMP).

    Parameters
    ----------
    d2, x2beg, key
        The lateral position of the traces: ``x2beg + i * d2``, or the header value ``key`` (a name or a function).
    x_above, t_above : arrays
        The positions and times of the top of the window (linearly interpolated, held at the ends); the top of the trace
        if they are not given.
    x_below, t_below : arrays
        The same for the bottom of the window, or ``wl``, the width of the window (the rest of the trace if 0).
    dt_resamp : float
        The sample interval of the resampling in the window (that of the traces by default).
    tmin : float
        The time of the first sample, if the traces do not have it.

    Differences from SU: the global maximum is found when all of the samples are negative (the program started from the
    smallest positive float), and the times of the window do not include one past its end.
    """
    if key is None and d2 is None:
        raise ValueError("Specify d2 or key")
    if (x_above is None) != (t_above is None):
        raise ValueError("lengths of x_above, t_above must be the same")
    if (x_below is None) != (t_below is None):
        raise ValueError("lengths of x_below, t_below must be the same")
    if x_below is not None and x_above is None:
        raise ValueError("Specify t_xabove")
    xa = None if x_above is None else np.asarray(x_above, dtype=np.float64)
    ta = None if t_above is None else np.asarray(t_above, dtype=np.float64)
    xb = None if x_below is None else np.asarray(x_below, dtype=np.float64)
    tb = None if t_below is None else np.asarray(t_below, dtype=np.float64)
    for a, b in ((xa, ta), (xb, tb)):
        if a is not None and (a.shape != b.shape or a.ndim != 1 or a.shape[0] == 0):
            raise ValueError("lengths of x and t must be the same")
    columns = {k: [] for k in ('x', 't_above', 't_below', 'max', 'tmax', 'min', 'tmin', 'abs', 'tabs', 'energy')}
    for number, trace in enumerate(as_trace_iterator(traces)):
        x = _real(trace, 'pickamp')
        nt = x.shape[0]
        dt = trace.d_sample
        if not dt:
            raise ValueError("dt header field must be set")
        t0 = trace.header['sample_start'] or tmin
        step = float(dt_resamp) if dt_resamp else float(dt)
        position = float(header_value(trace, key)) if key is not None else x2beg + number * d2
        top = t0 if xa is None else float(np.interp(position, xa, ta))
        end = t0 + (nt - 1) * dt
        if xb is not None:
            bottom = float(np.interp(position, xb, tb))
        elif wl:
            bottom = top + wl
        else:
            bottom = end
        top = builtins.max(top, t0)
        bottom = builtins.min(bottom, end)
        count = int(np.floor((bottom - top) / step + 1e-6)) + 1 if bottom >= top else 0
        if count < 1:
            raise ValueError(f"The window of trace {number} is empty.")
        times = top + step * np.arange(count)
        values = _resample.interpolate(np.ascontiguousarray(x), float(dt), float(t0), times.astype(np.float32)).astype(
            np.float64)
        columns['x'].append(position)
        columns['t_above'].append(top)
        columns['t_below'].append(bottom)
        columns['max'].append(values.max())
        columns['tmax'].append(times[values.argmax()])
        columns['min'].append(values.min())
        columns['tmin'].append(times[values.argmin()])
        columns['abs'].append(np.abs(values).max())
        columns['tabs'].append(times[np.abs(values).argmax()])
        columns['energy'].append((values * values).sum() * step)
    if not columns['x']:
        raise ValueError("There are no traces.")
    arrays = {k: np.array(v) for k, v in columns.items()}

    def best(name, time_name=None, largest=True):
        v = arrays[name]
        i = int(v.argmax() if largest else v.argmin())
        return (float(v[i]), i, float(arrays[time_name][i]) if time_name else None)

    return Amplitudes(
        global_max=best('max', 'tmax'), global_min=best('min', 'tmin', largest=False), global_abs=best('abs', 'tabs'),
        global_energy=best('energy'), **arrays)
