"""
Windowing and muting programs: SUMUTE, SUWIND and SUKILL (``su/main/windowing_sorting_muting``).

SU picks traces and positions on traces by header words (``key=offset``). Here the key is the name of one of the
values in ``trace.header`` (see `seispy.container.Trace`), or a function of a trace that gives the number. SU's default
keys, ``tracl`` and ``trid``, do not exist here: the number of the trace in its line is ``trace_id``.

Where a program plainly does not do what its documentation says (see the notes on each) this does what the
documentation says. The headers ``muts`` and ``mute`` that sumute sets, and sumute's ``hmute``, ``xfile``, ``tfile``
and ``twfile`` parameters, are not supported. (The tapers of the modes that mute a zone start at the first muted
sample on the early side, and the first unmuted one on the late side, as in sumute.)
"""
import math

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, header_value as _key_value, per_trace

__all__ = ['mute', 'wind', 'kill']

_abs, _min, _max = abs, min, max


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _half(n):
    # (integer division in C truncates towards 0)
    return int(n / 2)


# ----------------------------------------------------------------------------------------------------------- mute
def _mute_above(x, t, tmin, dt, taper, **_):
    """mode 0: mute above the curve"""
    nt = x.shape[0]
    nmute = _min(_nint((t - tmin) / dt), nt)
    if nmute > 0:
        x[:nmute] = 0
    for i in range(len(taper)):
        j = i + nmute
        if 0 < j < nt:
            x[j] *= taper[i]


def _mute_below(x, t, tmin, dt, taper, **_):
    """mode 1: mute below the curve"""
    nt = x.shape[0]
    nmute = _max(0, _nint((tmin + nt * dt - t) / dt))
    nzero = _min(nmute, nt)
    if nzero > 0:
        x[nt - nzero:] = 0
    for i in range(len(taper)):
        if nt > nmute + i and nmute + i > 0:
            x[nt - nmute - 1 - i] *= taper[i]


def _mute_zone(x, ntair, nmute, taper):
    """Mute a zone of nmute samples around sample ntair, tapering out of it on both sides"""
    nt = x.shape[0]
    half = _half(nmute)
    top = _min(_max(0, ntair - half), nt)
    bottom = _min(nt, ntair + half)
    if bottom > top:
        x[top:bottom] = 0
    for i in range(len(taper)):
        j = ntair - half - i
        if 0 < j < nt:
            x[j] *= taper[i]
    for i in range(len(taper)):
        j = ntair + half + i
        if 0 <= j < nt:
            x[j] *= taper[i]


def _mute_line(x, t, tmin, dt, taper, fval, linvel, tm0, **_):
    """mode 2: mute below and above a straight line (an air wave), t is the width of the zone"""
    nmute = _nint((tmin + t) / dt)
    ntair = _nint(tm0 / dt + fval / linvel / dt)
    _mute_zone(x, ntair, nmute, taper)


def _mute_hyperbola(x, t, tmin, dt, taper, fval, linvel, tm0, **_):
    """mode 3: mute below and above a hyperbola, t is the width of the zone"""
    nmute = _nint((tmin + t) / dt)
    ntair = _nint(math.sqrt((tm0 / dt) ** 2 + (fval / linvel / dt) ** 2))
    _mute_zone(x, ntair, nmute, taper)


def _mute_polygon(x, t, tmin, dt, taper, fval, xmute, twindow, **_):
    """mode 4: mute below and above the polygonal line, with the width of the zone from twindow"""
    # (sumute interpolates twindow at a variable that it never sets, here it is the key value like for tmute)
    tw = float(np.interp(fval, xmute, twindow, left=twindow[0], right=twindow[-1]))
    _mute_zone(x, _nint(t / dt), _nint(tw / dt), taper)


_MUTE_MODES = {0: _mute_above, 1: _mute_below, 2: _mute_line, 3: _mute_hyperbola, 4: _mute_polygon}


def _mute(
    upstream, xmute, tmute, *, mode=0, key='offset', ntaper=0, absolute=True, linvel=330.0, tm0=0.0, twindow=None,
):
    xmute = np.asarray(xmute, dtype=np.float64)
    tmute = np.asarray(tmute, dtype=np.float64)
    if xmute.ndim != 1 or xmute.shape != tmute.shape or xmute.shape[0] == 0:
        raise ValueError("xmute and tmute must be 1D arrays of the same length")
    if mode not in _MUTE_MODES:
        raise ValueError(f"mode = {mode!r} must be one of {sorted(_MUTE_MODES)}")
    if mode == 4:
        if twindow is None:
            raise ValueError("mode 4 needs a twindow")
        twindow = np.asarray(twindow, dtype=np.float64)
        if twindow.shape != xmute.shape:
            raise ValueError("lengths of xmute, twindow must be the same")
    if linvel == 0:
        raise ValueError("linear velocity can't be 0")
    if ntaper < 0:
        raise ValueError("ntaper must not be negative")
    k = np.arange(ntaper)
    taper = np.sin((k + 1) * np.pi / (2 * _max(ntaper, 1))) ** 2  # a sine squared taper
    apply_mode = _MUTE_MODES[mode]

    def mute_trace(trace):
        header = trace.header
        tmin = header['sample_start']
        dt = header['d_sample']
        if not dt:
            raise ValueError("The trace has no sample interval.")
        fval = _key_value(trace, key)
        # linear interpolation of the mute times, extrapolated to the left by the first time on the trace and to
        # the right by the last value given
        t = float(np.interp(fval, xmute, tmute, left=tmin, right=tmute[-1]))
        if absolute:
            fval = _abs(fval)
        x = np.array(trace)
        apply_mode(
            x, t=t, tmin=tmin, dt=dt, taper=taper.astype(np.float32), fval=fval,
            linvel=linvel, tm0=tm0, xmute=xmute, twindow=twindow,
        )
        return trace.replace(x)

    return per_trace(upstream, mute_trace)


def mute(xmute, tmute, *, mode=0, key='offset', ntaper=0, absolute=True, linvel=330.0, tm0=0.0, twindow=None):
    """Mute above (or below) a polygonal curve, along the distance given by a header value (SUMUTE).

    Parameters
    ----------
    xmute, tmute : arrays
        Positions (values of ``key``) and the corresponding times (s) that define the mute. The time is interpolated
        linearly, extrapolated to the left by the first time of the trace and to the right by the last ``tmute``.
        For the modes that mute a zone around a line they are the total duration of the muted zone.
    mode : {0, 1, 2, 3, 4}
        0 mutes above the curve, 1 below it. 2 mutes below and above a straight line, with the velocity ``linvel``
        (an air wave), 3 below and above a hyperbola of that velocity, and 4 below and above the polygonal line, with
        the widths of the zone in time given by ``twindow``.
    key : str or function
        The header value, or a function of a trace, with the positions, default ``'offset'``.
    ntaper : int
        Number of samples to taper (sine squared) before the hard mute.
    absolute : bool
        Take the absolute value of the key for modes 2, 3 and 4.
    linvel, tm0 : float
        Velocity, and the time shift at key=0, of the modes 2 and 3.
    twindow : array
        Widths of the muted zone in time, for mode 4 only.
    """
    kwargs = dict(mode=mode, key=key, ntaper=ntaper, absolute=absolute, linvel=linvel, tm0=tm0, twindow=twindow)
    _mute((), xmute, tmute, **kwargs)  # check the parameters now
    return Stage(_mute, xmute, tmute, parallelism='trace', name='mute', **kwargs)


# ----------------------------------------------------------------------------------------------------------- wind
def _wind(
    upstream, *, key='trace_id', min=None, max=None, j=1, s=0, skip=0, count=None, reject=(), accept=(),
    abs=False, ordered=0, dt=None, f1=None, tmin=None, tmax=None, itmin=None, itmax=None, nt=None,
):
    if j < 1:
        raise ValueError("j must be at least 1")
    if ordered not in (-1, 0, 1):
        raise ValueError("ordered must be -1, 0 or 1")
    if skip < 0:
        raise ValueError("skip must not be negative")
    if count is not None and count < 1:
        raise ValueError("count must be at least 1")
    low = -math.inf if min is None else min
    high = math.inf if max is None else max
    reject = list(reject)
    accept = list(accept)
    source = as_trace_iterator(upstream)

    def gen():
        traces = iter(source)
        for _ in range(skip):
            if next(traces, None) is None:
                return
        first = next(traces, None)
        if first is None:
            return

        # the window in time, from the first trace
        start = first.header['sample_start'] if f1 is None else f1
        sample_dt = first.d_sample if dt is None else dt
        if not sample_dt:
            raise ValueError("The first trace has no sample interval, and no `dt` was given.")
        if itmin is not None:
            first_sample = itmin
        elif tmin is not None:
            first_sample = _nint((tmin - start) / sample_dt)
        else:
            first_sample = 0
        if itmax is not None:
            last_sample = itmax
            n_out = last_sample - first_sample + 1
        elif tmax is not None:
            last_sample = _nint((tmax - start) / sample_dt)
            n_out = last_sample - first_sample + 1
        elif nt is not None:
            n_out = nt
            last_sample = first_sample + nt - 1
        else:
            last_sample = first.n_sample - 1
            n_out = last_sample - first_sample + 1
        if first_sample < 0:
            raise ValueError(f"itmin={first_sample} should be positive")
        if first_sample > last_sample:
            raise ValueError(f"itmin={first_sample}, itmax={last_sample} conflict")

        remaining = count
        for trace in _chain(first, traces):
            value = _key_value(trace, key)
            bad = value in reject
            good = value in accept
            ival = int(_abs(value) if abs else value)
            if (ordered == 1 and high < ival) or (ordered == -1 and low > ival):
                return
            if good or (low <= ival <= high and (ival - s) % j == 0 and not bad):
                if first_sample > 0 or n_out != trace.n_sample:
                    out = np.zeros(n_out, dtype=np.float32)
                    piece = np.asarray(trace)[first_sample:first_sample + n_out]
                    out[:piece.shape[0]] = piece
                    trace = trace.replace(
                        out, sample_start=trace.header['sample_start'] + first_sample * trace.d_sample,
                    )
                yield trace
                if remaining is not None:
                    remaining -= 1
                    if not remaining:
                        return

    # the number of traces that get through is not known
    return from_iterable(gen())


def _chain(first, rest):
    yield first
    yield from rest


def wind(
    *, key='trace_id', min=None, max=None, j=1, s=0, skip=0, count=None, reject=(), accept=(), abs=False,
    ordered=0, dt=None, f1=None, tmin=None, tmax=None, itmin=None, itmax=None, nt=None,
):
    """Window traces by a header value, and in time (SUWIND).

    Which traces pass: those whose ``key`` (a header value, or a function of a trace) is in ``min`` to ``max``
    (integers; the value is truncated and, with ``abs``, made positive first), is a multiple of ``j`` away from
    ``s``, and is not in ``reject``, as well as any that are in ``accept``, whatever else is true of them. ``skip``
    drops the first traces and ``count`` stops after that many have passed. With ``ordered=1`` (``-1``) the key is
    known to increase (decrease) along the traces, and everything after it passes ``max`` (``min``) is dropped.

    Windowing in time (the first trace sets the window for all of them): ``itmin`` or ``tmin`` (s, with ``f1`` the
    time of the first sample of the trace and ``dt`` the sample interval, by default from the first trace) for the
    start, and ``itmax``, ``tmax`` or ``nt`` for the end. A window that sticks out of a trace is padded with zeros.

    The number of traces that get through is not known and can depend on their order, so this can not be split across
    workers.
    """
    kwargs = dict(
        key=key, min=min, max=max, j=j, s=s, skip=skip, count=count, reject=reject, accept=accept, abs=abs,
        ordered=ordered, dt=dt, f1=f1, tmin=tmin, tmax=tmax, itmin=itmin, itmax=itmax, nt=nt,
    )
    _wind((), **kwargs)  # check the parameters now
    return Stage(_wind, parallelism='serial', name='wind', **kwargs)


# ----------------------------------------------------------------------------------------------------------- kill
def _kill(upstream, key=None, a=None, *, min=None, count=1):
    source = as_trace_iterator(upstream)

    def zeroed(trace):
        return trace.replace(np.zeros(trace.n_sample, dtype=np.float32))

    if min is None:
        # by the value of a header
        return per_trace(source, lambda trace: zeroed(trace) if _key_value(trace, key) == a else trace)

    def gen():
        n_killed = 0
        for k, trace in enumerate(source, start=1):
            if min <= k < min + count:
                n_killed += 1
                trace = zeroed(trace)
            yield trace
        if n_killed < count:
            raise ValueError(f"failed to get the requested trace #{min + n_killed}")

    return from_iterable(gen(), n_traces=source.n_traces)


def kill(key=None, a=None, *, min=None, count=1):
    """Zero out traces (SUKILL), either those whose header value ``key`` is ``a``, or a run of ``count`` traces from
    the ``min``-th (the first trace is number 1). ``min`` takes precedence.

    Selecting by position needs to know where each trace is in the stream, so it can not be split across workers.
    """
    if min is None and (key is None or a is None):
        raise ValueError("Give either `key` and `a`, or `min` (and `count`).")
    if min is not None and min < 1:
        raise ValueError(f"min = {min}, must be >= 1")
    if count < 1:
        raise ValueError("count must be at least 1")
    return Stage(_kill, key, a, parallelism='serial' if min is not None else 'trace', name='kill', min=min, count=count)
