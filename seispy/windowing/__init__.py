"""
Windowing and muting programs: SUMUTE, SUWIND and SUKILL (``su/main/windowing_sorting_muting``), and SUVLENGTH
(``su/main/operations``), and SUSORT and SUMIXGATHERS.

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
from .. import spool
from ..stage import Stage, header_value as _key_value, per_trace
from . import _mutec

__all__ = ['mute', 'wind', 'kill', 'vlength', 'sort', 'mixgathers', 'split', 'cleave', 'putgthr', 'getgthr', 'sorty']

from ._splitting import split, cleave, putgthr, getgthr, sorty  # noqa: E402

_abs, _min, _max = abs, min, max


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _half(n):
    # (integer division in C truncates towards 0)
    return int(n / 2)


# ----------------------------------------------------------------------------------------------------------- mute
def _on_parts(x, func):
    """func (which changes a float32 array in place) applied to x, or to the real and the imaginary part of a complex x"""
    if np.iscomplexobj(x):
        real, imag = np.ascontiguousarray(x.real, dtype=np.float32), np.ascontiguousarray(x.imag, dtype=np.float32)
        func(real)
        func(imag)
        return (real + 1j * imag).astype(np.complex64)
    x = np.ascontiguousarray(x, dtype=np.float32)
    func(x)
    return x


# (each mode is a function of the SU library, see _mutec.pyx and sumute.c)
def _mute_above(x, t, tmin, dt, taper, **_):
    """mode 0: mute above the curve"""
    return _on_parts(x, lambda a: _mutec.above(a, t, tmin, dt, taper))


def _mute_below(x, t, tmin, dt, taper, **_):
    """mode 1: mute below the curve"""
    return _on_parts(x, lambda a: _mutec.below(a, t, tmin, dt, taper))


def _mute_line(x, t, tmin, dt, taper, fval, linvel, tm0, **_):
    """mode 2: mute below and above a straight line (an air wave), t is the width of the zone"""
    return _on_parts(x, lambda a: _mutec.line(a, t, tmin, dt, taper, fval, linvel, tm0))


def _mute_hyperbola(x, t, tmin, dt, taper, fval, linvel, tm0, **_):
    """mode 3: mute below and above a hyperbola, t is the width of the zone"""
    return _on_parts(x, lambda a: _mutec.hyperbola(a, t, tmin, dt, taper, fval, linvel, tm0))


def _mute_polygon(x, t, tmin, dt, taper, fval, xmute, twindow, **_):
    """mode 4: mute below and above the polygonal line, with the width of the zone from twindow"""
    # (sumute interpolates twindow at a variable that it never sets, here it is the key value like for tmute)
    tw = float(np.interp(fval, xmute, twindow, left=twindow[0], right=twindow[-1]))
    return _on_parts(x, lambda a: _mutec.polygon(a, t, tw, dt, taper))


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
    taper = _mutec.taper_weights(ntaper)  # (a sine squared taper)
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
        x = apply_mode(
            np.array(trace), t=t, tmin=tmin, dt=dt, taper=taper, fval=fval,
            linvel=linvel, tm0=tm0, xmute=xmute, twindow=twindow,
        )
        return trace.replace(x)

    return per_trace(upstream, mute_trace, on_complex='native')


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
                    samples = np.asarray(trace)
                    out = np.zeros(n_out, dtype=samples.dtype)
                    piece = samples[first_sample:first_sample + n_out]
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
        return trace.replace(np.zeros_like(np.asarray(trace)))

    if min is None:
        # by the value of a header
        return per_trace(
            source, lambda trace: zeroed(trace) if _key_value(trace, key) == a else trace, on_complex='native'
        )

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


# ------------------------------------------------------------------------------------------------------- vlength
def _vlength(upstream, *, ns=None):
    source = as_trace_iterator(upstream)

    def traces():
        length = ns
        for trace in source:
            if length is None:
                length = trace.n_sample  # (the length of the first trace)
            n = trace.n_sample
            if n == length:
                yield trace
                continue
            x = np.asarray(trace)
            if n > length:
                out = x[:length]
            else:
                out = np.zeros(length, dtype=x.dtype)
                out[:n] = x
            yield trace.replace(out)

    return from_iterable(traces(), n_traces=source.n_traces)


def vlength(ns=None):
    """Make traces of different lengths the same length (SUVLENGTH): longer ones are cut, shorter ones are padded with
    zeros at the end.

    Parameters
    ----------
    ns : int, optional
        The number of samples of the output, by default the number of samples of the first trace. (Without it the
        length depends on the first trace of the stream, so such a stage can not be split across workers.)
    """
    if ns is not None and ns < 1:
        raise ValueError(f"ns={ns} must be at least 1")
    return Stage(_vlength, parallelism='serial' if ns is None else 'trace', name='vlength', ns=ns)


# --------------------------------------------------------------------------------------------------------- sort
def _sort_key(key):
    """(key, descending) from a header name that may have a + or - in front, a function of a trace, or a pair"""
    if isinstance(key, tuple):
        return key[0], bool(key[1])
    if callable(key):
        return key, False
    name = str(key)
    return name.lstrip('+-'), name.startswith('-')


def _sort(upstream, *keys, tmpdir=None, memory=None):
    keys = [_sort_key(k) for k in (keys or ('ensemble_number',))]
    source = as_trace_iterator(upstream)

    def sort_key(trace):
        # (a key that is sorted from the largest is wrapped, so that the keys can be compared together)
        values = (_key_value(trace, key) for key, _ in keys)
        return tuple(spool.Descending(v) if descending else v for v, (_, descending) in zip(values, keys))

    return from_iterable(
        spool.external_sort(source, sort_key, stage='sort', tmpdir=tmpdir, memory=memory), n_traces=source.n_traces
    )


def sort(*keys, tmpdir=None, memory=None):
    """Sort traces by header values (SUSORT): ``sort('ensemble_number', 'offset')`` for the gathers, and within each by
    offset. A ``-`` in front of a name (``'-offset'``) sorts that one from the largest. A key can also be a function of a
    trace, or a pair ``(function, descending)``. The default is ``'ensemble_number'`` (SU's cdp).

    All of the traces are read before the first is output. They are kept in memory while they fit in a budget of ``memory``
    bytes (see `seispy.spool`), and past that they are sorted in chunks that are put on disk and merged, so that data bigger
    than memory can be sorted. Traces with the same values stay in the order that they came in.

    Parameters
    ----------
    tmpdir : path, True or False, optional
        Where the chunks go, if there are any: a directory, or ``True`` for the default one (``seispy.spool.tmpdir``,
        ``SEISPY_TMPDIR``, ``CWP_TMPDIR``, or the system's). ``False`` never uses the disk. When the disk is used a
        warning (to the logger ``seispy.spool``) says which directory, and the directory is removed afterwards.
    memory : int, optional
        The number of bytes of traces to keep in memory before using the disk (default 1 GiB, or ``SEISPY_MEMORY``); 0 puts
        everything on disk, as SUSORT does.
    """
    for key in keys:
        _sort_key(key)
    spool.check_tmpdir(tmpdir)
    spool.resolve_memory(memory)
    return Stage(_sort, *keys, tmpdir=tmpdir, memory=memory, parallelism='serial', name='sort')


# ------------------------------------------------------------------------------------------------------ mixgathers
def _mixgathers(upstream, other, *, scaling=False):
    source = as_trace_iterator(upstream)
    second = as_trace_iterator(other)

    def present(offset, offsets):
        # (within a tenth of a percent of an offset of the first gather, as sumixgathers)
        for h in offsets:
            if h > 0 and 0.999 * h < offset < 1.001 * h:
                return True
            if h < 0 and 1.001 * h < offset < 0.999 * h:
                return True
            if h == 0 and offset == 0:
                return True
        return False

    def traces():
        panel = list(source)
        offsets = [t.header['offset'] for t in panel]
        extra = []
        for trace in second:
            offset = trace.header['offset']
            if present(offset, offsets):
                continue
            if scaling and offset != 0:
                # (a boost with the offset, which sumixgathers has for traces that are from an interpolation)
                trace = trace.replace(np.asarray(trace) * (1.0 + 0.03 * (abs(int(offset)) / 1000.0)))
            extra.append(trace)
        # the gather in order of offset (sumixgathers puts the new traces before all of the traces of the first gather)
        merged = sorted(panel + extra, key=lambda t: t.header['offset'])
        yield from merged

    return from_iterable(traces())


def mixgathers(other, *, scaling=False):
    """Fill the gaps of a gather with the traces of another (SUMIXGATHERS): the traces of the stream are kept, and the
    traces of ``other`` (an iterable of traces) are added where the stream has no trace at that offset.

    The gather that comes out is in order of offset. (sumixgathers writes the added traces first and then the gather.) With
    ``scaling`` the added traces are multiplied by ``1 + .03 |offset| / 1000``. Both are whole gathers: all of the traces
    are read before the first is output.
    """
    return Stage(_mixgathers, other, parallelism='serial', name='mixgathers', scaling=scaling)
