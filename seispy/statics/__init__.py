"""
Statics: SUSTATIC and SUSTATICRRS (elevation and refraction statics), SURANDSTAT and SUADDSTATICS (random statics errors) and
SURESSTAT (surface consistent residual statics) from ``su/main/statics``. (The programs of the csv files, which are driven by
SPS header words, are not here.)

A static is a time shift of a trace, in seconds (the SU programs have it in milliseconds): ``static`` applies a shift with
the 8 point sinc interpolation of the SU library, and puts the statics that it applied in the header (``source_static``,
``receiver_static`` and ``total_static``, which it adds to, so that corrections can be made one after another), where a
positive static moves the data earlier. The elevations are those of ``tx_loc`` and ``rx_loc`` (the third value), in
the units of the velocities, and not scaled with ``scalel``.

Where a program plainly does not do what its documentation says this does what the documentation says; the
differences are with each stage.
"""
import math
import time
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..noise import _rng
from ..stage import Stage, header_value, per_trace
from ..stretching import _resample
from . import _resstat

__all__ = ['static', 'staticrrs', 'randstat', 'addstatics', 'resstat']

F32 = np.float32
_MODES = ('field', 'header', 'regular', 'keyed')
_RRS_MODES = _MODES + ('refraction', 'residual')


def _nint(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _number(spec, trace):
    """A number that is a constant, the name of a value of the header, or a function of the trace"""
    if callable(spec) or isinstance(spec, str):
        return header_value(trace, spec)
    return float(spec)


def _shifted(trace, shift):
    """The samples of the trace at times that are ``shift`` later (the sinc interpolation of the SU library): zeros where
    they are outside of the trace, which is a static of ``shift`` (earlier for a positive one)"""
    nt, dt = trace.n_sample, trace.d_sample
    if not dt:
        raise ValueError("The trace has no sample interval.")
    tmin = trace.header['sample_start']
    x = np.asarray(trace)

    def of(part):
        return _resample.resample(np.ascontiguousarray(part, dtype=F32), F32(dt), F32(tmin - shift), nt, F32(dt), F32(tmin))

    if x.dtype.kind == 'c':
        return (of(x.real) + 1j * of(x.imag)).astype(np.complex64)
    return of(x)


def _lookup(table, key):
    """The static of a key from a mapping or a sequence (0 where there is none)"""
    if table is None:
        return 0.0
    if hasattr(table, 'get'):
        return float(table.get(key, 0.0))
    try:
        index = int(key)
        return float(table[index]) if 0 <= index < len(table) else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0


# ----------------------------------------------------------------------------------------------------- static, rrs
def _static_stage(upstream, *, v0=None, v1=None, mode='field', sign=1, source_datum=0.0, source_depth=0.0, uphole=0.0,
                  source_statics=None, receiver_statics=None, no=96, source_key='ensemble_number',
                  receiver_key='ensemble_trace_number', blvl=None, refr=None, fx=0.0, dx=1.0, v_r=None, vel=None,
                  dzv=None, report=None, tcdp=None, num=None, modes=_MODES):
    if mode not in modes:
        raise ValueError(f"mode={mode!r} must be one of {list(modes)}")
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    if mode == 'field' and v1 is None:
        raise ValueError("v1, the weathering velocity, is needed to work out the statics (the program takes it from the "
                         "header word swevel, which traces here do not have)")
    if mode == 'regular':
        if source_statics is None or receiver_statics is None:
            raise ValueError("source_statics and receiver_statics are needed for mode='regular'")
        if no < 1:
            raise ValueError("no must be at least 1")
    velocity_above = v0 if v0 is not None else v1
    if mode == 'refraction':
        for name, value in (('blvl', blvl), ('refr', refr), ('v_r', v_r), ('vel', vel), ('dzv', dzv)):
            if value is None:
                raise ValueError(f"{name} is needed for mode='refraction'")
        blvl, refr = np.asarray(blvl, dtype=np.float64), np.asarray(refr, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        if vel.ndim != 2 or blvl.ndim != 1 or blvl.shape != refr.shape:
            raise ValueError("vel is (mx, mz); blvl and refr are one value for each midpoint, and the same length")
        if dx == 0.0:
            raise ValueError("dx must not be 0")
        if report is None:
            report = {}
        report['tcdp'] = np.zeros(blvl.shape[0])
        report['num'] = np.zeros(blvl.shape[0])
    if mode == 'residual':
        if report is not None and tcdp is None:
            tcdp, num = report.get('tcdp'), report.get('num')
        if tcdp is None or num is None:
            raise ValueError("tcdp and num (or the report of mode='refraction') are needed for mode='residual'")
        tcdp, num = np.asarray(tcdp, dtype=np.float64), np.asarray(num, dtype=np.float64)
        if dx == 0.0:
            raise ValueError("dx must not be 0")
    state = {'io': 0, 'is': 0}
    source = as_trace_iterator(upstream)

    def index_of(x):
        return int((x - fx) / dx)  # (the program truncates)

    def checked(array, i, what):
        if not 0 <= i < array.shape[0]:
            raise ValueError(f"the {what} is outside of the model (index {i} of {array.shape[0]})")
        return i

    def static_of(trace):
        """(source static, receiver static, total static, the static to apply, and the headers' mode of change)"""
        header = trace.header
        tsd = trd = 0.0
        if mode == 'field':
            tx_z, rx_z = header['tx_loc'][2], header['rx_loc'][2]
            tsd = (-tx_z + _number(source_datum, trace) + _number(source_depth, trace)) / v1
            trd = tsd - _number(uphole, trace)
            total = tsd + trd + (tx_z - rx_z) / velocity_above
            return tsd, trd, total, total, 'add'
        if mode == 'header':
            total = header['total_static']
            if total == 0.0:
                total = header['source_static'] + header['receiver_static']
            return tsd, trd, total, total, 'keep'
        if mode == 'regular':
            n_s, n_r = len(source_statics), len(receiver_statics)
            ir = state['is'] + state['io']
            if state['is'] < n_s:
                tsd = float(source_statics[state['is']])
            if 0 <= ir < n_r:
                trd = float(receiver_statics[ir])
            state['io'] += 1
            if state['io'] > no - 1:
                state['io'], state['is'] = 0, state['is'] + 1
            return tsd, trd, tsd + trd, tsd + trd, 'add'
        if mode == 'keyed':
            tsd = _lookup(source_statics, header_value(trace, source_key))
            trd = _lookup(receiver_statics, header_value(trace, receiver_key))
            return tsd, trd, tsd + trd, tsd + trd, 'add'
        if mode == 'refraction':
            is_c = checked(blvl, index_of(header['tx_loc'][0]), 'source')
            ir_c = checked(blvl, index_of(header['rx_loc'][0]), 'receiver')
            n_z = vel.shape[1]
            tsd = float(np.sum(dzv / vel[checked(vel, is_c, 'source'), :n_z - 1])) - (blvl[is_c] - refr[is_c]) / v_r
            trd = float(np.sum(dzv / vel[checked(vel, ir_c, 'receiver'), :n_z - 1])) - (blvl[ir_c] - refr[ir_c]) / v_r
            ic_c = checked(report['tcdp'], index_of(header['ensemble_number']), 'cdp')
            report['tcdp'][ic_c] += tsd + trd
            report['num'][ic_c] += 1.0
            return 0.0, 0.0, tsd + trd, 0.0, 'refraction'
        # residual: what is left of the total, after the average of the CDP (that average is what is applied after stack)
        ic_c = checked(tcdp, index_of(header['ensemble_number']), 'cdp')
        if num[ic_c] >= 0.5:
            bulk = tcdp[ic_c] / num[ic_c]
            res = header['total_static'] - bulk
        else:
            bulk = res = 0.0
        return 0.0, res, bulk, res, 'residual'

    def apply(trace):
        if not trace.d_sample:
            raise ValueError("The trace has no sample interval.")
        tsd, trd, total, to_apply, change = static_of(trace)
        x = _shifted(trace, sign * to_apply) if to_apply != 0.0 or mode in ('field', 'regular', 'keyed') else None
        fields = {}
        header = trace.header
        if change == 'add':
            # (added to what the header has, so that statics can be applied one after the other)
            fields = dict(source_static=header['source_static'] + tsd, receiver_static=header['receiver_static'] + trd,
                          total_static=header['total_static'] + total)
        elif change == 'refraction':
            fields = dict(source_static=0.0, receiver_static=0.0, total_static=total)
        elif change == 'residual':
            fields = dict(source_static=0.0, receiver_static=trd, total_static=total)
        return trace.replace(x if x is not None else None, **fields)

    return per_trace(source, apply, on_complex='native')


_STATIC_HELP = """
    Parameters
    ----------
    v1 : float
        The subweathering velocity (m/s), for ``mode='field'``. (The program takes it from the header word swevel.)
    v0 : float, optional
        The weathering velocity, default ``v1`` (the program takes it from the header word wevel).
    mode : str
        ``'field'`` works out the statics from the elevations of the source and the receiver (``tx_loc`` and
        ``rx_loc``) and the values below: the program's ``hdrs=0``. ``'header'`` applies the static that the header has:
        ``total_static`` if it is not 0, otherwise the sum of ``source_static`` and ``receiver_static`` (``hdrs=1``; the
        header is not changed). ``'regular'`` takes them from the arrays ``source_statics`` (one for each source)
        and ``receiver_statics``, for a regular geometry where each source has ``no`` traces, and the receiver of a trace
        is ``source number + number of the trace in its source`` (``hdrs=2``). ``'keyed'`` takes them from the
        mappings (or arrays) ``source_statics`` and ``receiver_statics``, with the key ``source_key`` and
        ``receiver_key`` of each trace (``hdrs=3``, the output of ``resstat``).
    sign : {1, -1}
        -1 applies the negative of the statics.
    source_datum, source_depth, uphole : number, header name or function
        The values of the program's header words sdel, sdepth and sut (the uphole time, in seconds) for ``mode='field'``,
        default 0.
    source_statics, receiver_statics : arrays or mappings
        The statics (s) of ``'regular'`` and ``'keyed'``.
    source_key, receiver_key : header name or function
        What the statics of ``'keyed'`` are found by, default ``ensemble_number`` and ``ensemble_trace_number`` (the
        program's fldr and tracf).
"""


def static(v1=None, v0=None, *, mode='field', sign=1, source_datum=0.0, source_depth=0.0, uphole=0.0,
           source_statics=None, receiver_statics=None, no=96, source_key='ensemble_number',
           receiver_key='ensemble_trace_number'):
    """Elevation static corrections, from the elevations or from statics that are given (SUSTATIC).

    A trace is moved by its static with the 8 point sinc interpolation of the SU library, and the statics are added to
    the header. For ``mode='field'`` the statics are ``source = (datum of the source - elevation + depth) / v1``,
    ``receiver = source - uphole`` and ``total = source + receiver + (source elevation - receiver elevation) / v0``
    (as the program's documentation has them), for a datum at elevation 0: shift the elevations if it is not.
    """
    kwargs = dict(v0=v0, v1=v1, mode=mode, sign=sign, source_datum=source_datum, source_depth=source_depth,
                  uphole=uphole, source_statics=source_statics, receiver_statics=receiver_statics, no=no,
                  source_key=source_key, receiver_key=receiver_key)
    _static_stage((), **kwargs)  # check the parameters now
    return Stage(_static_stage, parallelism='serial' if mode == 'regular' else 'trace', name='static', **kwargs)


static.__doc__ += _STATIC_HELP + """
    The program's sustaticB (a revision of ``hdrs=2`` and ``3``) is what this does for them: the statics from files are
    in milliseconds in the program and were divided by 1000 more than once, which comes out right here (seconds), and
    a receiver index of 0 and a source index equal to the number of sources are not left out of the table.
    """


def staticrrs(v1=None, v0=None, *, mode='field', sign=1, source_datum=0.0, source_depth=0.0, uphole=0.0,
              source_statics=None, receiver_statics=None, no=96, source_key='ensemble_number',
              receiver_key='ensemble_trace_number', blvl=None, refr=None, fx=0.0, dx=1.0, v_r=None, vel=None, dzv=None,
              report=None, tcdp=None, num=None):
    """Elevation statics, with the splitting of refraction statics into residual and average ones (SUSTATICRRS).

    The modes of ``static``, and: ``'refraction'`` works out the statics of the near surface model ``vel`` (an array
    ``(mx, mz)`` of velocities, ``dzv`` apart in depth, one column for each ``dx`` from ``fx`` along the line) from its
    base ``blvl`` and the reference datum ``refr`` (one value for each midpoint) with the replacement velocity ``v_r``
    (Cox, 1999). The data are not moved; the statics are put in ``total_static``, and the sums of them for each CDP (the
    ``ensemble_number``, at ``fx + k dx``) are in ``report['tcdp']`` and ``report['num']`` (the program writes them to
    the files tcdp.b and num.b) when the stream has been read. ``'residual'`` (with ``tcdp`` and ``num``, or that
    ``report``) then applies what is left of the statics after the average of each CDP, which it puts in
    ``total_static``, and the residual in ``receiver_static`` (the program's gstat). The average statics are applied with
    ``'header'``, after stack.
    """
    kwargs = dict(v0=v0, v1=v1, mode=mode, sign=sign, source_datum=source_datum, source_depth=source_depth,
                  uphole=uphole, source_statics=source_statics, receiver_statics=receiver_statics, no=no,
                  source_key=source_key, receiver_key=receiver_key, blvl=blvl, refr=refr, fx=fx, dx=dx, v_r=v_r, vel=vel,
                  dzv=dzv, report=report, tcdp=tcdp, num=num, modes=_RRS_MODES)
    _static_stage((), **kwargs)  # check the parameters now
    serial = mode in ('regular', 'refraction')
    return Stage(_static_stage, parallelism='serial' if serial else 'trace', name='staticrrs', **kwargs)


# ------------------------------------------------------------------------------------------------------- randstat
def _randstat(upstream, *, max=None, seed=None, scale=1.0, dt=None):
    seed = int(time.time()) if seed is None else int(seed)
    generator = _rng.Uniform(seed)
    source = as_trace_iterator(upstream)

    def traces():
        for trace in source:
            sample_dt = trace.d_sample or dt
            if not sample_dt:
                raise ValueError("The trace has no sample interval, and no `dt` was given.")
            dt_ms = sample_dt * 1000.0
            limit = dt_ms if max is None else max
            # the random shift (ms) of this trace, as a delay of whole samples
            shift_ms = scale * 2.0 * limit * (0.5 - generator.next())
            delay = _nint(shift_ms / dt_ms)
            x = np.array(trace)
            n = x.shape[0]
            out = np.zeros_like(x)
            if delay >= 0:
                if delay < n:
                    out[delay:] = x[:n - delay]
            elif -delay < n:
                out[:n + delay] = x[-delay:]
            yield trace.replace(out)

    return from_iterable(traces(), n_traces=source.n_traces)


def randstat(*, max=None, seed=None, scale=1.0, dt=None):
    """Add random static errors to traces (SURANDSTAT): each trace is delayed by a random number of samples, from the
    uniform random numbers of the SU library.

    Parameters
    ----------
    max : float
        The greatest random shift (ms), default the sample interval.
    seed : int
        The seed of the random numbers, by default the time (not repeatable).
    scale : float
        A factor for the shifts, default 1.
    dt : float
        The sample interval for traces that do not have one.

    The shifts are in milliseconds, as the documentation says (the program uses them as samples), with one random number
    for each trace (the program starts the generator again with the seed for every trace, so that they all get the same
    shift), the samples that are moved in are zeros (the program leaves the old values, and reads and writes past the
    end of the trace), and the default of ``max`` is the sample interval (the program's ``(int) dt * 1000`` is 0).
    The stream is taken in order, so this can not be split across workers.
    """
    kwargs = dict(max=max, seed=seed, scale=scale, dt=dt)
    return Stage(_randstat, parallelism='serial', name='randstat', **kwargs)


# --------------------------------------------------------------------------------------------------- addstatics
def _addstatics(upstream, *, shift, seed=None, dt=None):
    if shift < 0.0:
        raise ValueError("shift must not be negative")
    seed = int(time.time()) if seed is None else int(seed)
    source = as_trace_iterator(upstream)

    def traces():
        traces = list(source)
        if not traces:
            return
        sources, receivers = {}, {}
        for trace in traces:
            tx, rx = trace.header['tx_loc'], trace.header['rx_loc']
            sources.setdefault((tx[0], tx[1]), len(sources))
            receivers.setdefault((rx[0], rx[1]), len(receivers))
        # a static (ms) for each source and then each receiver, in the order that they are first seen
        draws = _rng.Uniform(seed).draw(len(sources) + len(receivers)).astype(np.float64)
        statics = shift * (2.0 * draws - 1.0)
        for trace in traces:
            tx, rx = trace.header['tx_loc'], trace.header['rx_loc']
            total = statics[sources[(tx[0], tx[1])]] + statics[len(sources) + receivers[(rx[0], rx[1])]]
            # (the header has seconds)
            yield trace.replace(total_static=float(total) * 0.001)

    return from_iterable(traces(), n_traces=source.n_traces)


def addstatics(shift, *, seed=None, dt=None):
    """Add random surface consistent statics, to the header (SUADDSTATICS).

    Each source position and each receiver position (``tx_loc`` and ``rx_loc``) is given a random static in
    ``[-shift, +shift]``, and each trace gets the sum of its two in ``total_static``; ``static(mode='header')`` then
    makes the shifts.

    Parameters
    ----------
    shift : float
        The greatest static (ms).
    seed : int
        The seed of the random numbers (SU's), by default the time.

    The program needs the numbers of sources, receivers, CDPs and the maximum fold, and the name of the file, which are
    not needed here, and it rounds the static of each trace to a whole millisecond. The traces are held in memory.
    """
    kwargs = dict(shift=shift, seed=seed, dt=dt)
    _addstatics((), **kwargs)  # check the parameters now
    return Stage(_addstatics, parallelism='serial', name='addstatics', **kwargs)


# ------------------------------------------------------------------------------------------------------- resstat
def _midpoint(trace):
    tx, rx = trace.header['tx_loc'], trace.header['rx_loc']
    return (round(0.5 * (tx[0] + rx[0]), 3), round(0.5 * (tx[1] + rx[1]), 3))


def _key_of(trace, key):
    value = key(trace) if callable(key) else trace.header[key]
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(value)
    return value


def resstat(traces, *, shot='ensemble_number', receiver='ensemble_trace_number', cmp=None, ntpick=50, niter=5,
            sub=False, mode=0, perc=10.0):
    """Surface consistent residual statics of source and receiver positions (SURESSTAT, Ronen and Claerbout, 1985).

    This is a function of the traces (all of them, in memory) that returns the statics, which ``static(mode='keyed')``
    then applies. Each iteration builds the stack of the traces of each CMP, and for each shot (then each receiver) the
    cross-correlation of the supertrace of its traces with the supertrace of the stacks of their CMPs, picks the lag of
    its peak and applies it, so that the stacks improve.

    Parameters
    ----------
    traces : iterable of Trace
        All of the (prestack) traces, of the same length.
    shot, receiver, cmp : header name or function
        What identifies the shot (default ``ensemble_number``, the program's fldr), the receiver position (default
        ``ensemble_trace_number``, tracf) and the CMP (default the midpoint of ``tx_loc`` and ``rx_loc``) of a trace.
    ntpick : int
        The greatest shift (samples), default 50.
    niter : int
        The number of iterations, default 5.
    sub : bool
        Subtract the supertrace of the shot (or receiver) from that of the CMP (otherwise the statics are strongly
        biased to 0).
    mode : {0, 1}
        0 picks the global maximum of the cross-correlation, 1 the peak closest to zero lag that is within ``perc`` percent
        of it.

    Returns
    -------
    (shot_statics, receiver_statics) : dicts from the key to the static in seconds (to be applied with ``sign=1``, which
    moves the data earlier when it is positive). A shot or a receiver that has one trace has the static 0.

    The program's array sizes (``nshot``, ``nr``, ``nc``, ``sfold``, ``rfold``, ``cfold``, ``ntraces``) are not needed.
    It also reads samples 1 to ``nt`` of a trace (not 0 to ``nt - 1``), and gives a lag of ``-(ntpick + 1)`` for ``mode=1``
    when no peak is above the level: both are corrected.
    """
    if mode not in (0, 1):
        raise ValueError("mode must be 0 or 1")
    if ntpick < 0 or niter < 1:
        raise ValueError("ntpick must not be negative and niter must be at least 1")
    traces = list(traces)
    if not traces:
        return {}, {}
    cmp = _midpoint if cmp is None else cmp
    first = traces[0]
    dt = first.d_sample
    if not dt:
        raise ValueError("The trace has no sample interval.")
    n = first.n_sample
    if any(t.n_sample != n for t in traces):
        raise ValueError("The traces must all have the same number of samples.")
    if any(t.dtype.kind == 'c' for t in traces):
        raise TypeError("resstat works on real traces, but was given a complex one.")

    def numbered(key):
        values = {}
        index = np.empty(len(traces), dtype=np.int32)
        for i, trace in enumerate(traces):
            index[i] = values.setdefault(_key_of(trace, key), len(values))
        return values, index

    shots, ishot = numbered(shot)
    receivers, irec = numbered(receiver)
    cmps, icmp = numbered(cmp)
    data = np.ascontiguousarray(np.stack([np.asarray(t) for t in traces]), dtype=F32)
    source_samples, receiver_samples = _resstat.resstat(data, icmp, irec, ishot, len(cmps), len(receivers), len(shots),
                                                        ntpick, niter, mode, F32(perc), 1 if sub else 0)
    return ({k: float(source_samples[i]) * dt for k, i in shots.items()},
            {k: float(receiver_samples[i]) * dt for k, i in receivers.items()})
