"""
Programs that change the time axis of traces: SUSHIFT, SURESAMP, SUREDUCE, SUNMO, SUTAUPNMO, SULOG, SUILOG, SUTSQ, SUTTOZ and
SUZTOT (``su/main/stretching_moveout_resamp``).

Where a program plainly does not do what its documentation says (see the notes on each) this does what the
documentation says. Times are in seconds, including the ``dt`` of ``shift`` (which sushift has in microseconds).
"""
import numpy as np

from ..stage import Stage, header_value, per_trace, stage
from ..container import as_trace_iterator, from_iterable
from . import _nmo, _resample, _stretch, _taupnmo

__all__ = ['shift', 'resamp', 'reduce', 'nmo', 'taupnmo', 'log', 'ilog', 'tsq', 'ttoz', 'ztot']

F32 = np.float32


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _us(seconds):
    """seconds as an integer number of microseconds"""
    return _nint(1e6 * seconds)


# ----------------------------------------------------------------------------------------------------------- shift
def _shift(upstream, *, tmin=None, tmax=None, dt=None, fill=0.0):
    source = as_trace_iterator(upstream)
    fill = F32(fill)

    def window_of(trace):
        """The window, in microseconds, as sushift gets it: from the parameters and the first trace."""
        header = trace.header
        dt_us = _us(dt) if dt is not None else _us(header['d_sample'])
        if dt_us <= 0:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        utmin = _us(tmin) if tmin is not None else _us(header['sample_start'])
        utmax = _us(tmax) if tmax is not None else utmin + trace.n_sample * dt_us
        if utmin >= utmax:
            raise ValueError(f"utmin ({utmin}) >= utmax ({utmax})")
        return utmin, utmax, dt_us

    def shift_trace(trace, utmin, utmax, dt_us):
        n_out = (utmax - utmin) // dt_us
        ut1_in = _us(trace.header['sample_start'])
        ut2_in = ut1_in + dt_us * trace.n_sample
        x = np.asarray(trace)
        out = np.full(n_out, fill, dtype=x.dtype)
        if ut1_in != utmin or ut2_in != utmax:
            # where in the input the part of it that is in the window starts (it1) and ends up in the output (it2)
            if ut1_in < utmin:
                ut1, ut2 = utmin - ut1_in, 0
                ut = (ut2_in if ut2_in <= utmax else utmax) - utmin
            else:
                ut1, ut2 = 0, ut1_in - utmin
                ut = (ut2_in if ut2_in <= utmax else utmax) - ut1_in
            it1, it2, it = int(ut1 / dt_us), int(ut2 / dt_us), int(ut / dt_us)
            it = min(it, n_out - it2, x.shape[0] - it1)
            if it > 0:
                out[it2:it2 + it] = x[it1:it1 + it]
        else:
            out[:] = x
        return trace.replace(out, sample_start=utmin * 1e-6)

    if tmin is not None and tmax is not None and dt is not None:
        # everything is given, so every trace is on its own
        window = (_us(tmin), _us(tmax), _us(dt))
        if window[0] >= window[1]:
            raise ValueError(f"utmin ({window[0]}) >= utmax ({window[1]})")
        if window[2] <= 0:
            raise ValueError("dt must be positive")
        return per_trace(source, lambda trace: shift_trace(trace, *window), on_complex='native')

    def gen():
        window = None
        for trace in source:
            if window is None:
                window = window_of(trace)  # from the first trace
            yield shift_trace(trace, *window)

    return from_iterable(gen(), n_traces=source.n_traces)


def shift(*, tmin=None, tmax=None, dt=None, fill=0.0):
    """Shift and window traces in time, to a common time axis (SUSHIFT).

    The output has the samples between ``tmin`` and ``tmax`` (s), taken from where they are in time on the input
    traces, with ``fill`` where the trace does not reach. The defaults are from the first trace: ``tmin`` is its
    start time, and ``tmax`` is ``tmin`` plus its length. ``dt`` (s) is the sample interval, by default that of
    the first trace.

    The first trace gives the window for all the others, unless all of ``tmin``, ``tmax`` and ``dt`` are given, in
    which case the stage works trace by trace and can be split across workers.

    (sushift rounds the start time to the nearest millisecond, because that is how the header is stored.)
    """
    if tmin is not None and tmax is not None and tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) >= tmax ({tmax})")
    parallel = tmin is not None and tmax is not None and dt is not None
    return Stage(
        _shift, parallelism='trace' if parallel else 'serial', name='shift', tmin=tmin, tmax=tmax, dt=dt, fill=fill,
    )


# --------------------------------------------------------------------------------------------------------- resamp
def _resamp(upstream, *, nt=None, dt=None, tmin=None, rf=None, dt_in=None):
    if rf is not None and rf < 0.0:
        raise ValueError(f"factor rf={rf:g} must be positive")
    if nt is not None and nt < 1:
        raise ValueError("nt must be at least 1")
    if dt is not None and dt <= 0.0:
        raise ValueError("dt must be positive")

    def resamp_trace(trace):
        header = trace.header
        n_in = trace.n_sample
        d_in = header['d_sample'] or dt_in
        if not d_in:
            raise ValueError("The trace has no sample interval, and no `dt_in` was given.")
        t_in = header['sample_start']
        if rf:
            n_out = nt if nt is not None else _nint(n_in * rf)
            d_out = dt if dt is not None else d_in / rf
        else:
            n_out = nt if nt is not None else n_in
            d_out = dt if dt is not None else d_in
        first = t_in if tmin is None else tmin
        if n_out < 1:
            raise ValueError("The resampled trace would have no samples.")
        y = _resample.resample(np.asarray(trace), d_in, t_in, n_out, d_out, first)
        return trace.replace(y, d_sample=d_out, sample_start=first)

    return per_trace(upstream, resamp_trace, on_complex='linear')


# SURESAMP: resample in time, by sinc interpolation (the 8 point sinc interpolation of the SU library)
#
# nt : number of samples on the output, default that of the input
# dt : sample interval on the output (s), default that of the input
# tmin : time of the first output sample, default that of the input
# rf : resampling factor, if given the defaults for nt and dt are nt_in * rf and dt_in / rf (rf > 1 is downsampling
#      and rf < 1 upsampling). Remember to filter before downsampling!
# dt_in : sample interval of the input, for traces that do not have one
#
# Times outside of the input trace are 0.
resamp = stage(_resamp, parallelism='trace', name='resamp', validate=True)


# --------------------------------------------------------------------------------------------------------- reduce
def _reduce(upstream, *, rv=8.0, dt=None):
    if rv <= 0.0:
        raise ValueError(f"rv={rv:g}, must be > 0")

    def reduce_trace(trace):
        header = trace.header
        # (sureduce takes `dt` for traces that have none, and then overwrites it with the 0 in the header)
        d = header['d_sample'] or dt
        if not d:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        shift_samples = _nint(abs(F32(header['offset'])) / F32(rv * 1000.0) / F32(d))
        x = np.array(trace)
        n = x.shape[0]
        if shift_samples < n:
            x[:n - shift_samples] = x[shift_samples:].copy()
            x[n - shift_samples:] = 0
        else:
            x[:] = 0
        return trace.replace(x)

    return per_trace(upstream, reduce_trace, on_complex='native')


# SUREDUCE: convert traces to display in reduced time, by shifting them earlier by |offset| / rv
#
# rv : reducing velocity in km/s, default 8
# dt : sample interval (s) for traces that do not have one
reduce = stage(_reduce, parallelism='trace', name='reduce', validate=True)


# ------------------------------------------------------------------------------------------------------------ nmo
def _velocity_functions(tnmo, vnmo, cdp):
    """The (cdp, tnmo, vnmo) of the velocity functions, sorted by cdp: arrays of the same length for each cdp"""
    if cdp is None:
        cdps = [0.0]
        vs, ts = [vnmo], [tnmo]
    else:
        cdps = list(np.atleast_1d(np.asarray(cdp, dtype=np.float64)))
        vs, ts = vnmo, tnmo
        if vs is None or ts is None or len(vs) != len(cdps) or len(ts) != len(cdps):
            raise ValueError("a vnmo array and a tnmo array must be given for each cdp")
    functions = []
    for v, t in zip(vs, ts):
        v = np.atleast_1d(np.asarray(1500.0 if v is None else v, dtype=np.float64))
        t = np.atleast_1d(np.asarray(0.0 if t is None else t, dtype=np.float64))
        if v.shape[0] != t.shape[0] and not (v.shape[0] == 1 and t.shape[0] == 1):
            raise ValueError("number of vnmo and tnmo values must be equal")
        if (np.diff(t) <= 0).any():
            raise ValueError("tnmo values must increase monotonically")
        functions.append((t, v))
    order = np.argsort(cdps, kind='stable')
    return np.asarray(cdps)[order], [functions[i] for i in order]


class _CdpVelocities:
    """A function of the velocity (1/v^2 for nmo, v^2 for taupnmo) at the sample times of a trace, for any cdp"""

    def __init__(self, cdps, functions, transform):
        self.cdps = cdps
        self.functions = functions
        self.transform = transform
        self.key = None
        self.table = None

    def _table(self, nt, dt, ft):
        key = (nt, dt, ft)
        if self.key != key:
            times = ft + np.arange(nt) * dt
            table = np.empty((len(self.functions), nt), dtype=np.float32)
            for i, (t, v) in enumerate(self.functions):
                table[i] = self.transform(np.interp(times, t, v, left=v[0], right=v[-1]))
            self.key, self.table = key, table
        return self.table

    def at(self, nt, dt, ft, cdp):
        """The function for this cdp: constant extrapolation outside the cdps, linear interpolation between them"""
        table, cdps = self._table(nt, dt, ft), self.cdps
        if cdp <= cdps[0] or len(cdps) == 1:
            return table[0]
        if cdp >= cdps[-1]:
            return table[-1]
        i = int(np.searchsorted(cdps, cdp, side='right')) - 1
        a1 = np.float32((cdps[i + 1] - cdp) / (cdps[i + 1] - cdps[i]))
        a2 = np.float32((cdp - cdps[i]) / (cdps[i + 1] - cdps[i]))
        return a1 * table[i] + a2 * table[i + 1]


def _nmo_trace_function(cdps, functions, smute, lmute, sscale, invert, upward):
    sloths = _CdpVelocities(cdps, functions, lambda v: 1.0 / (v * v))
    last = [None, None]  # the key of the tables, and the tables

    def nmo_trace(trace):
        header = trace.header
        nt = trace.n_sample
        dt = header['d_sample']
        if not dt:
            raise ValueError("The trace has no sample interval.")
        ft = header['sample_start']
        offset = header['offset']
        cdp = header['ensemble_number'] if len(cdps) > 1 else 0
        # the tables only change with the time axis, the offset, and (if there is more than one) the cdp
        key = (nt, dt, ft, offset, cdp)
        if last[0] != key:
            ovvt = np.ascontiguousarray(sloths.at(nt, dt, ft, cdp), dtype=np.float32)
            last[:] = [key, _nmo.NMOTables(nt, dt, ft, offset, ovvt, smute, upward, invert, sscale)]
        x = np.array(trace)
        last[1].apply(x, lmute)
        return trace.replace(x)

    return nmo_trace


def _nmo_stage(upstream, *, tnmo=None, vnmo=1500.0, cdp=None, smute=1.5, lmute=25, sscale=True, invert=False,
               upward=False):
    if smute <= 0.0:
        raise ValueError("smute must be greater than 0.0")
    if lmute < 0:
        raise ValueError("lmute must not be negative")
    cdps, functions = _velocity_functions(tnmo, vnmo, cdp)
    return per_trace(
        upstream, _nmo_trace_function(cdps, functions, smute, lmute, sscale, invert, upward), on_complex='linear'
    )


def nmo(tnmo=None, vnmo=1500.0, *, cdp=None, smute=1.5, lmute=25, sscale=True, invert=False, upward=False):
    """NMO for an arbitrary velocity function of time and CDP (SUNMO).

    The offset of each trace is its ``offset`` header value and the CDP of a trace its ``ensemble_number``.

    Parameters
    ----------
    tnmo, vnmo : arrays
        NMO times (s) and the velocities for them. Velocities are linearly interpolated in time and constant outside
        the times given. For a constant velocity just give ``vnmo=constant``.
    cdp : array, optional
        The CDPs that the velocity functions are for. Then ``tnmo`` and ``vnmo`` are lists with an array for each
        CDP. Between CDPs, 1/velocity^2 is interpolated linearly, and outside of them the first and last functions are
        used.
    smute : float
        Samples with NMO stretch exceeding this are zeroed.
    lmute : int
        Length (in samples) of the linear ramp after the stretch mute.
    sscale : bool
        Divide the output samples by the NMO stretch factor.
    invert : bool
        An (approximate) inverse NMO. Exact inverse NMO is impossible, particularly for early times at large offsets
        and for frequencies near Nyquist with large interpolation errors.
    upward : bool
        Scan upward to find the first sample to kill (for velocities that decrease with time).

    sunmo's ``voutfile`` is not supported.
    """
    kwargs = dict(cdp=cdp, smute=smute, lmute=lmute, sscale=sscale, invert=invert, upward=upward)
    _nmo_stage((), tnmo=tnmo, vnmo=vnmo, **kwargs)  # check the parameters now
    return Stage(_nmo_stage, tnmo=tnmo, vnmo=vnmo, parallelism='trace', name='nmo', **kwargs)


# --------------------------------------------------------------------------------------------------------- taupnmo
def _taupnmo_ray_parameter(p, p0, dp):
    if p is not None and dp is not None:
        raise ValueError("Give the ray parameter (p) or its spacing (dp), not both.")
    if p is not None:
        return lambda trace: header_value(trace, p)
    if dp is None:
        raise ValueError("The ray parameter of the traces is needed: p (a header name or a function of a trace), or p0 and dp.")
    # (sutaupnmo takes it from the header, f2 + (tracr - 1) * d2)
    return lambda trace: p0 + (trace.header['trace_id'] - 1) * dp


def _taupnmo_stage(upstream, *, tnmo=None, vnmo=1500.0, p=None, p0=0.0, dp=None, cdp=None, smute=1.5, lmute=25, sscale=True):
    if smute <= 0.0:
        raise ValueError("smute must be greater than 0.0")
    if lmute < 0:
        raise ValueError("lmute must not be negative")
    ray_parameter = _taupnmo_ray_parameter(p, p0, dp)
    cdps, functions = _velocity_functions(tnmo, vnmo, cdp)
    velocities = _CdpVelocities(cdps, functions, lambda v: v * v)
    last = [None, None]  # the key of the tables, and the tables

    def taupnmo_trace(trace):
        header = trace.header
        nt = trace.n_sample
        dt = header['d_sample']
        if not dt:
            raise ValueError("The trace has no sample interval.")
        ft = header['sample_start']
        ray = ray_parameter(trace)
        cdp_of_trace = header['ensemble_number'] if len(cdps) > 1 else 0
        # the tables only change with the time axis, the ray parameter, and (if there is more than one) the cdp
        key = (nt, dt, ft, ray, cdp_of_trace)
        if last[0] != key:
            vvt = np.ascontiguousarray(velocities.at(nt, dt, ft, cdp_of_trace), dtype=np.float32)
            last[:] = [key, _taupnmo.TaupNMOTables(nt, dt, ft, ray, vvt, smute, sscale)]
        x = np.array(trace)
        last[1].apply(x, lmute)
        return trace.replace(x)

    return per_trace(upstream, taupnmo_trace, on_complex='linear')


def taupnmo(tnmo=None, vnmo=1500.0, *, p=None, p0=0.0, dp=None, cdp=None, smute=1.5, lmute=25, sscale=True):
    """NMO for an arbitrary velocity function of tau and CDP, of traces in the tau-p domain (SUTAUPNMO).

    The moveout is ``tau^2 (1 - p^2 v^2)``, with ``p`` the ray parameter of the trace (in the units of 1 / v). The CDP of a
    trace is its ``ensemble_number``.

    Parameters
    ----------
    tnmo, vnmo : arrays
        NMO times (tau, s) and the velocities for them. Velocities are linearly interpolated in time and constant outside
        the times given. For a constant velocity just give ``vnmo=constant``.
    p : header name or function, optional
        The ray parameter of a trace: the name of a value in ``trace.header``, or a function of a trace.
    p0, dp : float, optional
        Instead, the ray parameter of the traces is ``p0 + (trace_id - 1) * dp``. (This is what the program does, with the
        header words f2, d2 and tracr.)
    cdp : array, optional
        The CDPs that the velocity functions are for. Then ``tnmo`` and ``vnmo`` are lists with an array for each
        CDP. Between CDPs v^2 is interpolated linearly, and outside of them the first and last functions are used.
    smute : float
        Samples with NMO stretch exceeding this are zeroed (from the first sample that has it, to the end).
    lmute : int
        Length (in samples) of the linear ramp before the muted samples.
    sscale : bool
        Divide the output samples by the NMO stretch factor. (The program applies this to the samples that it has muted, which
        does nothing; here it is applied to the others.)
    """
    kwargs = dict(p=p, p0=p0, dp=dp, cdp=cdp, smute=smute, lmute=lmute, sscale=sscale)
    _taupnmo_stage((), tnmo=tnmo, vnmo=vnmo, **kwargs)  # check the parameters now
    return Stage(_taupnmo_stage, tnmo=tnmo, vnmo=vnmo, parallelism='trace', name='taupnmo', **kwargs)


# ---------------------------------------------------------------------------------------------------- log and ilog
def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _log(upstream, *, ntmin=None, m=3, ntau=None):
    if m < 1:
        raise ValueError("m must be at least 1")
    if ntau is not None and ntau < 1:
        raise ValueError("ntau must be at least 1")
    if ntmin is not None and ntmin < 1:
        raise ValueError("ntmin must be at least 1 (the log of 0 is not a number)")
    cache = {}

    def stretch_for(nt):
        if nt not in cache:
            first = int(ntmin if ntmin is not None else 0.1 * nt)
            if first < 1:
                raise ValueError(f"ntmin={first} is less than 1: there are too few samples ({nt}) for the default")
            n_tau_temp = m * nt
            dtau = np.log(nt / first) / n_tau_temp
            n_out = ntau if ntau is not None else _next_power_of_2(n_tau_temp)
            # the fractional sample of the input that each output sample is at
            positions = (first * np.exp(np.arange(n_out, dtype=np.float32) * np.float32(dtau))).astype(np.float32)
            cache.clear()
            cache[nt] = (_stretch.Stretch(positions, nt), dtau)
        return cache[nt]

    def log_trace(trace):
        x = np.asarray(trace)
        stretch, dtau = stretch_for(x.shape[0])
        return trace.replace(stretch.apply(x), d_sample=float(dtau))

    return per_trace(upstream, log_trace)


# SULOG: log-stretch of the time axis of the traces: the output sample i is at time ntmin * exp(i dtau), in samples.
#
# ntmin : the first sample of interest, default 0.1 nt. Everything before it is lost.
# m : the length of the stretched data is at least m * nt samples, default 3
# ntau : the number of samples of the output, which is a power of 2 that is at least m * nt by default
#
# The sample interval of the output is dtau = log(nt / ntmin) / (m nt), (which is not a time). `ilog` needs nt, ntmin
# and dt to undo this (sulog writes them to a parameter file).
log = stage(_log, parallelism='trace', name='log', validate=True)


def _ilog(upstream, nt, *, ntmin=None, dt=None):
    if nt < 1:
        raise ValueError("nt must be at least 1")
    first = int(0.1 * nt) if ntmin is None else int(ntmin)
    if first < 1:
        raise ValueError("ntmin must be at least 1")
    cache = {}

    def ilog_trace(trace):
        x = np.asarray(trace)
        dtau = trace.d_sample
        if not dtau:
            raise ValueError("The trace has no sample interval (dtau).")
        key = (x.shape[0], dtau)
        if key not in cache:
            positions = np.full(nt, -1.0, dtype=np.float32)  # (the samples before the first one are 0)
            positions[first:] = np.log(np.arange(first, nt) / first) / dtau  # (the fractional sample of the stretched trace)
            cache.clear()
            cache[key] = _stretch.Stretch(positions, x.shape[0])
        out = cache[key].apply(x)
        return trace.replace(out, d_sample=dt) if dt is not None else trace.replace(out)

    return per_trace(upstream, ilog_trace)


def ilog(nt, *, ntmin=None, dt=None):
    """Inverse log-stretch of the time axis (SUILOG).

    Parameters
    ----------
    nt : int
        The number of samples of the original data (the ``nt`` that ``log`` had).
    ntmin : int, optional
        The ``ntmin`` that ``log`` had, by default ``int(0.1 * nt)``, which is its default.
    dt : float, optional
        The sample interval (s) of the original data, to give to the output (which only has the dtau of the input, otherwise).
    """
    kwargs = dict(ntmin=ntmin, dt=dt)
    _ilog((), nt, **kwargs)  # check the parameters now
    return Stage(_ilog, nt, parallelism='trace', name='ilog', **kwargs)


# ----------------------------------------------------------------------------------------------------------- tsq
def _tsq(upstream, *, tmin=None, dt=None, flag=1):
    if flag not in (1, -1):
        raise ValueError("flag must be 1 (time to time squared) or -1 (time squared to time)")

    def tsq_trace(trace):
        x = np.asarray(trace)
        nt = x.shape[0]
        d_in = trace.d_sample
        if not d_in:
            raise ValueError("The trace has no sample interval.")
        if flag == 1:
            first = 0.1 * nt * d_in if tmin is None else tmin
            d_out = first * 2.0 * d_in
            t_max = nt * d_in
            n_out = 1 + int(t_max * t_max / d_out)
            # the times (in the time squared units, which the input is sampled in) that are evenly spaced squared
            times = np.sqrt(np.arange(n_out) * d_out)
        else:
            # (sutsq has the default of dt as the input's sample interval, and then sets it to 0 if it is given)
            d_out = 0.004 if dt is None else dt
            n_out = 1 + int(np.sqrt(nt * d_in) / d_out)
            times = np.arange(n_out) ** 2 * d_out ** 2
        y = _resample.interpolate(x, d_in, 0.0, times.astype(np.float32))
        return trace.replace(y, d_sample=d_out)

    return per_trace(upstream, tsq_trace)


# SUTSQ: time-squared stretch of the time axis, which makes events from hyperbolic moveouts have constant slope.
#
# flag : 1 for the forward transform, time to time squared (the default), or -1 for the inverse
# tmin : (forward) the minimum time of interest, default 0.1 nt dt. It sets the sampling in time squared.
# dt : (inverse) the sample interval (s) of the output, default .004
#
# The output of the forward transform always starts at time squared equal to 0, whatever the start time of the trace.
tsq = stage(_tsq, parallelism='trace', name='tsq', validate=True)


# --------------------------------------------------------------------------------------------------- ttoz and ztot
_METERS = 2  # the value of the header's sampling_unit for meters


def _velocity_function(axis, v):
    """The (positions, velocities) of an interval velocity function given as arrays (or one constant velocity)"""
    v = np.atleast_1d(np.asarray(1500.0 if v is None else v, dtype=np.float64))
    axis = np.atleast_1d(np.asarray(0.0 if axis is None else axis, dtype=np.float64))
    if axis.shape != v.shape:
        raise ValueError("number of times (or depths) and velocities must be equal")
    if (np.diff(axis) <= 0).any():
        raise ValueError("the times (or depths) of the velocity function must increase")
    return axis, v


def _vf_at(axis, v, positions):
    # linear interpolation, and constant extrapolation
    return np.interp(positions, axis, v, left=v[0], right=v[-1])


def _ttoz(upstream, *, t=None, v=None, nz=None, dz=None, fz=None):
    axis, velocities = _velocity_function(t, v)
    if dz is not None and dz <= 0:
        raise ValueError("dz must be positive")
    if nz is not None and nz < 1:
        raise ValueError("nz must be at least 1")

    def ttoz_trace(trace):
        x = np.asarray(trace)
        nt = x.shape[0]
        dt = trace.d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")
        ft = trace.header['sample_start']
        # (suttoz takes the times of the samples to start at 0, whatever the time of the first one, here they are the
        # times of the samples)
        vt = _vf_at(axis, velocities, ft + np.arange(nt) * dt)
        d_z = vt.min() * dt / 2.0 if dz is None else dz
        f_z = vt[0] * ft / 2.0 if fz is None else fz
        n_z = int(1 + (nt - 1) * dt * vt.max() / (2.0 * d_z)) if nz is None else nz
        # the time of each depth, from the SU library
        tz = _stretch.ttoz_table(vt.astype(np.float32), dt, ft, int(n_z), d_z, f_z)
        y = _resample.interpolate(x, dt, ft, tz)
        return trace.replace(y, d_sample=d_z, sample_start=f_z, sampling_unit=_METERS)

    return per_trace(upstream, ttoz_trace)


# SUTTOZ: resample from time to depth
#
# t, v : times (s) and the interval velocities for them. Linearly interpolated, and constant outside. Default 1500 m/s.
# dz : the depth sampling interval of the output, by default vmin dt / 2 (which avoids aliasing)
# fz : the first depth, by default v(ft) ft / 2
# nz : number of depth samples, by default 1 + (nt - 1) dt vmax / (2 dz)
#
# The output has a sample interval of dz, starts at fz and has meters for its units.
ttoz = stage(_ttoz, parallelism='trace', name='ttoz', validate=True)


def _ztot(upstream, *, z=None, v=None, nt=None, dt=None, ft=None):
    axis, velocities = _velocity_function(z, v)
    if dt is not None and dt <= 0:
        raise ValueError("dt must be positive")
    if nt is not None and nt < 1:
        raise ValueError("nt must be at least 1")

    def ztot_trace(trace):
        x = np.asarray(trace)
        n_z = x.shape[0]
        dz = trace.d_sample
        if not dz:
            raise ValueError("The trace has no sample interval (the depth sampling interval).")
        fz = trace.header['sample_start']
        # (sutztot takes the depths of the samples to start at 0)
        vz = _vf_at(axis, velocities, fz + np.arange(n_z) * dz)
        d_t = 2.0 * dz / vz.min() if dt is None else dt
        f_t = 2.0 * fz / vz[0] if ft is None else ft
        n_t = int(1 + (n_z - 1) * 2.0 * dz / (vz.max() * d_t)) if nt is None else nt
        # the depth of each time, from the SU library
        zt = _stretch.ztot_table(vz.astype(np.float32), dz, fz, int(n_t), d_t, f_t)
        y = _resample.interpolate(x, dz, fz, zt)
        return trace.replace(y, d_sample=d_t, sample_start=f_t, sampling_unit=1)

    return per_trace(upstream, ztot_trace)


# SUZTOT: resample from depth to time
#
# z, v : depths (m) and the interval velocities for them. Linearly interpolated, and constant outside. Default 1500 m/s.
# dt : the time sampling interval of the output, by default 2 dz / vmin (which avoids aliasing)
# ft : the first time, by default 2 fz / v(fz)
# nt : number of time samples, by default 1 + (nz - 1) 2 dz / (vmax dt)
#
# The velocity is piecewise linear in depth, which has to be remembered when using this as the inverse of `ttoz`.
ztot = stage(_ztot, parallelism='trace', name='ztot', validate=True)
