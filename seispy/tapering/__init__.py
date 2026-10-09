"""
Tapering programs: SUTAPER, SURAMP and SUGAUSSTAPER (``su/main/tapering``), with the taper functions of the SU library (``_taper.pyx``).

Where a program plainly does not do what its documentation says (see the notes on each) this does what the
documentation says.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, header_value, per_trace, stage
from . import _taperc as _c

__all__ = ['taper', 'ramp', 'gausstaper']

F32 = np.float32
_TAPER_TYPES = (1, 2, 3, 4, 5)


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


def _taper(upstream, tbeg=0.0, tend=0.0, *, type=1, tr1=0, tr2=None, min=0.0, ntr=None):
    if min > 1.0:
        raise ValueError("min must be less than 1")
    if type not in _TAPER_TYPES:
        raise ValueError(f"taper type {type!r} must be one of {list(_TAPER_TYPES)}")
    if tr2 is None:
        tr2 = tr1
    if tr1 < 0 or tr2 < 0:
        raise ValueError("tr1 and tr2 must not be negative")
    source = as_trace_iterator(upstream)

    def trace_weight(f):
        # (only the linear taper of whole traces knows about the minimum amplitude)
        return _c.envelope(type, f, min, 1.0)

    def gen():
        total = ntr if ntr is not None else source.n_traces
        if (tr1 or tr2) and not total:
            raise ValueError("ntr is neither known from the traces nor given.")
        for k, trace in enumerate(source):
            x = np.array(trace)
            if tr1 or tr2:
                # the first tr1 traces ramp up to (but not including) the full amplitude, and the last tr2
                # ramp down to the minimum on the very last trace
                fac = 1.0
                if k < tr1:
                    fac *= trace_weight(k / tr1)
                if tr2 and k >= total - tr2:
                    fac *= trace_weight((total - 1 - k) / tr2)
                x *= F32(fac)
            if tbeg != 0.0 or tend != 0.0:
                dt_ms = trace.d_sample * 1000.0
                if not dt_ms:
                    raise ValueError("The trace has no sample interval.")
                tlen = (x.shape[0] - 1) * dt_ms
                if tbeg + tend > tlen:
                    raise ValueError(f"sum of tapers tbeg={tbeg}, tend={tend} exceeds trace length ({tlen} ms)")
                x = _on_parts(x, lambda a: _c.time_taper(a, tbeg, tend, type, dt_ms))
            yield trace.replace(x)

    return from_iterable(gen(), n_traces=source.n_traces)


def taper(tbeg=0.0, tend=0.0, *, type=1, tr1=0, tr2=None, min=0.0, ntr=None):
    """Taper the edge traces of a data panel to zero, and/or the start and end of every trace (SUTAPER).

    Parameters
    ----------
    tbeg, tend : float
        Length (ms) of the taper at the start and at the end of each trace.
    type : {1, 2, 3, 4, 5}
        1 linear, 2 sine, 3 cosine, 4 gaussian (+/-3.8), 5 gaussian (+/-2.0).
    tr1, tr2 : int
        Number of traces to taper at the beginning and at the end (``tr2`` is ``tr1`` by default). This needs to
        know the position of each trace in the stream, so such a stage can not be split across workers.
    min : float
        Minimum amplitude factor of the taper of whole traces (only for the linear taper).
    ntr : int, optional
        Number of traces, if the upstream does not know it (the end of the stream is tapered).
    """
    kwargs = dict(type=type, tr1=tr1, tr2=tr2, min=min, ntr=ntr)
    _taper((), tbeg, tend, **kwargs)  # check the parameters now
    serial = bool(tr1 or (tr2 if tr2 is not None else tr1))
    return Stage(
        _taper, tbeg, tend, parallelism='serial' if serial else 'trace', name='taper', **kwargs
    )


# ------------------------------------------------------------------------------------------------------------ ramp
def _ramp(upstream, *, tmin=None, tmax=None, dt=None):
    def ramp_trace(trace):
        n = trace.n_sample
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        t_first = trace.header['sample_start']
        t_end = t_first + (n - 1) * sample_dt
        # (suramp has (nt-1)*dt as the default for tmax, which is not a no-op if the trace does not start at 0)
        up_end = t_first if tmin is None else tmin
        down_start = t_end if tmax is None else tmax
        n1 = _nint(max(0.0, (up_end - t_first) / sample_dt))
        n2 = _nint(max(0.0, (t_end - down_start) / sample_dt))
        x = _on_parts(np.array(trace), lambda a: _c.ramp(a, n1, n2))
        return trace.replace(x)

    return per_trace(upstream, ramp_trace, on_complex='native')


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


# SURAMP: linearly taper the start and/or end of traces to zero
#
# tmin : end of the starting ramp (s), default the start of the trace
# tmax : beginning of the ending ramp (s), default the end of the trace
# dt : sample interval (s) for traces that do not have one
#
# The default is a no-op.
ramp = stage(_ramp, parallelism='trace', name='ramp', validate=True)


# ------------------------------------------------------------------------------------------------------ gausstaper
def _gausstaper(upstream, *, key='offset', x0=300.0, xw=50.0):
    if not xw:
        raise ValueError("xw must not be 0")
    x0, xw = F32(x0), F32(xw)

    def gausstaper_trace(trace):
        # (the weight is made in double precision, as in the program)
        q = (F32(header_value(trace, key)) - x0) / xw
        weight = np.exp(-(np.float64(q) ** 2))
        return trace.replace((np.asarray(trace) * weight).astype(np.asarray(trace).dtype, copy=False))

    return per_trace(upstream, gausstaper_trace, on_complex='native')


# SUGAUSSTAPER: multiply traces with a symmetrical gaussian taper, w = exp(-((key - x0) / xw)**2)
#
# key : the header value (or a function of a trace), default 'offset'
# x0 : the value of the key at the center of the taper, default 300
# xw : the width of the taper, in units of the key, default 50
#
# Unlike taper, x0 is the center of the taper rather than the edges of the data.
gausstaper = stage(_gausstaper, parallelism='trace', name='gausstaper', validate=True)
