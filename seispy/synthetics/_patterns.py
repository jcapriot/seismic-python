"""
Test patterns: SUNHMOSPIKE (a gather of spikes with a choice of moveouts) and SUADDEVENT (adds an event to traces), from
``su/main/synthetics_waveforms_testpatterns``.

Where the program does not do what its documentation says this does what the documentation says (see the notes on each).
"""
import math

import numpy as np

from ..container import Trace, from_iterable
from ..stage import per_trace, stage
from . import _events

__all__ = ['nhmospike', 'addevent']

F32 = np.float32


# ------------------------------------------------------------------------------------------------------- nhmospike
_NHMO_EVENTS = ((0.0, 100.0, 1.0), (200.0, 100.0, 1.0), (0.0, 200.0, 1.0), (120.0, 200.0, 1.0))


def _gofx(gopt, offset, intercept_off, depthref):
    """g(x) of the moveout options, in single precision as the program has it"""
    offset = F32(offset) - F32(intercept_off)
    if gopt == 1:
        return offset * offset
    if gopt == 2:
        depth = F32(depthref)
        return F32(math.sqrt(float(depth * depth + offset * offset)))
    if gopt == 4:
        return F32(abs(offset))
    return offset


def nhmospike(*, nt=300, ntr=20, dt=0.001, offref=2000.0, gopt=1, depthref=400.0, offinc=100.0, nspk=4, events=None, cdp=1):
    """A gather of spikes with a choice of non-hyperbolic moveouts, for studies of impulse responses (SUNHMOSPIKE).

    The traces have offsets ``offinc``, ``2 offinc``, ... and each has the spikes of the events, at the times

    ``t + p g(x) / g(offref)`` (all in ms, ``p`` being the moveout at the reference offset),

    rounded down to a sample (a spike that falls on sample 0 or off the end of the trace is left out).

    Parameters
    ----------
    nt, ntr : int
        Number of time samples, and of traces.
    dt : float
        Sample interval in seconds.
    offref : float
        The reference offset, at which the moveouts ``p`` are given.
    gopt : {1, 2, 3, 4}
        The moveout: 1 parabolic (``g = x^2``), 2 Foster/Mosher pseudo-hyperbolic (``g = sqrt(depthref^2 + x^2)``), 3 linear
        tau-p (``g = x``), 4 linear using the absolute value of the offset.
    depthref : float
        The reference depth for ``gopt=2``. (The program documents ``depthref`` and reads ``refdepth``, which is ignored here.)
    offinc : float
        The increment of the offset.
    nspk : int
        The number of events to use, from the start of ``events`` (the program has up to four).
    events : sequence of (p, t, a)
        The moveout ``p`` in ms at the reference offset, the intercept time ``t`` in ms and the amplitude ``a`` of each event.
        The default is the four of the program: ``(0, 100, 1), (200, 100, 1), (0, 200, 1), (120, 200, 1)`` (the program's
        documented value for the last moveout is 120, and it uses 100).
    cdp : int
        The ``ensemble_number`` of the traces.
    """
    if gopt not in (1, 2, 3, 4):
        raise ValueError("gopt must be 1, 2, 3 or 4")
    if nt < 1:
        raise ValueError(f"nt={nt} must be at least 1")
    if ntr < 0:
        raise ValueError(f"ntr={ntr} must not be negative")
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")
    events = list(_NHMO_EVENTS if events is None else events)
    if any(len(e) != 3 for e in events):
        raise ValueError("each event is (p, t, a)")
    if not 0 <= nspk <= len(events):
        raise ValueError(f"nspk={nspk} must be from 0 to the number of events ({len(events)})")
    events = events[:nspk]

    dt32 = F32(dt)
    reference = _gofx(gopt, offref, 0.0, depthref)
    if reference == 0.0:
        raise ValueError("the reference offset gives g(x) = 0")
    # the intercept times in samples, and the moveouts in samples per unit of g (in the order of the program's arithmetic)
    times = [F32(t / (1000.0 * float(dt32))) for _, t, _ in events]
    moveouts = [F32(p) / ((reference * F32(1000)) * dt32) for p, _, _ in events]
    amplitudes = [F32(a) for _, _, a in events]

    def traces():
        for itr in range(ntr):
            data = np.zeros(nt, dtype=np.float32)
            offset = F32(itr + 1) * F32(offinc)
            g = _gofx(gopt, offset, 0.0, depthref)
            for t, p, a in zip(times, moveouts, amplitudes):
                index = int(t + p * g)
                if 0 < index < nt:
                    data[index] += a
            yield Trace(data, d_sample=dt).replace(trace_id=itr + 1, ensemble_number=int(cdp), offset=float(int(offset)))

    return from_iterable(traces(), n_traces=ntr)


# ------------------------------------------------------------------------------------------------------- addevent
def _addevent(upstream, *, type='nmo', t0=1.0, vel=3000.0, amp=1.0, dt=None):
    if type not in ('nmo', 'lmo'):
        raise ValueError("type must be 'nmo' (hyperbolic) or 'lmo' (linear)")
    if vel == 0.0:
        raise ValueError("vel must not be 0")

    def add(trace):
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        offset = trace.header['offset']
        if type == 'lmo':
            time = t0 + abs(offset) / vel
        else:
            time = math.sqrt(t0 * t0 + (offset / vel) ** 2)
        x = np.array(trace)
        x += _events.spike_response(time, amp, sample_dt, trace.header['sample_start'], x.shape[0])
        return trace.replace(x)

    return per_trace(upstream, add)


# SUADDEVENT: add a linear or hyperbolic moveout event to seismic data
#
# type : 'nmo' (hyperbolic, the default) or 'lmo' (linear)
# t0 : the zero-offset intercept time in seconds, default 1
# vel : the moveout velocity in m/s, default 3000
# amp : the amplitude of the event, default 1
# dt : the sample interval in seconds, for traces that have none (the program documents this option, but reads it after it has
#      used the interval, so it never takes effect there)
#
# The event is a band-limited spike (the 8 point sinc interpolation of the SU library) at sqrt(t0^2 + (offset / vel)^2), or at
# t0 + |offset| / vel, added to every trace.
addevent = stage(_addevent, parallelism='trace', name='addevent', validate=True)
