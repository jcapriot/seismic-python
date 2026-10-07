"""
Synthetic data that does not need the SU library: SUNULL and SURANDSPIKE.
"""
import time

import numpy as np

from ..container import Trace, from_iterable
from ..noise import _rng

__all__ = ['null', 'randspike']


def null(nt, *, ntr=5, dt=0.004):
    """Null traces: all zeros (SUNULL), ``ntr`` of them (default 5) with ``nt`` samples, ``dt`` apart (default 0.004 s).

    They are useful to separate the panels of a display.
    """
    if nt < 1:
        raise ValueError(f"nt={nt} must be at least 1")
    if ntr < 0:
        raise ValueError(f"ntr={ntr} must not be negative")
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")

    def traces():
        for i in range(ntr):
            yield Trace(np.zeros(nt, dtype=np.float32), d_sample=dt).replace(trace_id=i + 1)

    return from_iterable(traces(), n_traces=ntr)


def randspike(*, n1=500, n2=100, dt=0.002, nspk=20, amax=0.2, mode=1, seed=None):
    """A small data set of random spikes (SURANDSPIKE): ``n2`` traces of ``n1`` samples (a gather), with ``nspk`` spikes
    each, with random times and amplitudes in (-``amax``, ``amax``).

    With ``mode=1`` (the default) every trace has different spikes, with ``mode=2`` they are the same on every trace.
    ``seed`` makes it repeatable, by default it is the time. The random numbers are those of the generator of the SU
    library, so the spikes are those that surandspike makes for the same seed.

    The spikes are on samples 0 to ``n1 - 2``. (surandspike puts them on samples 1 to ``n1``, one of them past the
    end of the trace.) Where two spikes are on one sample the last one is kept.
    """
    if mode not in (1, 2):
        raise ValueError("mode must be 1 or 2")
    if n1 < 2:
        raise ValueError(f"n1={n1} must be at least 2")
    if n2 < 0 or nspk < 0:
        raise ValueError("n2 and nspk must not be negative")
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")
    amax = abs(amax)
    seed = int(time.time()) if seed is None else int(seed)
    f32 = np.float32
    t_max = f32(n1 - 1) * f32(dt)

    def spikes(generator):
        data = np.zeros(n1, dtype=np.float32)
        # (a time, then an amplitude, for each spike)
        draws = generator.draw(2 * nspk).reshape(nspk, 2)
        for u_time, u_amplitude in draws:
            time_of = f32(u_time) * t_max
            data[int(time_of / f32(dt))] = 2 * (f32(0.5) - f32(u_amplitude)) * f32(amax)
        return data

    def traces():
        generator = _rng.Uniform(seed)
        data = None
        for i in range(n2):
            if mode == 1 or data is None:
                data = spikes(generator)
            yield Trace(data.copy(), d_sample=dt, trace_type=1).replace(
                trace_id=i + 1, ensemble_number=1, ensemble_trace_number=i + 1
            )

    return from_iterable(traces(), n_traces=n2)
