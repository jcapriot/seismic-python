"""
Vibroseis sweeps: SUVIBRO (``su/main/synthetics_waveforms_testpatterns``), one function for each of its kinds of sweep:
``vibro_linear`` (sweep=1), ``vibro_segments`` (sweep=2), ``vibro_octave`` (sweep=3), ``vibro_hertz`` (sweep=4) and
``vibro_tpower`` (sweep=5). Each makes one trace, a modulated cosine, with the sweep taper of suvibro at the ends.

Where suvibro does not do what its documentation says this does what the documentation says: sweep=5 makes the
dB per Hertz sweep (it calls that function instead of the t-power one), a dB per Hertz sweep with a constant of 0 is
a linear sweep (suvibro makes a constant, the starting frequency), and the length of a sweep with segments is the end of
the last time segment (suvibro uses ``tv`` for the length of the trace). The end taper is on the last ``t2`` of the sweep:
suvibro puts it one sample after the end.
"""
import numpy as np

from ..container import Trace, from_iterable

__all__ = ['vibro_linear', 'vibro_segments', 'vibro_octave', 'vibro_hertz', 'vibro_tpower']

_EPS = 3.8090232  # exp(-EPS * EPS) = 5e-7

_TAPERS = {
    'linear': 1, 'sine': 2, 'cosine': 3, 'gaussian': 4, 'gaussian2': 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
}


def _envelope(f, kind):
    if kind == 1:
        return f
    if kind == 2:
        return np.sin(np.pi * f / 2.0)
    if kind == 3:
        return 0.5 * (1.0 - np.cos(np.pi * f))
    if kind == 4:
        return np.exp(-((_EPS * (1.0 - f)) ** 2))
    return np.exp(-((2.0 * (1.0 - f)) ** 2))


def _taper(samples, t1, t2, taper, tv, dt):
    kind = _TAPERS.get(taper)
    if kind is None:
        raise ValueError(f"taper={taper!r} must be one of 1 to 5 or {[k for k in _TAPERS if isinstance(k, str)]}")
    n = samples.shape[0]
    n1 = int(t1 / dt + 1)
    n2 = int(t2 / dt + 1)
    if n1 > 1:
        i = np.arange(min(n1, n))
        samples[i] *= _envelope(i / n1, kind)
    if n2 > 1:
        i = np.arange(min(n2, n))
        samples[n - 1 - i] *= _envelope(i / n2, kind)
    return samples


def _check(dt, tv, t1, t2, radians, phz):
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")
    if tv <= 0.0:
        raise ValueError(f"tv={tv} must be positive")
    if t1 < 0.0 or t2 < 0.0:
        raise ValueError("The tapers t1 and t2 must not be negative")
    if t1 + t2 > tv:
        raise ValueError(f"sum of tapers t1={t1}, t2={t2} exceeds tv={tv}")
    return phz if radians else phz * 2.0 * np.pi / 360.0


def _source(samples, dt):
    trace = Trace(samples.astype(np.float32), d_sample=dt, trace_type=1).replace(trace_id=1)
    return from_iterable([trace], n_traces=1)


def _finish(samples, dt, t1, t2, taper, tv):
    if t1 != 0.0 or t2 != 0.0:
        samples = _taper(samples, t1, t2, taper, tv, dt)
    return _source(samples, dt)


def vibro_linear(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A linear sweep, from ``f1`` to ``f2`` (Hz) in ``tv`` seconds, ``dt`` apart (SUVIBRO sweep=1).

    ``phz`` is the initial phase (radians, or degrees if ``radians`` is False). ``t1`` and ``t2`` are the lengths (s)
    of the tapers at the start and the end (0 for none), of kind ``taper``: 1 or ``'linear'``, 2 ``'sine'``,
    3 ``'cosine'``, 4 ``'gaussian'`` (+-3.8) or 5 ``'gaussian2'`` (+-2.0).
    """
    phz = _check(dt, tv, t1, t2, radians, phz)
    t = np.arange(int(tv / dt + 1)) * dt
    rate = (f2 - f1) / tv
    return _finish(np.cos(2.0 * np.pi * (f1 + rate / 2.0 * t) * t + phz), dt, t1, t2, taper, tv)


def vibro_segments(*, fseg=(10.0, 60.0), tseg=(0.0, 10.0), dt=0.004, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep that is made of linear segments (SUVIBRO sweep=2): the frequency is ``fseg[i]`` at the time
    ``tseg[i]``, and changes linearly between them. The length is ``tseg[-1]`` (the times must increase)."""
    fseg = np.asarray(fseg, dtype=np.float64)
    tseg = np.asarray(tseg, dtype=np.float64)
    if fseg.ndim != 1 or fseg.shape != tseg.shape or fseg.shape[0] < 2:
        raise ValueError("fseg and tseg must have the same number (at least 2) of values")
    if (np.diff(tseg) <= 0).any():
        raise ValueError("tseg must increase monotonically")
    tv = float(tseg[-1])
    phz = _check(dt, tv, t1, t2, radians, phz)
    nt = int(tv / dt + 1)
    samples = np.zeros(nt)
    start, phase = 0, 0.0
    for i in range(len(tseg) - 1):
        length = tseg[i + 1] - tseg[i]
        m = int(nt / tv * length)
        aa = 2.0 * np.pi * fseg[i] * dt
        ab = np.pi * (fseg[i + 1] - fseg[i]) * dt * dt / length
        j = np.arange(m)
        samples[start + j] = np.cos(j * (ab * j + aa) + phase + phz)
        phase = (ab * m + aa) * m + phase
        start += m
    return _finish(samples, dt, t1, t2, taper, tv)


def vibro_octave(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, swconst=0.0, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep with a boost in decibels per octave of ``swconst`` (SUVIBRO sweep=3)."""
    phz = _check(dt, tv, t1, t2, radians, phz)
    if swconst == -6.0:
        swconst = -5.999
    power = swconst / 6.0 + 1.0
    s = (power + 1.0) / power
    k1 = f1 ** power
    k2 = (f2 ** power - f1 ** power) / tv
    t = np.arange(int(tv / dt + 1)) * dt
    samples = np.cos(2.0 * np.pi / (s * k2) * (k1 + k2 * t) ** s + phz)
    return _finish(samples, dt, t1, t2, taper, tv)


def vibro_hertz(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, swconst=0.0, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep with a boost in decibels per Hertz of ``swconst`` (SUVIBRO sweep=4). With 0 it is a linear sweep."""
    if swconst == 0.0:
        return vibro_linear(f1=f1, f2=f2, tv=tv, dt=dt, phz=phz, radians=radians, t1=t1, t2=t2, taper=taper)
    phz = _check(dt, tv, t1, t2, radians, phz)
    k1 = 20.0 / (swconst * np.log(10.0))
    k2 = (np.exp(swconst * np.log(10.0) * (f2 - f1) / 20.0) - 1.0) / tv
    if k2 == 0.0:  # (the same frequency at both ends: the boost has nothing to move)
        return vibro_linear(f1=f1, f2=f2, tv=tv, dt=dt, phz=phz, radians=radians, t1=t1, t2=t2, taper=taper)
    t = np.arange(int(tv / dt + 1)) * dt
    u = 1.0 + t * k2
    samples = np.cos(2.0 * np.pi * (f1 * t + k1 / k2 * (u * np.log(u) - u)) + phz)
    return _finish(samples, dt, t1, t2, taper, tv)


def vibro_tpower(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, swconst=0.0, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep with a boost of time to the power of ``swconst`` (SUVIBRO sweep=5)."""
    phz = _check(dt, tv, t1, t2, radians, phz)
    if swconst <= -1.0:
        raise ValueError("swconst must be more than -1")
    t = np.arange(int(tv / dt + 1)) * dt
    s = t / tv
    samples = np.cos(2.0 * np.pi * t * (f1 + (f2 - f1) / (swconst + 1.0) * s ** swconst) + phz)
    return _finish(samples, dt, t1, t2, taper, tv)
