"""
Vibroseis sweeps: SUVIBRO (``su/main/synthetics_waveforms_testpatterns``), one function for each of its kinds of sweep:
``vibro_linear`` (sweep=1), ``vibro_segments`` (sweep=2), ``vibro_octave`` (sweep=3), ``vibro_hertz`` (sweep=4) and
``vibro_tpower`` (sweep=5). Each makes one trace, a modulated cosine, with the sweep taper of suvibro at the ends. The sweeps and the taper are
functions of the SU library.

Where suvibro does not do what its documentation says this does what the documentation says: sweep=5 makes the
dB per Hertz sweep (it calls that function instead of the t-power one), a dB per Hertz sweep with a constant of 0 is
a linear sweep (suvibro makes a constant, the starting frequency), and the length of a sweep with segments is the end of
the last time segment (suvibro uses ``tv`` for the length of the trace). The end taper is on the last ``t2`` of the sweep:
suvibro puts it one sample after the end.
"""
import numpy as np

from . import _waveforms
from ..container import Trace, from_iterable
from ..tapering import _taperc as _taper_c

__all__ = ['vibro_linear', 'vibro_segments', 'vibro_octave', 'vibro_hertz', 'vibro_tpower']

_TAPERS = {
    'linear': 1, 'sine': 2, 'cosine': 3, 'gaussian': 4, 'gaussian2': 5,
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
}


def _taper(samples, t1, t2, taper, tv, dt):
    kind = _TAPERS.get(taper)
    if kind is None:
        raise ValueError(f"taper={taper!r} must be one of 1 to 5 or {[k for k in _TAPERS if isinstance(k, str)]}")
    # (the taper of the SU library, which is that of sutaper)
    out = np.ascontiguousarray(samples, dtype=np.float32)
    _taper_c.time_taper(out, t1, t2, kind, dt)
    return out


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


def _count(tv, dt):
    """The number of samples, tv / dt + 1 (suvibro finds it in single precision, which makes 2500 of the 2501 for tv=10 and dt=.004)"""
    return int(tv / dt + 1.0e-6) + 1


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
    return _finish(_waveforms.vibro_linear(_count(tv, dt), f1, f2, tv, dt, phz), dt, t1, t2, taper, tv)


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
    nt = _count(tv, dt)
    samples = _waveforms.vibro_segments(
        nt, fseg.astype(np.float32), np.diff(tseg).astype(np.float32), tv, dt, phz,
    )
    return _finish(samples, dt, t1, t2, taper, tv)


def vibro_octave(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, swconst=0.0, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep with a boost in decibels per octave of ``swconst`` (SUVIBRO sweep=3)."""
    phz = _check(dt, tv, t1, t2, radians, phz)
    samples = _waveforms.vibro_octave(_count(tv, dt), f1, f2, tv, dt, swconst, phz)
    return _finish(samples, dt, t1, t2, taper, tv)


def vibro_hertz(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, swconst=0.0, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep with a boost in decibels per Hertz of ``swconst`` (SUVIBRO sweep=4). With 0 it is a linear sweep."""
    phz = _check(dt, tv, t1, t2, radians, phz)
    samples = _waveforms.vibro_hertz(_count(tv, dt), f1, f2, tv, dt, swconst, phz)
    return _finish(samples, dt, t1, t2, taper, tv)


def vibro_tpower(*, f1=10.0, f2=60.0, tv=10.0, dt=0.004, swconst=0.0, phz=0.0, radians=True, t1=1.0, t2=1.0, taper=1):
    """A sweep with a boost of time to the power of ``swconst`` (SUVIBRO sweep=5)."""
    phz = _check(dt, tv, t1, t2, radians, phz)
    if swconst <= -1.0:
        raise ValueError("swconst must be more than -1")
    samples = _waveforms.vibro_tpower(_count(tv, dt), f1, f2, tv, dt, swconst, phz)
    return _finish(samples, dt, t1, t2, taper, tv)
