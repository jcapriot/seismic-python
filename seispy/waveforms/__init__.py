"""
Seismic wavelets as one trace each: SUWAVEFORM (``akb``, ``berlage``, ``gauss``, ``gaussd``, ``ricker1``, ``ricker2``,
``spike`` and ``unit``, one for each of its types), SUDGWAVEFORM (``dgauss``) and SUVIBRO (the Vibroseis sweeps
``vibro_linear``, ``vibro_segments``, ``vibro_octave``, ``vibro_hertz`` and ``vibro_tpower``), from
``su/main/synthetics_waveforms_testpatterns``. These start a pipeline, like the synthetics do::

    ricker1(fpeak=15.0) | ...

The wavelets are computed by the SU library. The traces have 1 as their ``trace_id`` and ``trace_type`` (seismic data).

The wavelets that are zero phase (``ricker1``, ``gauss`` and ``ricker2`` without distortion) have the start time of the
trace set so that the peak is at time 0, as the documentation of suwaveform says (it sets a header word that has only
whole milliseconds, here the time is exact).
"""
import numpy as np

from ..container import Trace, from_iterable
from . import _waveforms
from ._sweeps import vibro_hertz, vibro_linear, vibro_octave, vibro_segments, vibro_tpower

__all__ = [
    'akb', 'berlage', 'gauss', 'gaussd', 'ricker1', 'ricker2', 'spike', 'unit', 'dgauss',
    'vibro_linear', 'vibro_segments', 'vibro_octave', 'vibro_hertz', 'vibro_tpower',
]


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _check(dt=None, fpeak=None, ns=None):
    if dt is not None and dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")
    if fpeak is not None and fpeak <= 0.0:
        raise ValueError(f"fpeak={fpeak} must be positive")
    if ns is not None and ns < 1:
        raise ValueError(f"ns={ns} must be at least 1")


def _source(samples, dt, sample_start=0.0):
    trace = Trace(samples, d_sample=dt, sample_start=sample_start, trace_type=1).replace(trace_id=1)
    return from_iterable([trace], n_traces=1)


def _fit(samples, ns):
    """The samples, cut or padded with zeros to ns samples, if that was asked for"""
    if ns is None or ns == samples.shape[0]:
        return samples
    out = np.zeros(ns, dtype=np.float32)
    n = min(ns, samples.shape[0])
    out[:n] = samples[:n]
    return out


def akb(*, fpeak=20.0, dt=0.004, ns=None):
    """The AKB wavelet (SUWAVEFORM type=akb) of Alford, Kelly and Boore: ``fpeak`` is its maximum frequency (Hz), its
    peak frequency is about a third of that. By default it has ``4 / (fpeak dt) + 1`` samples."""
    _check(dt, fpeak, ns)
    n = _nint(4.0 / (fpeak * dt)) + 1 if ns is None else ns
    return _source(_waveforms.akb(n, dt, fpeak), dt)


def berlage(*, fpeak=20.0, dt=0.004, ns=None, ampl=1.0, tn=2.0, decay=None, ipa=-90.0):
    """The Berlage wavelet (SUWAVEFORM type=berlage).

    ``fpeak`` is the peak frequency (Hz), ``tn`` the time exponent (how it begins), ``decay`` the exponential decay
    factor (1/s, default ``4 fpeak``) and ``ipa`` the initial phase angle (degrees, -90 or 90 for zero amplitude at the
    start). The number of samples by default is an estimate of the useful length.
    """
    _check(dt, fpeak, ns)
    decay = 4.0 * fpeak if decay is None else decay
    if tn < 0.0:
        raise ValueError(f"tn={tn} must be non-negative")
    if decay < 0.0:
        raise ValueError(f"decay={decay} must be non-negative")
    if ns is None:
        if decay:
            n = _nint(np.floor(fpeak * (8.0 + 2.0 * tn) / decay) / (dt * fpeak)) + 1
        else:
            n = _nint(2.0 / (dt * fpeak)) + 1
    else:
        n = ns
    return _source(_waveforms.berlage(max(n, 1), dt, fpeak, ampl, tn, decay, np.pi * ipa / 180.0), dt)


def gauss(*, fpeak=20.0, dt=0.004, ns=None):
    """The Gaussian wavelet (SUWAVEFORM type=gauss), zero phase, with its peak at time 0 (it is made with the peak at
    ``1 / fpeak``)."""
    _check(dt, fpeak, ns)
    n = _nint(2.0 / (fpeak * dt)) + 1 if ns is None else ns
    return _source(_waveforms.gauss(n, dt, fpeak), dt, -1.0 / fpeak)


def gaussd(*, fpeak=20.0, dt=0.004, ns=None):
    """The first derivative of the Gaussian wavelet (SUWAVEFORM type=gaussd)."""
    _check(dt, fpeak, ns)
    n = _nint(2.0 / (fpeak * dt)) + 1 if ns is None else ns
    return _source(_waveforms.gaussd(n, dt, fpeak), dt)


def ricker1(*, fpeak=20.0, dt=0.004, ns=None):
    """The Ricker wavelet with peak frequency ``fpeak`` (SUWAVEFORM type=ricker1, or ricker), zero phase, with its peak
    at time 0 (it is made with the peak at ``1 / fpeak``)."""
    _check(dt, fpeak, ns)
    n = _nint(2.0 / (fpeak * dt)) + 1 if ns is None else ns
    return _source(_waveforms.ricker1(n, dt, fpeak), dt, -1.0 / fpeak)


def ricker2(*, half=None, period=None, ampl=1.0, distort=0.0, fpeak=20.0, dt=0.004, ns=None):
    """The Ricker wavelet that is given by its half length, its period and a distortion (SUWAVEFORM type=ricker2).

    ``half`` is the half length (s, default ``1 / fpeak``), ``period`` the period (s, default ``0.77969680123 half``,
    which is ``sqrt(6) / pi`` times it), and ``distort`` the distortion factor (0 for a symmetric wavelet). The
    wavelet has ``2 hlw - 1`` samples, where ``hlw`` is ``half / dt`` as a whole number of samples (suwaveform
    allocates two more, that it does not set); ``ns`` cuts it or pads it with zeros. Without distortion the peak is at
    time 0.
    """
    _check(dt, fpeak, ns)
    half = 1.0 / fpeak if half is None else half
    period = 0.77969680123 * half if period is None else period
    if half <= 0.0:
        raise ValueError(f"half={half} must be positive")
    if period <= 0.0:
        raise ValueError(f"period={period} must be positive")
    hlw = max(_nint(half / dt), 1)
    samples = _fit(_waveforms.ricker2(hlw, dt, period, ampl, distort), ns)
    return _source(samples, dt, 0.0 if distort else -(hlw - 1) * dt)


def spike(*, tspike=0.0, dt=0.004, ns=None):
    """A spike at time ``tspike`` (SUWAVEFORM type=spike), by default with 5 samples after it."""
    _check(dt, None, ns)
    if tspike < 0.0:
        raise ValueError(f"tspike={tspike} must be non-negative")
    index = _nint(tspike / dt)
    n = index + 5 if ns is None else ns
    return _source(_waveforms.spike(n, index), dt)


def unit(*, half=None, fpeak=20.0, dt=0.004, ns=None):
    """A constant, one (SUWAVEFORM type=unit), by default ``half / dt + 1`` samples, where ``half`` is ``1 / fpeak``."""
    _check(dt, fpeak, ns)
    half = 1.0 / fpeak if half is None else half
    n = _nint(half / dt) + 1 if ns is None else ns
    return _source(_waveforms.unit(n), dt)


def dgauss(*, n=2, fpeak=35.0, sign=1, nfpeak=None, shift=0.0, nt=None):
    """The n-th derivative of a Gaussian, in double precision (SUDGWAVEFORM).

    ``fpeak`` is the peak frequency (Hz) and ``sign`` the polarity (1 or -1). The highest frequency is
    ``nfpeak fpeak`` (default ``n * n``), which sets the sample interval of the trace, ``0.5 / (nfpeak fpeak)``. The
    wavelet is delayed by ``sqrt(n) / fpeak + shift`` (s), and has ``2 t0 / dt + 1`` samples by default.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if fpeak <= 0.0:
        raise ValueError(f"fpeak={fpeak} must be positive")
    if sign not in (1, -1):
        raise ValueError("sign must be 1 or -1")
    nfpeak = n * n if nfpeak is None else nfpeak
    dt = 0.5 / (nfpeak * fpeak)
    if dt < 1e-6:
        raise ValueError("single-precision exceeded: reduce nfpeak or fpeak")
    t0 = shift + np.sqrt(n) / fpeak
    n_samples = int(2 * t0 / dt + 1) if nt is None else nt
    return _source(_waveforms.dgauss(dt, n_samples, t0, fpeak, n, sign), dt)
