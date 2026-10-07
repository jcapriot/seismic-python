"""
Random noise and random time shifts: SUADDNOISE (``addnoise``, ``addflatnoise``) and SUJITTER (``jitter``), from
``su/main/noise``.

These use numpy's random numbers (``seed`` makes them repeatable), so for the same seed they do not make the numbers
that SU's generator does.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import header_value, stage

__all__ = ['addnoise', 'addflatnoise', 'jitter']


# --------------------------------------------------------------------------------------------------------- noise
def _add_noise(upstream, draw, sn, seed, f, amps, dt):
    if sn <= 0:
        raise ValueError(f"sn={sn} must be positive")
    source = as_trace_iterator(upstream)
    random = np.random.default_rng(seed)
    bandlimit = f is not None or amps is not None

    def traces():
        data = list(source)
        if not data:
            return
        n = data[0].n_sample
        if any(t.n_sample != n for t in data):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in data):
            raise TypeError("This stage works on real traces, but was given a complex one.")

        # the largest amplitude of all of the data
        absmax = max((float(np.abs(np.asarray(t)).max()) if n else 0.0) for t in data)

        noise = draw(random, (len(data), n)).astype(np.float32)

        # noise that has the band of the signal
        if bandlimit:
            from ..filters import filter as band_filter

            noise_traces = [t.replace(row) for t, row in zip(data, noise)]
            filtered = list(from_iterable(noise_traces) | band_filter(f=f, amps=amps, dt=dt))
            noise = np.array([np.asarray(t) for t in filtered], dtype=np.float32)

        # a scale that gives the signal to noise ratio: the rms value of the largest signal over that of the noise
        power = float(np.mean(noise.astype(np.float64) ** 2))
        if absmax != 0.0 and power > 0.0:
            scale = (absmax / np.sqrt(2.0)) / (sn * np.sqrt(power))
        else:
            scale = 1.0
        for trace, row in zip(data, noise):
            yield trace.replace(np.asarray(trace) + np.float32(scale) * row)

    return from_iterable(traces(), n_traces=source.n_traces)


def _gauss(random, shape):
    return random.standard_normal(shape)


def _flat(random, shape):
    return 2.0 * random.random(shape) - 1.0


def _addnoise(upstream, *, sn=20, seed=None, f=None, amps=None, dt=None):
    return _add_noise(upstream, _gauss, sn, seed, f, amps, dt)


def _addflatnoise(upstream, *, sn=20, seed=None, f=None, amps=None, dt=None):
    return _add_noise(upstream, _flat, sn, seed, f, amps, dt)


# SUADDNOISE: add noise to the traces, Gaussian (``addnoise``, SU's noise=gauss) or uniform in [-1, 1]
# (``addflatnoise``, noise=flat).
#
# sn : signal to noise ratio, default 20
# seed : random number seed, by default it is not repeatable
# f, amps : frequencies and amplitudes (as in `filter`) that give the noise a band, before it is scaled
# dt : sample interval (s) for the filter, for traces that do not have one
#
# Output = signal + scale * noise, where scale = (1 / sn) (absmax / sqrt(2)) / rms(noise). absmax is the largest
# absolute value of all of the data, so the whole data set is read before the first trace is written, and the traces
# must be the same length.
addnoise = stage(_addnoise, parallelism='serial', name='addnoise', validate=True)
addflatnoise = stage(_addflatnoise, parallelism='serial', name='addflatnoise', validate=True)


# -------------------------------------------------------------------------------------------------------- jitter
def _jitter(upstream, *, min=1, max=1, pon=True, seed=None, key=None):
    if min > max:
        raise ValueError(f"min={min} is more than max={max}")
    source = as_trace_iterator(upstream)
    random = np.random.default_rng(seed)

    def new_shift():
        # (a whole number of samples, the part after the point of min + (max - min) u is cut off)
        shift = int(min + (max - min) * random.random())
        if pon:
            shift *= 1 if random.random() >= 0.5 else -1
        return shift

    def shifted(x, shift):
        out = np.zeros_like(x)
        n = x.shape[0]
        if shift == 0:
            return x.copy()
        if abs(shift) < n:
            if shift > 0:
                out[shift:] = x[:n - shift]  # (later)
            else:
                out[:n + shift] = x[-shift:]  # (earlier)
        return out

    def traces():
        shift = None
        last = None
        for trace in source:
            if key is None:
                shift = new_shift()  # a new shift for every trace
            else:
                value = header_value(trace, key)
                if shift is None or value != last:
                    shift = new_shift()  # a new shift when the key changes
                last = value
            yield trace.replace(shifted(np.asarray(trace), shift))

    return from_iterable(traces(), n_traces=source.n_traces)


# SUJITTER: random time shifts of the traces, which can simulate random statics.
#
# min, max : the least and the most (samples) of the shift, default 1. The shift is a whole number of samples, and
#            if `pon` it is positive (later) or negative (earlier) with equal chance, otherwise it is positive.
# key : by default every trace has a new shift. With a header value (or function of a trace) there is a new shift only
#       when it changes: SU's `fldr=1` is `key='ensemble_number'` or the number of the shot.
# seed : random number seed, by default it is not repeatable
#
# The data is moved and the samples that are left are 0. (sujitter leaves the samples at the start of the trace of a
# positive shift as they were, and those at the end of one of a negative shift are whatever was there.)
# Traces that change with the shift of the one before need it, so this stage can not be split across workers.
jitter = stage(_jitter, parallelism='serial', name='jitter', validate=True)
