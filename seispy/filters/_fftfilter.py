"""
SUFILTER: a zero-phase, sine-squared tapered frequency domain filter (``su/main/filters/sufilter.c``).

The filter is designed by ``polygonalFilter`` in the SU sources (``_filter.pyx``). The filtering is done with numpy's
Fourier transforms of the trace as it is: sufilter zero-pads each trace to a length that its prime-factor FFT is fast
for (``npfaro``), which is not needed here. The padding changes the frequency spacing a little, so the filter is a bit
different from sufilter's for a trace length that it would have padded. The filter is also rebuilt whenever the
length or sample interval of the traces changes, where sufilter builds it once from the first trace.
"""
import warnings

import numpy as np

from ..stage import per_trace
from . import _filter

# ratio of the default f1, f2, f3 and f4 to the Nyquist frequency
_DEFAULT_FRACTIONS = (0.10, 0.15, 0.45, 0.50)


def _filter_stage(upstream, *, f=None, amps=None, dt=None, inplace=False):
    if f is not None:
        f = np.atleast_1d(np.asarray(f, dtype=np.float32))
        if f.ndim != 1 or f.shape[0] == 0:
            raise ValueError("f must be a 1D array of frequencies.")
        if f.shape[0] < 2:
            warnings.warn(f"Only {f.shape[0]} value defining filter")
        if (f < 0.0).any() or (np.diff(f) < 0.0).any():
            raise ValueError("Bad filter parameters, f must be increasing and not negative.")
    if amps is not None:
        amps = np.atleast_1d(np.asarray(amps, dtype=np.float32))
        if amps.ndim != 1:
            raise ValueError("amps must be a 1D array.")
        if (amps < 0.0).any():
            raise ValueError("amp values must be positive")
        if not (amps > 0.0).any():
            raise ValueError("All amps values are zero")
        if (amps == amps[0]).all():
            warnings.warn("All amps values are the same")
        if f is not None and amps.shape[0] != f.shape[0]:
            raise ValueError("number of f values must = number of amps values")
        if f is None and amps.shape[0] != len(_DEFAULT_FRACTIONS):
            raise ValueError(
                f"number of f values must = number of amps values (the default f has {len(_DEFAULT_FRACTIONS)} values)"
            )

    cache = {}  # (number of samples, sample interval) -> the filter

    def filter_for(n, sample_dt):
        key = (n, sample_dt)
        if key not in cache:
            freqs = f
            if freqs is None:
                nyquist = 0.5 / sample_dt
                freqs = np.array([fraction * nyquist for fraction in _DEFAULT_FRACTIONS], dtype=np.float32)
            levels = amps
            if levels is None:
                # the default is a trapezoidal bandpass filter
                levels = np.ones(freqs.shape[0], dtype=np.float32)
                levels[0] = levels[-1] = 0.0
            cache.clear()
            cache[key] = _filter.design(freqs, levels, n, sample_dt)
        return cache[key]

    def filter_trace(trace):
        n = trace.n_sample
        if n == 0:
            return trace
        # (as SU does, .004 if there is nothing at all)
        sample_dt = trace.d_sample or dt or 0.004
        x = np.asarray(trace)
        y = np.fft.irfft(np.fft.rfft(x, n) * filter_for(n, sample_dt), n)
        if inplace:
            x[:] = y
            return trace
        return trace.replace(y.astype(np.float32))

    return per_trace(upstream, filter_trace, on_complex='linear')
