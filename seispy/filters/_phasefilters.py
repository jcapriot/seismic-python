"""
Filters that change the amplitude and phase of the Fourier transform of each trace: SUFRAC and SUPHASE
(``su/main/filters``). Both use the FFT of seispy (``seispy._fft``, SU's prime-factor FFT) of the trace zero-padded as the
programs do, to the length that is fast for it (``npfaro`` with a look ahead of twice the trace); the transform has
``exp(+i w t)`` as its forward kernel, as in the programs. What they do to a spectrum is a function of the SU library
(``_kernels.pyx``).
"""
import numpy as np

from .. import _fft
from ..stage import per_trace
from . import _kernels


# ------------------------------------------------------------------------------------------------------------ frac
def _frac(upstream, *, power=0.0, sign=-1.0, phasefac=0.0, dt=None):
    cache = {}

    def filter_for(n, sample_dt):
        # the complex power of i w (SU's sufrac, as a function of the SU library), for the frequencies of a transform of n samples
        key = (n, sample_dt)
        if key not in cache:
            cache.clear()
            cache[key] = _kernels.frac_filter(n // 2 + 1, n, sample_dt, power, sign, phasefac, 1.0)
        return cache[key]

    def frac_trace(trace):
        n = trace.n_sample
        if n == 0:
            return trace
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        nfft = _fft.good_length_within(n, 2 * n)
        spectrum = _fft.rfft(np.asarray(trace), nfft)
        y = _fft.irfft(spectrum * filter_for(nfft, sample_dt), nfft)[:n]
        return trace.replace(y)

    return per_trace(upstream, frac_trace)


# SUFRAC: a general (fractional) time derivative or integral, plus a phase shift. The input is causal.
#
# power : exponent of (-i w): 0 for a phase shift only, > 0 differentiates, < 0 integrates. Default 0.
# sign : the sign in front of i w. Default -1.
# phasefac : phase shift by phasefac * pi. Default 0.
# dt : sample interval (s), for traces that do not have one.
#
# g(t) = Re[IFFT{ ((sign) i w)^power FFT(f) }].  Large amplitude errors result if the data have too few points.
#
# Examples: `frac(power=.5, sign=1)` to correct 3D data for 2.5D migration, and `frac(phasefac=.25)` to correct data
# from 2D modeling for 2D migration.


# ----------------------------------------------------------------------------------------------------------- phase
def _phase(upstream, *, a=0.0, b=180.0 / np.pi, c=0.0):
    a_radians = a * np.pi / 180.0
    b_radians = b * np.pi / 180.0

    def phase_trace(trace):
        n = trace.n_sample
        if n == 0:
            return trace
        nfft = _fft.good_length_within(n, 2 * n)
        spectrum = _fft.rfft(np.asarray(trace), nfft)
        _kernels.phase_spectrum(spectrum, a_radians, b_radians, c)
        y = _fft.irfft(spectrum, nfft)[:n]
        return trace.replace(y)

    return per_trace(upstream, phase_trace)


# SUPHASE: phase manipulation by a linear transformation of the phase spectrum, new phase = a + b * old phase + c * i
# (where i is the number of the frequency).
#
# a : constant phase shift, in degrees. Default 0. (suphase's documentation says 90, and its code 0.)
# b : the slope, in degrees per degree. Default 180/pi, which is no change.
# c : the phase change per frequency sample (in radians). Default 0.
