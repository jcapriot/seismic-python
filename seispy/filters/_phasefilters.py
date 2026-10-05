"""
Filters that change the amplitude and phase of the Fourier transform of each trace: SUFRAC and SUPHASE
(``su/main/filters``). Both use numpy's Fourier transforms of the trace as it is, where the SU programs zero-pad it for
their prime-factor FFT. The SU transforms have ``exp(+i w t)`` as their forward kernel, which is the conjugate of
numpy's, so what SU multiplies a spectrum by is conjugated here.
"""
import numpy as np

from ..stage import per_trace

_FLT_MAX = float(np.finfo(np.float32).max)


# ------------------------------------------------------------------------------------------------------------ frac
def _frac(upstream, *, power=0.0, sign=-1.0, phasefac=0.0, dt=None):
    phase = phasefac * np.pi
    # the phase of the filter
    exparg = np.exp(1j * (sign * 0.5 * np.pi * power + phase))

    def frac_trace(trace):
        n = trace.n_sample
        if n == 0:
            return trace
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        omega = 2.0 * np.pi * np.arange(n // 2 + 1) / (n * sample_dt)
        if power < 0:
            omega[0] = _FLT_MAX  # (to handle omega = 0 for negative powers)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            amplitude = np.power(omega, power)
        spectrum = np.fft.rfft(np.asarray(trace), n)
        y = np.fft.irfft(spectrum * np.conj(exparg * amplitude), n)
        return trace.replace(y.astype(np.float32))

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
        spectrum = np.fft.rfft(np.asarray(trace), n)
        # as the SU transform has it: the numpy transform of a real trace is the conjugate of that
        sux = np.conj(spectrum)
        new_phase = a_radians + b_radians * np.angle(sux) + c * np.arange(sux.shape[0])
        new = np.abs(sux) * np.exp(1j * new_phase)
        y = np.fft.irfft(np.conj(new), n)
        return trace.replace(y.astype(np.float32))

    return per_trace(upstream, phase_trace)


# SUPHASE: phase manipulation by a linear transformation of the phase spectrum, new phase = a + b * old phase + c * i
# (where i is the number of the frequency).
#
# a : constant phase shift, in degrees. Default 0. (suphase's documentation says 90, and its code 0.)
# b : the slope, in degrees per degree. Default 180/pi, which is no change.
# c : the phase change per frequency sample (in radians). Default 0.
