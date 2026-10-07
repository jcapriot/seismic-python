"""
Transforms: SUHILB, SUZEROPHASE, SUANALYTIC, SUFFT, SUIFFT and SUAMP (``su/main/transforms``), and (in
``_cepstral.py``) SUCLOGFFT, SUICLOGFFT, SUCEPSTRUM, SUICEPSTRUM and SUWFFT, and the time-frequency panels (in
``_timefreq.py``) SUST, SUGABOR and SUCWT.

The stages that make complex traces (``analytic``, ``fft``), or take them apart (``ifft``, ``real``, ``imag``, ``amp``,
``logamp``, ``phase``), use traces with complex samples (see ``Trace.dtype``), where the SU programs have traces of
pairs of floats and a trace id that says that they are complex.

Where a program plainly does not do what its documentation says this does what the documentation says. Frequency
domain operations use numpy's Fourier transforms of the trace as it is: the SU programs zero-pad every trace to a
length that their prime-factor FFT is fast for. Note that the SU transforms have ``exp(+i w t)`` as their forward
kernel, which is the conjugate of numpy's, so what SU multiplies a spectrum by is conjugated here.
"""
import numpy as np

from ..stage import per_trace, stage
from . import _hilbert

__all__ = [
    'hilb', 'zerophase', 'analytic', 'fft', 'ifft', 'real', 'imag', 'amp', 'logamp', 'phase',
    'clogfft', 'iclogfft', 'cepstrum', 'icepstrum', 'wfft', 'st', 'gabor', 'cwt',
]


# ---------------------------------------------------------------------------------------------------------- hilb
def _hilb(upstream):
    return per_trace(
        upstream, lambda trace: trace.replace(_hilbert.hilbert(np.asarray(trace))), on_complex='linear'
    )


# SUHILB: Hilbert transform, computed in the time domain, by convolution with a 61 point windowed filter
hilb = stage(_hilb, parallelism='trace', name='hilb', validate=True)


# ------------------------------------------------------------------------------------------------------ zerophase
def _zerophase(upstream, *, t0=1.0, dt=None):
    def zerophase_trace(trace):
        n = trace.n_sample
        if n == 0:
            return trace
        # (as SU does, .004 if there is nothing at all)
        sample_dt = trace.d_sample or dt or 0.004
        spectrum = np.fft.rfft(np.asarray(trace), n)
        omega = 2.0 * np.pi * np.arange(n // 2 + 1) / (n * sample_dt)
        # the amplitude spectrum with the phase of a spike at t0
        y = np.fft.irfft(np.abs(spectrum) * np.exp(-1j * omega * t0), n)
        # (the peak is at sample t0 / dt, which is time 0)
        return trace.replace(y.astype(np.float32), sample_start=-t0)

    return per_trace(upstream, zerophase_trace)


# SUZEROPHASE: convert to the zero phase equivalent (the same amplitude spectrum, and a symmetric wavelet)
#
# t0 : time (s) of the peak of the output, which is delayed by that much, and given a start time of -t0
# dt : sample interval (s) for traces that do not have one, default .004
#
# The shifting is circular, so t0 must be well inside the trace.
zerophase = stage(_zerophase, parallelism='trace', name='zerophase', validate=True)


# ------------------------------------------------------------------------------------------------------- analytic
def _analytic(upstream, *, phaserot=None):
    if phaserot is not None:
        cos_rot = np.cos(np.pi * phaserot / 180.0)
        sin_rot = np.sin(np.pi * phaserot / 180.0)

    def analytic_trace(trace):
        x = np.asarray(trace)
        quadrature = _hilbert.hilbert(x)  # (SU's Hilbert transform has the opposite sign to the usual one)
        if phaserot is None:
            samples = x + 1j * quadrature
        else:
            samples = x * cos_rot + 1j * quadrature * sin_rot
        return trace.replace(samples.astype(np.complex64))

    return per_trace(upstream, analytic_trace)


# SUANALYTIC: the analytic (complex) trace, with the Hilbert transform as the imaginary part: trace + i hilb(trace)
#
# phaserot : phase rotation in degrees, which makes the trace cos(phaserot) trace + i sin(phaserot) hilb(trace)
#
# Use `amp`, `phase`, `real` and `imag` to get real traces out of it, and the stages of `seispy.attributes` for
# instantaneous phase, frequency, etc.
analytic = stage(_analytic, parallelism='trace', name='analytic', validate=True)


# the values of the header's sampling_domain for a trace that is in time (or space), and in the Fourier domain
_UNIT, _FOURIER = 1, 2


# ------------------------------------------------------------------------------------------------------------ fft
def _fft(upstream, *, sign=1, dt=None):
    if sign not in (1, -1):
        raise ValueError(f"sign = {sign} must be 1 or -1")

    def fft_trace(trace):
        n = trace.n_sample
        # (an even number of samples, so that the inverse knows how many there were)
        n_fft = n + n % 2
        # (as SU does, .004 if there is nothing at all)
        sample_dt = trace.d_sample or dt or 0.004
        spectrum = np.fft.rfft(np.asarray(trace), n_fft)
        if sign == 1:
            # the SU transform with sign = 1 has exp(+i w t) in it, numpy's has exp(-i w t)
            spectrum = np.conj(spectrum)
        return trace.replace(
            spectrum.astype(np.complex64), d_sample=1.0 / (n_fft * sample_dt), sample_start=0.0,
            sampling_domain=_FOURIER,
        )

    return per_trace(upstream, fft_trace)


# SUFFT: FFT real time traces to complex frequency traces, from 0 to the Nyquist frequency
#
# sign : the sign in the exponent of the transform, default 1 (as in SU)
# dt : sample interval (s) for traces that do not have one, default .004
#
# The frequency traces are marked as being in the Fourier domain, with the frequency spacing as their sample interval
# and 0 as their first sample. They have nfft / 2 + 1 samples, where nfft is the number of samples of the time trace,
# made even if it was odd (sufft pads the traces to a length that its FFT is fast for).
fft = stage(_fft, parallelism='trace', name='fft', validate=True)


# ----------------------------------------------------------------------------------------------------------- ifft
def _ifft(upstream, *, sign=-1):
    if sign not in (1, -1):
        raise ValueError(f"sign = {sign} must be 1 or -1")

    def ifft_trace(trace):
        if trace.dtype.kind != 'c':
            raise TypeError("ifft needs complex frequency traces, from fft.")
        n_freq = trace.n_sample
        if n_freq < 2:
            raise ValueError("ifft needs traces with at least 2 frequency samples.")
        n = 2 * (n_freq - 1)
        spectrum = np.asarray(trace)
        # (the inverse of the transform with sign = 1 is the one with sign = -1, which is numpy's)
        y = np.fft.irfft(np.conj(spectrum) if sign == -1 else spectrum, n)
        df = trace.d_sample
        return trace.replace(
            y.astype(np.float32), d_sample=1.0 / (n * df) if df else 0.0, sample_start=0.0, sampling_domain=_UNIT,
        )

    return per_trace(upstream, ifft_trace, on_complex='native')


# SUIFFT: FFT complex frequency traces to real time traces, normalized by 1 / nfft
#
# sign : the sign in the exponent of the inverse transform, default -1 (as in SU)
#
# `fft` | `ifft` gives back the traces (with an even number of samples).
ifft = stage(_ifft, parallelism='trace', name='ifft', validate=True)


# --------------------------------------------------------------------------------------------- real, imag, amp...
def _complex_to_real(name, compute):
    """A stage that makes real traces out of complex ones"""

    def make_iterator(upstream, **kwargs):
        def trace_function(trace):
            if trace.dtype.kind != 'c':
                raise TypeError(f"{name} needs complex traces.")
            return trace.replace(compute(np.asarray(trace), **kwargs).astype(np.float32))

        return per_trace(upstream, trace_function, on_complex='native')

    make_iterator.__name__ = '_' + name
    return make_iterator


def _amplitude(z, jack=False):
    amplitude = np.abs(z)
    if jack and amplitude.shape[0]:
        amplitude[0] /= 2.0
    return amplitude


def _log_amplitude(z, jack=False):
    amplitude = _amplitude(z, jack)
    # (suamp does not take the log of the first sample, and calls the log of 0 0)
    return np.log(np.where(amplitude == 0, 1.0, amplitude)) * (amplitude != 0)


real = stage(_complex_to_real('real', lambda z: z.real), parallelism='trace', name='real')
imag = stage(_complex_to_real('imag', lambda z: z.imag), parallelism='trace', name='imag')
amp = stage(_complex_to_real('amp', _amplitude), parallelism='trace', name='amp')
logamp = stage(_complex_to_real('logamp', _log_amplitude), parallelism='trace', name='logamp')
# (atan2, which is 0 where both parts are 0)
phase = stage(_complex_to_real('phase', lambda z: np.arctan2(z.imag, z.real)), parallelism='trace', name='phase')

# SUAMP: real traces from complex ones (usually the output of `fft`):
#
# real, imag : the real and the imaginary parts
# amp : the amplitude (modulus), logamp : its natural log, phase : the phase in (-pi, pi]
# jack : (for amp and logamp) divide the value at zero frequency by 2, which is right for the transform of a causal
#        function.


# (these need _UNIT and _FOURIER, from above)
from ._cepstral import clogfft, iclogfft, cepstrum, icepstrum, wfft  # noqa: E402
from ._timefreq import st, gabor, cwt  # noqa: E402
