"""
The complex logarithm of the spectrum and the cepstrum: SUCLOGFFT, SUICLOGFFT, SUCEPSTRUM and SUICEPSTRUM, and the
spectrum flattening of SUWFFT (``su/main/transforms``).

``clogfft`` makes a complex trace (the log of the amplitude spectrum, and the unwrapped phase, as its real and imaginary
parts), and ``iclogfft`` undoes it. Like the other transforms they use numpy's FFT of the trace as it is, made an even
length, where the SU programs pad it to a length that their FFT is fast for (and note that SU's forward kernel is
``exp(+i w t)``, the conjugate of numpy's, which is why the spectrum is conjugated for ``sign=1``).

Not supported: ``sucepstrum``'s ``smooth`` (damped least squares smoothing of the phase) and ``succepstrum`` (the complex
cepstrum), which does not finish what it does: it transforms the half spectrum as if it was a signal, and does not
update the length of its traces. ``suicepstrum`` calls its inverse transform with the number of frequencies for the
length, here it is the length of the transform that it is meant to have.
"""
import numpy as np

from ..stage import per_trace, stage
from . import _FOURIER, _UNIT, _kernels, _unwrap

__all__ = ['clogfft', 'iclogfft', 'cepstrum', 'icepstrum', 'wfft']

F32 = np.float32


def _check_sign(**signs):
    for name, value in signs.items():
        if value not in (1, -1):
            raise ValueError(f"{name} = {value} must be 1 or -1")


def _spectrum(x, n_fft, sign):
    """The spectrum of the SU transform that has exp(sign i w t) in its kernel (the real part, and the imaginary part)"""
    spectrum = np.fft.rfft(x, n_fft)
    return np.conj(spectrum) if sign == 1 else spectrum


def _real_trace(spectrum, n_fft, sign):
    """The (normalized) inverse of the transform that has exp(sign i w t) in its kernel: SU's inverse of sign = 1 is -1"""
    return np.fft.irfft(np.conj(spectrum) if sign == -1 else spectrum, n_fft)


def _log_spectrum(x, n_fft, sign, d1, mode, unwrap, trend, zeromean):
    """log|F| and the (unwrapped) phase of the spectrum of a trace"""
    spectrum = np.ascontiguousarray(_spectrum(x.astype(np.float64), n_fft, sign), dtype=np.complex64)
    log_amp, phase = _kernels.clogfft_spectrum(spectrum)  # (the function of the SU library)
    if unwrap:
        if mode == 'suphase':
            phase = _unwrap.simple(phase, int(trend), int(zeromean), F32(unwrap))
        else:
            phase = _unwrap.oppenheim(
                np.ascontiguousarray(spectrum.real, dtype=F32), np.ascontiguousarray(spectrum.imag, dtype=F32), F32(d1),
                int(trend), int(zeromean),
            )
    return log_amp.astype(np.float64), phase.astype(np.float64)


def _unwrap_parameters(mode, unwrap, **signs):
    _check_sign(**signs)
    if mode not in ('ouphase', 'suphase'):
        raise ValueError(f'unknown mode="{mode}", it is "ouphase" (Oppenheim) or "suphase" (simple)')
    if unwrap < 0:
        raise ValueError("unwrap must not be negative")


# ------------------------------------------------------------------------------------------------------- clogfft
def _clogfft(upstream, *, sign=1, mode='ouphase', unwrap=1.0, trend=True, zeromean=False, dt=None):
    _unwrap_parameters(mode, unwrap, sign=sign)

    def clogfft_trace(trace):
        n = trace.n_sample
        n_fft = n + n % 2
        sample_dt = trace.d_sample or dt or 0.004  # (as SU does, .004 if there is nothing at all)
        d1 = 1.0 / (n_fft * sample_dt)
        log_amp, phase = _log_spectrum(np.asarray(trace), n_fft, sign, d1, mode, unwrap, trend, zeromean)
        return trace.replace(
            (log_amp + 1j * phase).astype(np.complex64), d_sample=d1, sample_start=0.0, sampling_domain=_FOURIER,
        )

    return per_trace(upstream, clogfft_trace)


# SUCLOGFFT: the complex logarithm of the Fourier transform of real traces, log|F(w)| + i phi(w), where phi is the
# unwrapped phase, from 0 to the Nyquist frequency. It is a complex trace, in the Fourier domain, like that of `fft`.
#
# sign : the sign in the exponent of the transform, default 1
# mode : the way that the phase is unwrapped: 'ouphase' (the default, Oppenheim and Schafer, which integrates the
#        derivative of the phase) or 'suphase' (looks for jumps of more than pi / unwrap in the phase)
# unwrap : 1 by default, 0 does not unwrap the phase (suclogfft's documentation says that that is for 'suphase', but its
#          program does it for both)
# trend : remove the linear trend from the unwrapped phase, default True
# zeromean : the phase has zero mean, instead of starting at 0 (for 'suphase' only?), default False
# dt : sample interval (s) for traces that do not have one, default .004
#
# Where the spectrum is 0 (to 1e-5) the amplitude and phase are 0.
clogfft = stage(_clogfft, parallelism='trace', name='clogfft', validate=True)


# ------------------------------------------------------------------------------------------------------ iclogfft
def _iclogfft(upstream, *, sign=-1, sym=False):
    _check_sign(sign=sign)

    def iclogfft_trace(trace):
        if trace.dtype.kind != 'c':
            raise TypeError("iclogfft needs complex log frequency traces, from clogfft.")
        n_freq = trace.n_sample
        if n_freq < 2:
            raise ValueError("iclogfft needs traces with at least 2 frequency samples.")
        n_fft = 2 * (n_freq - 1)
        log_spectrum = np.asarray(trace)
        # (a log amplitude of exactly 0 is how a spectrum that is 0 is stored, so it is taken to be one, as in SU)
        spectrum = _kernels.iclogfft_spectrum(np.ascontiguousarray(log_spectrum, dtype=np.complex64), sym)
        df = trace.d_sample
        dt = 1.0 / (n_fft * df) if df else 0.0
        start = -(n_fft * dt / 2.0) if sym else 0.0
        return trace.replace(
            _real_trace(spectrum, n_fft, sign).astype(F32), d_sample=dt, sample_start=start, sampling_domain=_UNIT,
        )

    return per_trace(upstream, iclogfft_trace, on_complex='native')


# SUICLOGFFT: the inverse of `clogfft`: exp(log|F| + i phi), and the inverse Fourier transform of that, to real traces.
#
# sign : the sign in the exponent of the inverse transform, default -1
# sym : center the output (the transform is shifted by half of the trace), whose start time is then minus half of the
#       length of it
#
# `clogfft | iclogfft` is not quite the identity: the traces are made even in length, and a spectrum that is 1 in
# amplitude at a frequency is taken as 0 (see `clogfft`).
iclogfft = stage(_iclogfft, parallelism='trace', name='iclogfft', validate=True)


# -------------------------------------------------------------------------------------------------------- cepstrum
def _cepstrum(upstream, *, sign1=1, sign2=-1, mode='ouphase', unwrap=1.0, trend=True, zeromean=False, dt=None):
    _unwrap_parameters(mode, unwrap, sign1=sign1, sign2=sign2)

    def cepstrum_trace(trace):
        n = trace.n_sample
        n_fft = n + n % 2
        sample_dt = trace.d_sample or dt or 0.004
        d1 = 1.0 / (n_fft * sample_dt)
        log_amp, phase = _log_spectrum(np.asarray(trace), n_fft, sign1, d1, mode, unwrap, trend, zeromean)
        quefrency = _real_trace(log_amp + 1j * phase, n_fft, sign2)
        # (quefrency has the units of time, and starts at 0)
        return trace.replace(quefrency.astype(F32), d_sample=sample_dt, sample_start=0.0)

    return per_trace(upstream, cepstrum_trace)


# SUCEPSTRUM: the cepstrum of real traces: the inverse Fourier transform of the complex log of their Fourier transform,
# log|F(w)| + i phi(w) (see `clogfft`, whose parameters these are, with sign1 for the first transform and sign2 for the
# inverse one), to a real trace with the same sample interval, which is in quefrency (a time), and starts at 0.
cepstrum = stage(_cepstrum, parallelism='trace', name='cepstrum', validate=True)


# ------------------------------------------------------------------------------------------------------ icepstrum
def _icepstrum(upstream, *, sign1=1, sign2=-1, sym=False, dt=None):
    _check_sign(sign1=sign1, sign2=sign2)

    def icepstrum_trace(trace):
        n = trace.n_sample
        n_fft = n + n % 2
        sample_dt = trace.d_sample or dt or 0.004
        # the transform of the cepstrum is the complex log spectrum (log amplitude, phase), which is then exponentiated
        log_spectrum = np.ascontiguousarray(_spectrum(np.asarray(trace).astype(np.float64), n_fft, sign1), dtype=np.complex64)
        spectrum = _kernels.iclogfft_spectrum(log_spectrum, sym)
        out = _real_trace(spectrum, n_fft, sign2)
        start = -(trace.header['sample_start'] + n_fft * sample_dt / 2.0) if sym else trace.header['sample_start']
        return trace.replace(out.astype(F32), d_sample=sample_dt, sample_start=start)

    return per_trace(upstream, icepstrum_trace)


# SUICEPSTRUM: the inverse of `cepstrum`: the Fourier transform of the cepstrum is taken as the complex log spectrum
# (log amplitude, and phase), which is exponentiated and inverse transformed.
#
# sign1, sign2 : the signs in the exponents of the forward and the inverse transform, default 1 and -1
# sym : center the output (the start time is then minus half of the length of it)
# dt : sample interval (s) for traces that do not have one, default .004
icepstrum = stage(_icepstrum, parallelism='trace', name='icepstrum', validate=True)


# ----------------------------------------------------------------------------------------------------------- wfft
def _wfft(upstream, *, w0=0.75, w1=1.0, w2=0.75, sign=1, dt=None):
    _check_sign(sign=sign)

    def wfft_trace(trace):
        n = trace.n_sample
        n_fft = n + n % 2
        sample_dt = trace.d_sample or dt or 0.004
        spectrum = np.ascontiguousarray(_spectrum(np.asarray(trace).astype(np.float64), n_fft, sign), dtype=np.complex64)
        flat = _kernels.wfft_flatten(spectrum, w0, w1, w2)  # (the function of the SU library)
        return trace.replace(
            flat, d_sample=1.0 / (n_fft * sample_dt), sample_start=0.0, sampling_domain=_FOURIER,
        )

    return per_trace(upstream, wfft_trace)


# SUWFFT: the Fourier transform with the spectrum flattened: the spectrum divided by the weighted sum of the amplitude
# spectrum at the frequency and at the ones on each side of it, w0 |S(f - df)| + w1 |S(f)| + w2 |S(f + df)|.
#
# w0, w1, w2 : the weights, default .75, 1 and .75. With 0, 1 and 0 the amplitude spectrum is completely flat (1), with
#              other weights it keeps some of its shape
# sign : sign in the exponent of the transform, default 1
#
# The traces are in the Fourier domain, as with `fft` (so `wfft | ifft` follows it). The first and the last frequency
# are divided by their own amplitude only (suwfft reads past the ends of the spectrum there). Where the amplitudes are 0
# the result is 0.
wfft = stage(_wfft, parallelism='trace', name='wfft', validate=True)
