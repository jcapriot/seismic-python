"""
More transforms of ``su/main/transforms``: SUSLOWFT and SUSLOWIFT (a Fourier transform by the discrete sum, not an FFT),
SUPHASEVEL (a phase velocity dispersion image of a shot record), SUCCWT (the complex continuous wavelet transform) and
SURADON (the generalized Radon transform: tau-p transforms, and removal of multiples).

The Fourier transforms of ``succwt`` and ``suphasevel`` are those of ``seispy._fft`` (SU's prime-factor FFT) of the lengths
that the programs use; the Radon transform is the algorithm of ``suradon`` of the SU library (``_radon.pyx``).
"""
import warnings

import numpy as np

from .. import _fft
from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stacking import _at_midpoint
from ..stage import per_trace, stage
from . import _FOURIER, _UNIT, _radon

__all__ = ['slowft', 'slowift', 'phasevel', 'ccwt', 'radon']

F32 = np.float32


def _real_samples(trace, name):
    if trace.dtype.kind == 'c':
        raise TypeError(f"{name} works on real traces, but was given a complex one.")
    return np.ascontiguousarray(trace, dtype=F32)


# ------------------------------------------------------------------------------------------------------- slowft
def _dft(x, sign):
    """sum_k x[k] exp(sign 2 pi i k n / N) for every n, by the sum itself (in chunks of frequencies)"""
    n = x.shape[0]
    k = np.arange(n, dtype=np.int64)
    out = np.empty(n, dtype=np.complex128)
    x = x.astype(np.complex128)
    chunk = max(1, min(n, 2_000_000 // max(n, 1)))
    scale = sign * 2.0 * np.pi / n
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        # (the product k n is taken modulo N, which keeps the argument of the exponential accurate)
        phase = (np.outer(np.arange(start, stop, dtype=np.int64), k) % n) * scale
        out[start:stop] = np.exp(1j * phase) @ x
    return out


def _slowft(upstream, *, sign=1, dt=None):
    if sign not in (1, -1):
        raise ValueError(f"sign = {sign} must be 1 or -1")

    def slowft_trace(trace):
        n = trace.n_sample
        if n == 0:
            return trace
        sample_dt = trace.d_sample or dt or 0.004
        spectrum = _dft(_real_samples(trace, 'slowft'), sign)
        return trace.replace(spectrum.astype(np.complex64), d_sample=1.0 / (n * sample_dt), sample_start=0.0,
                             sampling_domain=_FOURIER)

    return per_trace(upstream, slowft_trace)


# SUSLOWFT: Fourier transform by the discrete sum (not an FFT, only for demonstration and small data)
#
# sign : the sign in the exponent of the transform, default 1
# dt : sample interval (s) for traces that do not have one, default .004
#
# The traces are not padded: the n samples become n complex samples (all of the frequencies, the second half being the
# negative ones), with the frequency spacing 1 / (n dt) as the sample interval.
slowft = stage(_slowft, parallelism='trace', name='slowft', validate=True)


def _slowift(upstream, *, sign=-1):
    if sign not in (1, -1):
        raise ValueError(f"sign = {sign} must be 1 or -1")

    def slowift_trace(trace):
        if trace.dtype.kind != 'c':
            raise TypeError("slowift needs complex frequency traces, from slowft.")
        n = trace.n_sample
        if n == 0:
            return trace
        # (the real part of the transform, divided by the number of samples)
        y = _dft(np.asarray(trace), sign).real / n
        df = trace.d_sample
        return trace.replace(y.astype(np.float32), d_sample=1.0 / (n * df) if df else 0.0, sample_start=0.0,
                             sampling_domain=_UNIT)

    return per_trace(upstream, slowift_trace, on_complex='native')


# SUSLOWIFT: inverse Fourier transform by the discrete sum, of the traces of slowft: real traces of the same length
#
# sign : the sign in the exponent of the transform, default -1
#
# The program divides by 2 / n, which makes slowft | slowift twice the input (its documentation says that it is not a
# no-op because of padding, which it does not do): it is divided by n here, so that it is the inverse.
slowift = stage(_slowift, parallelism='trace', name='slowift', validate=True)


# ----------------------------------------------------------------------------------------------------- phasevel
def _phasevel(upstream, *, fv=330.0, nv=100, dv=25.0, fmax=50.0, norm=False, dt=None, key='ensemble_number'):
    if nv < 1:
        raise ValueError("nv must be at least 1")
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = [_real_samples(t, 'phasevel') for t in group]
            nt = arrays[0].shape[0]
            if any(a.shape[0] != nt for a in arrays):
                raise ValueError("The traces of a gather must have the same number of samples.")
            first = group[0]
            sample_dt = dt or first.d_sample
            if not sample_dt:
                raise ValueError("The trace has no sample interval, and no `dt` was given.")
            offsets = np.abs(np.array([t.header['offset'] for t in group], dtype=F32))
            if not offsets.any():
                raise ValueError("The offsets are not set (they are 0): the traces need source and receiver positions.")
            n_fft = _fft.good_length(nt)
            n_freq = n_fft // 2 + 1
            spectra = _fft.rfft(np.stack(arrays), n_fft, axis=1)  # (the SU transform: exp(+i w t))
            # the frequencies: from 0 to fmax (not including it), or all of them
            domega = F32(F32(2.0 * np.pi) * F32(1.0 / n_fft) / F32(sample_dt))
            n_out = n_freq if fmax == 0 else min(n_freq, int(F32(2.0 * np.pi) * F32(fmax) / domega))
            spectra = spectra[:, :n_out]
            if norm:
                amplitude = np.abs(spectra)
                spectra = np.where(amplitude > 0, spectra / np.where(amplitude > 0, amplitude, 1), 0).astype(np.complex64)
            omega = np.arange(n_out, dtype=np.float64) * float(domega)
            df = 1.0 / (n_fft * sample_dt)
            for iv in range(nv):
                v = F32(fv) + F32(iv) * F32(dv)
                # the sum over the traces of the data, shifted by the phase of a wave of this phase velocity
                phase = -(omega / float(v))[None, :] * offsets.astype(np.float64)[:, None]
                row = (spectra * np.exp(1j * phase)).sum(axis=0).astype(np.complex64)
                yield first.replace(row, d_sample=df, sample_start=0.0, sampling_domain=_FOURIER,
                                    ensemble_trace_number=iv + 1, **_at_midpoint(first))

    return from_iterable(traces())


# SUPHASEVEL: multi-mode phase velocity dispersion map from shot records (Park, Miller and Xia, 1998)
#
# fv, nv, dv : the first phase velocity (m/s) and the number and the spacing of them, default 330, 100 and 25
# fmax : the greatest frequency (Hz) that is processed, default 50; 0 for all of them
# norm : divide the spectrum of each trace by its amplitude, default False (it makes aliasing and artifacts worse)
# dt : the sample interval for traces that do not have one
# key : the header value that the traces of a gather share, default 'ensemble_number'
#
# Each gather gives nv complex traces of frequencies (with the frequency spacing as their sample interval, in the
# Fourier domain), for the phase velocities fv + k dv, k = 0, ..., nv - 1 (``ensemble_trace_number`` is k + 1). The
# traces need offsets (``offset`` of the header, the sign does not matter).
phasevel = stage(_phasevel, parallelism='ensemble', name='phasevel', validate=True)


# --------------------------------------------------------------------------------------------------------- ccwt
def _ccwt(upstream, *, noct=5, nv=10, fmax=None, p=-0.5, c=None, k=1, fs=1, df=1.0, dt=None):
    if noct < 1 or nv < 1:
        raise ValueError("noct and nv must be at least 1")
    if k not in (1, 2):
        warnings.warn("Bad kernal wavelet choice ... use Morlet")
        k = 1
    if fs not in (1, 2):
        raise ValueError(f"fs = {fs} must be 1 or 2")
    if df <= 0.0:
        raise ValueError("df must be positive")
    source = as_trace_iterator(upstream)

    def traces():
        for trace in source:
            x = _real_samples(trace, 'ccwt')
            n = x.shape[0]
            if n == 0:
                continue
            sample_dt = dt or trace.d_sample
            if not sample_dt:
                raise ValueError("The trace has no sample interval, and no `dt` was given.")
            f_max = 1.0 / (2.0 * sample_dt) if fmax is None else fmax
            damping = 1.0 / (2.0 * f_max) if c is None else c
            n_scales = noct * nv
            if fs == 2:
                n_scales = int(int(f_max) / df)
                scales = np.array([n_scales / (i + 1.0) for i in range(n_scales)])
            else:
                scales = 2.0 ** (np.arange(n_scales) // nv + (np.arange(n_scales) % nv) / nv)
            if n_scales < 1:
                raise ValueError("There are no scales: fmax must be at least df.")
            n_pad = _fft.good_length(n, real=False)
            # the wavelets are centered on time 0, circularly (the program puts them at the middle of the transform and
            # rotates them: with an odd length and with the last sample it is off by one)
            index = np.arange(n_pad)
            tau = np.where(index <= n_pad // 2, index, index - n_pad) * sample_dt
            spectrum = _fft.fft(x, n_pad)
            ctr = F32(trace.header['trace_id'])
            for ia, a in enumerate(scales):
                center = f_max / a
                phase = np.exp(-2j * np.pi * center * tau)
                if k == 2:
                    wavelet = (a ** p) * phase
                else:
                    wavelet = (a ** p) * phase * np.exp(-(tau / (a * damping)) ** 2)
                basis = _fft.fft(wavelet.astype(np.complex64), n_pad)
                # (divided by the length: the program's inverse transform is not)
                row = _fft.ifft(basis * spectrum, sign=-1)[:n]
                yield trace.replace(row, ensemble_number=int(ctr), ensemble_trace_number=ia + 1)

    return from_iterable(traces())


# SUCCWT: complex continuous wavelet transform, a panel of complex traces for each trace, one for each scale
#
# noct, nv : the number of octaves (5) and of voices per octave (10): the scales are 2 ** (octave + voice / nv)
# fmax : the highest frequency (Hz) of the transform, default the Nyquist frequency
# p : the power of the scale that normalizes the transform, default -.5; 0 preserves the amplitude of a spectral
#     decomposition, -1 gives (within a constant factor) the inverse transform
# c : the damping parameter of the Gaussian of the Morlet wavelet in time, default 1 / (2 fmax)
# k : the kernel, 1 the complex Morlet wavelet (default), 2 the Fourier kernel exp(i 2 pi f t)
# fs : 1 for dyadic sampling of the frequencies, as above, 2 for linear, with ``int(fmax) / df`` frequencies fmax k / n
# df : the frequency interval of fs=2 (the program has it fixed at 1 in the scales, but it gives the number of them)
# dt : the sample interval for traces that do not have one
#
# The scale of ``ensemble_trace_number`` k (from 1) is ``2 ** ((k - 1) // nv + ((k - 1) % nv) / nv)``, or ``n / k``
# for fs=2, and its frequency ``fmax`` over that. In the headers of a panel ``ensemble_number`` is the ``trace_id``
# of the trace that it is from.
ccwt = stage(_ccwt, parallelism='trace', name='ccwt', validate=True)


# -------------------------------------------------------------------------------------------------------- radon
_RADON_CHOICES = (0, 1, 2, 3, 4)


def _radon_stage(upstream, *, choose=0, igopt=1, offref=2000.0, interoff=0.0, pmin=-200.0, pmax=400.0, dp=16.0,
                 pmula=80.0, pmulb=200.0, depthref=500.0, nwin=1, f1=60.0, f2=80.0, prewhite=0.1, ltaper=7, ninterp=0,
                 freq1=4.0, freq2=20.0, lagc=400, lent=5, lenx=7, xopt=1, muts=None, offsets=None, dt=None,
                 key='ensemble_number'):
    if choose not in _RADON_CHOICES:
        raise ValueError(f"choose = {choose} must be one of {_RADON_CHOICES}")
    if igopt not in (1, 2, 3, 4):
        raise ValueError(f"igopt = {igopt} must be 1, 2, 3 or 4")
    if dp == 0.0:
        raise ValueError("dp must not be 0")
    if ninterp < 0:
        raise ValueError("ninterp must not be negative")
    if choose == 4 and offsets is None:
        raise ValueError("choose=4 (the inverse transform) needs the offsets of the traces that it makes")
    if f2 <= f1:
        raise ValueError(f"f2={f2} must be more than f1={f1}")
    nk = max(int(nwin), 1)
    source = as_trace_iterator(upstream)

    fac = F32(1000.0) * F32(_radon.gofx(igopt, offref, interoff, depthref))
    pmin_n, pmax_n, dp_n = F32(pmin) / fac, F32(pmax) / fac, F32(dp) / fac
    pmula_n, pmulb_n = F32(pmula) / fac, F32(pmulb) / fac
    n_p = 1 + int((pmax_n - pmin_n) / dp_n)
    if n_p < 1:
        raise ValueError("Range of pmin and pmax invalid")
    ipa = int((pmula_n - pmin_n) / dp_n)
    ipb = int((pmulb_n - pmin_n) / dp_n)
    if choose in (0, 3, 4) or ipa < 0:
        ipa = 0
    ipb = min(ipb, n_p)
    # (the program's prewhitening is a fraction where its documentation says that it is a percent)
    white = F32(prewhite) / F32(100.0)

    def g_of(offs):
        return np.array([_radon.gofx(igopt, float(o), interoff, depthref) for o in offs], dtype=F32)

    def mute_times(group):
        if muts is None:
            if nk > 1:
                raise ValueError("nwin > 1 needs the mute times: give muts (ms, for each trace, or a function of a trace).")
            return np.zeros(len(group), dtype=F32)
        if callable(muts):
            return np.array([muts(t) for t in group], dtype=F32)
        values = np.asarray(muts, dtype=F32)
        if values.shape[0] != len(group):
            raise ValueError("muts needs a mute time for each trace of the gather.")
        return values

    def one_gather(group):
        n_x = len(group)
        first = group[0]
        if n_x < 2:
            return list(group)
        arrays = [_real_samples(t, 'radon') for t in group]
        n_t = arrays[0].shape[0]
        if any(a.shape[0] != n_t for a in arrays):
            raise ValueError("The traces of a gather must have the same number of samples.")
        sample_dt = dt or first.d_sample
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        sample_dt = F32(sample_dt)
        n_fft = _fft.good_length(n_t)
        data = np.ascontiguousarray(np.stack(arrays), dtype=F32)

        if choose == 4:
            # the traces are the tau-p panel (one for each ray parameter)
            padded = np.zeros((n_x, n_fft), dtype=F32)
            padded[:, :n_t] = data
            out_offsets = np.asarray(offsets, dtype=F32)
            inverse = _radon.inverse(padded, g_of(out_offsets), sample_dt, pmin_n, dp_n, 0, f1, f2)
            return [first.replace(row[:n_t].copy(), ensemble_trace_number=i + 1, offset=float(out_offsets[i]))
                    for i, row in enumerate(inverse)]

        offs = np.array([t.header['offset'] for t in group], dtype=F32)
        mutes = mute_times(group)
        g = g_of(offs)
        if ninterp > 0:
            data_i = _radon.xinterpolate(data, ninterp, freq1, freq2, lagc, lent, lenx, xopt, sample_dt)
            positions = (np.arange(data_i.shape[0]) * F32(1.0 / (1 + ninterp))).astype(F32)
            index = np.arange(n_x, dtype=F32)
            g_i = g_of(np.interp(positions, index, offs))
            mutes_i = np.interp(positions, index, mutes).astype(F32)
        else:
            data_i, g_i, mutes_i = data, g, mutes
        tau_p = _radon.forward(data_i, g_i, sample_dt, n_fft, n_p, pmin_n, dp_n, mutes_i, nk, f1, f2, white)
        if choose == 0:
            return [first.replace(row[:n_t].copy(), ensemble_trace_number=i + 1, **_at_midpoint(first))
                    for i, row in enumerate(tau_p)]
        if choose == 1:
            # (the mute of the primaries that are in the zone of the multiples)
            for ip in range(ipb):
                tau_p[ip] = _radon.taupmute(np.ascontiguousarray(tau_p[ip]), ip, ipa, ipb, ltaper)
        inverse = _radon.inverse(tau_p, g, sample_dt, pmin_n, dp_n, ipa, f1, f2)
        result = inverse[:, :n_t]
        if choose == 1:
            result = data - result
        return [t.replace(np.ascontiguousarray(row, dtype=F32)) for t, row in zip(group, result)]

    def traces():
        for group in group_by(source, key):
            yield from one_gather(group)

    return from_iterable(traces())


# SURADON: the generalized Radon transform of a gather (tau-p transforms, and the removal of multiples), after Anderson
#
# choose : 0 the forward transform (a panel of ``(pmax - pmin) / dp + 1`` traces, one for each ray parameter, the trace k
#          being for ``pmin + k dp``, at the reference offset ``offref`` in ms), 1 the data minus multiples, 2 the estimate
#          of the multiples, 3 the forward and the inverse transform, 4 the inverse transform (of the traces of a panel
#          from choose=0, to the traces at ``offsets``, which is given for it: the program takes them from the headers of
#          the traces of the panel, which it set itself)
# igopt : g(x) of the transform: 1 parabolic (offset ** 2), 2 pseudo hyperbolic sqrt(depthref ** 2 + offset ** 2), 3
#         linear (offset), 4 linear with the absolute value of the offset
# offref, interoff, depthref : the reference offset (m) of the moveouts, the offset of the intercept times, and the depth
#                              for igopt=2
# pmin, pmax, dp : the range of moveouts (ms at the reference offset) and the increment, defaults -200, 400, 16
# pmula, pmulb : the moveouts where the multiples begin at the maximum time and at zero time, defaults 80 and 200
# nwin : the number of windows through the mute zone (the traces need to be sorted by offset, and ``muts`` given)
# f1, f2 : the frequencies (Hz) where the taper off starts and where it ends, defaults 60 and 80
# prewhite : the prewhitening in percent, default .1
# ltaper : the length (samples) of the taper of the mute, default 7
# ninterp : the number of traces to interpolate between traces first, along dips picked from the frequencies from freq1
#           to freq2 (defaults 4 and 20 Hz) in an AGC window of lagc (400) ms, with smoothers of lent and lenx samples
#           (5 and 7), and the spatial derivative by differences (xopt=1) or by an FFT (0): for linear tau-p of
#           aliased data
# muts : the mute times (ms) that nwin > 1 needs: for each trace of a gather, or a function of a trace
# offsets : the offsets of the traces that choose=4 makes
# dt : the sample interval for traces that do not have one
# key : the header value that the traces of a gather share, default 'ensemble_number'
#
# Where the program does not do what its documentation says this does that: ``ninterp`` is 0 by default (the program
# makes it 1, by reading the parameter a second time), ``prewhite`` is a percent (the program uses it as a fraction),
# the taper of the mute is a ramp (the program's integer division makes it 0), and the mute times of an interpolated
# gather are interpolated with it (the program indexes them as if they were of the interpolated traces). Gathers of
# one trace are passed as they are. The p values and the dp of the headers f2 and d2 are not set (there are none).
radon = stage(_radon_stage, parallelism='ensemble', name='radon', validate=True)
