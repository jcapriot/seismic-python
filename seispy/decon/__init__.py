"""
Deconvolution and shaping (``su/main/decon_shaping``): SUPEF and SUSHAPE (Wiener, least squares, with the Toeplitz
solver of the SU library), SUCDDECON (by division in the frequency domain), SUPHIDECON (phase inversion) and SUFXDECON
(f-x prediction filtering, from the SU library).

Not supported: supef's ``cdp`` tables (lags that change with the CDP), ``wienerout`` and ``outpar``.
"""
import warnings

import numpy as np

from .. import _fft
from ..container import as_trace_iterator, from_iterable
from ..convolution import _as_filter, _convolve, _correlate
from ..stage import Stage, per_trace
from ..transforms import _hilbert
from . import _toeplitz
from ._fxpredict import fxdecon as _kernel

__all__ = ['pef', 'shape', 'cddecon', 'phidecon', 'fxdecon']

F32 = np.float32


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _samples(wavelet):
    """The samples of a wavelet that is a Trace, or an array"""
    samples = np.array(wavelet, dtype=np.float32) if hasattr(wavelet, 'header') else wavelet
    samples = np.atleast_1d(np.asarray(samples, dtype=np.float32))
    if samples.ndim != 1 or samples.shape[0] == 0:
        raise ValueError("A wavelet must be a Trace, or a 1D array with some samples.")
    return samples


# ----------------------------------------------------------------------------------------------------------- shape
def _shape(upstream, w, d, *, nshape=None, pnoise=0.001):
    w, d = _samples(w), _samples(d)
    n_w, n_d = w.shape[0], d.shape[0]
    # (centering: the filter is applied with this shift. A C integer division, truncating towards 0)
    shift = int((n_w - n_d) / 2)
    if nshape is not None and nshape < 1:
        raise ValueError("nshape must be at least 1")
    shapers = {}

    def shaper_of(n_shape):
        if n_shape not in shapers:
            autocorr = _correlate(w, w, 0, n_shape)  # for the matrix
            crosscorr = _correlate(w, d, 0, n_shape)  # the right hand side
            if autocorr[0] == 0.0:
                raise ValueError("can't shape with zero wavelet")
            autocorr[0] *= 1.0 + pnoise  # whiten
            shapers[n_shape] = _toeplitz.solve(autocorr, crosscorr)[0]
        return shapers[n_shape]

    def shape_trace(trace):
        x = np.asarray(trace)
        n = x.shape[0]
        shaper = shaper_of(n if nshape is None else nshape)
        full = _convolve(x, shaper)
        # z[i] = sum_k shaper[k] x[i - k - shift], which is the sample i - shift of the full convolution
        out = np.zeros(n, dtype=np.float32)
        i = np.arange(n)
        j = i - shift
        valid = (j >= 0) & (j < full.shape[0])
        out[valid] = full[j[valid]]
        return trace.replace(out)

    return per_trace(upstream, shape_trace)


def shape(w, d, *, nshape=None, pnoise=0.001):
    """Wiener shaping filter (SUSHAPE): a filter that turns the wavelet ``w`` into the desired wavelet ``d``, in the
    least squares sense, applied to every trace.

    Parameters
    ----------
    w : array or Trace
        The input wavelet.
    d : array or Trace
        The desired output wavelet.
    nshape : int
        Length of the shaping filter, by default the length of the trace.
    pnoise : float
        Relative additive (white) noise level, default .001.
    """
    kwargs = dict(nshape=nshape, pnoise=pnoise)
    _shape((), w, d, **kwargs)  # check the parameters now
    return Stage(_shape, w, d, parallelism='trace', name='shape', **kwargs)


# ------------------------------------------------------------------------------------------------------------ pef
def _pef(upstream, *, minlag=None, maxlag=None, pnoise=0.001, mincorr=None, maxcorr=None, mix=None):
    weights = np.asarray([1.0] if mix is None else mix, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] == 0:
        raise ValueError("mix must be a 1D array of weights")
    n_mix = weights.shape[0]
    weights = (weights / n_mix).astype(np.float32)  # (divided by the number of traces that are averaged)
    for name, value in (('minlag', minlag), ('maxlag', maxlag), ('mincorr', mincorr), ('maxcorr', maxcorr)):
        if value is not None and value < 0:
            raise ValueError(f"{name}={value} must not be negative")
    if minlag is not None and maxlag is not None and minlag > maxlag:
        raise ValueError(f"minlag={minlag} is more than maxlag={maxlag}")
    if mincorr is not None and maxcorr is not None and mincorr >= maxcorr:
        raise ValueError(f"mincorr={mincorr}, maxcorr={maxcorr}")

    # the autocorrelations (whitened, before they are mixed) of the traces before this one, the latest first
    history = []

    def pef_trace(trace):
        x = np.asarray(trace)
        nt = x.shape[0]
        dt = trace.d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")
        i_minlag = 1 if minlag is None else _nint(minlag / dt)
        i_maxlag = _nint(0.05 * nt) if maxlag is None else _nint(maxlag / dt)
        i_mincorr = 0 if mincorr is None else _nint(mincorr / dt)
        i_maxcorr = nt if maxcorr is None else _nint(maxcorr / dt)
        if i_maxcorr > nt:
            raise ValueError(f"maxcorr={maxcorr} too large")
        if i_mincorr >= i_maxcorr:
            raise ValueError(f"mincorr={mincorr}, maxcorr={maxcorr}")
        n_lag = i_maxlag - i_minlag + 1
        if n_lag < 1:
            raise ValueError(f"The filter has no samples: lags {i_minlag} to {i_maxlag}.")
        l_corr = i_maxlag + 1

        # the autocorrelation in the window (that has one more sample than i_maxcorr - i_mincorr, as in supef)
        window = x[i_mincorr:i_maxcorr + 1]
        autocorr = _correlate(window, window, 0, l_corr)
        if autocorr[0] == 0.0:
            return trace  # (a trace of zeros is left alone, and does not count as one of the traces that are mixed)
        autocorr[0] *= 1.0 + pnoise  # whiten

        # a weighted moving average of the autocorrelations of this trace and the ones before it
        if history and history[0].shape[0] != l_corr:
            history.clear()
        mixed = autocorr * weights[0]
        for k, past in enumerate(history[:n_mix - 1], start=1):
            mixed += past * weights[k]
        history.insert(0, autocorr.copy())
        del history[n_mix - 1:]

        # the Wiener filter, and the prediction error filter that is 1 and then minus the Wiener filter
        wiener = _toeplitz.solve(np.ascontiguousarray(mixed[:n_lag]), np.ascontiguousarray(mixed[i_minlag:i_minlag + n_lag]))[0]
        kernel = np.zeros(i_maxlag + 1, dtype=np.float32)
        kernel[i_minlag:] = wiener
        out = x - _convolve(x, kernel, nt)
        return trace.replace(out)

    return per_trace(upstream, pef_trace)


def pef(*, minlag=None, maxlag=None, pnoise=0.001, mincorr=None, maxcorr=None, mix=None):
    """Wiener (least squares) predictive error filtering (SUPEF), also deconvolution to a spike.

    Parameters
    ----------
    minlag : float
        First lag of the prediction filter (s), by default one sample. It is the gap of gapped (predictive)
        deconvolution.
    maxlag : float
        Last lag of the prediction filter (s), by default 5% of the length of the trace. Spiking deconvolution is a
        ``minlag`` of one sample, and a ``maxlag`` that is the width of the autocorrelation wavelet.
    pnoise : float
        Relative additive noise level, default .001.
    mincorr, maxcorr : float
        The start and the end (s) of the window that the autocorrelation is found in, by default the whole trace.
    mix : array
        Weights of a moving average of the autocorrelations of the traces (for example [1, 2, 1]), by default none. This
        needs the traces in order, so such a stage can not be split across workers.
    """
    kwargs = dict(minlag=minlag, maxlag=maxlag, pnoise=pnoise, mincorr=mincorr, maxcorr=maxcorr, mix=mix)
    _pef((), **kwargs)  # check the parameters now
    mixing = mix is not None and len(np.atleast_1d(mix)) > 1
    return Stage(_pef, parallelism='serial' if mixing else 'trace', name='pef', **kwargs)


# ------------------------------------------------------------------------------------------------------- cddecon
def _complex_division(x, f, pnoise, sym):
    """The trace x deconvolved by the filter f by division in the frequency domain, with white noise added"""
    n_t, n_f = x.shape[0], f.shape[0]
    half = n_f // 2  # (NINT of an integer division)
    # the length of the program: padded, with room for the shift of the data by half of the filter, and for the filter
    need = max(n_t + half, n_f)
    n_fft = _fft.good_length_within(n_t, 5 * n_t)
    if n_fft < need:
        n_fft = _fft.good_length_within(need, 5 * need)
    padded = np.zeros(n_fft, dtype=np.float32)
    padded[half:half + n_t] = x
    filt = np.zeros(n_fft, dtype=np.float32)
    filt[:n_f] = f
    ct = _fft.rfft(padded)
    cf = _fft.rfft(filt)
    power = cf.real.astype(np.float64) ** 2 + cf.imag.astype(np.float64) ** 2
    delta = pnoise * power.mean()
    out = (ct * np.conj(cf) / (power + delta)).astype(np.complex64)
    if sym:
        out[1::2] = -out[1::2]  # (flipping the sign of every other frequency is a shift by half of the transform)
    y = _fft.irfft(out, n_fft)
    return y[half:half + n_t]


def _cddecon(upstream, filter, *, panel=False, pnoise=0.001, sym=False):
    if pnoise < 0.0:
        raise ValueError(f"pnoise={pnoise} must not be negative")
    source = as_trace_iterator(upstream)

    def decon(trace, samples, start):
        out = _complex_division(np.asarray(trace, dtype=np.float32), samples, pnoise, sym)
        # (the time delay of the filter is taken from the one of the trace)
        return trace.replace(out, sample_start=trace.header['sample_start'] - start)

    if not panel:
        samples, start = _as_filter(filter)
        return per_trace(source, lambda trace: decon(trace, samples, start))

    def gen():
        filters = iter(filter)
        for trace in source:
            one = next(filters, None)
            if one is None:
                warnings.warn("There are fewer filters than traces, the later traces are left as they are.")
                yield trace
                continue
            yield decon(trace, *_as_filter(one))

    return from_iterable(gen(), n_traces=source.n_traces)


def cddecon(filter, *, panel=False, pnoise=0.001, sym=False):
    """Deconvolution with a filter that is given, by complex division in the frequency domain (SUCDDECON).

    The spectrum of each trace is multiplied by the conjugate of the spectrum of the filter and divided by its power
    spectrum plus a white noise term, ``pnoise`` times the mean of the power spectrum of the filter.

    Parameters
    ----------
    filter : array or Trace
        The filter. If it is a Trace its start time is subtracted from the start times of the traces (which makes this
        the inverse of ``conv``, which adds it).
    panel : bool
        ``filter`` is an iterable of Traces (or arrays), one for each trace, which are matched up in order. This needs
        to know the position of each trace, so it can not be split across workers.
    pnoise : float
        The white noise factor, default .001.
    sym : bool
        Center the output on each trace.

    Where the program does not do what its documentation says this does: the noise term is the mean of the power
    spectrum, of the filter of each trace (the program adds up the amplitudes, and keeps adding them from trace to trace,
    so that the noise grows with the number of traces), and the transform is divided by its length (the program's output is
    that many times too big). The traces keep their length (the program's documentation says that they get longer, its code
    does not do that).
    """
    if pnoise < 0.0:
        raise ValueError(f"pnoise={pnoise} must not be negative")
    if not panel:
        _as_filter(filter)  # check it now
    return Stage(_cddecon, filter, parallelism='serial' if panel else 'trace', name='cddecon', panel=panel,
                 pnoise=pnoise, sym=sym)


# ------------------------------------------------------------------------------------------------------ phidecon
def _phidecon(upstream, *, tm=-0.1, tp=0.4, pnoise=0.001, pad=50.0, dt=None):
    if pnoise < 0.0 or pad < 0.0:
        raise ValueError("pnoise and pad must not be negative")

    def phidecon_trace(trace):
        x = np.asarray(trace, dtype=np.float32)
        n_t = x.shape[0]
        if n_t == 0:
            return trace
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval.")
        # (C: float / float, converted to int by truncation)
        itmin = int(-F32(tp) / F32(sample_dt))
        itmax = int(-F32(tm) / F32(sample_dt))
        n_wind = itmax - itmin + 1
        if n_wind < 1:
            raise ValueError(f"tm={tm} must be less than tp={tp}")
        # the pseudo cepstrum: padded, from npfaro (the program's npfao can give a length that the real transform does not
        # take)
        n_padded = _nint(n_t * (1.0 + pad / 100.0))
        n_fft = _fft.good_length_within(n_padded, 4 * n_padded)
        if n_wind > n_fft:
            raise ValueError(f"The window of tm={tm} to tp={tp} is longer than the transform.")
        n_f = n_fft // 2 + 1
        window = (0.5 * (1.0 - np.cos(2.0 * np.pi * (np.arange(n_wind) + 1) / (n_wind + 1)))).astype(np.float32)

        scaled = np.zeros(n_fft, dtype=np.float32)
        scaled[:n_t] = x * F32(1.0 / n_fft)
        spectrum = _fft.rfft(scaled, sign=-1)
        real = np.ascontiguousarray(spectrum.real, dtype=np.float32)
        # (the imaginary part is the Hilbert transform of the real part, along frequency)
        imag = (F32(-1.0 / np.pi) * _hilbert.hilbert(real)).astype(np.float32)
        z = real.astype(np.float64) + 1j * imag.astype(np.float64)
        modulus = np.abs(z)
        logged = np.zeros(n_f, dtype=np.complex128)
        nonzero = modulus != 0.0
        logged[nonzero] = np.log(z[nonzero])
        full = np.zeros(n_fft, dtype=np.complex64)
        full[:n_f] = logged
        full[n_f:] = np.conj(logged[n_f - 2:0:-1])
        cepstrum = _fft.fft(full, sign=1)  # (not divided by the length)
        # a hanning window from itmin to itmax (the times -tp to -tm), wrapped around zero
        cepstrum = np.roll(cepstrum, itmax + 1)
        cepstrum[:n_wind] *= window
        cepstrum[n_wind:] = 0.0
        cepstrum = np.roll(cepstrum, -(itmax + 1))
        wavelet = np.exp(_fft.ifft(cepstrum, sign=-1).astype(np.complex128))
        wavelet[~(np.isfinite(wavelet.real) & np.isfinite(wavelet.imag))] = 0.0
        wavelet = wavelet.astype(np.complex64)

        # deconvolution: the transform of the trace is divided by that of the wavelet (with the real part increased by a
        # fraction of the mean amplitude)
        padded_trace = np.zeros(n_fft, dtype=np.float32)
        padded_trace[:n_t] = x
        ct = _fft.rfft(padded_trace, sign=-1) * F32(1.0 / n_fft)
        w = wavelet[:n_f]
        prewhite = pnoise * (np.abs(w).astype(np.float64).sum() / n_f)
        denominator = (w.real.astype(np.float64) + prewhite) + 1j * w.imag.astype(np.float64)
        ratio = (ct / denominator).astype(np.complex64)
        out = _fft.irfft(ratio, n_fft, sign=1, scale=False)  # (as the program: not divided by the length)
        return trace.replace(out[:n_t])

    return per_trace(upstream, phidecon_trace)


def phidecon(*, tm=-0.1, tp=0.4, pnoise=0.001, pad=50.0, dt=None):
    """Phase inversion deconvolution (SUPHIDECON), of Lichman and Northwood.

    The wavelet is separated from the reflectivity (and noise) by their different smoothness in a pseudo cepstrum
    domain, which keeps the wavelet and the multiples that are in the window from ``-tp`` to ``-tm``. It is then taken
    out of the trace in the frequency domain.

    Parameters
    ----------
    tm : float
        The time (s) before zero that the wavelet has (its maximum phase part), default -.1.
    tp : float
        The time (s) after zero (its minimum phase part and the multiples), default .4.
    pnoise : float
        Pre-whitening: a fraction of the mean amplitude spectrum of the wavelet that is added to it before it is divided
        out, default .001.
    pad : float
        The percentage of the length of the trace that it is padded by before the cepstrum is found, default 50.
    dt : float
        The sample interval, for traces that do not have one.

    The parameters that the program documents do not match its code. The code has ``pad`` (default .1) as the
    pre-whitening percentage (that is ``pnoise`` here, .001 as the documentation says) and a padding of 50% that can not
    be set (here ``pad``, which is a percentage as the documentation says, but 50 by default). The pre-whitening of the
    program is also reduced every trace, so that by the second trace it is nothing: that is not done here. The amplitudes
    are not calibrated (as in the program they depend on the length of the transform, because the cepstrum window
    attenuates the constant term): the relative amplitudes of the events are what is kept.
    """
    kwargs = dict(tm=tm, tp=tp, pnoise=pnoise, pad=pad, dt=dt)
    _phidecon((), **kwargs)  # check the parameters now
    return Stage(_phidecon, parallelism='trace', name='phidecon', **kwargs)


# ------------------------------------------------------------------------------------------------------- fxdecon
def _fxdecon(upstream, *, taper=0.1, fmin=6.0, fmax=None, twlen=None, ntrw=10, ntrf=4, dt=None):
    if taper < 0.0 or ntrw < 0 or ntrf < 0:
        raise ValueError("taper, ntrw and ntrf must not be negative")
    source = as_trace_iterator(upstream)

    def traces():
        traces = list(source)
        if not traces:
            return
        first = traces[0]
        n_sample = first.n_sample
        if any(t.n_sample != n_sample for t in traces):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in traces):
            raise TypeError("fxdecon works on real traces, but was given a complex one.")
        # (as the program: dt given overrides the header, and .004 if there is none)
        sample_dt = dt if dt is not None else (first.d_sample or 0.004)
        if n_sample == 0:
            yield from traces
            return
        n_taper = taper if taper != 0.0 else 0.004
        f_max = 0.6 / (2 * sample_dt) if fmax is None or fmax == 0.0 else fmax
        window = (n_sample - 1) * sample_dt if twlen is None else twlen
        window = max(window, 0.3)  # (the program says that a window can not be shorter than that)
        data = np.ascontiguousarray(np.stack([np.asarray(t) for t in traces]), dtype=np.float32)
        out = _kernel(data, F32(sample_dt), F32(n_taper), F32(6.0 if fmin == 0.0 else fmin), F32(f_max), F32(window),
                      ntrw or 10, ntrf or 4)
        for trace, row in zip(traces, out):
            yield trace.replace(row)

    return from_iterable(traces(), n_traces=source.n_traces)


def fxdecon(*, taper=0.1, fmin=6.0, fmax=None, twlen=None, ntrw=10, ntrf=4, dt=None):
    """Random noise attenuation by f-x deconvolution (SUFXDECON).

    The traces of the stream are one panel, held in memory. Each time window of each trace is transformed to the
    frequency domain, and for each frequency a complex Wiener prediction filter (with unit prediction distance in space)
    predicts every trace from the ones beside it (windows of ``ntrw`` traces, filters of ``ntrf`` traces); the
    predictions, which are what is coherent, are transformed back and joined by tapers.

    Parameters
    ----------
    taper : float
        Length (s) of the taper between time windows, default .1.
    fmin, fmax : float
        The range of frequencies (Hz) that is processed, default 6 and ``.6 / (2 dt)``.
    twlen : float
        Length (s) of the time windows, by default the whole trace; it is at least .3.
    ntrw : int
        Number of traces in a window, default 10.
    ntrf : int
        Number of traces of the filter, default 4 (the program says to keep it less than ``ntrw``).
    dt : float
        Sample interval (s) (that of the first trace if it is not given; .004 if it has none).

    Differences from the program, which are its bugs: with fewer than ``ntrw`` traces, or a window of more than 1.5 times
    the length of a trace, it gives all zeros (here one window is used); a single window of more than ``ntrw`` traces
    gets the last trace for the traces after the first ``ntrw`` (here each takes its own); ``ntrf=0`` is changed to 4
    as the program says that it does, and a window that is all zeros gives zeros (the program's matrix is singular).
    The autocorrelations are of the traces from the start of the context that is added to the window, as the program
    does it. The program's temporary files are not used.
    """
    kwargs = dict(taper=taper, fmin=fmin, fmax=fmax, twlen=twlen, ntrw=ntrw, ntrf=ntrf, dt=dt)
    _fxdecon((), **kwargs)  # check the parameters now
    return Stage(_fxdecon, parallelism='serial', name='fxdecon', **kwargs)
