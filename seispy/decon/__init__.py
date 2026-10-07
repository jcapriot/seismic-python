"""
Wiener (least squares) deconvolution and shaping: SUPEF and SUSHAPE (``su/main/decon_shaping``), with the Toeplitz
solver of the SU library.

Not supported: supef's ``cdp`` tables (lags that change with the CDP), ``wienerout`` and ``outpar``.
"""
import numpy as np

from ..convolution import _convolve, _correlate
from ..stage import Stage, per_trace
from . import _toeplitz

__all__ = ['pef', 'shape']

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
