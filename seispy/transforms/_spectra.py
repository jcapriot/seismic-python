"""
Amplitude spectra: SUSPECFX, SUSPECFK and SUSPECK1K2 (``su/main/transforms``).

``specfx`` is the amplitude spectrum of each trace. ``specfk`` is the f-k spectrum, and ``speck1k2`` the 2D (k1, k2) spectrum,
of the traces of the stream taken as one panel (held in memory). The transforms are numpy's, of the traces as they are (the
first dimension is made even by a zero), see the notes in ``seispy.transforms``.

The traces that these make are marked as being in the Fourier domain, and their sample interval and first sample are those of
their (frequency or wavenumber) axis. The axis across the traces is not in the trace header; ``specfk`` has traces from
wavenumber ``-1 / (2 dx)`` in steps of ``1 / (nk dx)`` (``nk`` is the number of traces), and ``speck1k2`` from
``-1 / (2 dx2) + 1 / (nk dx2)`` in steps of ``1 / (nk dx2)``.

Where the program plainly does not do what it says: SUSPECK1K2 has header values (``d1``, ``f1``, ``d2``, ``f2``) that are twice
those of the axes of what it outputs (which runs from -Nyquist to +Nyquist), and the positive k1 half of each of its traces
starts again at k1 = 0, so that the zero wavenumber is there twice and the rest is shifted by one sample. Here the headers
are those of the axes, and the positive half starts at the first wavenumber after 0.
"""
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage, per_trace

__all__ = ['specfx', 'specfk', 'speck1k2']

_FOURIER = 2  # (the sampling domain of the traces of a Fourier transform, as in seispy.transforms)


def _real(trace, name):
    x = np.asarray(trace)
    if x.dtype.kind == 'c':
        raise TypeError(f"{name} works on real traces, but was given a complex one.")
    return x.astype(np.float32)


# ------------------------------------------------------------------------------------------------------------ specfx
def _specfx(upstream, *, dt=None):
    def specfx_trace(trace):
        x = _real(trace, 'specfx')
        n = x.shape[0]
        n_fft = n + n % 2
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            warnings.warn("dt not set, assumed to be .004")
            sample_dt = 0.004
        amplitude = np.abs(np.fft.rfft(x, n_fft)).astype(np.float32)
        amplitude[0] /= 2.0  # (as in the program)
        return trace.replace(amplitude, d_sample=1.0 / (n_fft * sample_dt), sample_start=0.0, sampling_domain=_FOURIER)

    return per_trace(upstream, specfx_trace)


def specfx(dt=None):
    """The Fourier amplitude spectrum of each trace (SUSPECFX).

    The traces have ``nfft / 2 + 1`` samples (``nfft`` is the number of samples, made even), from 0 Hz in steps of
    ``1 / (nfft * dt)``, which is their sample interval, and are marked as being in the Fourier domain. As in the
    program, the amplitude at 0 Hz is halved.

    Parameters
    ----------
    dt : float, optional
        The sample interval (s) for traces that do not have one, otherwise .004.
    """
    _specfx((), dt=dt)
    return Stage(_specfx, parallelism='trace', name='specfx', dt=dt)


# ------------------------------------------------------------------------------------------------------------ specfk
def _panel(source, name):
    traces = list(source)
    if not traces:
        return traces, None
    n = traces[0].n_sample
    if any(t.n_sample != n for t in traces):
        raise ValueError("The traces must all have the same number of samples.")
    return traces, np.stack([_real(t, name) for t in traces])


def _specfk(upstream, *, dt=None, dx=None):
    source = as_trace_iterator(upstream)

    def traces():
        traces, x = _panel(source, 'specfk')
        if x is None:
            return
        nx, nt = x.shape
        sample_dt = traces[0].d_sample or dt
        if not sample_dt:
            warnings.warn("The traces have no sample interval, assuming dt=0.004")
            sample_dt = 0.004
        sample_dx = dx
        if sample_dx is None:
            warnings.warn("dx not given, assuming dx=1.0")
            sample_dx = 1.0
        ntfft = nt + nt % 2
        # (the odd traces are negated to center the transform across the traces)
        signs = np.where(np.arange(nx) % 2 == 1, -1.0, 1.0).astype(np.float32)
        # the SU transform in time has exp(+i w t) in it, the one across the traces exp(-i k x)
        ct = np.conj(np.fft.rfft(x * signs[:, None], ntfft, axis=1))
        ct = np.fft.fft(ct, axis=0)
        amplitude = np.abs(ct).astype(np.float32)
        d1 = 1.0 / (ntfft * sample_dt)
        for ik, row in enumerate(amplitude):
            yield traces[0].replace(row, trace_id=ik + 1, d_sample=d1, sample_start=0.0, sampling_domain=_FOURIER)

    return from_iterable(traces())


def specfk(dt=None, dx=None):
    """The f-k Fourier amplitude spectrum of a data set (SUSPECFK).

    The traces of the stream are one panel of ``nx`` traces, which gives ``nx`` traces of ``nfft / 2 + 1`` samples (the
    frequencies from 0 in steps of ``1 / (nfft dt)``), one for each wavenumber, from ``-1 / (2 dx)`` in steps of
    ``1 / (nx dx)``. Their phase is ``i (w t - k x) = 2 pi i (F t - K x)``.

    Parameters
    ----------
    dt : float, optional
        The sample interval (s) for traces that do not have one, otherwise .004.
    dx : float
        The spatial sampling interval (the program takes it from the header word d2, which traces here do not have), 1 if
        not given.
    """
    _specfk((), dt=dt, dx=dx)
    return Stage(_specfk, parallelism='serial', name='specfk', dt=dt, dx=dx)


# -------------------------------------------------------------------------------------------------------- speck1k2
def _speck1k2(upstream, *, d1=None, d2=None):
    source = as_trace_iterator(upstream)

    def traces():
        traces, x = _panel(source, 'speck1k2')
        if x is None:
            return
        nx2, nx1 = x.shape
        dx1 = d1 if d1 is not None else (traces[0].d_sample or None)
        if dx1 is None:
            warnings.warn("The traces have no sample interval and d1 is not given, assuming d1=1.0")
            dx1 = 1.0
        dx2 = d2
        if dx2 is None:
            warnings.warn("d2 not given, assuming d2=1.0")
            dx2 = 1.0
        n1fft = nx1 + nx1 % 2
        n2fft = nx2
        nk1 = n1fft // 2 + 1
        signs = np.where(np.arange(nx2) % 2 == 1, -1.0, 1.0).astype(np.float32)
        c = np.fft.rfft(x * signs[:, None], n1fft, axis=1)
        c = np.fft.fft(c, axis=0)
        mag = np.abs(c)
        nk2 = n2fft
        # (the wavenumber of the first sample is -Nyquist, and the zero wavenumber is on the sample n1fft / 2)
        out_d1 = 1.0 / (n1fft * dx1)
        out_f1 = -1.0 / (2.0 * dx1)
        for ik2 in range(nk2):
            row = np.empty(n1fft, dtype=np.float32)
            # the upper half of the K-plane
            row[:nk1] = mag[nk2 - 1 - ik2, ::-1]
            # the positive k1, from the same wavenumbers of the next k2 by symmetry (k1 = 0 is on the sample before)
            row[nk1:] = mag[ik2 + 1 if ik2 < nk2 - 1 else 0, 1:n1fft - nk1 + 1]
            yield traces[0].replace(row, trace_id=ik2 + 1, d_sample=out_d1, sample_start=out_f1,
                                    sampling_domain=_FOURIER)

    return from_iterable(traces())


def speck1k2(d1=None, d2=None):
    """The 2D (k1, k2) Fourier amplitude spectrum of a (x1, x2) data set (SUSPECK1K2).

    The traces of the stream are one panel: the samples of a trace run along x1 (the fast dimension), the traces along x2.
    The result has one trace for each wavenumber k2, of ``nfft`` samples (``nfft`` is the number of samples, made even) for
    k1, running from -Nyquist to +Nyquist in both dimensions: k1 from ``-1 / (2 d1)`` in steps of ``1 / (nfft d1)`` (the
    sample interval and first sample of the traces), k2 from ``-1 / (2 d2) + 1 / (n d2)`` in steps of ``1 / (n d2)``, for
    the ``n`` traces.

    Parameters
    ----------
    d1 : float, optional
        The spatial sampling interval along the traces, by default the sample interval of the first trace (1 if it has
        none).
    d2 : float, optional
        The spatial sampling interval across the traces (the program takes it from the header word d2, which traces here
        do not have), 1 if not given.
    """
    _speck1k2((), d1=d1, d2=d2)
    return Stage(_speck1k2, parallelism='serial', name='speck1k2', d1=d1, d2=d2)
