"""
SUDIPFILT: dip (slope) filter in the f-k domain (``su/main/filters``).

The traces of the stream are one panel (held in memory). They are Fourier transformed in time and across the traces (numpy's
FFT, of the traces as they are: see the notes in ``seispy.transforms``), multiplied by an amplitude that is a function of the
slope ``k / w``, and transformed back.
"""
import numpy

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage

__all__ = ['dipfilt']


def _dipfilt(upstream, *, slopes=0.0, amps=None, bias=0.0, dt=None, dx=None):
    slopes = numpy.atleast_1d(numpy.asarray(slopes, dtype=numpy.float64))
    if amps is None:
        amps = numpy.ones(1)  # (the program says that it is doing nothing)
    amps = numpy.atleast_1d(numpy.asarray(amps, dtype=numpy.float64))
    if slopes.ndim != 1 or amps.shape != slopes.shape:
        raise ValueError(f"number of slopes ({slopes.size}) must equal number of amps ({amps.size})")
    if (numpy.diff(slopes) <= 0).any():
        raise ValueError("slopes must be monotonically increasing")
    source = as_trace_iterator(upstream)

    def traces():
        traces = list(source)
        if not traces:
            return
        first = traces[0]
        nt, nx = first.n_sample, len(traces)
        if any(t.n_sample != nt for t in traces):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in traces):
            raise TypeError("dipfilt works on real traces, but was given a complex one.")
        sample_dt = first.d_sample or dt
        if not sample_dt:
            raise ValueError("dt field is zero and dt is not given")
        if not dx:
            raise ValueError("dx is needed (the program takes it from the header word d2, which traces here do not have)")
        data = numpy.stack([numpy.asarray(t) for t in traces]).astype(numpy.float64)

        ntfft = nt + nt % 2
        nw = ntfft // 2 + 1
        dw = 2.0 * numpy.pi / (ntfft * sample_dt)
        dk = 2.0 * numpy.pi / (nx * dx)
        w = numpy.arange(nw) * dw
        x = numpy.arange(nx) * dx
        # the SU transform in time has exp(+i w t) in it, the one across the traces exp(-i k x)
        spectrum = numpy.conj(numpy.fft.rfft(data, ntfft, axis=1))
        # the bias: a linear moveout that is removed by a phase shift, and restored after filtering
        shift = numpy.exp(-1j * x[:, None] * w[None, :] * bias)
        spectrum = numpy.fft.fft(spectrum * shift, axis=0)
        k = numpy.fft.fftfreq(nx, d=1.0 / nx) * dk  # (ik for ik <= nk/2, ik - nk above)
        # (the frequency of the first sample is not 0, to avoid dividing by it)
        w_nonzero = 1e-6 * dw + w
        slope = k[:, None] / w_nonzero[None, :] + bias
        amp = numpy.interp(slope, slopes, amps)
        spectrum = numpy.fft.ifft(spectrum * amp, axis=0)
        spectrum = spectrum / shift
        out = numpy.fft.irfft(numpy.conj(spectrum), ntfft, axis=1)[:, :nt].astype(numpy.float32)
        for trace, row in zip(traces, out):
            yield trace.replace(row)

    return from_iterable(traces(), n_traces=source.n_traces)


def dipfilt(slopes=0.0, amps=None, bias=0.0, *, dt=None, dx=None):
    """A dip (or better, slope) filter in the f-k domain (SUDIPFILT).

    The traces of the stream are one panel held in memory, ``dx`` apart. The slope of an event is ``delta_t / delta_x``,
    in the units of ``dt`` and ``dx``: it is sometimes useful to give ``dt=1`` and ``dx=1``, avoiding units and small
    slopes. Linear interpolation and constant extrapolation are used for the amplitudes of the slopes that are not given.

    Parameters
    ----------
    slopes : array
        Monotonically increasing slopes.
    amps : array
        The amplitudes for them; the default is 1, which does nothing.
    bias : float
        A slope that is made horizontal before filtering (and restored after). It can be useful for spatially aliased
        data. The ``slopes`` do not change with the bias.
    dt : float, optional
        The sample interval, for traces that do not have one.
    dx : float
        The distance between traces (the program takes it from the header word d2, which traces here do not have).
    """
    kwargs = dict(slopes=slopes, amps=amps, bias=bias, dt=dt, dx=dx)
    _dipfilt((), **kwargs)  # check the parameters now
    return Stage(_dipfilt, parallelism='serial', name='dipfilt', **kwargs)
