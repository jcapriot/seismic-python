"""
SUDIPDIVCOR: dip-dependent divergence (spreading) correction (``su/main/amplitudes``).

The correction is made in the wavenumber domain: the traces of the stream (which are the panel) are Fourier transformed in x
(numpy's FFT, of the length that SU would pad them to), each wavenumber is corrected by the dip filter of the SU library
(``_dipdivcor.pyx``), and they are transformed back.

Where the program plainly does not do what its documentation says this does what the documentation says: the rms velocity
function ``tmig``, ``vmig`` is interpolated linearly with constant extrapolation, where the program replaces that with
a monotonic cubic spline (that extrapolates) unless there is one value for every sample.
"""
import math

import numpy

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage
from . import _dipdivcor as _c

__all__ = ['dipdivcor']

F32 = numpy.float32


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _velocity(nt, dt, tmig, vmig, vt):
    if vt is not None:
        vt = numpy.ascontiguousarray(vt, dtype=numpy.float32)
        if vt.shape != (nt,):
            raise ValueError(f"vt must have {nt} velocities, one for each sample")
        return vt
    tmig = numpy.atleast_1d(numpy.asarray(tmig, dtype=numpy.float64))
    vmig = numpy.atleast_1d(numpy.asarray(vmig, dtype=numpy.float64))
    if tmig.shape != vmig.shape or tmig.ndim != 1:
        raise ValueError("number of tmig and vmig must be equal")
    if (numpy.diff(tmig) <= 0).any():
        raise ValueError("tmig must increase monotonically")
    times = numpy.arange(nt, dtype=numpy.float32) * F32(dt)
    return numpy.interp(times, tmig, vmig).astype(numpy.float32)


def _dipdivcor(upstream, *, dxcdp, np=50, tmig=0.0, vmig=1500.0, vt=None, conv=False, trans=False, norm=True):
    if not dxcdp:
        raise ValueError("dxcdp is required")
    if np < 2:
        raise ValueError("np must be at least 2")
    # (the velocity function is checked now, with a trace of 2 samples)
    _velocity(2, 1.0, tmig, vmig, None if vt is None else numpy.zeros(2))
    source = as_trace_iterator(upstream)

    def traces():
        traces = list(source)
        if not traces:
            return
        first = traces[0]
        nt, dt = first.n_sample, first.d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")
        if any(t.n_sample != nt for t in traces):
            raise ValueError("The traces must all have the same number of samples.")
        if any(t.dtype.kind == 'c' for t in traces):
            raise TypeError("dipdivcor works on real traces, but was given a complex one.")
        x = numpy.stack([numpy.asarray(t) for t in traces]).astype(numpy.float32)
        nx = x.shape[0]

        velocity = _velocity(nt, dt, tmig, vmig, vt)
        divcor, v0 = _c.divcor_table(np, dt, velocity, bool(trans), bool(norm))
        if conv:
            # only the conventional correction: no dip filtering
            out = x * divcor[0]
        else:
            ntfft, nxfft = _c.fft_sizes(nt, nx)
            nk = nxfft // 2 + 1
            dx = F32(dxcdp) * F32(0.001)
            dk = F32(2.0 * math.pi / (nxfft * dx))
            padded = numpy.zeros((nxfft, nt), dtype=numpy.float32)
            padded[:nx] = x
            ptk = numpy.ascontiguousarray(numpy.fft.rfft(padded, axis=0), dtype=numpy.complex64)
            # the wavenumbers that are corrected, and the sampling of the slopes
            nkmax = min(nk, _nint(math.pi / dt / v0 / dk))
            dpx = F32(1.0 / (np - 1) / v0)
            # wavenumber 0
            ptk[0] *= divcor[0]
            _c.dip_filter(ptk, dk, dpx, F32(dt), np, ntfft, nkmax, divcor)
            out = numpy.fft.irfft(ptk, n=nxfft, axis=0)[:nx]
        out = numpy.ascontiguousarray(out, dtype=numpy.float32)
        for trace, row in zip(traces, out):
            yield trace.replace(row)

    return from_iterable(traces(), n_traces=source.n_traces)


def dipdivcor(dxcdp, *, np=50, tmig=0.0, vmig=1500.0, vt=None, conv=False, trans=False, norm=True):
    """Dip-dependent divergence (spreading) correction (SUDIPDIVCOR).

    The whole stream is one panel of traces of the same length, which are ``dxcdp`` apart, and is held in memory.

    Parameters
    ----------
    dxcdp : float
        Distance between successive CDPs, in meters.
    np : int
        Number of slopes.
    tmig, vmig : arrays
        Times and the rms velocities for them. Linear interpolation, and constant extrapolation.
    vt : array, optional
        Instead, the rms velocity for every sample (the program reads these from a file, ``vfile``).
    conv : bool
        Only apply the conventional divergence correction (which over-amplifies reflections from dipping beds).
    trans : bool
        Include transmission factors (for data that is migrated with a reverse time migration based on the constant
        density wave equation).
    norm : bool
        Normalize by the correction at zero reflection slope and the first sample interval. Without it, zero-offset
        ``synlv`` amplitudes after this equal exploding reflector ones (with ``trans``).
    """
    kwargs = dict(np=np, tmig=tmig, vmig=vmig, vt=vt, conv=conv, trans=trans, norm=norm)
    _dipdivcor((), dxcdp=dxcdp, **kwargs)  # check the parameters now
    return Stage(_dipdivcor, dxcdp=dxcdp, parallelism='serial', name='dipdivcor', **kwargs)
