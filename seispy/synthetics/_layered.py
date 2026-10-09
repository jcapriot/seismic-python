"""
Layered media: SUGOUPILLAUDPO, the primaries-only impulse response of a lossless Goupillaud medium, and SUSYNCZ, zero-offset data over
dipping interfaces of constant velocity layers (``su/main/synthetics_waveforms_testpatterns``).

A Goupillaud medium has layers of the same two-way traveltime. The traces that go in are series of reflection coefficients

    r[i] = (impedance[i] - impedance[i+1]) / (impedance[i] + impedance[i+1]),

``r[0]`` being that of the surface (as seen from above) and the last one that of the deepest interface, and the traces that come
out are the seismograms for a source at the top of layer ``l`` and a receiver at the top of layer ``k``. The sample interval of
the reflectivity is the two-way traveltime of the layers, and the seismogram has the same.

Where the program does not do what its documentation says this does what the documentation says (see the notes on each). It reads
only the first trace of its input; here every trace is a reflectivity series that gets its seismogram.
"""
import warnings

import numpy as np

from ..container import Trace, from_iterable
from ..stage import per_trace, stage
from . import _goupillaud, _syncz

__all__ = ['goupillaudpo', 'syncz']


def _goupillaudpo(upstream, *, l=1, k=1, tmax=None, pV=1):
    if pV not in (1, -1):
        raise ValueError("The field-type flag pV should be either 1 or -1.")
    if k < 1:
        raise ValueError("Receiver layer k must be >=1 (k=1 corresponds to surface seismogram).")
    if l < 1:
        raise ValueError("Source layer l must be >= 1 (l=1 corresponds to a surface source).")
    if tmax is not None and tmax < 0:
        raise ValueError("The number of the output time samples tmax cannot be negative.")

    def seismogram(trace):
        r = np.ascontiguousarray(np.asarray(trace), dtype=np.float32)
        n = r.shape[0] - 1
        if n < 0:
            raise ValueError("The reflectivity has no samples.")
        if n == 0:
            warnings.warn("WARNING: model without subsurface reflectors!")
        if l > n + 1:
            raise ValueError("The current version of the program requires l<=n+1.")
        if k > n + 1:
            raise ValueError("The receiver layer k must be at most n+1 (the number of reflection coefficients).")
        length = _goupillaud.goupillaudpo_tmax(n, l, k) if tmax is None else tmax
        status, out, odd = _goupillaud.goupillaudpo(r, l, k, length, pV)
        if status == -1:
            raise ValueError("Invalid reflection coefficient encountered.")
        if status == -2:
            raise ValueError("Seismogram length too small -- cannot observe any signal.")
        if status != 0:
            raise ValueError("The parameters are not allowed.")
        # (an odd distance between the layers puts the seismogram half a sample later)
        return trace.replace(out, sample_start=trace.d_sample / 2.0 if odd else 0.0, trace_type=1)

    return per_trace(upstream, seismogram)


# SUGOUPILLAUDPO: the primaries-only impulse response of a lossless Goupillaud medium for plane waves at normal incidence
#
# l : the source layer, 1 <= l <= n + 1 for n + 1 reflection coefficients; the source is at the top of the layer, default 1
# k : the receiver layer, 1 <= k <= n + 1, at the top of the layer, default 1
# tmax : the number of output samples, by default enough to catch all of the primaries
# pV : 1 for a vector field (displacement, velocity, acceleration), -1 for pressure
#
# For a vector field a buried source makes a spike of amplitude 1 going down and -1 going up; a buried pressure source makes 1 in
# both directions; a surface source makes only a downgoing spike of amplitude 1.
goupillaudpo = stage(_goupillaudpo, parallelism='trace', name='goupillaudpo', validate=True)


# ------------------------------------------------------------------------------------------------------------ syncz
def syncz(*, ninf=4, zint=None, dip=None, v=None, rho=None, nline=1, ntr=32, dx=10.0, tdelay=0.0, dt=0.004, nt=128):
    """Zero-offset data over dipping interfaces in a medium of layers of constant velocity, true amplitude (primaries only) for
    2.5-D (SUSYNCZ). It is a source.

    The interfaces are planes that dip in the x direction; the traces are at x = 0, ``dx``, ... and the source and receiver are at
    each of them. The amplitudes include the reflection and transmission coefficients (from the densities), and the geometrical
    spreading.

    Parameters
    ----------
    ninf : int
        The number of interfaces (not counting the surface).
    zint : array of ninf numbers
        The depths of the interfaces at x = 0, increasing. Default ``100 i``.
    dip : array of ninf numbers
        The dips of the interfaces in degrees, from -90 to 90 (and not differing by more than 90 from the one above). Default ``5 i``.
    v, rho : array of ninf + 1 numbers
        The velocities and densities of the layers: the one above the first interface, and below each. Defaults ``1500 + 500 i`` and 1.
    nline : int
        The number of (identical) lines, each of ``ntr`` traces. Their ``ensemble_number`` is the line, from 1.
    dx, dt, nt, tdelay : float
        The trace interval, the sample interval, the number of samples and the delay of the recording time (the first sample is at
        time ``tdelay``).

    Where the program does not do what its documentation says this does what the documentation says: it leaves the deepest
    interface out of the table of angles (so a model of one interface has none) and the deepest layer out of the checks of the
    velocities and densities, and its final check of the interfaces stops after the first. Interfaces that meet in the region that is
    observed are an error.
    """
    if ninf < 0:
        raise ValueError("ninf must not be negative")
    if nline < 0 or ntr < 0:
        raise ValueError("nline and ntr must not be negative")
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")
    f32 = np.float32
    zint = [100.0 * i for i in range(1, ninf + 1)] if zint is None else list(zint)
    dip = [5.0 * i for i in range(1, ninf + 1)] if dip is None else list(dip)
    v = [1500.0 + 500.0 * i for i in range(ninf + 1)] if v is None else list(v)
    rho = [1.0] * (ninf + 1) if rho is None else list(rho)
    if len(zint) != ninf or len(dip) != ninf:
        raise ValueError(f"zint and dip must have ninf = {ninf} numbers")
    if len(v) != ninf + 1 or len(rho) != ninf + 1:
        raise ValueError(f"v and rho must have ninf + 1 = {ninf + 1} numbers")
    model = _syncz.SynczModel(ninf, ntr, dx, np.array([0.0] + zint, dtype=np.float32), np.array([0.0] + dip, dtype=np.float32),
                              np.array(v, dtype=np.float32), np.array(rho, dtype=np.float32), nt, dt, tdelay)

    def traces():
        for iline in range(nline):
            x = f32(0.0)
            for itr in range(ntr):
                yield Trace(model.trace(x), d_sample=dt).replace(
                    sample_start=tdelay, trace_id=itr + 1, ensemble_number=iline + 1, tx_loc=[float(x), 0.0, 0.0], rx_loc=[float(x), 0.0, 0.0],
                )
                x = f32(x + f32(dx))

    return from_iterable(traces(), n_traces=nline * ntr)
