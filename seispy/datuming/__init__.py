"""
Datuming (``su/main/datuming``): SUDATUMFD (``datumfd``: zero offset finite difference datuming of shot gathers by the
one-way wave equation, with a zero velocity layer for an irregular surface), and the Kirchhoff datuming programs
SUDATUMK2DR, SUDATUMK2DS, SUKDMDCR and SUKDMDCS (``datumk2dr``, ``datumk2ds``, ``kdmdcr``, ``kdmdcs``, from traveltime
tables).

Where a program plainly does not do what its documentation says this does what the documentation says; the differences
are with each stage.
"""
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage
from . import _kernels
from ._kirchhoff import datumk2dr, datumk2ds, kdmdcr, kdmdcs

__all__ = ['datumfd', 'datumk2dr', 'datumk2ds', 'kdmdcr', 'kdmdcs']

F32 = np.float32


def _datumfd(upstream, *, vel1, vel2, nx, nsx, nz, dz, dx=None, dt=None, buff=5, tap_len=5, x_0=0.0):
    vel1 = np.ascontiguousarray(vel1, dtype=F32)
    vel2 = np.ascontiguousarray(vel2, dtype=F32)
    if vel1.ndim != 2 or vel1.shape != vel2.shape:
        raise ValueError("vel1 and vel2 are two arrays of the same shape (mx, mz).")
    if nx < 4 or nsx < 1 or nz < 0:
        raise ValueError("nx must be at least 4, nsx at least 1 and nz not negative.")
    if nz > vel1.shape[1]:
        raise ValueError("nz is more than the number of depth samples of the velocity models.")
    if dz <= 0.0:
        raise ValueError("dz must be positive")
    if dx is not None and dx <= 0.0:
        raise ValueError("dx must be positive")
    if buff < 0 or tap_len < 0:
        raise ValueError("buff and tap_len must not be negative")
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        n_traces = nx * nsx
        if len(panel) % n_traces:
            raise ValueError(f"There are {len(panel)} traces, a multiple of nx * nsx = {n_traces} is needed.")
        if any(t.dtype.kind == 'c' for t in panel):
            raise TypeError("datumfd works on real traces, but was given a complex one.")
        n_t = panel[0].n_sample
        if any(t.n_sample != n_t for t in panel):
            raise ValueError("The traces must all have the same number of samples.")
        sample_dt = panel[0].d_sample or dt
        if not sample_dt:
            sample_dt = 0.004
            warnings.warn("The traces have no sample interval and dt was not given: assuming dt=0.004")
        spacing = dx
        if spacing is None:
            spacing = 1.0
            warnings.warn("dx was not given: assuming dx=1.0")
        outside = False
        for start in range(0, len(panel), n_traces):
            block = panel[start:start + n_traces]
            data = np.ascontiguousarray(np.stack([np.asarray(t) for t in block]).reshape(nsx, nx, n_t), dtype=F32)
            gx = np.array([t.header['rx_loc'][0] for t in block], dtype=F32).reshape(nsx, nx)
            out, status = _kernels.datumfd(data, gx, vel1, vel2, F32(sample_dt), F32(dz), int(nz), F32(spacing),
                                           F32(x_0), int(buff), int(tap_len))
            outside = outside or bool(status)
            for trace, row in zip(block, out.reshape(n_traces, n_t)):
                yield trace.replace(np.ascontiguousarray(row))
        if outside:
            warnings.warn("A receiver is outside of the velocity models: the nearest column was used.")

    return from_iterable(traces())


def datumfd(vel1, vel2, *, nx, nsx, nz, dz, dx=None, dt=None, buff=5, tap_len=5, x_0=0.0):
    """Zero offset finite difference datuming by the one-way wave equation (SUDATUMFD, after Beasley and Lynn).

    The traces are ``nsx`` shot gathers of ``nx`` receivers each (the receivers fastest; more than one such block of
    traces can follow). Each gather is continued downward in ``nz`` steps of ``dz`` by a 45 degree implicit finite
    difference scheme (a thin lens term and a diffraction term) for zero offset data (the exploding reflector model, with
    half the velocity). Two velocity models can be given so that a zero velocity layer takes the data from an irregular
    surface.

    Parameters
    ----------
    vel1, vel2 : arrays
        The velocities (m/s) for the thin lens term and for the diffraction term: arrays ``(mx, mz)`` ([ix][iz]).
    nx, nsx, nz : int
        The number of receivers per gather, of gathers, and of depth steps (at most ``mz``).
    dz : float
        The depth step (m).
    dx : float
        The horizontal sampling interval (m) of the velocity models and of the receivers, by default 1.
    dt : float
        The sample interval (s) for traces that do not have one, by default .004.
    buff : int
        The number of zero traces added to each side of a gather, default 5.
    tap_len : int
        The length (in traces) of the taper at the ends of the gather, default 5.
    x_0 : float
        The x coordinate (m) of the first column of the velocity models.

    The column of the velocity of a receiver is from the x of its ``rx_loc``, truncated to an integer as the program
    does; a receiver outside of the models takes the nearest column (the program reads outside of the array). The
    frequencies of the program are 2 pi k / ((n - 1) dt) for the n of the transform, where they are 2 pi k / (n dt):
    that is corrected. The headers pass through unchanged.
    """
    kwargs = dict(vel1=vel1, vel2=vel2, nx=nx, nsx=nsx, nz=nz, dz=dz, dx=dx, dt=dt, buff=buff, tap_len=tap_len, x_0=x_0)
    _datumfd((), **kwargs)  # check the parameters now
    return Stage(_datumfd, parallelism='serial', name='datumfd', **kwargs)
