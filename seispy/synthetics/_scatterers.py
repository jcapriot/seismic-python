"""
Shot records of a scatterer by the Born integral equation: SUIMP2D (a line scatterer embedded in three dimensions) and SUIMP3D (a
point scatterer, in-plane records), from ``su/main/synthetics_waveforms_testpatterns``.

These are sources. The shots are ``dsx``, ``dsy`` and ``dsz`` apart from the first one, the receivers of each shot ``dgx``, ``dgy``
and ``dgz`` apart from the first, in meters (``z`` is depth); the traces are in the order of the shots, then the receivers. The
shot of a trace is its ``ensemble_number`` (from 1), its number in the shot ``ensemble_trace_number``, and the source and receiver
are ``tx_loc`` and ``rx_loc`` with the elevation ``-z``. (The programs round the locations to whole meters, which their trace
headers need; here they are kept.)

The traces are made with the omega^(3/2) (2-D) or omega^2 (3-D) filters of the programs, in the FFT length of the SU library (which
the amplitudes depend on).
"""
import math

import numpy as np

from ..container import Trace, from_iterable
from . import _born

__all__ = ['imp2d', 'imp3d']

F32 = np.float32


def _check(nshot, nrec, nt, c, dt):
    if nshot < 0 or nrec < 0:
        raise ValueError("nshot and nrec must not be negative")
    if nt < 1:
        raise ValueError(f"nt={nt} must be at least 1")
    if c <= 0.0:
        raise ValueError(f"c={c} must be positive")
    if dt <= 0.0:
        raise ValueError(f"dt={dt} must be positive")


def _distance(*deltas):
    """The length of the vector, in single precision as the programs have it"""
    return F32(math.sqrt(float(sum((d * d for d in deltas), F32(0.0)))))


def imp2d(*, nshot=1, nrec=1, c=5000.0, dt=0.004, nt=256, x0=1000.0, z0=1000.0, sxmin=0.0, szmin=0.0, gxmin=0.0, gzmin=0.0,
          dsx=100.0, dsz=0.0, dgx=100.0, dgz=0.0):
    """Shot records for a line scatterer embedded in three dimensions, by the Born integral equation (SUIMP2D).

    Parameters
    ----------
    nshot, nrec : int
        Number of shots, and of receivers of each.
    c : float
        The speed.
    dt, nt : float, int
        Sample interval (s) and number of samples.
    x0, z0 : float
        The location of the scatterer.
    sxmin, szmin, gxmin, gzmin : float
        The first shot and receiver locations; ``dsx``, ``dsz``, ``dgx`` and ``dgz`` are the steps in them.
    """
    _check(nshot, nrec, nt, c, dt)
    born = _born.BornResponse(nt, dt, c)

    def traces():
        number = 0
        for s in range(nshot):
            sx, sz = F32(sxmin) + F32(s) * F32(dsx), F32(szmin) + F32(s) * F32(dsz)
            rs = _distance(sx - F32(x0), sz - F32(z0))
            for g in range(nrec):
                gx, gz = F32(gxmin) + F32(g) * F32(dgx), F32(gzmin) + F32(g) * F32(dgz)
                rg = _distance(gx - F32(x0), gz - F32(z0))
                number += 1
                yield Trace(born.response2d(rs, rg), d_sample=dt).replace(
                    trace_id=number, ensemble_number=s + 1, ensemble_trace_number=g + 1,
                    tx_loc=[float(sx), 0.0, -float(sz)], rx_loc=[float(gx), 0.0, -float(gz)],
                )

    return from_iterable(traces(), n_traces=nshot * nrec)


def imp3d(*, nshot=1, nrec=1, c=5000.0, dt=0.004, nt=256, x0=1000.0, y0=0.0, z0=1000.0, dir=0, sxmin=0.0, symin=0.0, szmin=0.0,
          gxmin=0.0, gymin=0.0, gzmin=0.0, dsx=100.0, dsy=0.0, dsz=0.0, dgx=100.0, dgy=0.0, dgz=0.0):
    """In-plane shot records for a point scatterer embedded in three dimensions, by the Born integral equation (SUIMP3D).

    The parameters are those of `imp2d`, with the ``y`` coordinates (``y0``, ``symin``, ``gymin``, ``dsy``, ``dgy``) added, and

    dir : {0, 1}
        1 to include the direct arrival (from the horizontal distance between the shot and the receiver; a shot and receiver
        at the same place make it infinite).
    """
    _check(nshot, nrec, nt, c, dt)
    if dir not in (0, 1):
        raise ValueError("dir must be 0 or 1")
    born = _born.BornResponse(nt, dt, c)

    def traces():
        number = 0
        for s in range(nshot):
            sx, sy, sz = (F32(sxmin) + F32(s) * F32(dsx), F32(symin) + F32(s) * F32(dsy), F32(szmin) + F32(s) * F32(dsz))
            rs = _distance(sx - F32(x0), sy - F32(y0), sz - F32(z0))
            for g in range(nrec):
                gx, gy, gz = (F32(gxmin) + F32(g) * F32(dgx), F32(gymin) + F32(g) * F32(dgy), F32(gzmin) + F32(g) * F32(dgz))
                rg = _distance(gx - F32(x0), gy - F32(y0), gz - F32(z0))
                rd = _distance(gx - sx, gy - sy)
                number += 1
                with np.errstate(divide='ignore', invalid='ignore'):
                    data = born.response3d(rs, rg, bool(dir), rd)
                yield Trace(data, d_sample=dt).replace(
                    trace_id=number, ensemble_number=s + 1, ensemble_trace_number=g + 1,
                    tx_loc=[float(sx), float(sy), -float(sz)], rx_loc=[float(gx), float(gy), -float(gz)],
                )

    return from_iterable(traces(), n_traces=nshot * nrec)
