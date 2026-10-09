from ._gain import _gain
from ._pertrace import zero, nan, normalize, weight, ai2r, r2ai, divcor, centsamp, impedance
from ._scan import pgc
from ._dipdiv import dipdivcor
from ..stage import Stage

__all__ = ['gain', 'zero', 'nan', 'normalize', 'weight', 'ai2r', 'r2ai', 'divcor', 'centsamp', 'pgc', 'impedance', 'dipdivcor']


def gain(*, panel=False, **kwargs):
    """Apply various types of gain (SUGAIN).

    ``out(t) = scale * BAL{CLIP[AGC{[t^tpow * exp(epow * t^etpow) * (in(t) - bias)]^gpow}]}``

    Parameters
    ----------
    panel : bool
        Gain the whole data set at once instead of trace by trace. As in SU, all the traces are then treated as one
        long trace. This needs the whole stream (and every trace to be the same length), so such a stage can not be
        split across workers.
    tpow, epow, etpow, gpow : float
        Multiply by ``t**tpow``, by ``exp(epow * t**etpow)``, and take the signed ``gpow`` power.
    agc, gagc : bool
        Automatic gain control, with a box window (``agc``) or a gaussian taper (``gagc``) of ``wagc`` seconds.
    trap, clip, pclip, nclip, qclip : float
        Zero values larger than ``trap`` in magnitude, clip at ``clip`` in magnitude, clip values above ``pclip`` /
        below ``nclip``, and clip at the ``qclip`` quantile of the magnitudes on each trace.
    qbal, pbal, mbal, maxbal : bool
        Balance traces by the ``qclip`` quantile, by the rms value, by subtracting the mean, or the maximum.
    scale, norm, bias : float
        Overall scale factor (divided by ``norm`` if it is given) and bias (added before everything else).
    jon : bool
        ``tpow=2, gpow=0.5, qclip=0.95``
    vred : float
        Reducing velocity, used to shift the times used by ``tpow`` by ``|offset| / vred``.
    dt : float, optional
        Sample interval in seconds for traces that do not have one.
    inplace : bool
        Overwrite the samples of the incoming traces instead of making new traces.

    Not supported: SU's ``mark``, ``tmpdir`` and ``verbose`` parameters.
    """
    try:
        # catch bad parameters now, not when something finally gets pulled through the pipe
        _gain((), panel=panel, **kwargs)
    except TypeError as err:
        raise TypeError(f"gain: {err}") from None
    return Stage(
        _gain, panel=panel, parallelism='serial' if panel else 'trace', name='gain', **kwargs
    )
