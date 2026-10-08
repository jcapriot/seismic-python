"""
Instantaneous trace attributes, the modes of SUATTRIBUTES (``su/main/attributes_parameter_estimation``), each as a stage
of its own: ``amp`` (the envelope), ``phase``, ``freq``, ``normamp``, ``fdenv``, ``sdenv``, ``bandwidth`` and ``q``::

    from seispy import attributes

    source | attributes.amp()

They are made from the complex trace ``trace + i hilbert(trace)``, with the Hilbert transform of the SU library
(``seispy.transforms.hilb``), by the functions of the SU library in ``_attributes.pyx``. The modes ``uphase``, ``freqw`` and ``thin`` are not here.

The programs that report on a data set instead of making traces (``mean``, ``max``, ``quantile``, ``histogram`` and ``cmp``, of
SUMEAN, SUMAX, SUQUANTILE, SUHISTOGRAM and SUCMP) are functions that return their results, in ``_stats.py``.

Where the program plainly does not do what its documentation says this does what the documentation says: ``bandwidth``
and ``q`` run over 2 nt - 1 samples of a trace of nt (``ntout``), here it is nt.
"""
import numpy as np

from ..stage import Stage, per_trace
from ..transforms import _hilbert
from . import _attributes as _c

__all__ = ['amp', 'phase', 'freq', 'normamp', 'fdenv', 'sdenv', 'bandwidth', 'q']


def _quadrature(trace):
    """The real part, and the imaginary part (the Hilbert transform), of the analytic trace"""
    x = np.asarray(trace)
    return x, _hilbert.hilbert(x)


# one function for each attribute: (re, im, dt, unwrap) -> the attribute, from the SU library (_attributes.pyx)
def _amp(re, im, dt, unwrap):
    return _c.envelope(re, im)


def _phase(re, im, dt, unwrap):
    return _c.phase(re, im, unwrap)


def _freq(re, im, dt, unwrap):
    return _c.freq(re, im, dt, unwrap)


def _normamp(re, im, dt, unwrap):
    return _c.normamp(re, im)


def _fdenv(re, im, dt, unwrap):
    return _c.fdenv(re, im, dt)


def _sdenv(re, im, dt, unwrap):
    return _c.sdenv(re, im, dt)


def _bandwidth(re, im, dt, unwrap):
    return _c.bandwidth(re, im, dt)


def _q(re, im, dt, unwrap):
    return _c.q(re, im, dt, unwrap)


class _AttributeFactory:
    """What a stage of one of these attributes calls to make its iterator (a class, so that it can be pickled)."""

    def __init__(self, name, func, default_unwrap):
        self.name = name
        self.func = func
        self.default_unwrap = default_unwrap

    def __call__(self, upstream, *, unwrap=None):
        func = self.func
        if unwrap is None:
            unwrap = self.default_unwrap

        def attribute_trace(trace):
            dt = trace.d_sample
            if not dt:
                raise ValueError(f"{self.name} needs the trace to have a sample interval.")
            if trace.n_sample < 2:
                raise ValueError(f"{self.name} needs traces with at least 2 samples.")
            re, im = _quadrature(trace)
            return trace.replace(func(re, im, dt, unwrap).astype(np.float32))

        return per_trace(upstream, attribute_trace)


def _define(name, func, doc, unwrap=None):
    """Make the function that makes stages of an attribute. ``unwrap`` is the default of the parameter, if it has one"""
    factory = _AttributeFactory(name, func, 0 if unwrap is None else unwrap)

    def build(unwrap_value):
        if unwrap is not None and unwrap_value < 0:
            raise ValueError("unwrap must be positive (or 0, for none)")
        kwargs = {} if unwrap is None or unwrap_value == unwrap else {'unwrap': unwrap_value}
        return Stage(factory, parallelism='trace', name=name, **kwargs)

    if unwrap is None:
        def make():
            return build(0)
    else:
        def make(unwrap=unwrap):
            return build(unwrap)

    make.__name__ = make.__qualname__ = name
    make.__module__ = __name__
    make.__doc__ = doc
    return make


_UNWRAP = """
    unwrap : float
        The smallest change of phase taken to be the result of wrapping is pi / unwrap (so a bigger number makes the
        unwrapping more sensitive). 0 does not unwrap. The unwrapping is crude, it makes the phase increase.
"""

amp = _define('amp', _amp, "The envelope (instantaneous amplitude).")
phase = _define('phase', _phase, "The instantaneous phase." + _UNWRAP + "    Default 0.", unwrap=0)
freq = _define(
    'freq', _freq,
    "The instantaneous frequency, the derivative of the unwrapped phase, folded back below the Nyquist frequency."
    + _UNWRAP + "    Default 1.", unwrap=1,
)
normamp = _define('normamp', _normamp, "The normalized amplitude, the cosine of the instantaneous phase.")
fdenv = _define('fdenv', _fdenv, "The first derivative of the envelope (with an extra factor of 2 pi, as in SU).")
sdenv = _define('sdenv', _sdenv, "The second derivative of the envelope (with an extra factor of 2 pi per derivative).")
bandwidth = _define(
    'bandwidth', _bandwidth, "The instantaneous bandwidth, |d(envelope)/dt| / (2 pi envelope) (Barnes 1992)."
)
q = _define(
    'q', _q, "The instantaneous Q factor, -pi f(t) envelope / (d envelope / dt) (Barnes 1992)." + _UNWRAP + "    Default 1.",
    unwrap=1,
)

# (not in __all__: `from seispy.attributes import *` would replace the builtin max, and these are not stages)
from ._stats import mean, max, quantile, histogram, cmp  # noqa: E402,F401
