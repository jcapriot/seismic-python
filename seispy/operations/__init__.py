"""
Unary arithmetic operations on traces, the operations of SUOP (``suop op=...``), and (in ``_panels.py``) the operations on
panels: SUMIX and SUOP2.

Each operation is a stage of its own, with its own parameters::

    from seispy import operations as op

    source | op.abs() | op.mean(nw=11)

Every stage here works trace by trace, so they can be split across workers. They all take ``inplace``, to overwrite the
samples of the incoming traces instead of making new traces, and the window operations (``mean``, ``std``, ``var``,
``despike``) also take ``nw``, the number of samples in the window (rounded up to be odd).

Most are plain array arithmetic (see ``_numpy_ops.py``). ``saf``, ``freq`` and ``despike`` are functions in the SU
sources (``su/main/operations/suop.c``).
"""
# (not `from ._ops import _ops`, which would hide the submodule)
from . import _ops as _o
from . import _numpy_ops as _n
from ..stage import Stage

_DEFAULT_NW = 21

# The operations that are defined for complex samples. The others need an order, a sign, or a median of the samples
# (posonly, ssqrt, sgn, slog, mod2pi, spike, saf, despike, ...), which complex numbers do not have. abs and db give a
# real trace (the modulus, and its level in dB), the window statistics std and var also, the rest are complex.
_COMPLEX_OK = {
    'abs', 'sqr', 'exp', 'db', 'cos', 'sin', 'tan', 'cosh', 'sinh', 'tanh', 'neg', 'nop', 'inv', 's2v', 's2vm',
    'd2m', 'cnorm', 'norm', 'avg', 'rmsamp', 'sum', 'integ', 'refl', 'diff', 'drv2', 'drv4', 'mean', 'std', 'var',
}


class _OpFactory:
    """What a stage of one of these operations calls to make its iterator (a class, so that it can be pickled)."""

    def __init__(self, name, func, **flags):
        self.name = name
        self.func = func
        self.flags = flags

    def __call__(self, upstream, **kwargs):
        return _o._ops(upstream, self.func, name=self.name, **self.flags, **kwargs)


def _define(name, func, doc, *, window=False, **flags):
    """Make the function that makes stages of one operation."""
    factory = _OpFactory(name, func, complex_ok=name in _COMPLEX_OK, **flags)

    def build(nw, inplace):
        kwargs = {}
        if nw != _DEFAULT_NW:
            kwargs['nw'] = nw
        if inplace:
            kwargs['inplace'] = True
        factory((), **kwargs)  # check the parameters now
        return Stage(factory, parallelism='trace', name=factory.name, **kwargs)

    if window:
        def make(nw=_DEFAULT_NW, *, inplace=False):
            return build(nw, inplace)
    else:
        # (nw means nothing to this operation)
        def make(*, inplace=False):
            return build(_DEFAULT_NW, inplace)

    make.__name__ = make.__qualname__ = name
    make.__module__ = __name__
    make.__doc__ = doc
    return make


# --------------------------------------------------------------------------------------------------- pointwise
abs = _define('abs', _n.op_abs, "Absolute value.")
ssqrt = _define('ssqrt', _n.op_ssqrt, "Signed square root.")
sqr = _define('sqr', _n.op_sqr, "Square.")
ssqr = _define('ssqr', _n.op_ssqr, "Signed square.")
sgn = _define('sgn', _n.op_sgn, "Signum function (1 for 0, as in SU).")
exp = _define('exp', _n.op_exp, "Exponentiate.")
sexp = _define('sexp', _n.op_sexp, "Signed exponentiate.")
slog = _define('slog', _n.op_slog, "Signed natural log (0 where the data is 0).")
slog2 = _define('slog2', _n.op_slog2, "Signed log base 2 (0 where the data is 0).")
slog10 = _define('slog10', _n.op_slog10, "Signed common log (0 where the data is 0).")
db = _define('db', _n.op_db, "20 * slog10(data).")
cos = _define('cos', _n.op_cos, "Cosine.")
sin = _define('sin', _n.op_sin, "Sine.")
tan = _define('tan', _n.op_tan, "Tangent.")
cosh = _define('cosh', _n.op_cosh, "Hyperbolic cosine.")
sinh = _define('sinh', _n.op_sinh, "Hyperbolic sine.")
tanh = _define('tanh', _n.op_tanh, "Hyperbolic tangent.")
neg = _define('neg', _n.op_neg, "Negate.")
nop = _define('nop', _n.op_nop, "No operation.")
posonly = _define('posonly', _n.op_posonly, "Pass only positive values (the rest become 0).")
negonly = _define('negonly', _n.op_negonly, "Pass only negative values (the rest become 0).")
inv = _define('inv', _n.op_inv, "Inverse (0 where the data is 0).")
mod2pi = _define('mod2pi', _n.op_mod2pi, "Modulo 2 pi, into [0, 2 pi).")
s2v = _define('s2v', _n.op_s2v, "Sonic to velocity (ft/s) conversion.")
s2vm = _define('s2vm', _n.op_s2vm, "Sonic to velocity (m/s) conversion.")
d2m = _define('d2m', _n.op_d2m, "Density (g/cc) to metric (kg/m^3) conversion.")
cnorm = _define('cnorm', _n.op_cnorm, "Normalize complex samples (pairs of numbers) by their modulus.")

# ----------------------------------------------------------------------------------------------- whole trace
norm = _define('norm', _n.op_norm, "Divide the trace by its maximum magnitude.")
avg = _define('avg', _n.op_avg, "Remove the average value.")
rmsamp = _define('rmsamp', _n.op_rmsamp, "The rms amplitude, in the first sample (the rest are 0).")
sum = _define('sum', _n.op_sum, "Running sum, trace integration.")
integ = _define('integ', _n.op_integ, "Top-down integration.")

# ---------------------------------------------------------------------------------------- neighbouring samples
refl = _define('refl', _n.op_refl, "(v[i] - v[i-1]) / (v[i] + v[i-1]), the reflectivity of a velocity trace.")
diff = _define(
    'diff', _n.op_diff, "Running difference, differentiation (needs the sample interval of the traces).",
    min_samples=3, needs_dt=True,
)
drv2 = _define(
    'drv2', _n.op_drv2,
    "2nd order vertical derivative: (x[i-1] - x[i]) / (2 dt), leaving the ends of the trace alone.",
    min_samples=3, needs_dt=True,
)
drv4 = _define(
    'drv4', _n.op_drv4,
    "4th order vertical derivative, the negative of the usual one, leaving the ends of the trace alone.",
    min_samples=5, needs_dt=True,
)
spike = _define('spike', _n.op_spike, "Local extrema to spikes.", min_samples=3)
lnza = _define('lnza', _n.op_lnza, "Preserve least non-zero amplitudes.", min_samples=3)
saf = _define('saf', _o.saf, "Spike and fill to the next spike.", min_samples=3)
freq = _define('freq', _o.freq, "Local dominant frequency.", min_samples=3, needs_dt=True)

# --------------------------------------------------------------------------------------------------- windows
mean = _define('mean', _n.op_mean, "Arithmetic mean in a window of nw samples.", window=True)
std = _define('std', _n.op_std, "Standard deviation in a window of nw samples.", window=True)
var = _define('var', _n.op_var, "Variance in a window of nw samples.", window=True)
despike = _define(
    'despike', _o.despike, "Despiking with a median filter of nw samples.", window=True, needs_window=True,
)

from ._panels import mix, sum2, diff2, prod2, quo2, ptsum, ptdiff, ptprod, ptquo, zipper, zippol  # noqa: E402
from ._dataset import flip, vcat  # noqa: E402

__all__ = [
    'abs', 'ssqrt', 'sqr', 'ssqr', 'sgn', 'exp', 'sexp', 'slog', 'slog2', 'slog10', 'db', 'cos', 'sin', 'tan', 'cosh',
    'sinh', 'tanh', 'neg', 'nop', 'posonly', 'negonly', 'inv', 'mod2pi', 's2v', 's2vm', 'd2m', 'cnorm',
    'norm', 'avg', 'rmsamp', 'sum', 'integ',
    'refl', 'diff', 'drv2', 'drv4', 'spike', 'lnza', 'saf', 'freq',
    'mean', 'std', 'var', 'despike',
    'mix', 'sum2', 'diff2', 'prod2', 'quo2', 'ptsum', 'ptdiff', 'ptprod', 'ptquo', 'zipper', 'zippol',
    'flip', 'vcat',
]
