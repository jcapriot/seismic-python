"""
The operations of SUOP that are plain array arithmetic.

Each is a function of its own, ``op_<name>(x, dt, nw)``, that changes the 1D float32 array ``x`` in place (the stage hands
over a zero-copy view of the samples of a trace). ``dt`` is the sample interval in seconds, and ``nw`` the (odd)
number of samples in the window of the window operations. They follow suop.c, including its habits: the signum
function is 1 at 0, logs of 0 are 0, the window operations are 0 where the window sticks out of the trace, and drv2
and drv4 leave the ends of the trace alone and are the negative of what you might expect.

The operations that are more than that (saf, freq, despike) are functions in the SU sources, see ``_ops.pyx``.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

F32 = np.float32
TWOPI = 2.0 * np.pi

def _sgn(x):
    """SU's SGN macro: -1 for negative numbers, and 1 for everything else (including 0)."""
    return np.where(x < 0, F32(-1), F32(1))


def _signed_log(x, log):
    """sgn(x) * log(|x|), or 0 where x is 0 (log(1) is 0, so that falls out by using 1 for the zeros)"""
    a = np.abs(x)
    return _sgn(x) * log(np.where(a > 0, a, F32(1)))


# ------------------------------------------------------------------------------------------------------ pointwise

def op_abs(x, dt, nw):
    np.abs(x, out=x)


def op_ssqrt(x, dt, nw):
    x[:] = _sgn(x) * np.sqrt(np.abs(x))


def op_sqr(x, dt, nw):
    np.multiply(x, x, out=x)


def op_ssqr(x, dt, nw):
    x[:] = _sgn(x) * x * x


def op_sgn(x, dt, nw):
    x[:] = _sgn(x)


def op_exp(x, dt, nw):
    np.exp(x, out=x)


def op_sexp(x, dt, nw):
    x[:] = _sgn(x) * np.exp(np.abs(x))


def op_slog(x, dt, nw):
    x[:] = _signed_log(x, np.log)


def op_slog2(x, dt, nw):
    x[:] = _signed_log(x, np.log2)


def op_slog10(x, dt, nw):
    x[:] = _signed_log(x, np.log10)


def op_db(x, dt, nw):
    x[:] = F32(20) * _signed_log(x, np.log10)


def op_cos(x, dt, nw):
    np.cos(x, out=x)


def op_sin(x, dt, nw):
    np.sin(x, out=x)


def op_tan(x, dt, nw):
    np.tan(x, out=x)


def op_cosh(x, dt, nw):
    np.cosh(x, out=x)


def op_sinh(x, dt, nw):
    np.sinh(x, out=x)


def op_tanh(x, dt, nw):
    np.tanh(x, out=x)


def op_neg(x, dt, nw):
    np.negative(x, out=x)


def op_nop(x, dt, nw):
    pass


def op_posonly(x, dt, nw):
    x[~(x > 0)] = 0


def op_negonly(x, dt, nw):
    x[~(x < 0)] = 0


def op_inv(x, dt, nw):
    nonzero = x != 0
    x[nonzero] = F32(1) / x[nonzero]


def op_mod2pi(x, dt, nw):
    x[:] = np.mod(x.astype(np.float64), TWOPI)


def op_s2v(x, dt, nw):
    # sonic to velocity (ft/s)
    with np.errstate(divide='ignore'):
        x[:] = F32(1000000.0) / x


def op_s2vm(x, dt, nw):
    # sonic to velocity (m/s)
    with np.errstate(divide='ignore'):
        x[:] = F32(304800.0) / x


def op_d2m(x, dt, nw):
    # density (g/cc) to metric (kg/m^3)
    x *= F32(1000.0)


def op_cnorm(x, dt, nw):
    # normalize complex samples (pairs of numbers) by their modulus
    pairs = x[:x.shape[0] // 2 * 2].reshape(-1, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        pairs /= np.hypot(pairs[:, 0], pairs[:, 1])[:, None]


# ------------------------------------------------------------------------------------------------ whole trace

def op_norm(x, dt, nw):
    biggest = np.abs(x).max()
    if biggest != 0:
        x /= biggest


def op_avg(x, dt, nw):
    x -= x.mean()


def op_rmsamp(x, dt, nw):
    rms = np.sqrt(np.mean(x * x))
    x[:] = 0
    x[0] = rms


def op_sum(x, dt, nw):
    # running sum, trace integration
    np.cumsum(x, out=x)


def op_integ(x, dt, nw):
    # top-down integration
    np.cumsum(x, out=x)


# --------------------------------------------------------------------------------------- neighbouring samples

def op_refl(x, dt, nw):
    # (v[i] - v[i-1]) / (v[i] + v[i-1]), with a denominator of 0 taken as 1
    numer = x[1:] - x[:-1]
    denom = x[1:] + x[:-1]
    denom[denom == 0] = 1
    x[1:] = numer / denom
    x[0] = 0


def op_diff(x, dt, nw):
    # centered differences, and one sided ones at the ends
    n = x.shape[0]
    t = x.copy()
    dt = F32(dt)
    twobydt = F32(2) * dt
    x[2:n - 2] = (t[3:n - 1] - t[1:n - 3]) / twobydt
    x[0] = (t[1] - t[0]) / dt
    x[n - 1] = (t[n - 1] - t[n - 2]) / dt
    x[1] = (t[2] - t[0]) / twobydt
    x[n - 2] = (t[n - 1] - t[n - 3]) / twobydt


def op_drv2(x, dt, nw):
    # 2nd order vertical derivative. Note the sign: it is (x[i-1] - x[i]) / (2 dt), and the ends are left alone
    n = x.shape[0]
    t = x.copy()
    x[1:n - 1] = (t[:n - 2] - t[1:n - 1]) / (F32(2) * F32(dt))


def op_drv4(x, dt, nw):
    # 4th order vertical derivative, also negative, and the ends are left alone
    n = x.shape[0]
    t = x.copy()
    stencil = t[:n - 4] - F32(8) * t[1:n - 3] + F32(8) * t[3:n - 1] - t[4:]
    x[2:n - 2] = -stencil / (F32(12) * F32(dt))


def op_spike(x, dt, nw):
    # local extrema as spikes
    t = x.copy()
    before, here, after = t[:-2], t[1:-1], t[2:]
    extremum = ((before < here) & (after < here)) | ((before > here) & (after > here))
    x[:] = 0
    x[1:-1] = np.where(extremum, here, F32(0))


def op_lnza(x, dt, nw):
    # preserve least non-zero amplitudes
    t = x.copy()
    s = _sgn(t)
    a, b, c = t[:-2], t[1:-1], t[2:]
    sa, sb, sc = s[:-2], s[1:-1], s[2:]
    same_ab = sa == sb
    x[1:-1] = np.select(
        [same_ab & (sb == sc), same_ab & (sb != sc), ~same_ab & (sb == sc)],
        [F32(0), np.where(np.abs(b) < np.abs(c), b, c), np.where(np.abs(a) < np.abs(b), a, b)],
        default=b,
    )
    x[0] = 0
    x[-1] = 0


# ------------------------------------------------------------------------------------------------------ windows

def _window_op(x, nw, stat):
    """Apply stat(windows) to windows of nw samples, which is 0 where the window sticks out of the trace.

    (SU also leaves the sample that is exactly half a window from the start at 0)
    """
    n = x.shape[0]
    half = (nw - 1) // 2
    t = x.copy()
    x[:] = 0
    if n > nw:
        windows = sliding_window_view(t, nw)[1:]
        x[half + 1:n - half] = stat(windows)


def op_mean(x, dt, nw):
    _window_op(x, nw, lambda w: w.mean(axis=1))


def op_std(x, dt, nw):
    _window_op(x, nw, lambda w: w.std(axis=1))


def op_var(x, dt, nw):
    _window_op(x, nw, lambda w: w.var(axis=1))
