"""
Amplitude programs that work on one trace at a time and are plain array arithmetic: SUZERO, SUNAN and SUNORMALIZE.

They follow the SU programs (``su/main/amplitudes``). Where a program plainly does not do what its documentation says
(see the notes on each) this does what the documentation says.
"""
import numpy as np

from ..stage import header_value, per_trace, stage

F32 = np.float32


# ------------------------------------------------------------------------------------------------------------ zero
def _zero(upstream, itmax, *, itmin=0, value=0.0):
    if itmin < 0:
        raise ValueError(f"itmin = {itmin}, must not be negative")
    if itmax < itmin:
        raise ValueError("itmax < itmin, not allowed")
    value = F32(value)

    def zero_trace(trace):
        n = trace.n_sample
        # (suzero only checks itmax > nt, so for itmax == nt it writes one past the end of the trace)
        if itmax >= n:
            raise ValueError(f"itmax = {itmax}, must be < nt = {n}")
        x = np.array(trace)
        x[itmin:itmax + 1] = value
        return trace.replace(x)

    return per_trace(upstream, zero_trace, on_complex='native')


# SUZERO: zero-out (or set constant) data within a time window
#
# itmax : last time sample to zero out
# itmin : first time sample to zero out, default 0
# value : value to set, default 0
zero = stage(_zero, parallelism='trace', name='zero', validate=True)


# ------------------------------------------------------------------------------------------------------------- nan
def _fix_nans(x, value, interp):
    """Replace the NaNs and Infs of x (in place, one after the other, so a replaced sample can be a neighbor)"""
    n = x.shape[0]
    for i in np.flatnonzero(~np.isfinite(x)):
        if interp:
            if i == 0:
                if n > 1 and np.isfinite(x[1]):
                    x[0] = x[1]
                    continue
            elif i == n - 1:
                if np.isfinite(x[i - 1]):
                    x[i] = x[i - 1]
                    continue
            elif np.isfinite(x[i - 1]) and np.isfinite(x[i + 1]):
                x[i] = (x[i - 1] + x[i + 1]) / F32(2.0)
                continue
        x[i] = value


def _nan(upstream, *, value=0.0, interp=False):
    value = F32(value)

    def nan_trace(trace):
        x = np.array(trace)
        _fix_nans(x, value, interp)
        return trace.replace(x)

    return per_trace(upstream, nan_trace, on_complex='native')


# SUNAN: remove NaNs and Infs
#
# value : what NaNs and Infs are replaced with, default 0
# interp : replace them with the average of their neighbors instead (the first and last samples take the one neighbor
#          they have), when those are finite. Otherwise they are replaced by `value`.
#
# (sunan has an `else` missing, so that it sets every NaN to `value` even after interpolating, and for the last
# sample it looks at the sample before the previous one. This does what its documentation says.)
nan = stage(_nan, parallelism='trace', name='nan', validate=True)


# ------------------------------------------------------------------------------------------------------- normalize
def _norm_rms(w):
    return np.sqrt(np.mean(np.abs(w).astype(np.float64) ** 2))


def _norm_max(w):
    return np.max(np.abs(w))


def _norm_median(w):
    if np.iscomplexobj(w):
        raise TypeError("The median of complex samples is not defined.")
    return np.median(w)


# name -> (what to find in the window, how to apply it: divide or subtract)
_NORMS = {
    'rms': (_norm_rms, 'divide'),
    'max': (_norm_max, 'divide'),
    'med': (_norm_median, 'divide'),
    'median': (_norm_median, 'divide'),
    'balmed': (_norm_median, 'subtract'),
}


def _normalize(upstream, norm='rms', *, t0=0.0, t1=None, dt=None):
    if norm not in _NORMS:
        raise ValueError(f"unknown norm {norm!r}, expected one of {sorted(_NORMS)}")
    measure, how = _NORMS[norm]

    def normalize_trace(trace):
        n = trace.n_sample
        sample_dt = trace.d_sample or dt
        if not sample_dt:
            raise ValueError("The trace has no sample interval, and no `dt` was given.")
        it0 = int(F32(t0) / F32(sample_dt))
        # (the whole trace by default. sunormalize has ns*dt/dt, which can round to just under ns)
        it1 = n if t1 is None else int(F32(t1) / F32(sample_dt))
        if not 0 <= it0 < it1 <= n:
            raise ValueError(f"The window of samples {it0} to {it1} is not inside the {n} samples of the trace.")
        x = np.array(trace)
        level = F32(measure(x[it0:it1]))
        if how == 'divide':
            x /= level if level != 0 else F32(1)
        else:
            x -= level
        return trace.replace(x)

    return per_trace(upstream, normalize_trace, on_complex='native')


# SUNORMALIZE: trace normalization by rms, max, or median, or median balancing
#
# norm : 'rms' (divide by the rms amplitude), 'max' (by the maximum magnitude), 'med' or 'median' (by the median
#        value), 'balmed' (subtract the median value). A level of 0 is taken to be 1.
# t0, t1 : the time window the level is found in, default the whole trace
# dt : sample interval in seconds for traces that do not have one
#
# (sunormalize accumulates sums over traces without resetting them, so that its "rms" is a growing sum of squares
# and its "max" the largest of every trace so far, and takes the wrong sample as the median of an even number of
# samples. This takes the levels of every trace on its own.)
normalize = stage(_normalize, parallelism='trace', name='normalize', validate=True)


# ----------------------------------------------------------------------------------------------------------- weight
def _weight(upstream, *, key='offset', a=1.0, b=0.0005, key2=None, scale=0.0001, inv=False):
    a, b, scale = F32(a), F32(b), F32(scale)

    def weight_trace(trace):
        if key2 is None:
            factor = a + F32(header_value(trace, key)) * b
        else:
            factor = F32(header_value(trace, key2)) * scale
        x = np.array(trace)
        with np.errstate(divide='ignore', invalid='ignore'):
            if inv:
                x /= factor
            else:
                x *= factor
        return trace.replace(x)

    return per_trace(upstream, weight_trace, on_complex='native')


# SUWEIGHT: weight traces by a header value, such as the offset
#
# key : the header value (or a function of a trace), default 'offset'. The traces are multiplied by a + b * key.
# a, b : constants, default 1 and .0005
# key2 : instead, use the values of this header value, times `scale` (default .0001), as the weights
# inv : divide by the weights instead of multiplying
weight = stage(_weight, parallelism='trace', name='weight', validate=True)


# ------------------------------------------------------------------------------------------- ai2r and r2ai
def _ai2r(upstream):
    def ai2r_trace(trace):
        x = np.array(trace)
        r = np.zeros_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            r[:-1] = -(x[1:] - x[:-1]) / (x[1:] + x[:-1])
        return trace.replace(r)

    return per_trace(upstream, ai2r_trace)


# SUAI2R: acoustic impedance to reflectivity (forward modeling), -(a[i+1] - a[i]) / (a[i+1] + a[i]). The last
# sample is 0.
ai2r = stage(_ai2r, parallelism='trace', name='ai2r', validate=True)


def _r2ai(upstream):
    def r2ai_trace(trace):
        r = np.asarray(trace)
        n = r.shape[0]
        ai = np.empty(n, dtype=np.float32)
        if n:
            ai[0] = 1.0
        if n > 1:
            used = r[:-1]
            bad = np.flatnonzero(np.abs(used) > 1.0)
            if bad.size:
                raise ValueError(f"Illegal Reflectivity={used[bad[0]]:g} ... abs value must be <1")
            with np.errstate(divide='ignore', invalid='ignore'):
                ai[1:] = np.cumprod((F32(1.0) - used) / (F32(1.0) + used), dtype=np.float32)
        # (sur2ai gives every trace the header of the first one)
        return trace.replace(ai)

    return per_trace(upstream, r2ai_trace)


# SUR2AI: reflectivity to acoustic impedance (inversion), a[i+1] = a[i] * (1 - r[i]) / (1 + r[i]), starting from 1
r2ai = stage(_r2ai, parallelism='trace', name='r2ai', validate=True)


# ---------------------------------------------------------------------------------------------------------- divcor
def _divcor(upstream, *, trms=None, vrms=None, vt=None):
    n_t = 0 if trms is None else np.atleast_1d(trms).shape[0]
    n_v = 0 if vrms is None else np.atleast_1d(vrms).shape[0]
    if n_t != n_v:
        raise ValueError("number of trms and vrms must be equal")
    if vt is None:
        if n_t > 1 and (np.diff(np.atleast_1d(trms)) <= 0).any():
            raise ValueError("trms must increase monotonically")
    else:
        vt = np.asarray(vt, dtype=np.float64)
        if vt.ndim != 1 or vt.shape[0] < 2:
            raise ValueError("vt must have a velocity for each sample, at least 2 of them.")
    cache = {}

    def correction(n, dt, tmin):
        key = (n, dt, tmin)
        if key not in cache:
            t = tmin + np.arange(n) * dt
            if vt is not None:
                if vt.shape[0] != n:
                    raise ValueError(f"vt has {vt.shape[0]} velocities, the trace has {n} samples")
                v = vt
                denom = (tmin + dt) * vt[1] ** 2
            else:
                times = np.array([0.0, dt]) if trms is None else np.atleast_1d(np.asarray(trms, dtype=np.float64))
                velocities = np.array([1500.0, 1500.0]) if vrms is None else np.atleast_1d(np.asarray(vrms, dtype=np.float64))
                if times.shape[0] == 1:  # (one value is a constant velocity)
                    times = np.array([times[0], times[0] + dt])
                    velocities = np.array([velocities[0], velocities[0]])
                v = np.interp(t, times, velocities)  # (linear, and constant outside)
                # (the second pick, as in sudivcor)
                denom = times[1] * velocities[1] ** 2
            cache.clear()
            cache[key] = (t * v * v / denom).astype(F32)
        return cache[key]

    def divcor_trace(trace):
        x = np.asarray(trace)
        dt = trace.d_sample
        if not dt:
            raise ValueError("The trace has no sample interval.")
        return trace.replace(x * correction(x.shape[0], dt, trace.header['sample_start']))

    return per_trace(upstream, divcor_trace, on_complex='native')


# SUDIVCOR: divergence (spreading) correction, which multiplies the trace by t v(t)^2 / (t1 v1^2), where v is the rms
# velocity function.
#
# trms, vrms : times (s) that must increase and the rms velocities for them. Linearly interpolated, constant outside.
#              One pair is a constant velocity. Default 1500 m/s.
# vt : a velocity for every sample of the traces (which makes trms and vrms not used)
#
# t1 and v1 are the second time and velocity of the function (as sudivcor has it, which is not the first: with no
# function given this is t / dt). The time of the first sample is the start time of the trace.
divcor = stage(_divcor, parallelism='trace', name='divcor', validate=True)


# -------------------------------------------------------------------------------------------------------- centsamp
def _centsamp(upstream, *, nvals_min=1, dt=None):
    if nvals_min < 1:
        raise ValueError("nvals_min must be at least 1")
    from ._centsamp import centsamp as _centsamp_trace

    def centsamp_trace(trace):
        sample_dt = trace.d_sample or dt or 0.004  # (as SU does, .004 if there is nothing at all)
        return trace.replace(_centsamp_trace(np.ascontiguousarray(trace, dtype=np.float32), sample_dt, nvals_min))

    return per_trace(upstream, centsamp_trace)


# SUCENTSAMP: replace each lobe of the trace (between zero crossings, and where it has an inflection) with a spike at its
# centroid, which is the area of the lobe.
#
# nvals_min : the least number of samples in a lobe for it to be kept, default 1
# dt : sample interval (s) for traces that do not have one
#
# (The last lobe of a trace is not ended by a zero crossing, so it is not given a spike.)
centsamp = stage(_centsamp, parallelism='trace', name='centsamp', validate=True)


# ------------------------------------------------------------------------------------------------------ impedance
def _impedance(upstream, *, v0=1500.0, rho0=1.0e6):
    z0 = v0 * rho0

    def impedance_trace(trace):
        r = np.clip(np.asarray(trace).astype(np.float64), -0.9999, 0.9999)
        # z[k] = z[k - 1] (1 + r[k - 1]) / (1 - r[k - 1]), and z[0] = v0 rho0
        ratios = (1.0 + r) / (1.0 - r)
        z = z0 * np.concatenate(([1.0], np.cumprod(ratios[:-1])))
        return trace.replace(z.astype(F32))

    return per_trace(upstream, impedance_trace)


# SUIMPEDANCE: reflection coefficients to impedances, [1 - R(k)] Z(k) = [1 + R(k)] Z(k - 1), starting from
# v0 rho0 at the first sample.
#
# v0 : the velocity (m/s) at the first sample, default 1500
# rho0 : the density (g/m^3) at the first sample, default 1e6
#
# The reflection coefficients are limited to +-.9999. (`r2ai` is the same with the opposite sign convention for the
# reflection coefficient, and the impedance starting at 1.)
impedance = stage(_impedance, parallelism='trace', name='impedance', validate=True)
