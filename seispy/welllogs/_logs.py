"""
Well logs (``su/main/well_logs``): LAS2SU, SUBACKUS, SUBACKUSH, SULPRIME, SUWELLRF and SUGASSMANN.

A log is a trace of the depth samples of a well (the sample interval ``d_sample`` is the depth interval, ``sample_start`` the
first depth). The stages that take logs (``backus``, ``backush``) take the first three (or four) traces of the stream, which
are the logs in the order that the programs read them: vp, vs, rho (and the gamma ray log). ``las2su`` makes the logs from
a LAS file.

Where the program plainly does not do what its documentation says (or what could not be intended), this does what the
documentation says, and the differences are listed with each function.
"""
import collections
import itertools
import math
import warnings

import numpy as np

from ..container import Trace, as_trace_iterator, from_iterable
from ..stage import stage
from ..stretching import _resample
from . import _gassmann

__all__ = ['_backus', '_backush', 'lprime', 'wellrf', 'las2su', 'gassmann', 'rock_functions']

F32 = np.float32
_FEET_PER_METER = 3.28084


def _nint(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# ---------------------------------------------------------------------------------------------------- backus averages
def _check_window(navg, nz):
    if navg < 1:
        raise ValueError("navg must be at least 1.")
    half = (navg - 1) // 2
    if nz < 2 * half + 2:
        raise ValueError(f"The logs have {nz} samples: fewer than the {navg + 1} that the averaging window needs.")
    return half


def _backus_average(p, half):
    """The mean of p in a centered window of 2 * half + 1 samples. The samples at the ends, where the window does not
    fit, are not averaged (they are p; see ``_handle_ends``)"""
    nz = p.shape[0]
    avg = np.array(p, dtype=np.float64)
    c = np.concatenate(([0.0], np.cumsum(p, dtype=np.float64)))
    i = np.arange(half + 1, nz - half)  # (i - half > 0 and i + half < nz)
    avg[i] = (c[i + half + 1] - c[i - half]) / (2 * half + 1)
    return avg


def _handle_ends(p, half):
    """The first valid value of the average, copied to each end for a distance of the half window"""
    nz = p.shape[0]
    m = half + 1
    out = np.array(p)
    out[:m] = p[m]
    out[nz - m:] = p[nz - m]
    return out


def _first_logs(source, n, names):
    logs = list(itertools.islice(source, n))
    if len(logs) < n:
        raise ValueError(f"Needs {n} logs ({', '.join(names)}), the stream has {len(logs)}.")
    nz = logs[0].n_sample
    for log in logs:
        if log.dtype.kind == 'c':
            raise TypeError("The logs must be real, one is complex.")
        if log.n_sample != nz:
            raise ValueError("The logs must all have the same number of samples.")
    return logs, [np.asarray(log, dtype=np.float64) for log in logs]


def _backus(upstream, *, navg=201, all=False, ang=30.0):
    if navg < 1:
        raise ValueError("navg must be at least 1.")
    navg = navg // 2 * 2 + 1  # (odd)
    source = as_trace_iterator(upstream)

    def traces():
        logs, (vp, vs, rho) = _first_logs(source, 3, ('vp', 'vs', 'rho'))
        nz = vp.shape[0]
        half = _check_window(navg, nz)

        def average(p):
            return _backus_average(p, half)

        mu = rho * vs * vs
        lp2mu = rho * vp * vp
        lam = rho * (vp * vp - 2 * vs * vs)
        lpmu = rho * (vp * vp - vs * vs)
        blam = average(lam)
        bmu = average(mu)
        # c11, c13, c33, c44 and c66 of the equivalent medium
        a_term = average(lam / lp2mu) ** 2  # <lam / (lam + 2 mu)>^2
        c33_inv = average(1.0 / lp2mu)
        b_term = 4.0 * average(mu * lpmu / lp2mu)
        c11 = a_term / c33_inv + b_term
        c13 = average(lam / lp2mu) / average(1.0 / lp2mu)
        c33 = 1.0 / c33_inv
        c44 = 1.0 / average(1.0 / mu)
        c66 = average(mu)
        rhoa = average(rho)

        sina = math.sin(ang * 3.1415927 / 180.0)
        cosa = math.cos(ang * 3.1415927 / 180.0)
        sina2, cosa2 = sina * sina, cosa * cosa
        sina4 = sina2 * sina2
        eps = (c11 - c33) / (2.0 * c33)
        delta = ((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2.0 * c33 * (c33 - c44))
        gamma = (c66 - c44) / (2.0 * c44)
        vp0 = np.sqrt(c33 / rhoa)
        vs0 = np.sqrt(c44 / rhoa)
        eta = (eps - delta) / (1.0 + 2.0 * delta)
        u = 1.0 - (vs0 * vs0) / (vp0 * vp0)
        ds = (2.0 * delta - eps) * u
        dds = 0.5 * (np.sqrt(u * u + 4 * ds * sina2 * cosa2 + 4 * (u + eps) * eps * sina4) - u)
        vang = np.sqrt(1.0 + eps * sina2 + dds)  # vp(ang) / vp0

        averaged = [eps, delta, gamma]
        outputs = [handled for handled in (_handle_ends(x, half) for x in averaged)]
        if all:
            more = [vp0, vs0, rhoa, eta, vang, c11, c13, c33, c44, a_term, b_term, blam, bmu]
            outputs += [_handle_ends(x, half) for x in more]
            outputs += [vp, vs, rho]
            outputs.append(_handle_ends(c66, half))
            outputs += [lp2mu, lam, lp2mu, mu, mu]
        first = logs[0]
        for number, out in enumerate(outputs, start=1):
            yield first.replace(np.ascontiguousarray(out, dtype=F32), trace_id=number)

    return from_iterable(traces())


def _backush(upstream, *, navg=101, gr=False, grs=100.0, grc=10.0, smode=1, se=0.209, sd=0.033, sg=0.203):
    if navg < 1:
        raise ValueError("navg must be at least 1.")
    if smode not in (0, 1):
        raise ValueError("smode is 0 (the anisotropy of pure shale only) or 1 (proportional to the shale volume).")
    if grs == grc:
        raise ValueError("grs and grc must differ.")
    navg = navg // 2 * 2 + 1
    source = as_trace_iterator(upstream)

    def traces():
        names = ('vp', 'vs', 'rho', 'gr') if gr else ('vp', 'vs', 'rho')
        logs, arrays = _first_logs(source, len(names), names)
        vp, vs, rho = arrays[:3]
        nz = vp.shape[0]
        half = _check_window(navg, nz)

        def average(p):
            return _backus_average(p, half)

        # the shale volume (1 or more is pure shale, 0 or less has no shale; no shale without a gamma ray log)
        if gr:
            x = (arrays[3] - grc) / (grs - grc)
            vsh = np.clip(0.33 * (np.power(2.0, 2.0 * x) - 1.0), 0.0, 1.0)
        else:
            vsh = np.zeros(nz)

        def shale(value):
            # the intrinsic anisotropy of the layers: of pure shale only, or proportional to the shale volume
            if not gr:
                return np.zeros(nz)
            if smode == 0:
                return np.where(vsh >= 1.0, value, 0.0)
            return value * vsh

        epso, deltao, gammao = shale(se), shale(sd), shale(sg)
        ao = (1.0 + 2.0 * epso) * rho * vp * vp
        ftmp = (vp * vp - vs * vs) * ((1.0 + 2.0 * deltao) * vp * vp - vs * vs)
        with np.errstate(invalid='ignore'):
            fo = rho * (np.sqrt(ftmp) - vs * vs)
        co = rho * vp * vp
        lo = rho * vs * vs
        mo = (1.0 + 2.0 * gammao) * rho * vs * vs

        cb = 1.0 / average(1.0 / co)
        btmp = average(fo / co)
        fb = cb * btmp
        ab = average(ao - fo * fo / co) + cb * btmp * btmp
        lb = 1.0 / average(1.0 / lo)
        mb = average(mo)
        rhob = average(rho)

        vp0 = np.sqrt(cb / rhob)
        vs0 = np.sqrt(lb / rhob)
        epsb = (ab - cb) / (2.0 * cb)
        deltab = ((fb + lb) ** 2 - (cb - lb) ** 2) / (2.0 * cb * (cb - lb))
        gammab = (mb - lb) / (2.0 * lb)
        outputs = [_handle_ends(x, half) for x in (vp0, vs0, rhob, epsb, deltab, gammab)]
        outputs += [vsh, epso]
        first = logs[0]
        for number, out in enumerate(outputs, start=1):
            yield first.replace(np.ascontiguousarray(out, dtype=F32), trace_id=number)

    return from_iterable(traces())


# SUBACKUS: the Thomsen anisotropy parameters of the Backus average of well logs (vp, vs, rho in metric units; the first
# three traces of the stream). Traces: 1 epsilon, 2 delta, 3 gamma (trace_id) and, with all=True, 4 to 8: vp0, vs0, the
# averaged density, eta and vang = vp(ang) / vp0, 9 to 12: c11, c13, c33, c44 (Backus averaged), 13 and 14: <lam / (lam
# + 2 mu)>^2 and 4 <mu (lam + mu) / (lam + 2 mu)>, 15 and 16: the averaged lambda and mu, 17 to 19: the logs, 20: the
# averaged c66, 21 to 25: c11, c13, c33, c44 and c66 of the logs.
#
# navg : the samples of the averaging window (made odd), default 201; ang : the angle (degrees) of vang, default 30
# The window needs the logs to have more samples than it has (the program wrote and read outside of its arrays otherwise).
backus = stage(_backus, parallelism='serial', name='backus', validate=True)

# SUBACKUSH: like backus, with the intrinsic anisotropy of shale layers (Wang 2002) if there is a gamma ray log (gr=True: the
# fourth trace of the stream). Traces: 1 vp0, 2 vs0, 3 the averaged density, 4 epsilon, 5 delta, 6 gamma of the average,
# 7 the shale volume (Larionov), 8 the epsilon of the layers (se * volume if smode=1, se for pure shale if smode=0).
# (The program left the shale volume unset when there is no gamma ray log, and gave the epsilon of the layers that of pure
# shale in more cases than it should.)
backush = stage(_backush, parallelism='serial', name='backush', validate=True)


LPrime = collections.namedtuple('LPrime', 'backus_number navg vs0_min lengths numbers vs0_mins')


def lprime(vs, rho, *, b=2.0, dz=1.0, f=60.0, nmax=301):
    """The Backus averaging length for a log suite, a frequency and a purpose (SULPRIME).

    Parameters
    ----------
    vs, rho : array
        The shear velocity and density logs (the program takes them as the second and third traces).
    b : float
        The target Backus number: 2 is the transmission limit (processing, migration), 0.3 the scattering limit
        (modelling).
    dz : float
        The depth sample interval (ft).
    f : float
        The frequency (Hz): the dominant, or the maximum to be safe.
    nmax : int
        The largest averaging length (samples).

    Returns
    -------
    A named tuple: ``backus_number`` and ``navg`` (samples) of the length whose Backus number is closest to ``b``, ``vs0_min``
    the smallest averaged vertical shear velocity there, and ``lengths``, ``numbers`` and ``vs0_mins`` of all the odd
    lengths up to ``nmax`` that were tried.
    """
    vs = np.asarray(vs, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    if vs.ndim != 1 or rho.shape != vs.shape:
        raise ValueError("vs and rho must be 1D arrays of the same length.")
    if nmax < 1:
        raise ValueError("nmax must be at least 1.")
    lo = rho * vs * vs
    lengths, numbers, mins = [], [], []
    best = (math.inf, 0, 0.0, 0.0)
    for navg in range(1, nmax + 1, 2):
        half = (navg - 1) // 2
        with np.errstate(divide='ignore', invalid='ignore'):
            lb = 1.0 / _backus_average(1.0 / lo, half)
            rhob = _backus_average(rho, half)
            vs0 = np.sqrt(lb / rhob)
        minvs0 = float(vs0.min())
        number = f * navg * dz * 0.3048 / minvs0
        error = abs(b - number)
        lengths.append(navg)
        numbers.append(number)
        mins.append(minvs0)
        if error < best[0]:
            best = (error, navg, number, minvs0)
    return LPrime(best[2], best[1], best[3], np.array(lengths), np.array(numbers), np.array(mins))


# ------------------------------------------------------------------------------------------------------------ wellrf
def wellrf(depth, velocity, density=None, *, dtout=None, ntr=1):
    """A normal incidence reflectivity function of time from well logs of depth, velocity and density (SUWELLRF).

    A raw impedance reflectivity is made with the smallest two-way traveltime of a layer as its sample interval, and
    resampled to ``dtout`` with the 8 point sinc interpolation of the SU library.

    Parameters
    ----------
    depth, velocity, density : array
        The values at the depths, which must not decrease. The density is 1 where it is not given.
    dtout : float
        The sample interval of the trace, half the smallest two-way traveltime by default.
    ntr : int
        The number of copies of the trace in the stream.

    Differences from the program: the trace is not limited to SU_NFLTS samples, and the last reflection is not written
    after the end of the raw trace (the program wrote one sample past it).
    """
    d = np.asarray(depth, dtype=F32)
    v = np.asarray(velocity, dtype=F32)
    r = np.ones_like(d) if density is None else np.asarray(density, dtype=F32)
    if d.ndim != 1 or v.shape != d.shape or r.shape != d.shape:
        raise ValueError("depth, velocity and density must be 1D arrays of the same length.")
    nval = d.shape[0]
    if nval < 2:
        raise ValueError("Needs at least 2 values.")
    if (np.diff(d) < 0).any():
        raise ValueError("depth not monotonically increasing!")
    if (v[:-1] <= 0).any():
        raise ValueError("The velocities must be positive.")
    if ntr < 1:
        raise ValueError("ntr must be at least 1.")
    dt_layers = F32(2.0) * np.diff(d) / v[:-1]
    dtmin = F32(dt_layers.min())
    tmax = F32(0.0)
    for x in dt_layers:  # (accumulated in single precision as the program does)
        tmax = F32(tmax + x)
    if dtmin <= 0:
        raise ValueError("A layer has no thickness (or its traveltime is 0): no sample interval can be made.")
    dt_out = F32(dtout) if dtout is not None else F32(dtmin / F32(2.0))
    if dt_out <= 0:
        raise ValueError("dtout must be positive.")
    ntmax = _nint(float(tmax / dtmin))
    ntout = _nint(float(tmax / dt_out))
    if ntmax < 1 or ntout < 1:
        raise ValueError("The logs are too short for this sampling.")
    raw = np.zeros(ntmax + 1, dtype=F32)
    ttotal = F32(0.0)
    for i in range(1, nval):
        ttotal = F32(ttotal + dt_layers[i - 1])
        k = _nint(float(F32(ntmax) * ttotal / tmax))
        z1, z0 = F32(v[i] * r[i]), F32(v[i - 1] * r[i - 1])
        raw[k] = (z1 - z0) / (z1 + z0)
    trace = _resample.resample(raw, float(dtmin), 0.0, ntout, float(dt_out), 0.0)
    template = Trace(trace, d_sample=float(dt_out))
    return from_iterable((template.replace() for _ in range(ntr)), n_traces=ntr)


# ------------------------------------------------------------------------------------------------------------ las2su
def las2su(las, *, ncurve=None, dz=0.5, m=False, ss=False, null=None):
    """The curves of a LAS (version 2) file as logs (LAS2SU): a stream of traces, one for each curve.

    Parameters
    ----------
    las : path, file object, or iterable of lines
        The LAS file. The first curve must be the depth.
    ncurve : int
        The number of curves, from the curve section by default.
    dz : float
        The depth sample interval (ft) of the file, 0.5 as in the program.
    m : bool
        Make the depth interval and first depth in meters (the file is in feet).
    ss : bool
        Pass every other sample.
    null : float
        A null value of the file (-999.25 is common) that is made NaN. The program leaves them as they are.

    The traces have ``trace_id`` the number of the curve from 1 (1 is the depth), ``d_sample`` the interval and
    ``sample_start`` the first depth. Differences from the program: the number of curves does not count the line of the
    ~A section when there is no ~P section after the curves, and ``ss`` makes every other sample (the program kept every
    fourth); the number of samples is not limited to that of a SEG-Y trace, 32767.
    """
    if hasattr(las, 'read'):
        lines = las.read().splitlines()
    elif isinstance(las, (str, bytes)) or hasattr(las, '__fspath__'):
        with open(las, 'r', errors='replace') as stream:
            lines = stream.read().splitlines()
    else:
        lines = [str(line).rstrip('\r\n') for line in las]
    curves = 0
    in_curves = False
    data_start = None
    for number, line in enumerate(lines):
        if line.startswith('~A'):
            data_start = number + 1
            break
        if line.startswith('~'):
            in_curves = line.startswith('~C')
        elif in_curves and line.strip() and not line.startswith('#'):
            curves += 1
    if data_start is None:
        raise ValueError("The file has no ~A (ASCII log data) section.")
    count = curves if ncurve is None else int(ncurve)
    if count < 1:
        raise ValueError("The file has no curves.")
    rows = []
    for line in lines[data_start:]:
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) < count:
            raise ValueError(f"A line of data has {len(tokens)} values, the curves are {count}.")
        rows.append([float(token) for token in tokens[:count]])
    if not rows:
        raise ValueError("The file has no data.")
    x = np.array(rows, dtype=np.float64)
    if null is not None:
        x[x == null] = np.nan
    step = float(dz)
    if ss:
        x = x[::2]
        step *= 2.0
    first = float(x[0, 0])
    if m:
        step /= _FEET_PER_METER
        first /= _FEET_PER_METER
    return from_iterable(
        (Trace(np.ascontiguousarray(x[:, i], dtype=F32), d_sample=step, sample_start=first, sampling_unit='m').replace(
            trace_id=i + 1) for i in range(count)),
        n_traces=count)


# --------------------------------------------------------------------------------------------------------- gassmann
_DEFAULTS = dict(h=20.0, kmin=37900.0, mumin=44300.0, rmin=2.67, phi=0.24, a=1.0, b=15.0, c=1.0, api=50.0, g=0.56, s=0.35,
                 so=0.7, sg=0.2, t=140.0, p=20.0, v1=3300.0, r1=2.6)


def _rock_function(name):
    names = _gassmann.FUNCTIONS[name]

    def function(**kwargs):
        unknown = set(kwargs) - set(names)
        if unknown:
            raise TypeError(f"{name} takes {', '.join(names)}, not {', '.join(sorted(unknown))}.")
        values = [np.asarray(kwargs.get(n, _DEFAULTS[n]), dtype=F32) for n in names]
        shape = np.broadcast(*values).shape
        args = np.ascontiguousarray(np.stack([np.broadcast_to(v, shape).reshape(-1) for v in values], axis=1), dtype=F32)
        if 'so' in names and 'sg' in names:
            so, sg = values[names.index('so')], values[names.index('sg')]
            if (so + sg > 1).any():
                raise ValueError("so + sg > 1...impossible")
        return _gassmann.evaluate(name, args).reshape(shape)

    function.__name__ = name
    function.__doc__ = (
        f"{name.replace('_', ' ')} (Gassmann's fluid substitution and the Batzle and Wang fluid properties of SUGASSMANN).\n\n"
        f"Keyword arguments, scalars or arrays that broadcast: {', '.join(names)}.\n"
        "Temperature t is in degrees C and pressure p in MPa; the moduli kmin and mumin are in MPa and the densities in g/cc;\n"
        "salinity s is the weight fraction of NaCl, saturations so and sg are 0 to 1 (water is 1 - so - sg). Defaults are those\n"
        "of the program.")
    return function


rock_functions = {name: _rock_function(name) for name in _gassmann.FUNCTIONS}


def gassmann(*, mode=0, nsamples=200, ntr=200, p=0.15, v1=3300.0, r1=2.6, temp=140.0, pres=20.0, h=20.0, **rock):
    """A reflection coefficient, or a two-way time, from rock and fluid properties (SUGASSMANN).

    mode 0 makes one trace: the normal incidence reflection coefficient of a reservoir under a caprock (``v1``, ``r1``) as a
    percentage of the one at gas saturation ``sg``, for the gas saturations from (1 - p) sg to (1 + p) sg. Its
    ``sample_start`` and ``d_sample`` are the first gas saturation and the step. mode 2 makes ``ntr`` traces of the two-way
    time through a reservoir of thickness ``h``, as a function of the pressure (from 100 MPa down, ``nsamples`` samples)
    for each of ``ntr`` temperatures up to 350 degrees C. The properties of the rock and fluid are the arguments of the
    functions in ``rock_functions``: kmin, mumin, rmin, phi, a, b, c, api, g, s, so, sg (defaults of the program).

    The pressure is 20 MPa by default, as the documentation of the program has it (the code had 200 MPa, where the oil
    density formula goes negative); the caprock ``v1`` and ``r1`` are the 3300 m/s and 2.6 g/cc of the code (the
    documentation has the values of the rock mineral's bulk modulus and density by mistake).

    Differences from the program: mode 2 can be used (it was changed to 0), the step of the temperature is not an integer
    division (350 / ntr), and mode 1 (which the program documents as embedding the reflection coefficient in a random
    series but does nothing) is an error.
    """
    if mode == 1:
        raise NotImplementedError("mode 1 does nothing in SUGASSMANN: it is not implemented.")
    if mode not in (0, 2):
        raise ValueError("mode is 0 (the reflection coefficient) or 2 (the two-way time).")
    props = dict(_DEFAULTS)
    unknown = set(rock) - set(props)
    if unknown:
        raise TypeError(f"Unknown rock or fluid properties: {', '.join(sorted(unknown))}.")
    props.update(rock)
    rock_args = {k: props[k] for k in ('kmin', 'mumin', 'rmin', 'phi', 'a', 'b', 'c', 'api', 'g', 's', 'so')}
    if mode == 0:
        x = props['sg']
        n = int(nsamples)
        it = np.arange(n, dtype=np.float64)
        y = (1.0 - p) * x + 2.0 * it * p * x / n
        reflectivity = rock_functions['rock_reflectivity']
        r_y = reflectivity(v1=v1, r1=r1, sg=y, t=temp, p=pres, **rock_args)
        r_0 = reflectivity(v1=v1, r1=r1, sg=x, t=temp, p=pres, **rock_args)
        trace = (100.0 * r_y / r_0).astype(F32)
        t = Trace(np.ascontiguousarray(trace), d_sample=2.0 * p * x / n, sample_start=(1.0 - p) * x)
        return from_iterable([t], n_traces=1)
    n = int(nsamples)
    count = int(ntr)
    twt = rock_functions['rock_twt']
    pressures = 100.0 - (100.0 - 0.1) / n * np.arange(1, n + 1)
    traces = []
    for i in range(count):
        temperature = 350.0 / count * (i + 1)
        samples = twt(h=h, sg=props['sg'], t=temperature, p=pressures, **rock_args)
        traces.append(Trace(np.ascontiguousarray(samples, dtype=F32), d_sample=(100.0 - 0.1) / n, sample_start=100.0))
    return from_iterable(traces, n_traces=count)
