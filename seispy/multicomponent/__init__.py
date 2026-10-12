"""
Multicomponent data (``su/main/multicomponent``): SUHROT (``hrot``), SUPOLAR (``polar``), SUPOFILT (``pofilt``), SUEIPOFI
(``eipofi``), SULTT (``ltt``) and SUALFORD (``alford``).

SU and SEG-Y have no model of a multicomponent station: the programs take three adjacent traces as one three-component
station (the vertical component, then the two horizontal ones), and the shear wave programs four volumes. Here a station is
a group of traces, found by one of these:

* ``key=None`` (default): consecutive traces, three (or four) at a time, as the programs do;
* ``key``: the name of a value of the header (or a function of a trace) that the traces of a station share, as for a gather
  of ``velan``; each group needs exactly three (or four) traces;
* ``components``: the ``trace_type`` of each component, to put the traces of a station in order: SEG-Y has 12 for a vertical
  sensor, 13 for the cross-line and 14 for the in-line horizontal ones (and 15, 16 and 17 for the rotated vertical,
  transverse and radial ones), so ``components=(12, 14, 13)`` makes an in-line component the "north" of the station.

The shear wave programs take the traces of a station in the order S11, S12, S21, S22 (inline source and receiver, inline
source and cross-line receiver, cross-line source and inline receiver, cross-line source and receiver).

Where the program plainly does not do what its documentation says this does what the documentation says; the differences are
with each stage.
"""
import math
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stage import Stage, header_value
from . import _kernels

__all__ = ['hrot', 'polar', 'pofilt', 'eipofi', 'ltt', 'alford']

F32 = np.float32

# the trace_type codes of SEG-Y (and SU) for the rotated components
ROTATED_VERTICAL, TRANSVERSE, RADIAL = 15, 16, 17

_ANGLE_UNITS = {'rad': 1.0, 'deg': 180.0 / math.pi, 'gon': 200.0 / math.pi}


def _real(trace, name):
    if trace.dtype.kind == 'c':
        raise TypeError(f"{name} works on real traces, but was given a complex one.")
    return np.ascontiguousarray(np.asarray(trace), dtype=F32)


def _ordered(group, components, size):
    if components is None:
        return list(group)
    if len(components) != size:
        raise ValueError(f"components needs a trace_type for each of the {size} components")
    out = []
    for code in components:
        found = [t for t in group if t.header['trace_type'] == code]
        if len(found) != 1:
            raise ValueError(f"the station has {len(found)} traces of trace_type {code}, and needs 1")
        out.append(found[0])
    return out


def _stations(source, size, key, components, name):
    """The stations of a stream: lists of ``size`` traces (see the module's notes)"""
    if key is None:
        pending = []
        for trace in source:
            pending.append(trace)
            if len(pending) == size:
                yield _ordered(pending, components, size)
                pending = []
        if pending:
            warnings.warn(f"{name}: the last {len(pending)} trace(s) do not make a station, and are left out")
        return
    for group in group_by(source, key):
        if len(group) != size:
            raise ValueError(f"{name}: a station needs {size} traces, but there are {len(group)} with the same key")
        yield _ordered(group, components, size)


def _check_station_lengths(samples):
    if any(s.shape != samples[0].shape for s in samples):
        raise ValueError("The traces of a station must have the same number of samples.")


# ------------------------------------------------------------------------------------------------------------ hrot
def _hrot(upstream, *, angle='rad', inv=False, a=None, x=None, angle_key='ensemble_trace_number', key=None,
          components=None):
    if angle not in _ANGLE_UNITS:
        raise ValueError(f"unknown angle={angle!r}: rad, deg or gon")
    factor = _ANGLE_UNITS[angle]
    if a is not None:
        a_values = np.atleast_1d(np.asarray(a, dtype=np.float64)) / factor
        x_values = np.zeros(1) if x is None else np.atleast_1d(np.asarray(x, dtype=np.float64))
        if x_values.shape != a_values.shape:
            raise ValueError("number of values in x and a must be equal")
        if (np.diff(x_values) == 0).any() or not (np.all(np.diff(x_values) > 0) or np.all(np.diff(x_values) < 0)):
            raise ValueError("x must be monotonically increasing or decreasing")
        if x_values[0] > x_values[-1]:
            x_values, a_values = x_values[::-1], a_values[::-1]
    source = as_trace_iterator(upstream)

    def traces():
        for station in _stations(source, 3, key, components, 'hrot'):
            z, n, e = (_real(t, 'hrot') for t in station)
            _check_station_lengths([z, n, e])
            if a is not None:
                where = header_value(station[0], angle_key)
                phi = float(np.interp(where, x_values, a_values))
            else:
                # the azimuth of the receiver from the source (clockwise from north), by the geometry of the last trace
                tx, rx = station[-1].header['tx_loc'], station[-1].header['rx_loc']
                dx, dy = rx[0] - tx[0], rx[1] - tx[1]
                if dx == 0.0 and dy == 0.0:
                    phi = 0.0
                else:
                    phi = math.atan2(dx, dy)
                    if phi < 0.0:
                        phi += 2.0 * math.pi
            if inv:
                phi = -phi
            # the axes are rotated clockwise by phi: the radial one is the azimuth phi
            c, s = math.cos(phi), math.sin(phi)
            radial = c * n + s * e
            transverse = -s * n + c * e
            yield station[0].replace(z, trace_type=ROTVERT)
            yield station[1].replace(radial.astype(F32), trace_type=RADIAL)
            yield station[2].replace(transverse.astype(F32), trace_type=TRANSVERSE)

    return from_iterable(traces())


ROTVERT = ROTATED_VERTICAL


def hrot(*, angle='rad', inv=False, a=None, x=None, angle_key='ensemble_trace_number', key=None, components=None):
    """Horizontal rotation of three-component data from Z-North-East to Z-Radial-Transverse (SUHROT).

    Parameters
    ----------
    angle : {'rad', 'deg', 'gon'}
        The unit of ``a`` (the angles that are given).
    inv : bool
        The inverse rotation (counter-clockwise).
    a, x : arrays, optional
        Rotation angles ``a`` for the values ``x`` (monotonic) of ``key`` of the first trace of a station, interpolated
        linearly and constant outside of them. By default the angle of each station is worked out from its geometry: the
        azimuth of the receiver from the source, clockwise from north (``tx_loc`` and ``rx_loc`` of the last trace of the
        station, where x is east and y is north).
    angle_key : header name or function
        What the angles ``x`` are for (the program's ``key=tracf``), default ``ensemble_trace_number``.
    key, components :
        As in the notes of this module: what the traces of a station share, and their ``trace_type`` (the vertical, north
        and east ones) to put them in order.

    The output has three traces for each station: the vertical (``trace_type`` 15), the radial (17) and the transverse
    (16), with the headers of the traces that they come from. The radial component is along the azimuth of the receiver
    from the source: the program's rotation is by minus that angle (the radial trace of a wave along the azimuth ``phi`` is
    ``cos(2 phi)`` of it, not all of it), and it does not rotate by 90 degrees (or 270) for the stations where the receiver
    is along a line of one axis (it sets the angle to 0 whenever one of ``dx`` and ``dy`` is 0, after it has worked it
    out): both are corrected here.
    """
    kwargs = dict(angle=angle, inv=inv, a=a, x=x, angle_key=angle_key, key=key, components=components)
    _hrot((), **kwargs)  # check the parameters now
    return Stage(_hrot, parallelism='serial', name='hrot', **kwargs)


# ---------------------------------------------------------------------------------------------------------- polar
_FLAG_NAMES = ('rl', 'theta', 'phi', 'tau', 'ellip', 'pln', 'f1', 'l1', 'amp', 'dir')


def polar(traces, *, wl=0.1, win='boxcar', rlq=1.0, rl=1, theta=0, phi=0, tau=0, ellip=0, pln=0, f1=0, l1=0, amp=0,
          dir=1, angle='rad', all=0, dt=None, key=None, components=None):
    """Polarization analysis of three-component data (SUPOLAR, Jurkevics 1988).

    The covariance matrix of the three components, in a window of ``wl`` seconds around each sample, is diagonalized, and
    the attributes are made from its eigenvalues and eigenvectors. This is a function of all of the traces: it returns a
    dictionary, where the program writes a file for each attribute.

    Parameters
    ----------
    traces : iterable of Trace
        The stations (see the notes of this module), the vertical component first.
    wl : float
        The correlation window (s), default .1.
    win : {'boxcar', 'bartlett', 'hanning', 'welsh'}
        Its shape. (A tapered window needs more length: the effective widths are 1/2 for Bartlett and Hanning, and 1/3 for
        Welsh.)
    rlq : float
        The contrast of the rectilinearity, default 1.
    rl, theta, phi : int
        The options of the rectilinearity (1, 2, 3: Kanasewich, Jurkevics and Meyer), of the incidence angle (1, 2, 3) and
        of the horizontal azimuth (1, 2, 3) of the principal axis; 0 for none.
    tau, ellip, pln, f1, l1, amp, dir : int
        1 for the global polarization parameter, the ellipticities (``e21``, ``e31`` and ``e32``), the planarity, the
        flatness and linearity coefficients, the amplitude parameters (the eigenresultant ``er``, and the instantaneous
        and quadratic resultants ``ir`` and ``qr``), and the three components of the direction of polarization.
    angle : {'rad', 'deg', 'gon'}
        The unit of the angles ``theta`` and ``phi``.
    all : int
        If it is not 0, the flags that are not given are set to it (1, 2 or 3).
    key, components :
        As in the module notes.

    Returns
    -------
    A dict with a list of traces for each attribute that was asked for, one for each station (``'dir'`` has the three
    components of each: ``'dir'`` is a list of 3 * stations traces), with the header of the first trace of the station.
    The first and last ``wl / (2 dt)`` samples of the attributes are 0.

    The program's ``phi=1`` uses the wrong element of the eigenvector (its test is the vertical component of the second
    one, where it means the north component of the first), and the quadratic resultant is made over a window that is
    half a window too long (reading past the end of the trace for the last samples).
    """
    if win not in _kernels.WINDOWS:
        raise ValueError(f"unknown win={win!r}")
    if angle not in _ANGLE_UNITS:
        raise ValueError(f"unknown angle={angle!r}: rad, deg or gon")
    flags = dict(rl=rl, theta=theta, phi=phi, tau=tau, ellip=ellip, pln=pln, f1=f1, l1=l1, amp=amp, dir=dir)
    if all:
        # (the program: the unset ones are `all` for rl, theta and phi and 1 for the others)
        flags = dict(rl=rl if rl != 1 else all, theta=theta or all, phi=phi or all, tau=tau or 1, ellip=ellip or 1,
                     pln=pln or 1, f1=f1 or 1, l1=l1 or 1, amp=amp or 1, dir=dir or 1)
    results = {}

    def want(name):
        return results.setdefault(name, [])

    for station in _stations(as_trace_iterator(traces), 3, key, components, 'polar'):
        samples = [_real(t, 'polar') for t in station]
        _check_station_lengths(samples)
        n = samples[0].shape[0]
        sample_dt = station[0].d_sample or dt
        if not sample_dt:
            sample_dt = 0.004
            warnings.warn("dt not set, assuming dt=0.004")
        iwl = int(wl / sample_dt + 0.5) if wl > 0 else 0
        if iwl < 1:
            raise ValueError(f"wl={wl} must be positive")
        if iwl > n:
            raise ValueError(f"wl={wl} too long for the trace")
        out = _kernels.polar_station(np.stack(samples), iwl, _kernels.WINDOWS[win], F32(rlq), int(flags['rl']),
                                     int(flags['theta']), int(flags['phi']), int(flags['tau']), int(flags['ellip']),
                                     int(flags['pln']), int(flags['f1']), int(flags['l1']), int(flags['amp']),
                                     int(flags['dir']), F32(_ANGLE_UNITS[angle]))
        header = station[0]

        def emit(name, row):
            want(name).append(header.replace(np.ascontiguousarray(out[row])))

        if flags['rl']:
            emit('rl', 0)
        if flags['theta']:
            emit('theta', 1)
        if flags['phi']:
            emit('phi', 2)
        if flags['tau']:
            emit('tau', 3)
        if flags['ellip']:
            emit('e21', 4)
            emit('e31', 5)
            emit('e32', 6)
        if flags['pln']:
            emit('pln', 7)
        if flags['f1']:
            emit('f1', 8)
        if flags['l1']:
            emit('l1', 9)
        if flags['amp']:
            emit('er', 10)
            emit('ir', 11)
            emit('qr', 12)
        if flags['dir']:
            for i, component in enumerate(station):
                want('dir').append(component.replace(np.ascontiguousarray(out[13 + i])))
    return results


# --------------------------------------------------------------------------------------------------------- pofilt
def _smooth(x, isl):
    """The average of the isl samples from it - isl / 2, 0 for the samples that do not have all of them"""
    n = x.shape[0]
    out = np.zeros(n)
    half = isl // 2
    if n > 2 * half:
        csum = np.concatenate([[0.0], np.cumsum(x.astype(np.float64))])
        it = np.arange(half, n - half)
        out[half:n - half] = (csum[it + half] - csum[it - half]) / isl
    return out


def _pofilt(upstream, *, direction, weights, dt=None, sl=0.05, smooth=True, wpow=1.0, dpow=1.0, key=None,
            components=None):
    if sl <= 0.0 and smooth:
        raise ValueError(f"sl={sl} must be positive")
    directions = [np.abs(np.asarray(d, dtype=np.float64)) for d in direction]
    values = [np.asarray(w, dtype=np.float64) for w in weights]
    source = as_trace_iterator(upstream)

    def traces():
        for station_index, station in enumerate(_stations(source, 3, key, components, 'pofilt')):
            samples = [_real(t, 'pofilt') for t in station]
            _check_station_lengths(samples)
            n = samples[0].shape[0]
            sample_dt = station[0].d_sample or dt
            if not sample_dt:
                sample_dt = 0.004
                warnings.warn("dt not set, assuming dt=0.004")
            isl = int(sl / sample_dt + 0.5)
            if smooth and isl < 1:
                raise ValueError(f"sl={sl} must be positive")
            if smooth and isl > n:
                raise ValueError(f"sl={sl} too long for the trace")
            if 3 * station_index + 2 >= len(directions) or station_index >= len(values):
                raise ValueError("There are fewer direction and weight traces than stations")
            w = values[station_index] ** wpow
            if smooth:
                w = _smooth(w, isl)
            for i, (trace, x) in enumerate(zip(station, samples)):
                d = directions[3 * station_index + i] ** dpow
                if smooth:
                    d = _smooth(d, isl)
                if d.shape[0] != n or w.shape[0] != n:
                    raise ValueError("The direction and weight traces must have the length of the traces")
                yield trace.replace((x * d * w).astype(F32))

    return from_iterable(traces())


def pofilt(direction, weights, *, dt=None, sl=0.05, smooth=True, wpow=1.0, dpow=1.0, key=None, components=None):
    """Polarization filter of three-component data (SUPOFILT), with the results of ``polar``.

    Each component of a station is multiplied by the (smoothed) absolute value of the corresponding component of the
    direction of polarization, to the power ``dpow``, and by the (smoothed) weighting function (usually the
    rectilinearity), to the power ``wpow``.

    Parameters
    ----------
    direction : list of Trace or arrays
        The three components of the direction of polarization of each station (``polar(...)['dir']``).
    weights : list of Trace or arrays
        The weighting function of each station (``polar(...)['rl']``).
    sl : float
        The length (s) of the smoothing window of the filter, default .05.
    smooth : bool
        Smooth the filter operators, default True.
    wpow, dpow : float
        The powers of the weighting and the directivity functions, default 1.
    dt : float
        The sample interval for traces that do not have one.
    """
    kwargs = dict(direction=direction, weights=weights, dt=dt, sl=sl, smooth=smooth, wpow=wpow, dpow=dpow, key=key,
                  components=components)
    _pofilt((), **kwargs)  # check the parameters now
    return Stage(_pofilt, parallelism='serial', name='pofilt', **kwargs)


# --------------------------------------------------------------------------------------------------------- eipofi
def _eipofi(upstream, *, wl=0.1, pwr=1.0, interp='cubic', dt=None, report=None, key=None, components=None):
    if interp not in ('cubic', 'linear'):
        raise ValueError(f"unknown interp={interp!r}: cubic or linear")
    if pwr < 0.0:
        raise ValueError(f"pwr={pwr} must be positive")
    source = as_trace_iterator(upstream)
    if report is not None:
        for name in ('rl1', 'rl2', 'pln'):
            report.setdefault(name, [])

    def traces():
        for station in _stations(source, 3, key, components, 'eipofi'):
            samples = [_real(t, 'eipofi') for t in station]
            _check_station_lengths(samples)
            n = samples[0].shape[0]
            sample_dt = station[0].d_sample or dt
            if not sample_dt:
                sample_dt = 0.004
                warnings.warn("dt not set, assuming dt=0.004")
            iwl = int(wl / sample_dt + 0.5)
            if iwl < 3:
                raise ValueError(f"wl={wl} must be at least {3 * sample_dt} s (3 samples)")
            if iwl > n:
                raise ValueError(f"wl={wl} too long for the trace")
            cubic = interp == 'cubic' and n // iwl >= 2
            filtered, weights = _kernels.eipofi_station(np.stack(samples), iwl, F32(pwr), cubic)
            if report is not None:
                for name, row in zip(('rl1', 'rl2', 'pln'), weights):
                    report[name].append(row.copy())
            for trace, row in zip(station, filtered):
                yield trace.replace(np.ascontiguousarray(row))

    return from_iterable(traces())


def eipofi(*, wl=0.1, pwr=1.0, interp='cubic', dt=None, report=None, key=None, components=None):
    """Polarization filter of three-component data by the singular value decomposition of eigenimages (SUEIPOFI, after
    de Franco and Musacchio).

    The filter is the sum of the first two eigenimages of the SVD of each (non overlapping) time window, weighted by
    functions of the rectilinearity along the two axes, and of the planarity, interpolated between the centers of the
    windows. It assumes that the particle motion is mostly in a plane (elliptical polarization).

    Parameters
    ----------
    wl : float
        The length (s) of the windows, default .1 (at least 3 samples).
    pwr : float
        The exponent of the weights, default 1.
    interp : {'cubic', 'linear'}
        The interpolation of the weights (cubic ones are clipped to 0 to 1). Cubic needs 2 windows.
    report : dict, optional
        If given, it is given the lists ``'rl1'``, ``'rl2'`` and ``'pln'`` of the weights of each station (the program
        writes them to files).

    The samples after the last whole window are not filtered (they are left as they are), as in the program, and the
    linear interpolation reads the end value from the right place (the program reads past the weights).
    """
    kwargs = dict(wl=wl, pwr=pwr, interp=interp, dt=dt, report=report, key=key, components=components)
    _eipofi((), **kwargs)  # check the parameters now
    return Stage(_eipofi, parallelism='serial', name='eipofi', **kwargs)


# ------------------------------------------------------------------------------------------------------------ ltt
def _mixed(history, count):
    """The mean of the last ``count`` entries (all of the ones there are, at the start)"""
    return np.mean(np.stack(history[-count:]), axis=0)


def _running_sum(x, window):
    """The sums over a window of an odd number of samples (zero outside of the trace)"""
    kernel = np.ones(window)
    return np.convolve(x, kernel, mode='same')


def _ltt_attributes(s11, s12, s21, s22, wl, mode):
    window = wl if wl % 2 == 1 else wl + 1  # (the running window always has an odd number of samples)
    n = s11.shape[0]
    alpha = np.zeros(n)
    theta = np.zeros(n)
    gamma = np.zeros(n)
    if mode == 1:
        zeta, chi = s11 + s22, s21 - s12
        z2 = _running_sum(zeta * zeta, window)
        c2 = _running_sum(chi * chi, window)
        zc = _running_sum(zeta * chi, window)
        # the eigenvalues of the matrix (Asym_analysis)
        d = np.sqrt(np.maximum((z2 - c2) ** 2 - 4 * zc ** 2, 0.0))
        major = 0.5 * (z2 + c2 + d)
        minor = 0.5 * (z2 + c2 - d)
        with np.errstate(divide='ignore', invalid='ignore'):
            gamma = np.where(major > 1e-10, minor / np.where(major > 1e-10, major, 1.0), 0.0)
        theta = np.degrees(0.5 * np.arctan2(2 * zc, z2 - c2))
        return alpha, theta, gamma
    xi, eta = s11 - s22, s12 + s21
    if mode == 2:
        xi_s = _running_sum(xi, window)
        eta_s = _running_sum(eta, window)
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = 0.5 * np.degrees(np.arctan(eta_s / xi_s))
        alpha = np.nan_to_num(alpha)
        return alpha, theta, gamma
    xi2 = _running_sum(xi * xi, window)
    eta2 = _running_sum(eta * eta, window)
    xieta = _running_sum(xi * eta, window)
    principal = np.degrees(0.5 * np.arctan2(2 * xieta, xi2 - eta2))
    return 0.5 * principal, theta, gamma


def _ltt(upstream, *, mode=2, wl=3, ntraces=5, key=None, components=None):
    if mode not in (1, 2, 3):
        raise ValueError(f"mode={mode} must be 1, 2 or 3")
    if wl < 1:
        raise ValueError("wl must be at least 1")
    if ntraces < 1:
        raise ValueError("ntraces must be at least 1")
    source = as_trace_iterator(upstream)

    def traces():
        history = [[], [], [], []]
        for station in _stations(source, 4, key, components, 'ltt'):
            samples = [_real(t, 'ltt') for t in station]
            _check_station_lengths(samples)
            for h, x in zip(history, samples):
                h.append(x.astype(np.float64))
                del h[:-ntraces]
            mixed = [_mixed(h, ntraces) for h in history]
            alpha, theta, gamma = _ltt_attributes(*mixed, int(wl), mode)
            if mode == 1:
                yield station[0].replace(theta.astype(F32))
                yield station[0].replace(gamma.astype(F32))
            else:
                yield station[0].replace(alpha.astype(F32))

    return from_iterable(traces())


def ltt(*, mode=2, wl=3, ntraces=5, key=None, components=None):
    """Instantaneous polarization angles of shear wave data by the linear transform technique of Li and Crampin (SULTT).

    A station is four traces, S11, S12, S21, S22 (see the notes of this module). The ``ntraces`` neighbouring stations
    are averaged, and the attributes are found from sliding window sums of the combinations of the four components.

    Parameters
    ----------
    mode : {1, 2, 3}
        1 the asymmetry indexes: two traces for each station, the misalignment angle ``theta`` (degrees) between the
        sources and the receivers, and the asymmetry ``gamma`` of the medium; 2 the polarization angle ``alpha`` (degrees)
        from an arctangent; 3 the same from eigenvalues, which is more robust.
    wl : int
        The sliding window (samples), always odd (an even one is made 1 longer, as the program's documentation says),
        default 3.
    ntraces : int
        The number of stations averaged, default 5.

    The output traces have the header of the S11 trace. The average is of the stations that there are at the start of
    the stream (the program averages with zeros for the stations before the first, so that the first ones are small); the
    program also does not make the last station and is only able to read the volumes together as it does.
    """
    kwargs = dict(mode=mode, wl=wl, ntraces=ntraces, key=key, components=components)
    _ltt((), **kwargs)  # check the parameters now
    return Stage(_ltt, parallelism='serial', name='ltt', **kwargs)


# --------------------------------------------------------------------------------------------------------- alford
def _alford(upstream, *, angle_inc=5.0, tmin=None, tmax=None, taper=3, taperwin=10, maxlag=40.0, ntraces=5, report=None,
            dt=None, key=None, components=None):
    if ntraces < 1:
        raise ValueError("ntraces must be at least 1")
    source = as_trace_iterator(upstream)
    if report is not None:
        for name in ('azimuth', 'quality', 'input_quality', 'lag', 'xcorr'):
            report.setdefault(name, [])

    def traces():
        history = [[], [], [], []]
        for station in _stations(source, 4, key, components, 'alford'):
            samples = [_real(t, 'alford') for t in station]
            _check_station_lengths(samples)
            n = samples[0].shape[0]
            sample_dt = station[0].d_sample or dt
            if not sample_dt:
                raise ValueError("The trace has no sample interval, and no `dt` was given.")
            ft = station[0].header['sample_start']
            itmin = 0 if tmin is None else int(np.floor((tmin - ft) / sample_dt + 0.5))
            itmax = n - 1 if tmax is None else int(np.floor((tmax - ft) / sample_dt + 0.5))
            if itmin < 0:
                raise ValueError(f"itmin = {itmin} should not be negative")
            if itmin > itmax or itmax >= n:
                raise ValueError(f"itmin={itmin}, itmax={itmax} conflict")
            for h, x in zip(history, samples):
                h.append(x.astype(np.float64))
                del h[:-ntraces]
            mixed = [np.ascontiguousarray(_mixed(h, ntraces), dtype=F32) for h in history]
            angle, quality, input_quality, lag, xcorr = _kernels.alford_angle(
                *mixed, F32(sample_dt), itmin, itmax, F32(angle_inc), int(taperwin))
            # a lag that is too long (or no quality) is not a rotation
            if lag * 1000.0 > maxlag or quality <= 0.0:
                angle = quality = lag = 0.0
            rotated = [x.copy() for x in samples]
            if angle != 0.0:
                _kernels.alford_rotate(*rotated, F32(angle), itmin, int(taper))
            if report is not None:
                report['azimuth'].append(angle)
                report['quality'].append(quality)
                report['input_quality'].append(input_quality)
                report['lag'].append(lag)
                report['xcorr'].append(xcorr)
            for trace, row in zip(station, rotated):
                yield trace.replace(row)

    return from_iterable(traces())


def alford(*, angle_inc=5.0, tmin=None, tmax=None, taper=3, taperwin=10, maxlag=40.0, ntraces=5, report=None, dt=None,
           key=None, components=None):
    """Alford rotation of shear wave data into the natural coordinates, trace by trace (SUALFORD, after Schuck).

    A station is four traces, S11, S12, S21, S22 (see the notes of this module). The angle that maximizes the energy of the
    diagonal traces and minimizes that of the off-diagonal ones (the quality factor of the rotation) is searched in steps of
    ``angle_inc`` degrees in the window of analysis of the (mix of the neighbouring) stations and refined by a quadratic fit;
    the lag between the fast and the slow shear waves, from the cross-correlation of the rotated traces, picks which of the
    two is the fast one. The traces are then rotated by it from the top of the window to the end.

    Parameters
    ----------
    angle_inc : float
        The angle increment (degrees), from 1, default 5.
    tmin, tmax : float
        The window of analysis (s), by default the whole trace.
    taper : int
        ``2 * taper + 1`` is the overlap (samples) between the rotated and the unrotated part above the window: -1 for none,
        0 for one sample, from 1 a cosine scale. Default 3.
    taperwin : int
        The taper (samples) of the window of analysis, default 10.
    maxlag : float
        The greatest lag (ms) between the fast and slow shear waves: past it the station is not rotated (angle, quality
        and lag are 0), default 40.
    ntraces : int
        The number of neighbouring stations that are mixed to find the angle, default 5.
    report : dict, optional
        If given, it is given the lists ``azimuth`` (degrees: the direction of the fast shear wave), ``quality`` and
        ``input_quality`` (of the rotation and of none), ``lag`` (s) and ``xcorr`` (the maximum of the normalized
        cross-correlation) of each station: the program stores them in header words that are chosen with ``Az_key`` and
        the others.

    The program does not use the angle that it finds consistently: when the window is the whole trace it outputs
    traces that are rotated by the angle before the 90 degrees were added and swapped, and then not rotated again if the
    lag is too long. Here the traces of a station (not the mix) are rotated by the angle that is found, whatever the
    window is, and the cosine scale of the overlap is a real one (the program's is an integer division).
    """
    kwargs = dict(angle_inc=angle_inc, tmin=tmin, tmax=tmax, taper=taper, taperwin=taperwin, maxlag=maxlag,
                  ntraces=ntraces, report=report, dt=dt, key=key, components=components)
    _alford((), **kwargs)  # check the parameters now
    return Stage(_alford, parallelism='serial', name='alford', **kwargs)
