"""
More programs of ``su/main/windowing_sorting_muting``: SUMUTECSV (mutes that change with the CDP), SUWINDPOLY (windowing
by a polygon), SUPADMISSING (dead traces for the values of a key that are missing), SUBRAID (merging ordered streams) and
SUGPRFB (removing the first break from GPR data).

Where the program plainly does not do what its documentation says this does what the documentation says; the
differences are listed with each stage.
"""
import heapq
import math
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stage import Stage, header_value, stage

__all__ = ['mutecsv', '_windpoly', '_padmissing', 'braid', '_gprfb']

F32 = np.float32


def _nint(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


# -------------------------------------------------------------------------------------------------------- mutecsv
def _extrapolating_interp(x, xs, ys, extrap):
    """Linear interpolation of the points (xs, ys) at x. Outside of them the end values are held, unless extrap is 1 (both
    ends), 2 (the lower end) or 3 (the higher end), which extrapolate with the slope of the end segment"""
    n = xs.shape[0]
    if n == 1:
        return float(ys[0])
    if x < xs[0]:
        if extrap in (1, 2):
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            return float(ys[0] + slope * (x - xs[0]))
        return float(ys[0])
    if x > xs[-1]:
        if extrap in (1, 3):
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            return float(ys[-1] + slope * (x - xs[-1]))
        return float(ys[-1])
    return float(np.interp(x, xs, ys))


def _mutecsv(upstream, *, cdp, offs, tims, mode=0, offkey='offset', absolute=True, ntaper=0, vel=330.0, tzero=0.0,
             extrapi=0, extrapt=0, cdpkey='ensemble_number'):
    from . import _MUTE_MODES, _mutec  # (this module is imported by the package)
    from ..stage import per_trace

    cdps = np.atleast_1d(np.asarray(cdp, dtype=np.float64))
    if cdps.ndim != 1 or cdps.shape[0] == 0:
        raise ValueError("cdp must be a list of at least one cdp")
    time_lists = [np.atleast_1d(np.asarray(t, dtype=np.float64)) for t in tims]
    if len(time_lists) != cdps.shape[0]:
        raise ValueError("There must be the same number of tims lists as numbers in the cdp list")
    if len(offs) and np.isscalar(offs[0]):
        offs = [offs] * cdps.shape[0]
    offset_lists = [np.atleast_1d(np.asarray(o, dtype=np.float64)) for o in offs]
    if len(offset_lists) != cdps.shape[0]:
        raise ValueError("There must be the same number of offs lists as numbers in the cdp list")
    for o, t in zip(offset_lists, time_lists):
        if o.shape != t.shape or o.shape[0] == 0:
            raise ValueError("Each offs list and its tims list must have the same number of values")
        if (np.diff(o) <= 0).any():
            raise ValueError("The offsets in the offs arrays must be monotonically increasing")
    if mode not in (0, 1, 2, 3):
        raise ValueError("mode must be 0, 1, 2 or 3")
    if vel == 0:
        raise ValueError("vel can't be 0")
    if ntaper < 0:
        raise ValueError("ntaper must not be negative")
    for e in (extrapi, extrapt):
        if e not in (0, 1, 2, 3):
            raise ValueError("The extrapolation options are 0, 1, 2 and 3")
    order = np.argsort(cdps, kind='stable')
    cdps = cdps[order]
    if (np.diff(cdps) == 0).any():
        raise ValueError("Two sets of values input for the same cdp")
    time_lists = [time_lists[i] for i in order]
    offset_lists = [offset_lists[i] for i in order]
    # times are in milliseconds if any is over 100
    tscale = 0.001 if any((t > 100.0).any() for t in time_lists) else 1.0
    tm0 = tzero * tscale
    apply_mode = _MUTE_MODES[mode]
    taper = _mutec.taper_weights(ntaper)

    def mute_trace(trace):
        header = trace.header
        tmin = header['sample_start']
        dt = header['d_sample']
        if not dt:
            raise ValueError("The trace has no sample interval.")
        fval = header_value(trace, offkey)
        if absolute:
            fval = abs(fval)
        place = header_value(trace, cdpkey)
        # the mute time at the offset for each function, then between the functions (as the program does)
        times = [_extrapolating_interp(fval, o, t, extrapt) for o, t in zip(offset_lists, time_lists)]
        t = _extrapolating_interp(place, cdps, np.asarray(times), extrapi) * tscale
        x = apply_mode(np.array(trace), t=t, tmin=tmin, dt=dt, taper=taper, fval=fval, linvel=vel, tm0=tm0, xmute=None,
                       twindow=None)
        return trace.replace(x)

    return per_trace(upstream, mute_trace, on_complex='native')


def mutecsv(cdp, offs, tims, *, mode=0, offkey='offset', absolute=True, ntaper=0, vel=330.0, tzero=0.0, extrapi=0, extrapt=0,
            cdpkey='ensemble_number'):
    """Mute above (or below) polygonal curves that are interpolated between CDPs and offsets (SUMUTECSV).

    Parameters
    ----------
    cdp : list
        The CDPs that the mute functions are given for.
    offs, tims : lists
        For each cdp, the offsets (increasing) and the times of the mute (s, or ms if any is over 100). If ``offs`` is one
        list it is used for every cdp.
    mode : {0, 1, 2, 3}
        Mute above (0) or below (1) the curve; 2 and 3 mute a zone around a straight line of velocity ``vel`` or a
        hyperbola, the times being the lengths of the zone (as in ``seispy.windowing.mute``).
    offkey, cdpkey : str or function
        The header values of the offset and of the cdp (default ``'offset'`` and ``'ensemble_number'``); the absolute
        value of the offset is used if ``absolute``.
    ntaper, vel, tzero
        The samples of the sine squared taper, the velocity of modes 2 and 3, and the time shift (in the units of
        ``tims``) of the line or hyperbola at offset 0.
    extrapi, extrapt : {0, 1, 2, 3}
        In the directions of the cdp and of the offset: hold the end values (0), extrapolate at both ends (1), at the
        lower end (2) or at the higher end (3).

    The q-file input (``qin``) and the 3D grids (``rfile``, ``grid_*``) of the program are not supported.
    """
    kwargs = dict(mode=mode, offkey=offkey, absolute=absolute, ntaper=ntaper, vel=vel, tzero=tzero, extrapi=extrapi,
                  extrapt=extrapt, cdpkey=cdpkey, cdp=cdp, offs=offs, tims=tims)
    _mutecsv((), **kwargs)  # check the parameters now
    return Stage(_mutecsv, parallelism='trace', name='mutecsv', **kwargs)


# --------------------------------------------------------------------------------------------------------- windpoly
def _in_polygon(x0, y0, xs, ys):
    """O'Rourke's crossing test: 0 outside, 1 a vertex, 2 on an edge, 3 interior. The polygon is closed (first = last)"""
    n = len(xs)
    right = left = 0
    for i in range(n):
        if xs[i] == x0 and ys[i] == y0:
            return 1
        j = (i + n - 1) % n
        if (ys[i] > y0 and ys[j] <= y0) or (ys[j] > y0 and ys[i] <= y0):
            x = ((xs[i] - x0) * (ys[j] - y0) - (xs[j] - x0) * (ys[i] - y0)) / ((ys[j] - y0) - (ys[i] - y0))
            if x > 0:
                right += 1
            if x < 0:
                left += 1
    if right % 2 != left % 2:
        return 2
    return 3 if right % 2 == 1 else 0


def _polyline_transform(x0, y0, xs, ys, arc):
    """For each segment that the point is over (on its normal): the arc length of the foot, and the signed distance
    (positive on the right of the direction of the line)"""
    out = []
    for i in range(1, len(xs)):
        dx, dy = xs[i] - xs[i - 1], ys[i] - ys[i - 1]
        length2 = dx * dx + dy * dy
        if length2 == 0.0:
            continue
        u = ((x0 - xs[i - 1]) * dx + (y0 - ys[i - 1]) * dy) / length2
        if u < 0.0 or u > 1.0:
            continue
        foot_x, foot_y = xs[i - 1] + u * dx, ys[i - 1] + u * dy
        distance = math.hypot(x0 - foot_x, y0 - foot_y)
        # the cross product of the direction and of the point from the start of the segment: negative on the right
        cross = dx * (y0 - ys[i - 1]) - dy * (x0 - xs[i - 1])
        out.append((arc[i - 1] + u * math.sqrt(length2), distance if cross < 0 else -distance))
    return out


def _windpoly(upstream, *, x, y, xkey, ykey, pass_=0, dx=1.0, dy=1.0, dw=1.0):
    source = as_trace_iterator(upstream)

    def traces():
        if pass_ == 0:
            xs = [float(v) * dx for v in x]
            ys = [float(v) * dy for v in y]
            arc = [0.0]
            for i in range(1, len(xs)):
                arc.append(arc[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
            for trace in source:
                x0 = header_value(trace, xkey) * dx
                y0 = header_value(trace, ykey) * dy
                for inline, crossline in _polyline_transform(x0, y0, xs, ys, arc):
                    if abs(crossline) <= dw:
                        yield trace.replace(iline=int(inline), xline=int(crossline))
        else:
            polygons = []
            for px, py in polygons_of:
                px, py = [float(v) for v in px], [float(v) for v in py]
                if px[-1] != px[0] or py[-1] != py[0]:
                    px.append(px[0])
                    py.append(py[0])
                polygons.append((px, py))
            for trace in source:
                x0 = header_value(trace, xkey)
                y0 = header_value(trace, ykey)
                inside = any(_in_polygon(x0, y0, px, py) for px, py in polygons)
                if inside == (pass_ == 1):
                    yield trace

    if pass_ not in (0, 1, 2):
        raise ValueError("Invalid pass option")
    if pass_ == 0:
        if len(x) != len(y) or len(x) < 2:
            raise ValueError("x and y must be the same number (2 or more) of values")
    else:
        # (a polygon is lists of x and y; for pass 2 there may be several: the traces outside of all of them)
        if len(x) and np.isscalar(x[0]):
            polygons_of = [(x, y)]
        else:
            polygons_of = list(zip(x, y))
        if pass_ == 1 and len(polygons_of) != 1:
            raise ValueError("pass=1 takes one polygon: use one stage for each")
        for px, py in polygons_of:
            if len(px) != len(py) or len(px) < 3:
                raise ValueError("x and y must be the same number (3 or more) of values of a polygon")
    return from_iterable(traces())


# SUWINDPOLY: window the traces on or within a polygonal line, or a polygon, whose vertices are (x, y) in the header values
# xkey and ykey (names, or functions of a trace; SU's defaults fldr and ep do not exist here).
# pass_=0 : the traces within dw (in the units of x and y, which are scaled by dx and dy) of the polyline; a trace is output
# once for each segment that it is within dw of (on the normal to), with iline the arc length along the line and xline the
# distance from it, positive on the right of the direction of the line. pass_=1 : the traces inside of the polygon (it closes
# itself), pass_=2 : the traces outside of all the polygons (x and y may be lists of lists).
# Differences from SU: the sign of the distance from a vertical segment is that of the other segments (the program had it
# reversed), and the polygons are not scaled by dx and dy (the documentation says they are for the polyline only).
windpoly = stage(_windpoly, parallelism='serial', name='windpoly', validate=True)


# ----------------------------------------------------------------------------------------------------- padmissing
def _padmissing(upstream, *, key, incval=1.0, minval=None, maxval=None):
    if incval <= 0.0:
        raise ValueError("incval must be greater than 0.0")
    if minval is not None and maxval is not None and minval > maxval:
        raise ValueError("minval cannot be greater than maxval")
    source = as_trace_iterator(upstream)

    def make_pad(template, value):
        pad = template.replace(np.zeros(template.n_sample, dtype=np.float32), fold=0)
        try:
            return pad.replace(**{key: type(template.header[key])(value)})
        except (TypeError, KeyError):
            raise ValueError(f"key {key!r} is not a header value that can be set") from None

    def traces():
        base = None if minval is None else float(minval)
        n = 0
        previous = None
        last = None  # the last trace that came out, and the last that was before minval, for the header of the padding
        dropped = None

        def expected():
            return base + n * incval

        for trace in source:
            value = header_value(trace, key)
            if previous is not None and value <= previous:
                raise ValueError(f"key={key} is not increasing. Next trace={value}  Previous={previous}")
            previous = value
            if base is None:
                base = value
            if value < base and not math.isclose(value, base):
                dropped = trace
                continue
            if maxval is not None and value > maxval and not math.isclose(value, maxval):
                while expected() <= maxval or math.isclose(expected(), maxval):
                    yield make_pad(trace, expected())
                    n += 1
                return
            while expected() < value and not math.isclose(expected(), value):
                yield make_pad(trace, expected())
                n += 1
            yield trace
            n += 1
            last = trace
        template = last if last is not None else dropped
        if maxval is not None and template is not None:
            while expected() <= maxval or math.isclose(expected(), maxval):
                yield make_pad(template, expected())
                n += 1

    return from_iterable(traces())


# SUPADMISSING: dead traces (zeros, fold 0, the header of the next trace, with key set to the missing value) for the
# values of key (a header value that is an integer or a number, increasing in the input) that are missing from the series
# minval + n incval, from minval (the first trace's by default) to maxval (the last's by default).
# Differences from SU: none for traces that are on the grid; an input trace that is not takes the place of the value that it
# is nearest after, as in the program.
padmissing = stage(_padmissing, parallelism='serial', name='padmissing', validate=True)


# --------------------------------------------------------------------------------------------------------- braid
def braid(*sources, key='ensemble_number', renumber=True):
    """Combine ordered streams of traces into one ordered stream (SUBRAID).

    Parameters
    ----------
    sources : iterables of traces
        At least two. The traces of each are in the order of ``key``.
    key : str, function, or list of them
        The keys that order the traces: names of header values or functions of a trace. A name that begins with ``-`` is
        in decreasing order. The first key is the most significant.
    renumber : bool
        Number the output traces from 1 in ``trace_id`` (the program resets tracl and tracr).

    The order of each input is checked on the fly, and a ``ValueError`` is raised if a trace is out of it. Traces with equal
    keys come out in the order of the sources.
    """
    if len(sources) < 2:
        raise ValueError("There must be at least 2 input files.")
    keys = [key] if (callable(key) or isinstance(key, str)) else list(key)
    if not keys:
        raise ValueError("key must not be empty")
    getters = []
    for k in keys:
        if callable(k):
            getters.append((k, 1.0))
        elif isinstance(k, str) and k.startswith('-'):
            getters.append((k[1:], -1.0))
        else:
            getters.append((k, 1.0))

    def values(trace):
        return tuple(sign * header_value(trace, k) for k, sign in getters)

    def traces():
        iterators = [iter(as_trace_iterator(s)) for s in sources]
        heap = []
        last = [None] * len(iterators)
        count = 0

        def push(index):
            try:
                trace = next(iterators[index])
            except StopIteration:
                return
            v = values(trace)
            if last[index] is not None and v < last[index]:
                raise ValueError(f"The traces of input {index + 1} are not in the order of key {keys!r}.")
            last[index] = v
            heapq.heappush(heap, (v, index, count_holder[0], trace))
            count_holder[0] += 1

        count_holder = [0]
        for i in range(len(iterators)):
            push(i)
        while heap:
            v, index, _, trace = heapq.heappop(heap)
            push(index)
            count += 1
            yield trace.replace(trace_id=count) if renumber else trace

    return from_iterable(traces())


# ----------------------------------------------------------------------------------------------------------- gprfb
def _remove_first_break(trace, pilot, report):
    n = pilot.shape[0]
    first = trace[:n]
    # the lag of the trace relative to the pilot, by cross-correlation (lags -n/2 to n/2 - 1)
    lags = np.arange(-(n // 2), n - n // 2)
    corr = np.empty(lags.shape[0])
    for i, lag in enumerate(lags):
        if lag >= 0:
            corr[i] = np.dot(pilot[:n - lag], first[lag:]) if n - lag > 0 else 0.0
        else:
            corr[i] = np.dot(pilot[-lag:], first[:n + lag]) if n + lag > 0 else 0.0
    lag = int(lags[int(np.argmax(corr))])
    # the scale: least squares of the aligned samples of the trace on those of the pilot
    if lag >= 0:
        a_part, b_part = first[lag:], pilot[:n - lag]
    else:
        a_part, b_part = first[:n + lag], pilot[-lag:]
    denom = float(np.dot(b_part, b_part))
    scale = float(np.dot(a_part, b_part) / denom) if denom > 0 else 1.0
    # a ramp that fades the pilot out over the last fifth
    ramps = _nint(n - n // 5)
    ramp = np.ones(n)
    for it in range(ramps, n):
        ramp[it] = 1.0 - (it - ramps) / max(n - ramps - 1, 1)
    out = np.array(trace, dtype=np.float64)
    u = np.arange(n)
    source = u - lag
    ok = (source >= 0) & (source < n)
    out[:n][ok] -= pilot[source[ok]] * scale * ramp[ok]
    if report is not None:
        report.append((_nint((1.0 - scale) * 100.0), -lag))
    return out.astype(np.float32)


def _gprfb(upstream, *, nx=51, fbt=60, report=None):
    if nx < 1:
        raise ValueError("nx must be at least 1")
    if fbt < 1:
        raise ValueError("fbt must be at least 1")
    if nx % 2 == 0:
        nx += 1
        warnings.warn(f" nx has been changed to {nx} to be odd.")
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        if len(panel) < nx:
            raise ValueError(f"There are {len(panel)} traces, nx={nx} are needed")
        nt = panel[0].n_sample
        if fbt > nt:
            raise ValueError("fbt is longer than the traces")
        first = np.stack([np.asarray(t, dtype=np.float64)[:fbt] for t in panel])
        half = nx // 2
        total = np.concatenate([np.zeros((1, fbt)), np.cumsum(first, axis=0)])
        for i, trace in enumerate(panel):
            # the pilot is the mean of the first break of the traces of the window around the trace (the first and last
            # windows for the traces at the ends)
            centre = min(max(i, half), len(panel) - 1 - half)
            lo = centre - half
            pilot = (total[lo + nx] - total[lo]) / nx
            yield trace.replace(_remove_first_break(np.asarray(trace), pilot, report))

    return from_iterable(traces(), n_traces=source.n_traces)


# SUGPRFB: remove the first break (direct wave) from GPR data: the first fbt samples of the nx traces around a trace are
# stacked into a pilot, which is shifted and scaled to fit the trace and subtracted from it (a ramp fades it out). `report`,
# a list, gets (the percent of the trace that was removed, the shift in samples) for each trace (the program stored them in
# the header words trwf and grnors). Differences from SU: the pilot is shifted the same way that it was fitted (the program
# fitted the shifted pilot for a negative shift by reading before its array, and shifted it the other way for the other).
gprfb = stage(_gprfb, parallelism='serial', name='gprfb', validate=True)
