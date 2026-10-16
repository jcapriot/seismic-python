"""
Noise attenuation (``su/main/noise``): SUHARLAN (``harlan``: signal and noise separation by the invertible linear
transformation of Harlan), SUSEMBASE (``sembase``: semblance based structure oriented interpolation and smoothing) and
SUSEMGROUP (``semgroup``: the same smoothing in groups of traces, without new traces).

The two outputs of the programs that are files (the noise of ``harlan``, the dips, powers and semblances of the others) are
lists in the dictionary ``report`` that is given. Where a program plainly does not do what its documentation says this does
what the documentation says; the differences are with each stage.
"""
import math
import time
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import Stage
from . import _kernels

F32 = np.float32


def _real_traces(traces, name):
    if any(t.dtype.kind == 'c' for t in traces):
        raise TypeError(f"{name} works on real traces, but was given a complex one.")
    n = traces[0].n_sample
    if any(t.n_sample != n for t in traces):
        raise ValueError("The traces must all have the same number of samples.")
    return n


# ---------------------------------------------------------------------------------------------------------- harlan
def _harlan(upstream, *, niter=1, anenv=1, scl=0, norm=1, rgt=2, sts=1, dx=20.0, fx=0.0, dt=None, gopt=1, pmin1=-400.0,
            pmax1=400.0, pmin2=None, pmax2=None, nslopes=100, prewhite=0.01, offref=2000.0, depthref=500.0, interoff=0.0,
            pmula=None, pmulb=None, ninterp=0, nintlh=50, sditer=5, c=0.04, rel1=0.5, rel2=0.75, r1=10.0, r2=2.0,
            seed=None, report=None, output='signal'):
    if output not in ('signal', 'noise', 'both'):
        raise ValueError("output is 'signal', 'noise' or 'both'.")
    if gopt not in (1, 2, 3, 4, 5):
        raise ValueError(f"gopt={gopt} must be 1 to 5")
    if nslopes < 2 or nintlh < 3 or niter < 1:
        raise ValueError("nslopes must be at least 2, nintlh at least 3 and niter at least 1.")
    if rgt not in (1, 2):
        raise ValueError("rgt is 1 (uniform) or 2 (gaussian).")
    if pmax1 == pmin1:
        raise ValueError("pmax1 and pmin1 must differ.")
    source = as_trace_iterator(upstream)
    pmin2 = pmin1 if pmin2 is None else pmin2
    pmax2 = pmax1 if pmax2 is None else pmax2
    pmula = pmax1 if pmula is None else pmula
    pmulb = pmax1 if pmulb is None else pmulb
    if report is not None:
        report.setdefault('noise', [])

    def traces():
        panel = list(source)
        if len(panel) < 2:
            if panel:
                raise ValueError("harlan needs at least two traces.")
            return
        n = _real_traces(panel, 'harlan')
        sample_dt = float(dt) if dt is not None else (panel[0].d_sample or 0.004)
        data = np.ascontiguousarray(np.stack([np.asarray(t) for t in panel]), dtype=F32)
        seed_value = int(time.time()) if seed is None else int(seed)
        signal, noise = _kernels.harlan(data, rgt, sts, norm, anenv, scl, seed_value, sditer, niter, nintlh, F32(sample_dt),
                                        gopt, F32(prewhite), ninterp, F32(offref), F32(depthref), F32(interoff),
                                        F32(pmin1), F32(pmax1), F32(pmin2), F32(pmax2), nslopes, F32(dx), F32(c),
                                        F32(rel1), F32(rel2), F32(r1), F32(r2), F32(fx), F32(pmula), F32(pmulb))
        if report is not None:
            report['noise'] = [t.replace(np.ascontiguousarray(row)) for t, row in zip(panel, noise)]
        if output in ('signal', 'both'):
            for trace, row in zip(panel, signal):
                yield trace.replace(np.ascontiguousarray(row))
        if output in ('noise', 'both'):
            for trace, row in zip(panel, noise):
                yield trace.replace(np.ascontiguousarray(row))

    return from_iterable(traces())


def harlan(*, niter=1, anenv=1, scl=0, norm=1, rgt=2, sts=1, dx=20.0, fx=0.0, dt=None, gopt=1, pmin1=-400.0, pmax1=400.0,
           pmin2=None, pmax2=None, nslopes=100, prewhite=0.01, offref=2000.0, depthref=500.0, interoff=0.0, pmula=None,
           pmulb=None, ninterp=0, nintlh=50, sditer=5, c=0.04, rel1=0.5, rel2=0.75, r1=10.0, r2=2.0, seed=None,
           report=None, output='signal'):
    """Signal and noise separation by the invertible linear transformation of Harlan (SUHARLAN).

    The stream is one gather (a shot gather: the traces are at ``fx + i dx``). The data are transformed by the
    tau-p (Radon) transform; the samples of its output that are not focused (those that are as likely to be noise as
    signal, from histograms of the data and of the same data with random sign reversals of the traces) are zeroed,
    and the inverse transform makes the signal. The noise is the data less the signal (and is refined, by ``niter``
    iterations, with ``rel1`` and then ``rel2``).

    Parameters
    ----------
    niter : int
        The number of iterations, default 1 (the first uses ``rel1``, the others ``rel2``).
    anenv : {0, 1, 2, 3}
        0 no analytic envelopes (not recommended); 1 signed, 2 positive, 3 negative analytic envelopes. (The
        documentation of the program says that 1 is the positive ones: it is the signed ones in the code, which is what
        works with the histograms of signed samples, and it is that here.)
    scl : {0, 1, 2, 3}
        Scale the extracted signal to the data by trace (1), by time slice (2), or both (3); not recommended.
    norm : {0, 1}
        Normalize the reliability, default 1.
    rgt : {1, 2}
        The random numbers for the sign reversals: uniform or gaussian (default), SU's generators.
    sts : {0, 1}
        Damped least squares smoothing of the mask (default 1).
    dx, fx : float
        The offset interval and the offset of the first trace (m), by default 20 and 0.
    dt : float
        The sample interval (s), by default that of the traces (.004 if they do not have one).
    gopt, pmin1, pmax1, pmin2, pmax2, nslopes, prewhite, offref, depthref, interoff, pmula, pmulb, ninterp :
        The tau-p transform, as in :func:`seispy.transforms.radon` (``nslopes`` is the program's ``np``). The moveout
        limits are in ms at the reference offset.
    nintlh : int
        The number of bins of the histograms, default 50.
    sditer : int
        The number of steepest descent iterations for the signal histogram, default 5.
    c : float
        The largest noise in a sample of signal (fraction), default .04 (the documented value; the program used .05).
    rel1, rel2 : float
        The reliability required for the first and the following iterations, default .5 and .75. The most critical
        parameter: start small (.1 or .01).
    r1, r2 : float
        The vertical and horizontal damped least squares smoothing, default 10 and 2.
    seed : int
        The seed of the random number generator, by default the time (as the program does).
    report : dict, optional
        If given, its ``'noise'`` is the list of the extracted noise traces (the program writes it to ``noise=``).
    output : {'signal', 'noise', 'both'}
        What the stream is: the signal (default, the program's ``signal=`` file), the noise, or the signal gather then the
        noise gather.

    The plots and the temporary files of the program are not here. The state of the golden search of the program was in
    static variables, which were not reset between the calls: it is here for each call.
    """
    kwargs = dict(niter=niter, anenv=anenv, scl=scl, norm=norm, rgt=rgt, sts=sts, dx=dx, fx=fx, dt=dt, gopt=gopt,
                  pmin1=pmin1, pmax1=pmax1, pmin2=pmin2, pmax2=pmax2, nslopes=nslopes, prewhite=prewhite, offref=offref,
                  depthref=depthref, interoff=interoff, pmula=pmula, pmulb=pmulb, ninterp=ninterp, nintlh=nintlh,
                  sditer=sditer, c=c, rel1=rel1, rel2=rel2, r1=r1, r2=r2, seed=seed, report=report, output=output)
    _harlan((), **kwargs)  # check the parameters now
    return Stage(_harlan, parallelism='serial', name='harlan', **kwargs)


# --------------------------------------------------------------------------------------------- sembase / semgroup
def _key_value(trace, key):
    return float(key(trace)) if callable(key) else float(trace.header[key])


def _sem_setup(program, lockey, groupkey, locmaxscan, locmaxsum, dipuse, dipext, scanstep, scanmax, scanmin, indip,
               defaults):
    """The common parameters of the two programs, with their non-hierarchical checks"""
    scanning_given = any(v is not None for v in (scanstep, scanmax, scanmin, dipext))
    if locmaxscan is not None and locmaxscan < 0.0:
        raise ValueError("locmaxscan must be greater than or equal to 0.")
    scan_range = defaults['locmaxscan'] if locmaxscan is None else float(locmaxscan)
    if scan_range == 0.0 and (scanning_given or locmaxsum is None):
        raise ValueError("locmaxscan=0 does not allow scanstep, scanmax, scanmin, dipext, and requires locmaxsum")
    if indip is not None and (scanning_given or locmaxscan is not None or locmaxsum is None):
        raise ValueError("indip does not allow scanstep, scanmax, scanmin, dipext and locmaxscan, and requires "
                         "locmaxsum")
    sum_range = scan_range if locmaxsum is None else float(locmaxsum)
    if sum_range <= 0.0:
        raise ValueError("locmaxsum must be greater than 0.")
    if dipuse not in (0, 1):
        raise ValueError("dipuse is 0 or 1.")
    ext = defaults['dipext'] if dipext is None else float(dipext)
    if ext < 0.0:
        raise ValueError("dipext must be equal to or greater than 0.0")
    step = defaults['scanstep'] if scanstep is None else float(scanstep)
    if step <= 0.0 and scan_range > 0.0 and indip is None:
        raise ValueError("scanstep must be greater than 0.")
    smax = defaults['scanmax'] if scanmax is None else float(scanmax)
    smin = -smax if scanmin is None else float(scanmin)
    if smax < smin:
        raise ValueError("scanmax must be equal to or greater than scanmin")
    return scan_range, sum_range, ext, step, smin, smax


class _Dips:
    """The scanning of one group of traces, and the dips of its traces (scanned once)"""

    def __init__(self, data, locs, scan_range, smin_s, step_s, jsmax, minsd, maxsd):
        self.data, self.locs = data, locs
        self.scan_range, self.smin, self.step, self.jsmax, self.minsd, self.maxsd = scan_range, smin_s, step_s, jsmax, \
            minsd, maxsd
        self.cache = {}

    def at(self, location):
        """(smoothed dip, power, semblance) at the location, in samples per location unit"""
        dip, power, sem = _kernels.sem_scan(self.data, self.locs, float(location), self.scan_range, self.smin, self.step,
                                            self.jsmax)
        return _kernels.sem_smooth(dip, power, self.minsd, self.maxsd), power, sem

    def of(self, j):
        if j not in self.cache:
            self.cache[j] = self.at(self.locs[j])
        return self.cache[j]


def _shifted(trace_data, dip, loc, outputloc):
    times = (np.arange(trace_data.shape[0], dtype=np.float32) + dip * F32(loc - outputloc)).astype(F32)
    return _kernels.shift(np.ascontiguousarray(trace_data, dtype=F32), np.ascontiguousarray(times))


def _sample_dt_ms(trace, dt):
    value = float(dt) * 1000.0 if dt is not None else float(trace.d_sample) * 1000.0
    if value <= 0.0:
        raise ValueError("The traces have no sample interval, and no `dt` was given.")
    return value


def _read_dip(indip_iter, trace_index, expected, key, name, dt_ms, n):
    try:
        trd = next(indip_iter)
    except StopIteration:
        raise ValueError(f"The seismic trace {trace_index} needs a trace from indip, which has no more.") from None
    if abs(_key_value(trd, key) - expected) > 1e-9:
        raise ValueError(f"The lockey value of the indip trace ({_key_value(trd, key)}) and of the {name} "
                         f"({expected}) are not the same.")
    arr = np.asarray(trd, dtype=F32)
    if arr.shape[0] != n:
        raise ValueError("The indip traces must have the samples of the seismic traces.")
    return arr / F32(dt_ms)


_SUMOPTS = {'one': 1, 'two': 2, 'twolinear': 2, 'stack': 3, 'sum': 4, 'dadstack': 5, 'dadsum': 6}


def _sembase(upstream, *, lockey='ensemble_number', groupkey=None, outputinc=1.0, outputalign=0.0, outputshift=0.0,
             locmaxscan=None, scanstep=None, scanmax=None, scanmin=None, dipext=None, dipuse=1, locmaxsum=None,
             sumopt='stack', dadscale=None, dadext=None, dadnoise=None, indip=None, bypass=0, report=None, dt=None):
    if not isinstance(lockey, str):
        raise TypeError("lockey is the name of a settable header value: the locations are written to it.")
    if sumopt not in _SUMOPTS:
        raise ValueError(f"unknown sumopt={sumopt!r}: {sorted(_SUMOPTS)}")
    isumopt = _SUMOPTS[sumopt]
    ilinear = sumopt == 'twolinear'
    if isumopt < 5 and any(v is not None for v in (dadscale, dadext, dadnoise)):
        raise ValueError("only sumopt=dadstack or dadsum allows dadscale, dadext and dadnoise")
    dadscale = 1.0 if dadscale is None else float(dadscale)
    dadext = 12.0 if dadext is None else float(dadext)
    dadnoise = 0.0 if dadnoise is None else float(dadnoise)
    if dadext < 0.0 or dadnoise < 0.0:
        raise ValueError("dadext and dadnoise must be equal to or greater than 0.0")
    if bypass not in (-1, 0, 1):
        raise ValueError("bypass is -1, 0 or 1.")
    if outputinc <= 0.0:
        raise ValueError("outputinc must be greater than 0.0")
    if outputalign < 0.0:
        raise ValueError("outputalign must be equal to or greater than 0.0")
    outputcni = outputinc if outputinc < 1.0 else 1.0
    align = math.fmod(outputalign, outputinc)
    scan_range, sum_range, ext, step, smin, smax = _sem_setup(
        'sembase', lockey, groupkey, locmaxscan, locmaxsum, dipuse, dipext, scanstep, scanmax, scanmin, indip,
        dict(locmaxscan=3.0, dipext=12.0, scanstep=1.0, scanmax=12.0))
    source = as_trace_iterator(upstream)
    if report is not None:
        for name in ('dip', 'pow', 'sem'):
            report.setdefault(name, [])

    def process_group(group, state):
        """The output traces of a group: its input traces with their locations, in the order given"""
        n_t = state['n_t']
        dt_ms = state['dt_ms']
        order = sorted(range(len(group)), key=lambda k: state['locs'][k])  # (stable: duplicates keep their order)
        locs = np.array([state['locs'][k] for k in order], dtype=np.int32)
        data = np.ascontiguousarray(np.stack([np.asarray(group[k]) for k in order]), dtype=F32)
        members = [group[k] for k in order]
        sstep, sminv = step / dt_ms, smin / dt_ms
        jsmax = int(1 + (smax / dt_ms - sminv) / sstep) if step > 0 else 1
        maxsd = int((ext + 0.01) / dt_ms)
        maxdad = int((dadext + 0.01) / dt_ms)
        dips = _Dips(data, locs, scan_range, F32(sminv), F32(sstep) if step > 0 else F32(1.0), jsmax, -maxsd, maxsd)
        if indip is not None and dipuse > 0:
            for pos, k in enumerate(order):
                dips.cache[pos] = (state['indips'][k], None, None)
        reported = set()
        lo, hi = int(locs[0]), int(locs[-1])
        k_first = math.ceil((lo - sum_range - align) / outputinc - 1e-9)
        k_last = math.floor((hi + sum_range - align) / outputinc + 1e-9)
        for k in range(k_first, k_last + 1):
            outputloc = k * outputinc + align
            sel = np.nonzero((locs >= outputloc - sum_range) & (locs <= outputloc + sum_range))[0]
            if sel.size == 0:
                continue
            onedip = onepow = onesem = None
            if dipuse < 1:
                if indip is None:
                    onedip, onepow, onesem = dips.at(outputloc)
                else:
                    expected = (outputloc - align) / outputcni + outputshift
                    onedip = _read_dip(state['indip_iter'], len(state['output']), expected, lockey, 'output location',
                                       dt_ms, n_t)
            shifted = []
            nearest = second = None
            diff1 = diff2 = 1e30
            for pos, j in enumerate(sel):
                loc = float(locs[j])
                if dipuse > 0:
                    dip, power, sem = dips.of(int(j))
                    if report is not None and int(j) not in reported and indip is None:
                        reported.add(int(j))
                        for name, vals in (('dip', dip * F32(dt_ms)), ('pow', power), ('sem', sem)):
                            report[name].append(members[j].replace(np.ascontiguousarray(vals, dtype=F32)))
                else:
                    dip = onedip
                shifted.append(_shifted(data[j], dip, loc, outputloc))
                diff = abs(loc - outputloc)
                if diff < diff1:
                    diff2, second = diff1, nearest
                    nearest, diff1 = pos, diff
                elif diff < diff2:
                    diff2, second = diff, pos
            jsum = len(sel)
            out_header = members[int(sel[nearest])]
            location = (outputloc - outputalign) / outputcni + outputshift
            iput = True
            if bypass > 0 and diff1 < 1e-9:
                result = np.asarray(out_header, dtype=F32).copy()  # (the input trace)
            elif bypass < 0 and diff1 > 1e-9:
                iput = False
                result = None
            elif isumopt == 1 or jsum == 1:
                result = shifted[nearest]
            elif isumopt == 2 or jsum == 2:
                a, b = shifted[nearest], shifted[second]
                la, lb = float(locs[sel[nearest]]), float(locs[sel[second]])
                if not ilinear or abs(outputloc - la) == abs(outputloc - lb):
                    result = F32(0.5) * (a + b)
                elif outputloc == la:
                    result = a
                elif outputloc == lb:
                    result = b
                else:
                    ratio = abs(outputloc - la) / (abs(outputloc - la) + abs(outputloc - lb))
                    result = F32(1.0 - ratio) * a + F32(ratio) * b
            elif isumopt in (3, 4):
                total = np.sum(np.stack(shifted), axis=0, dtype=F32)
                result = total / F32(jsum) if isumopt == 3 else total
            else:
                result = _dad_sum(np.stack(shifted), locs[sel].astype(float), sum_range, dadscale, dadnoise, maxdad,
                                  isumopt == 5)
            if iput and int(location) < 0:
                # (header values here are not negative: the program writes the wrapped value)
                state['dropped'] += 1
                iput = False
            if iput:
                header = {lockey: int(location)}
                state['output'].append(out_header.replace(np.ascontiguousarray(result, dtype=F32), **header))
                if dipuse < 1 and report is not None and indip is None:
                    for name, vals in (('dip', onedip * F32(dt_ms)), ('pow', onepow), ('sem', onesem)):
                        report[name].append(state['output'][-1].replace(np.ascontiguousarray(vals, dtype=F32)))

    def traces():
        state = None
        group = []
        group_locs = []
        group_dips = []
        this_group = None
        prev_loc = None
        dups = 0
        indip_iter = iter(as_trace_iterator(indip)) if indip is not None else None
        n_traces = 0
        for trace in source:
            if state is None:
                if trace.dtype.kind == 'c':
                    raise TypeError("sembase works on real traces, but was given a complex one.")
                state = dict(n_t=trace.n_sample, dt_ms=_sample_dt_ms(trace, dt), output=[], indip_iter=indip_iter, dropped=0)
            elif trace.n_sample != state['n_t']:
                raise ValueError("The traces must all have the same number of samples.")
            loc = int(_key_value(trace, lockey))
            new_group = groupkey is not None and _key_value(trace, groupkey) != this_group
            if new_group or (this_group is None and groupkey is not None):
                if group:
                    state['locs'], state['indips'] = group_locs, group_dips
                    process_group(group, state)
                    group, group_locs, group_dips = [], [], []
                    yield from state['output']
                    state['output'] = []
                this_group = _key_value(trace, groupkey)
                prev_loc = loc - 1
            elif prev_loc is None:
                prev_loc = loc - 1
            if loc <= prev_loc:
                if loc == prev_loc:
                    dups += 1
                else:
                    raise ValueError(f"input lockey {loc} is less than that of the previous trace ({prev_loc})")
            prev_loc = loc
            group.append(trace)
            group_locs.append(loc)
            if indip_iter is not None and dipuse > 0:
                group_dips.append(_read_dip(indip_iter, n_traces, loc, lockey, 'seismic trace', state['dt_ms'],
                                            state['n_t']))
            n_traces += 1
        if group:
            state['locs'], state['indips'] = group_locs, group_dips
            process_group(group, state)
            yield from state['output']
        if dups:
            warnings.warn(f"There were {dups} input traces with duplicate lockey values (within their groups).")
        if state is not None and state['dropped']:
            warnings.warn(f"{state['dropped']} output traces have a negative lockey, which a header value cannot hold: "
                          "they are not output (use outputshift to move the output locations).")

    return from_iterable(traces())


def _dad_sum(shifted, locs, sum_range, dadscale, dadnoise, maxdad, average):
    """The sum with the range of locations from the detection of amplitude differences (the dadstack options)"""
    jsum, n_t = shifted.shape
    centloc = float(np.mean(locs))
    absolute = np.abs(shifted)
    weighted = np.sum(absolute * (locs - centloc)[:, None].astype(F32), axis=0, dtype=F32)
    total = (np.sum(absolute, axis=0, dtype=F32) + F32(1e-30)).astype(F32)
    quiet = total / F32(jsum) <= F32(dadnoise)
    located = np.where(quiet, F32(0.0), weighted / total).astype(F32)
    total = np.where(quiet, F32(1e-30), total).astype(F32)
    smooth = _kernels.sem_smooth(np.ascontiguousarray(located), np.ascontiguousarray(total), -maxdad, maxdad)
    span = 2.0 * sum_range
    rel = (2.0 * smooth.astype(np.float64) * dadscale + span).astype(F32)
    rel = np.where(rel > span + 0.1, rel - F32(span) - F32(1.0), rel).astype(F32)
    out = np.zeros(n_t, dtype=F32)
    for i in range(n_t):
        if rel[i] > span / 2.0 - 0.5:
            jfirst, jlast = 0, min(int(1.5 + rel[i]), jsum)
        else:
            jfirst, jlast = max(int(1.5 + rel[i]), 0), jsum
        if jlast > jfirst:
            out[i] = np.sum(shifted[jfirst:jlast, i], dtype=F32)
            if average and jlast - jfirst > 1:
                out[i] /= F32(jlast - jfirst)
    return out


def sembase(*, lockey='ensemble_number', groupkey=None, outputinc=1.0, outputalign=0.0, outputshift=0.0, locmaxscan=None,
            scanstep=None, scanmax=None, scanmin=None, dipext=None, dipuse=1, locmaxsum=None, sumopt='stack', dadscale=None,
            dadext=None, dadnoise=None, indip=None, bypass=0, report=None, dt=None):
    """Semblance based structure oriented interpolation and smoothing (SUSEMBASE, after Latour).

    The traces are scanned in the direction of their locations (``lockey``) for the (apparent) dips of events, by
    semblance, and are summed after the dips have lined up the events, which also makes traces between the input ones.
    Traces must be in increasing (or equal) order of ``lockey``, within each ``groupkey`` group.

    Parameters
    ----------
    lockey : str
        The header value that is the location, default ``'ensemble_number'`` (the CDP). It is written with the location
        of the output trace.
    groupkey : str, optional
        A header value whose change starts a new set of locations (3-D stacks: the in-line or cross-line index).
    outputinc, outputalign, outputshift : float
        The increment of the output locations (default 1), their alignment (output locations are at ``k * outputinc +
        outputalign``) and the number added to the output ``lockey``. When ``outputinc`` is below 1 the output ``lockey``
        is the index ``k`` (so that it can be an integer), above 1 it is the location (less the alignment).
    locmaxscan : float
        The largest relative location to scan for dips, default 3 (0: no scan, which needs ``locmaxsum``).
    scanstep, scanmax, scanmin : float
        The step (default 1), largest (12) and smallest (``-scanmax``) dips scanned, ms per location unit.
    dipext : float
        The range (ms, default 12) of the power weighted average of the dips in time.
    dipuse : {1, 0}
        1 (default) dips are made and used at the input locations, 0 at the output ones.
    locmaxsum : float
        The largest relative location included in the sum (default ``locmaxscan``).
    sumopt : str
        ``'stack'`` (default; the dip corrected traces within the range, averaged), ``'sum'``, ``'one'`` (the nearest
        trace only), ``'two'``, ``'twolinear'`` (the two nearest, averaged or by distance weights), ``'dadstack'`` and
        ``'dadsum'`` (the range of locations from the detected rapid amplitude changes: ``dadscale`` (1), ``dadnoise``
        (0) and ``dadext`` (ms, 12)).
    indip : iterable of traces, optional
        Dips (ms per location unit; the ``'dip'`` of an earlier ``report``) to use instead of scanning: one for each input
        trace (``dipuse=1``) or output location (``dipuse=0``). The scan parameters are then an error.
    bypass : {0, 1, -1}
        1 copies the input trace where an input is exactly at an output location, -1 outputs only those locations.
    report : dict, optional
        If given, it gets the lists ``'dip'`` (ms per location unit), ``'pow'`` and ``'sem'`` of traces (the files
        ``outdip``, ``outpow`` and ``outsem`` of the program).
    dt : float
        The sample interval (s) for traces that do not have one.

    Output traces whose ``lockey`` would be negative are dropped with a warning (a header value here cannot be
    negative; use ``outputshift``).

    The program kept only ``numrolling`` traces at a time and read its input dips without dividing by the sample interval
    (``indip`` for ``dipuse=1`` was in the wrong units); neither is the case here. Output locations are
    ``k * outputinc + outputalign`` as documented (the program started them at the first location times ``outputinc``,
    which is wrong for an ``outputinc`` that is not 1 for negative locations or an ``outputinc`` above 1), and ties
    for the nearest trace go to the lower location.
    """
    kwargs = dict(lockey=lockey, groupkey=groupkey, outputinc=outputinc, outputalign=outputalign,
                  outputshift=outputshift, locmaxscan=locmaxscan, scanstep=scanstep, scanmax=scanmax, scanmin=scanmin,
                  dipext=dipext, dipuse=dipuse, locmaxsum=locmaxsum, sumopt=sumopt, dadscale=dadscale, dadext=dadext,
                  dadnoise=dadnoise, indip=indip, bypass=bypass, report=report, dt=dt)
    _sembase((), **kwargs)  # check the parameters now
    return Stage(_sembase, parallelism='serial', name='sembase', **kwargs)


def _semgroup(upstream, *, lockey='offset', groupkey='ensemble_number', maxfold=999, locmaxscan=None, scanstep=None,
              scanmax=None, scanmin=None, dipext=None, dipuse=1, locmaxsum=None, indip=None, report=None, dt=None):
    if maxfold < 2:
        raise ValueError("maxfold must be greater than 1")
    scan_range, sum_range, ext, step, smin, smax = _sem_setup(
        'semgroup', lockey, groupkey, locmaxscan, locmaxsum, dipuse, dipext, scanstep, scanmax, scanmin, indip,
        dict(locmaxscan=75.0, dipext=12.0, scanstep=0.05, scanmax=1.1))
    source = as_trace_iterator(upstream)
    if report is not None:
        for name in ('dip', 'pow', 'sem'):
            report.setdefault(name, [])

    def process(group, locs_list, indips, dt_ms):
        n = len(group)
        locs = np.array(locs_list, dtype=np.int32)
        data = np.ascontiguousarray(np.stack([np.asarray(t) for t in group]), dtype=F32)
        sstep, sminv = step / dt_ms, smin / dt_ms
        jsmax = int(1 + (smax / dt_ms - sminv) / sstep) if step > 0 else 1
        maxsd = int((ext + 0.01) / dt_ms)
        dips = _Dips(data, locs, scan_range, F32(sminv), F32(sstep) if step > 0 else F32(1.0), jsmax, -maxsd, maxsd)
        if indip is None:
            for j in range(n):
                dip, power, sem = dips.of(j)
                if report is not None:
                    for name, vals in (('dip', dip * F32(dt_ms)), ('pow', power), ('sem', sem)):
                        report[name].append(group[j].replace(np.ascontiguousarray(vals, dtype=F32)))
        else:
            for j in range(n):
                dips.cache[j] = (indips[j], None, None)
        out = []
        for jloc in range(n):
            outputloc = float(locs[jloc])
            sel = np.nonzero((locs >= outputloc - sum_range) & (locs <= outputloc + sum_range))[0]
            total = np.zeros(data.shape[1], dtype=F32)
            for j in sel:
                dip = dips.of(int(j))[0] if dipuse > 0 else dips.of(jloc)[0]
                total += _shifted(data[j], dip, float(locs[j]), outputloc)
            out.append(group[jloc].replace(np.ascontiguousarray(total / F32(len(sel)), dtype=F32)))
        return out

    def traces():
        group, locs, indips = [], [], []
        this_group = None
        dt_ms = None
        n_t = None
        indip_iter = iter(as_trace_iterator(indip)) if indip is not None else None
        count = 0
        for trace in source:
            if dt_ms is None:
                if trace.dtype.kind == 'c':
                    raise TypeError("semgroup works on real traces, but was given a complex one.")
                dt_ms = _sample_dt_ms(trace, dt)
                n_t = trace.n_sample
            elif trace.n_sample != n_t:
                raise ValueError("The traces must all have the same number of samples.")
            if groupkey is not None:
                value = _key_value(trace, groupkey)
                if this_group is None or value != this_group:
                    if group:
                        yield from process(group, locs, indips, dt_ms)
                        group, locs, indips = [], [], []
                    this_group = value
            if len(group) >= maxfold:
                raise ValueError(f"There are too many traces in groupkey value {this_group}: maxfold={maxfold}")
            loc = int(_key_value(trace, lockey))
            group.append(trace)
            locs.append(loc)
            if indip_iter is not None:
                indips.append(_read_dip(indip_iter, count, loc, lockey, 'seismic trace', dt_ms, n_t))
            count += 1
        if group:
            yield from process(group, locs, indips, dt_ms)

    return from_iterable(traces())


def semgroup(*, lockey='offset', groupkey='ensemble_number', maxfold=999, locmaxscan=None, scanstep=None, scanmax=None,
             scanmin=None, dipext=None, dipuse=1, locmaxsum=None, indip=None, report=None, dt=None):
    """Semblance based structure oriented smoothing in groups of traces, without new traces (SUSEMGROUP, after Latour).

    As :func:`sembase` for the traces of a group (a CDP gather by ``'offset'``, or a stack by its CDPs with
    ``groupkey=None``): each trace is replaced by the average of the dip corrected traces within ``locmaxsum`` of it,
    which does not smear the events as the plain average does. The defaults are those of the program for gathers:
    ``locmaxscan=75``, ``scanstep=.05``, ``scanmax=1.1`` ms per location unit, and for stacks they are about
    ``scanstep=.5``, ``scanmax=12`` and ``locmaxscan=3`` with ``lockey`` the CDP and ``groupkey=None``.

    Parameters
    ----------
    lockey, groupkey : str
        The location within a group (default ``'offset'``) and the group (default ``'ensemble_number'``; ``None``: one
        group of all the traces).
    maxfold : int
        The largest number of traces in a group (default 999).
    locmaxscan, scanstep, scanmax, scanmin, dipext, locmaxsum, dipuse, indip, report, dt :
        As for :func:`sembase` (``dipuse=0`` is the dip of the trace that is being output, 1 that of each trace
        that is summed).

    The input dips of ``indip`` are one for each input trace, in ms per location unit.
    """
    kwargs = dict(lockey=lockey, groupkey=groupkey, maxfold=maxfold, locmaxscan=locmaxscan, scanstep=scanstep,
                  scanmax=scanmax, scanmin=scanmin, dipext=dipext, dipuse=dipuse, locmaxsum=locmaxsum, indip=indip,
                  report=report, dt=dt)
    _semgroup((), **kwargs)  # check the parameters now
    return Stage(_semgroup, parallelism='serial', name='semgroup', **kwargs)
