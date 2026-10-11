"""
More programs of ``su/main/stacking``: SURECIP (sum the traces of opposite offsets) and SUCVS4FOWLER (constant velocity
stacks of CDP gathers, the input of the Fowler DMO and migration codes).

A gather is the run of traces that have the same value of ``key`` (``ensemble_number``, the CDP, by default).
"""
import math
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stage import stage
from ..stretching import _resample

__all__ = ['_recip', '_cvs4fowler']

F32 = np.float32


# ---------------------------------------------------------------------------------------------------------- recip
def _recip(upstream, *, norm=True, key='ensemble_number'):
    source = as_trace_iterator(upstream)

    def traces():
        for gather in group_by(source, key):
            # the traces of the same offset (the program assumed the absolute offsets, sorted by cdp and offset)
            groups = {}
            for trace in gather:
                groups.setdefault(abs(float(trace.header['offset'])), []).append(trace)
            for offset in sorted(groups):
                members = groups[offset]
                total = np.zeros(members[0].n_sample, dtype=np.float64)
                for trace in members:
                    if trace.n_sample != total.shape[0]:
                        raise ValueError("The traces of an offset must have the same number of samples.")
                    if trace.dtype.kind == 'c':
                        raise TypeError("recip works on real traces, but was given a complex one.")
                    total += np.asarray(trace)
                fold = len(members)
                if norm and fold != 1:
                    total /= fold
                first = members[0]
                tx, rx = list(first.header['tx_loc']), list(first.header['rx_loc'])
                fields = dict(fold=fold)
                if tx[0] > rx[0]:  # (the source is at the smaller x: the program swapped sx and gx)
                    fields.update(tx_loc=rx, rx_loc=tx)
                yield first.replace(np.ascontiguousarray(total, dtype=F32), **fields)

    return from_iterable(traces())


# SURECIP: sum the traces that have the same CDP and the same absolute offset (reciprocity holds), one trace for each; its
# `fold` is the number of traces, and norm=True divides by it. The source of a trace is at the smaller x of the two.
# Differences from SU: the traces of a cdp do not have to be sorted by offset (the program expected suabshw's output sorted),
# the stacked traces come out in the order of increasing absolute offset.
recip = stage(_recip, parallelism='ensemble', name='recip', validate=True)


# ----------------------------------------------------------------------------------------------------- cvs4fowler
def _taper(lxtaper, lbtaper, nx, ix, nt, trace):
    """Taper a trace near the left and right sides of the window of nx traces (ix of them), and its bottom"""
    if lxtaper <= 0:
        xtaper = 1.0
    elif ix < lxtaper:
        xtaper = 0.54 + 0.46 * math.cos(math.pi * (lxtaper - ix) / lxtaper)
    elif ix >= nx - lxtaper:
        xtaper = 0.54 + 0.46 * math.cos(math.pi * (lxtaper + ix + 1 - nx) / lxtaper)
    else:
        xtaper = 1.0
    if xtaper != 1.0:
        trace = trace * xtaper
    if lbtaper > 0:
        first = max(0, nt - lbtaper)
        it = np.arange(first, nt)
        trace = trace.copy()
        trace[first:] *= 0.54 + 0.46 * np.cos(np.pi * (lbtaper + it + 1 - nt) / lbtaper)
    return trace


def _cvs4fowler(upstream, *, ncdps=None, vminstack=1500.0, nvstack=180, lmute=24, nonhyp=True, vhyp=2500.0, lbtaper=0,
                lstaper=0, dtout=None, mxfold=120, salias=0.8, key='ensemble_number'):
    if nvstack < 6:
        raise ValueError("nvstack must be at least 6")
    if vminstack <= 0:
        raise ValueError("vminstack must be positive")
    source = as_trace_iterator(upstream)
    salias = min(float(salias), 1.0)

    def traces():
        for icmp, gather in enumerate(group_by(source, key)):
            if ncdps is not None and icmp >= ncdps:
                break
            if len(gather) > mxfold:
                raise ValueError("cvstacks: input cdp has more traces than mxfold")  # (the program wrote past its arrays)
            first = gather[0]
            nt = first.n_sample
            dt = first.d_sample
            if not dt:
                raise ValueError("The trace has no sample interval.")
            if first.header['sample_start'] != 0.0:
                raise ValueError("The traces must start at time 0.")
            out_dt = float(dtout) if dtout else 1.5 * dt
            ntout = int(1 + nt * dt / out_dt)
            fmax = salias * 0.5 / out_dt
            taper_length = int(1 + 0.001 * lmute / out_dt)
            bottom = int(lbtaper / (1000.0 * dt))
            if lstaper > 0 and ncdps is None:
                raise ValueError("lstaper needs ncdps, the number of cdps")
            dp2 = 1.0 / (vminstack * vminstack * (nvstack - 5))
            data, offsets, mutes = [], [], []
            for trace in gather:
                if trace.n_sample != nt:
                    raise ValueError("The traces must all have the same number of samples.")
                if trace.dtype.kind == 'c':
                    raise TypeError("cvs4fowler works on real traces, but was given a complex one.")
                x = np.asarray(trace, dtype=np.float32)
                if lbtaper > 0 or lstaper > 0:
                    x = _taper(int(lstaper), bottom, int(ncdps or 0), icmp, nt, x).astype(np.float32)
                offset = float(trace.header['offset'])
                data.append(np.ascontiguousarray(x))
                offsets.append(offset)
                m = fmax * offset * offset * dp2
                if nonhyp:
                    m = max(m, 2.0 * abs(offset) / vhyp)
                mutes.append(m)
            times = np.arange(ntout) * out_dt
            template = first
            middle = 0.5 * (first.header['tx_loc'][0] + first.header['rx_loc'][0])
            positions = dict(tx_loc=[middle, first.header['tx_loc'][1], first.header['tx_loc'][2]],
                             rx_loc=[middle, first.header['rx_loc'][1], first.header['rx_loc'][2]])
            cdp = first.header['ensemble_number']
            for iv in range(nvstack):
                p2 = iv * dp2
                stack = np.zeros(ntout)
                fold = np.zeros(ntout)
                for x, offset, mute in zip(data, offsets, mutes):
                    itmute = int(mute / out_dt)
                    if itmute >= ntout:
                        continue
                    factor = offset * offset * p2
                    ttn = np.sqrt(times[itmute:] ** 2 + factor).astype(np.float32)
                    # NMO by 8 point sinc interpolation
                    q = _resample.interpolate(x, float(dt), 0.0, np.ascontiguousarray(ttn)).astype(np.float64)
                    # a linear ramp tapers the mute
                    n = min(taper_length, ntout - itmute)
                    q[:n] *= (np.arange(n) + 1.0) / taper_length
                    stack[itmute:] += q
                    fold[itmute:] += 1.0
                fold[fold == 0.0] = 1.0
                yield template.replace(np.ascontiguousarray(stack / fold, dtype=F32), d_sample=out_dt,
                                       ensemble_number=cdp, ensemble_trace_number=iv + 1, **positions)

    return from_iterable(traces())


# SUCVS4FOWLER: constant velocity stacks of the CDP gathers, for the Fowler DMO codes: nvstack traces for each gather, the
# stack of the NMO correction with the constant slowness^2 iv * dp2, dp2 = 1 / (vminstack^2 (nvstack - 5)), with
# ensemble_trace_number the number of the panel from 1 (zero offset, d_sample dtout).
#
# ncdps : the number of gathers to do (all; needed for lstaper), vminstack : the smallest velocity (m/s), nvstack : the number
# of panels (180), lmute : the length (ms) of the taper of the mute (24), nonhyp, vhyp : mute at 2 |offset| / vhyp to
# avoid non-hyperbolic moveout, lbtaper : the length (ms) of the taper of the bottom, lstaper : the length (traces) of the
# side taper, dtout : the sample interval of the output (1.5 dt), mxfold : the most traces in a gather (120), salias : the
# fraction of the output frequencies kept within the sloth antialias limit (0.8).
# Differences from SU: the non-hyperbolic mute uses the absolute offset; a gather over mxfold traces is an error before it
# overruns the arrays; the stacks have the header of the gather's first trace, at its midpoint.
cvs4fowler = stage(_cvs4fowler, parallelism='ensemble', name='cvs4fowler', validate=True)
