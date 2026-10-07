"""
Stacking: SUSTACK, SUDIVSTACK, SUPWS and SUSTACKUP (``su/main/stacking``).

These make one trace from the traces of each gather. A gather is the run of traces that have the same value of ``key``
(a name from ``trace.header``, or a function of a trace, like the keys of ``seispy.parallel.group_by``): by default
``ensemble_number``, the CDP of SU. ``stack``, ``divstack`` and ``pws`` need the traces of a gather to be next to each
other, ``stackup`` does not (it reads all of the traces first). The stacked trace has the header of the first
trace of its gather (of the trace with the smallest offset for ``stackup``), and ``fold``, the number of traces that were
stacked.

A stage of the first three can be split between gathers across workers: ``pmap(stack(), key='ensemble_number')``.
What is done with the traces of a gather is done by functions of the SU library (``_kernels.pyx``).
The stacked traces are not renumbered (the SU programs number them from 1), so that they do not depend on the
traces before them.

A stack of traces that were made at the offsets of a CDP has no offset, so ``stack`` (and ``pws`` of CDPs) moves the
source and the receiver of the stacked trace to the midpoint of the first trace of the gather: its offset is 0.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from ..stage import header_value, stage
from ..transforms import _hilbert
from . import _kernels

__all__ = ['stack', 'divstack', 'pws', 'stackup']

F32 = np.float32


def _nint(x):
    return int(x + 0.5) if x > 0.0 else int(x - 0.5)


def _at_midpoint(trace):
    """The values of the header to give a stacked trace its zero offset: the source and receiver at the midpoint"""
    tx, rx = trace.header['tx_loc'], trace.header['rx_loc']
    middle = [0.5 * (tx[0] + rx[0]), 0.5 * (tx[1] + rx[1])]
    return dict(tx_loc=middle + [tx[2]], rx_loc=middle + [rx[2]])


def _floats(x):
    """The samples of a trace as the floats that the functions of the SU library take (2 for each complex sample)"""
    x = np.ascontiguousarray(x)
    return x.view(np.float32) if x.dtype.kind == 'c' else x.astype(np.float32, copy=False)


def _fold_of(trace):
    return trace.header['fold'] or 1


def _samples(group):
    """The traces of a gather as float64 (or complex128) arrays, which have to be the same length"""
    arrays = [np.asarray(t) for t in group]
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("The traces of a gather must have the same number of samples to be stacked.")
    return arrays


# ---------------------------------------------------------------------------------------------------------- stack
def _stack(upstream, *, key='ensemble_number', normpow=1.0, nrepeat=1):
    if nrepeat < 1:
        raise ValueError("nrepeat must be at least 1")
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _samples(group)
            dtype = arrays[0].dtype
            ncomp = 2 if dtype.kind == 'c' else 1
            n = arrays[0].shape[0]
            total = np.zeros(n * ncomp, dtype=np.float32)
            nonzero = np.zeros(n, dtype=np.intc)
            for a in arrays:
                _kernels.stack_add(_floats(a), ncomp, total, nonzero)  # (the functions of the SU library)
            fold = sum(_fold_of(t) for t in group)
            if normpow and fold != 1:
                # each sample is divided by the number of values that were not 0 that went into it, to a power
                _kernels.stack_normalize(total, nonzero, ncomp, normpow)
            header = {} if key == 'offset' else _at_midpoint(group[0])
            out = group[0].replace(total.view(np.complex64) if ncomp == 2 else total, fold=fold, **header)
            for _ in range(nrepeat):
                yield out

    return from_iterable(traces())


# SUSTACK: stack adjacent traces that have the same key
#
# key : the header value to stack on, default 'ensemble_number' (the CDP)
# normpow : each sample is divided by the number of values that were not 0 that were stacked, to this power. The default
#           1 is the mean of them, 0 is not to divide.
# nrepeat : repeat each stacked trace this many times, default 1 (SU's repeat=1 and nrepeat=10 are for the VSP
#           corridor stack)
#
# `fold` of the stacked trace is the sum of those of the traces (1 for a trace that has none).
stack = stage(_stack, parallelism='ensemble', name='stack', validate=True)


# ------------------------------------------------------------------------------------------------------- divstack
def _divstack(upstream, *, key='ensemble_trace_number', winlen=None, peak=False, dt=None):
    if winlen is not None and winlen <= 0.0:
        raise ValueError("winlen must be positive")
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            arrays = _samples(group)
            n = arrays[0].shape[0]
            if winlen is None:
                n_window = n
            else:
                sample_dt = group[0].d_sample or dt
                if not sample_dt:
                    raise ValueError("The trace has no sample interval, and no `dt` was given.")
                n_window = min(int(winlen / sample_dt) + 1, n)
            ncomp = 2 if arrays[0].dtype.kind == 'c' else 1
            scaled = np.zeros(n * ncomp, dtype=np.float32)
            scalers = np.zeros(n, dtype=np.float32)
            for a in arrays:
                x = _floats(a)
                power = _kernels.divstack_power(x, ncomp, n_window, bool(peak))  # (the functions of the SU library)
                _kernels.divstack_add(x, ncomp, power, scaled, scalers)
            stacked = _kernels.divstack_finish(scaled, scalers, ncomp)
            fold = sum(_fold_of(t) for t in group)
            yield group[0].replace(stacked.view(np.complex64) if ncomp == 2 else stacked, fold=fold)

    return from_iterable(traces())


# SUDIVSTACK: diversity stack: each trace is scaled by the inverse of its power (the average, or with `peak` the
# largest, of the squares of the samples) in windows, which is interpolated between the windows, and the sum is divided
# by the sum of the scalers. This reduces the noise of duplicate data.
#
# key : the header value to stack on, default 'ensemble_trace_number' (SU's tracf, the trace number within the field
#       record, to stack duplicate shots, after `sort`ing on it)
# winlen : the length (s) of the windows, default the whole trace. Typical: .064, .128, ..., 4.096
# peak : use the peak power, not the average
# dt : sample interval (s) for traces that do not have one
divstack = stage(_divstack, parallelism='ensemble', name='divstack', validate=True)


# ------------------------------------------------------------------------------------------------------------ pws
def _pws(upstream, *, key='ensemble_number', pwr=1.0, sl=0.0, ps=False, dt=None):
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            if any(t.dtype.kind == 'c' for t in group):
                raise TypeError("This stage works on real traces, but was given a complex one.")
            arrays = [a.astype(F32) for a in _samples(group)]
            n = arrays[0].shape[0]
            sample_dt = group[0].d_sample or dt or 0.004
            isl = _nint(abs(sl) / sample_dt)
            if isl > n:
                raise ValueError(f"sl={abs(sl)} is too long for the trace")
            ordinary = np.zeros(n, dtype=np.float32)
            phase_stack = np.zeros(2 * n, dtype=np.float32)
            for a in arrays:
                a = np.ascontiguousarray(a)
                # (the Hilbert transform of SU, and the functions of the SU library)
                _kernels.pws_accumulate(a, _hilbert.hilbert(a), ordinary, phase_stack)
            fold = len(group)
            weights = _kernels.pws_weights(phase_stack, fold, pwr, isl)
            if ps:
                result = weights
            else:
                result = ordinary
                _kernels.pws_apply(weights, fold, result)
            header = _at_midpoint(group[0]) if (key == 'ensemble_number' and not ps) else {}
            yield group[0].replace(result.astype(F32), fold=fold, **header)

    return from_iterable(traces())


# SUPWS: phase stack, or phase weighted stack (PWS), of adjacent traces that have the same key. The phase stack is the
# modulus of the mean of the unit phasors of the traces (from their instantaneous phase), 0 to 1 where they are
# incoherent to coherent. The PWS is the mean of the traces multiplied by it, which removes incoherent noise.
#
# key : the header value to stack on, default 'ensemble_number' (the CDP)
# pwr : raise the phase stack to this power, default 1
# sl : the length (s) of a window to smooth the weights with, default 0 (none)
# ps : output the phase stack (the weights), not the PWS
# dt : sample interval (s) for traces that do not have one, default .004
#
# A PWS of CDPs has its offset set to 0 (as in `stack`).
pws = stage(_pws, parallelism='ensemble', name='pws', validate=True)


# ---------------------------------------------------------------------------------------------------------- stackup
def _in_range(value, low, high):
    """In [low, high) when low < high, outside of [high, low) otherwise (for angles that wrap around)"""
    if low < high:
        return low <= value < high
    return value >= low or value < high


def _stackup(upstream, *, keyloc=('ensemble_number',), keyabs=None, keyabs2=None, minabs=None, maxabs=None,
             keysign=None, keysign2=None, minsign=None, maxsign=None, keep=True):
    if isinstance(keyloc, str) or callable(keyloc):
        keyloc = (keyloc,)
    keyloc = tuple(keyloc)
    if not keyloc:
        raise ValueError("keyloc must have at least one key")
    if keyabs is None and (keyabs2 is not None or minabs is not None or maxabs is not None):
        raise ValueError("keyabs2, minabs and maxabs need keyabs")
    if keysign is None and (keysign2 is not None or minsign is not None or maxsign is not None):
        raise ValueError("keysign2, minsign and maxsign need keysign")
    low_abs = -1.0e31 if minabs is None else minabs
    high_abs = 1.0e31 if maxabs is None else maxabs
    low_sign = -1.0e31 if minsign is None else minsign
    high_sign = 1.0e31 if maxsign is None else maxsign
    source = as_trace_iterator(upstream)

    def live(trace):
        if keyabs is not None:
            value = header_value(trace, keyabs)
            if keyabs2 is not None:
                value -= header_value(trace, keyabs2)
            if not _in_range(abs(value), low_abs, high_abs):
                return False
        if keysign is not None:
            value = header_value(trace, keysign)
            if keysign2 is not None:
                value -= header_value(trace, keysign2)
            if not _in_range(value, low_sign, high_sign):
                return False
        return True

    def traces():
        stacks = {}  # location -> [header trace (the smallest offset), sum, samples of data, fold]
        for trace in source:
            is_live = live(trace)
            if not is_live and not keep:
                continue
            location = tuple(header_value(trace, k) for k in keyloc)
            offset = abs(trace.header['offset'])
            x = np.asarray(trace)
            if x.dtype.kind == 'c':
                raise TypeError("This stage works on real traces, but was given a complex one.")
            entry = stacks.get(location)
            if entry is None:
                entry = stacks[location] = [trace, offset, np.zeros(x.shape, dtype=np.float64), np.zeros(x.shape, dtype=np.float32), 0]
            elif offset < entry[1]:
                entry[0], entry[1] = trace, offset
            if is_live:
                if x.shape != entry[2].shape:
                    raise ValueError("The traces must all have the same number of samples to be stacked.")
                _kernels.stackup_add(np.ascontiguousarray(x, dtype=np.float32), entry[2], entry[3])  # (the function of the SU library)
                entry[4] += 1
        for location in sorted(stacks):
            header_trace, _, total, sample_fold, fold = stacks[location]
            yield header_trace.replace(_kernels.stackup_finish(total, sample_fold), fold=fold)

    return from_iterable(traces())


# SUSTACKUP: stack to any combination of keys, from traces in any order. All of the traces that have the same values of the
# keys of `keyloc` are stacked. Each sample is the mean of the values that are not 0.
#
# keyloc : the keys (names, or functions of a trace) that give the location of a stacked trace, default ('ensemble_number',).
#          The stacked traces come out in order of their values.
# keyabs, keyabs2, minabs, maxabs : only traces for which |keyabs - keyabs2| is in [minabs, maxabs) are stacked
#          (usually keyabs='offset'). With minabs >= maxabs the range is outside of [maxabs, minabs), which is for angles.
# keysign, keysign2, minsign, maxsign : the same, with the sign kept, for keysign - keysign2.
# keep : output a trace (of zeros, with fold 0) for every location, even if the ranges leave it with no trace
#
# The header of a stacked trace is that of the trace with the smallest absolute offset of the location (in the range
# or not), and its fold is the number of traces that were stacked. (sustackup makes its offset positive, here it is left as
# it is.) All of the traces are read before the first is output.
stackup = stage(_stackup, parallelism='serial', name='stackup', validate=True)
