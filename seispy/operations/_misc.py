"""
Programs of ``su/main/operations`` that are not arithmetic on a trace by itself: SUPERMUTE (a panel is a 3D cube to
transpose), SURECMO (mix with the traces before it, for a streamer that moves) and SUSIMMAT (a similarity matrix of two
traces).

Where the program plainly does not do what its documentation says this does what the documentation says; the
differences are listed with each stage.
"""
import math
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import stage

__all__ = ['_permute', '_recmo', '_simmat']


# --------------------------------------------------------------------------------------------------------- permute
def _permute(upstream, *, n1=None, n2=None, n3=1, o=(1, 2, 3), d=(1, 1, 1)):
    o = tuple(int(x) for x in o)
    d = tuple(int(x) for x in d)
    if sorted(o) != [1, 2, 3]:
        raise ValueError(f"invalid permutation {o}: o is a permutation of 1, 2 and 3")
    if len(d) != 3 or 0 in d:
        raise ValueError(f"invalid interval {d}: d has three values that are not 0")
    if n3 is not None and n3 < 1:
        raise ValueError("invalid dimensions: n3 must be at least 1")
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        nt = panel[0].n_sample
        c1 = nt if n1 is None else int(n1)
        c3 = int(n3)
        c2 = len(panel) // c3 if n2 is None else int(n2)
        if c1 < 1 or c1 > nt or c2 < 1:
            raise ValueError(f"invalid dimensions: {c1} {c2} {c3}")
        if len(panel) < c2 * c3:
            raise ValueError(f"wrong amount of data: {c2 * c3} traces of {c1} samples are needed, there are {len(panel)}")
        cube = np.stack([np.asarray(t)[:c1] for t in panel[:c2 * c3]]).reshape(c3, c2, c1)
        # the axes of the cube are (3, 2, 1), the dimensions of the program: the new fast axis is the old dimension o1
        new = np.transpose(cube, (3 - o[2], 3 - o[1], 3 - o[0]))
        # a negative interval reverses the dimension, and the output is every |d|th sample
        for axis, step in zip((2, 1, 0), d):
            if step < 0:
                new = np.flip(new, axis=axis)
        new = new[::abs(d[2]), ::abs(d[1]), ::abs(d[0])]
        template = panel[0]
        for row in new.reshape(-1, new.shape[2]):
            yield template.replace(np.ascontiguousarray(row, dtype=np.float32), d_sample=1.0, sample_start=1.0)

    return from_iterable(traces())


# SUPERMUTE: permute (transpose) a 3D cube of traces: n1 samples (default all) in each of n2 traces (default all of them
# if n3 is 1) in each of n3 panels. o is the new order of the dimensions: o=(2, 1, 3) is the transpose of the panels. d are
# the output intervals in the new dimensions (a negative one reverses it). The traces have d_sample and sample_start 1.
# Differences from SU: the number of traces in the header (ntr) is not set.
permute = stage(_permute, parallelism='serial', name='permute', validate=True)


# ----------------------------------------------------------------------------------------------------------- recmo
def _recmo(upstream, *, vb, dx, dt=None, fill=0.0):
    if dx == 0:
        raise ValueError("dx must not be 0")
    source = as_trace_iterator(upstream)

    def traces():
        previous = None  # the traces before: [1] the one before this one, [2] the one before that
        warned = False
        for trace in source:
            if trace.dtype.kind == 'c':
                raise TypeError("recmo works on real traces, but was given a complex one.")
            x = np.asarray(trace, dtype=np.float64)
            nt = x.shape[0]
            sample_dt = float(dt) if dt else trace.d_sample
            if not sample_dt:
                raise ValueError("The trace has no sample interval, give dt.")
            if previous is None or previous[0].shape[0] != nt:
                previous = [np.full(nt, float(fill)), np.full(nt, float(fill))]
            t = np.arange(nt) * sample_dt
            # the streamer has moved vb * t in the time of the sample: the weights of this trace, the one before and the
            # one before that (valid while that is less than two channel intervals)
            m1 = vb * t / dx
            if not warned and m1.max() > 2.0:
                warnings.warn("recmo is valid for data where tmax < 2*dx/vb")
                warned = True
            m0 = 1.0 - m1
            m2 = np.zeros(nt)
            over = m1 > 1.0
            m2[over] = m1[over] - 1.0
            m1[over] = 1.0 - m2[over]
            m0[over] = 0.0
            out = x * m0 + previous[0] * m1 + previous[1] * m2
            previous = [x, previous[0]]
            yield trace.replace(np.ascontiguousarray(out, dtype=np.float32))

    return from_iterable(traces(), n_traces=source.n_traces)


# SURECMO: compensate for the continuously moving streamer of marine acquisition, mixing each trace with the two before it
# with weights that depend on the time (the far offset trace first). vb : the boat speed (m/s), dx : the channel spacing (m),
# dt : the sample interval in seconds (that of the traces by default), fill : the value of the traces before the first one.
# Differences from SU: dt is in seconds, not microseconds; fill is used (the program did not read it).
recmo = stage(_recmo, parallelism='serial', name='recmo', validate=True)


# ---------------------------------------------------------------------------------------------------------- simmat
def _window_sums(a, b, hw_matrix, hw0, weight_for):
    """For every (icol, irow): the sums over the window of half length hw (telescoped at the edges, hw_matrix) of
    a[icol + s] * b[irow + s] * w(s), a[icol + s]^2 and b[irow + s]^2, and the plain sums of a, b, a * b"""
    nt = a.shape[0]
    shape = (nt, nt)
    sums = {k: np.zeros(shape) for k in ('ab', 'aa', 'bb', 'a', 'b', 'abp', 'aap', 'bbp', 'n')}
    cols = np.arange(nt)
    for s in range(-hw0, hw0 + 1):
        inside = np.abs(s) <= hw_matrix
        ia = np.clip(cols + s, 0, nt - 1)
        av = a[ia][:, None] * np.ones(nt)[None, :]
        bv = np.ones(nt)[:, None] * b[ia][None, :]
        w = weight_for(s, hw_matrix)
        m = inside.astype(np.float64)
        sums['ab'] += av * bv * w * m
        sums['aa'] += av * av * m
        sums['bb'] += bv * bv * m
        sums['a'] += av * m
        sums['b'] += bv * m
        sums['abp'] += av * bv * m
        sums['aap'] += av * av * m
        sums['bbp'] += bv * bv * m
        sums['n'] += m
    return sums


def _similarity_matrix(trace, data, mt, method, taper, eps):
    nt = trace.shape[0]
    hw0 = (mt - 1) // 2
    idx = np.arange(nt)
    # the half length of the window: the one asked for, as long as it fits (telescoping boundary conditions)
    edge = np.minimum(idx, nt - 1 - idx)
    hw = np.minimum(np.minimum(edge[:, None], edge[None, :]), hw0)

    def weight_for(s, hw_matrix):
        if not taper:
            return 1.0
        # exponential taper: 1/e at the ends of the window
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(hw_matrix > 0, s / np.maximum(hw_matrix, 1), 0.0)
        return np.exp(-r * r)

    s = _window_sums(trace, data, hw, hw0, weight_for)
    x0 = s['ab']
    if method == 1:
        return x0
    scale = np.sqrt(s['aa'] * s['bb']) + eps
    if method == 2:
        return x0 / scale
    if method == 4:
        return np.degrees(np.arccos(np.clip(x0 / scale, -1.0, 1.0)))
    # method 3: the correlation coefficient of the linear regression of the window of the data on that of the trace
    n = s['n']
    num = n * s['abp'] - s['a'] * s['b']
    den = np.sqrt(np.maximum(n * s['aap'] - s['a'] ** 2, 0.0) * np.maximum(n * s['bbp'] - s['b'] ** 2, 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(den > 0, num / den, 0.0)


def _simmat(upstream, *, other, panel=False, mt=21, eps=1e-3, taper=False, method=1):
    if method not in (1, 2, 3, 4):
        raise ValueError("Bad method")
    if mt < 1:
        raise ValueError("mt must be at least 1")
    if mt != 21 and mt != 1:
        mt = mt // 2 * 2 + 1  # (odd)
    if mt < 5 and method == 3:
        raise ValueError("Need more than 5 points for linear regression")
    source = as_trace_iterator(upstream)

    def traces():
        if panel:
            others = iter(as_trace_iterator(other))
            single = None
        else:
            single = None
            for single in as_trace_iterator(other):
                break
            if single is None:
                raise ValueError("other has no trace")
        for trace in source:
            if trace.dtype.kind == 'c':
                raise TypeError("simmat works on real traces, but was given a complex one.")
            if panel:
                try:
                    partner = next(others)
                except StopIteration:
                    raise ValueError("number of traces in other are not the same as the number of traces on the input") from None
            else:
                partner = single
            a = np.asarray(trace, dtype=np.float64)
            b = np.asarray(partner, dtype=np.float64)
            if a.shape != b.shape:
                raise ValueError("The traces and the trace of other must have the same number of samples.")
            matrix = _similarity_matrix(a, b, mt, method, bool(taper), float(eps))
            # horizontal axis is the time of the trace, vertical that of the other: one output trace for each time of
            # the trace, over the times of the other
            for icol in range(a.shape[0]):
                yield trace.replace(np.ascontiguousarray(matrix[icol], dtype=np.float32), trace_id=icol + 1)

    return from_iterable(traces())


# SUSIMMAT: the correlation similarity matrix of a trace with another one: one output trace for each sample (time) of the
# trace, whose samples are over the times of the other. method 1: the zero lag of the cross-correlation of windows of mt
# samples (odd), 2: that normalized by the rms (and eps), 3: the correlation coefficient of the linear regression, 4: the
# angle (degrees) of the mt dimensional vectors. taper : an exponential taper of the windows (1/e at the ends, method 1).
# other : the other trace (the first of it, or a trace stream with a trace for each input trace if panel=True).
# Differences from SU: the window is shortened at both ends (the program did not shorten it at the end of the trace, and read
# past the trace), method 2 is the normalized correlation (it was the angle of method 4), and the taper is the one of the
# documentation (it was zero but at the ends of the windows, by an integer division).
simmat = stage(_simmat, parallelism='serial', name='simmat', validate=True)
