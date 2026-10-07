"""
SUPGC: one gain function for all the traces, found from the first traces of the data.
"""
import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..stage import stage
from . import _pgc as _pgc_c

__all__ = ['pgc']


def _pgc(upstream, *, ntrscan=200, lwindow=1.0):
    if ntrscan < 1:
        raise ValueError(f"ntrscan={ntrscan} must be at least 1")
    if lwindow < 0.0:
        raise ValueError(f"lwindow={lwindow} must not be negative")
    source = as_trace_iterator(upstream)

    def gain_function(total, count, dt):
        lw = int(0.5 * lwindow / dt + 0.5)
        # (the function of the SU library)
        return _pgc_c.gain_function(total.astype(np.float32), count, lw)

    def traces():
        scanned = []
        total = None
        for trace in source:
            x = np.asarray(trace)
            if total is None:
                total = np.zeros(x.shape[0])
            elif x.shape[0] != total.shape[0]:
                raise ValueError("pgc needs every trace to have the same number of samples.")
            total += np.abs(x)
            scanned.append(trace)
            if len(scanned) == ntrscan:
                break
        if not scanned:
            return
        dt = scanned[0].d_sample
        if not dt:
            raise ValueError("The first trace has no sample interval.")
        g = gain_function(total, len(scanned), dt)
        for trace in scanned:
            yield trace.replace(np.asarray(trace) * g)
        for trace in source:
            x = np.asarray(trace)
            if x.shape[0] != g.shape[0]:
                raise ValueError("pgc needs every trace to have the same number of samples.")
            yield trace.replace(x * g)

    return from_iterable(traces(), n_traces=source.n_traces)


# SUPGC: programmed gain control. An AGC like gain that is the same for all of the traces, so that their relative
# amplitudes are kept.
#
# ntrscan : the number of traces (from the start) that the gain function is found from, default 200
# lwindow : the length (s) of the window of the gain, default 1.0
#
# The gain function at each sample is the number of samples in the window around it times the number of traces
# scanned, divided by the sum of the amplitudes in the window over the scanned traces. (Where the sum is 0 it is 1.)
# The scanned traces are held until the gain is known, the rest go through one at a time.
pgc = stage(_pgc, parallelism='serial', name='pgc', validate=True)
