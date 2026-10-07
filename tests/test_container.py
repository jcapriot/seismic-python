import gc
import sys

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, TraceCollection
from seispy.synthetics import spike


def test_trace_constructor():
    data = np.arange(10, dtype=np.float64)
    tr = Trace(data, d_sample=0.004)
    assert tr.n_sample == 10
    assert len(tr) == 10
    assert tr.d_sample == 0.004
    npt.assert_equal(np.asarray(tr), data.astype(np.float32))


def test_empty_trace_has_an_empty_buffer():
    tr = Trace(np.zeros(0), d_sample=0.004)
    assert np.asarray(tr).shape == (0,)
    assert np.asarray(tr).dtype == np.float32
    assert memoryview(tr).nbytes == 0
    # and a real one reports its size in bytes
    assert memoryview(Trace(np.zeros(5), d_sample=0.004)).nbytes == 20


def test_trace_bad_units():
    with pytest.raises(KeyError):
        Trace(np.zeros(4), d_sample=1.0, sampling_unit='parsecs')


def test_collection_from_arrays():
    data = np.random.default_rng(0).normal(size=(5, 16))
    coll = TraceCollection(data, d_sample=0.002)
    assert coll.in_memory
    assert len(coll) == 5
    for row, tr in zip(data, coll):
        npt.assert_allclose(np.asarray(tr), row.astype(np.float32))


def test_collection_file_roundtrip(tmp_path):
    path = tmp_path / "out.spy"
    ref = spike().to_memory()
    spike().to_file(path)

    # A fresh collection that only knows about the file
    coll = TraceCollection.from_file(path)
    assert coll.on_disk
    assert len(coll) == len(ref)
    n = 0
    for tr_a, tr_b in zip(ref, coll):
        npt.assert_equal(np.asarray(tr_a), np.asarray(tr_b))
        n += 1
    assert n == len(ref)


def _rss_bytes():
    """Resident set size of this process (tracemalloc can't see C mallocs)."""
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
            ]

        counters = PMC()
        counters.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            wintypes.HANDLE(k32.GetCurrentProcess()),
            ctypes.byref(counters), counters.cb,
        )
        if not ok:
            pytest.skip("Unable to query process memory.")
        return counters.WorkingSetSize
    try:
        import resource
    except ImportError:
        pytest.skip("Unable to query process memory.")
    with open('/proc/self/statm') as f:
        return int(f.read().split()[1]) * resource.getpagesize()


def test_trace_buffers_are_freed():
    # Each trace is 64 KiB of data, if the buffers were leaked 2000 of them
    # would hold on to ~128 MiB.
    def run():
        for _ in range(2000):
            for tr in spike(nt=16384, ntr=1):
                pass

    run()  # warm up
    gc.collect()
    before = _rss_bytes()
    run()
    gc.collect()
    growth = _rss_bytes() - before
    assert growth < 20_000_000


# --------------------------------------------------------------------------- derived header values (not stored)
def test_offset_is_the_distance_from_the_source_to_the_receiver():
    import numpy as np
    from seispy.container import Trace

    t = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004, tx_loc=[10.0, 0.0, 0.0], rx_loc=[13.0, 4.0, 0.0])
    assert t.header['offset'] == 5.0
    assert t.replace(rx_loc=[7.0, 4.0, 0.0]).header['offset'] == -5.0  # (before the source)
    assert t.replace(rx_loc=[10.0, -2.0, 0.0]).header['offset'] == -2.0  # (same x: by y)
    assert Trace(np.zeros(4, dtype=np.float32), d_sample=0.004).header['offset'] == 0.0
    assert 'mid_point' not in t.header


def test_replace_offset_moves_the_receiver():
    import numpy as np
    import pytest
    from seispy.container import Trace

    t = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004, tx_loc=[10.0, 3.0, 1.0], rx_loc=[0.0, 0.0, 2.0])
    moved = t.replace(offset=-250.0)
    assert moved.header['offset'] == -250.0
    assert moved.header['tx_loc'] == [10.0, 3.0, 1.0]
    assert moved.header['rx_loc'] == [-240.0, 3.0, 2.0]
    with pytest.raises(TypeError):
        t.replace(offset=1.0, rx_loc=[1.0, 1.0, 1.0])
