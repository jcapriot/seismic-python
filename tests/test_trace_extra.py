import pickle

import numpy as np

import seispy
from seispy.container import Trace


def make():
    return Trace(np.arange(8, dtype=np.float32), d_sample=0.004).replace(extra={"fldr": 7, "tracl": 3})


def test_extra_defaults_empty_and_is_per_trace():
    tr = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004)
    assert tr.extra == {}
    tr.extra["fldr"] = 1
    assert Trace(np.zeros(4, dtype=np.float32), d_sample=0.004).extra == {}


def test_replace_carries_a_copy():
    tr = make()
    new = tr.replace(data=np.ones(8, dtype=np.float32), d_sample=0.002)
    assert new.extra == {"fldr": 7, "tracl": 3}
    new.extra["fldr"] = 9
    assert tr.extra["fldr"] == 7
    assert tr.replace(extra={"fldr": 8}).extra == {"fldr": 8, "tracl": 3}


def test_pickle_roundtrip():
    tr = make()
    assert pickle.loads(pickle.dumps(tr)).extra == {"fldr": 7, "tracl": 3}


def test_extra_survives_cython_stages():
    from seispy import amplitudes, filters, operations

    tr = make().replace(data=np.random.default_rng(0).standard_normal(64).astype(np.float32))
    for stage in (
        amplitudes.gain(tpow=1.0),
        filters.bfilt(f3db_low=5.0, n_poles_low=2),
        operations.abs(),
    ):
        (out,) = list(seispy.container.as_trace_iterator([tr]) | stage)
        assert out.extra == {"fldr": 7, "tracl": 3}, stage
