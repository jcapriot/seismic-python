import pickle

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import TraceCollection, BaseTraceIterator
from seispy.filters import bfilt
from seispy.stage import Pipeline
from seispy.synthetics import spike


def _arr(it):
    return np.array([np.asarray(t) for t in it])


def test_pipe_matches_bind():
    s = bfilt(f_pass_low=10.0, f_stop_low=5.0)
    npt.assert_equal(_arr(spike() | s), _arr(s.bind(spike())))


def test_pipe_collection_and_result_type():
    coll = spike().to_memory()
    out = coll | bfilt()
    assert isinstance(out, BaseTraceIterator)
    assert len(out.to_memory()) == len(coll)


def test_stage_composition_flattens():
    s = bfilt() | bfilt(zerophase=False) | bfilt()
    assert isinstance(s, Pipeline)
    assert len(s.stages) == 3


def test_pipeline_equals_chaining_one_by_one():
    s1, s2 = bfilt(zerophase=False), bfilt()
    npt.assert_equal(_arr(spike() | (s1 | s2)), _arr(spike() | s1 | s2))
    npt.assert_equal(_arr(spike() | (s1 | s2)), _arr(s2.bind(s1.bind(spike()))))


def test_stage_reusable_and_picklable():
    s = bfilt(f_pass_low=10.0, f_stop_low=5.0) | bfilt()
    s2 = pickle.loads(pickle.dumps(s))
    npt.assert_equal(_arr(spike() | s), _arr(spike() | s2))
    # reusable: binding twice gives independent iterators
    npt.assert_equal(_arr(spike() | s), _arr(spike() | s))


def test_generator_into_stage():
    def gen():
        for tr in spike():
            yield tr

    npt.assert_equal(_arr(gen() | bfilt()), _arr(spike() | bfilt()))


def test_list_into_stage():
    traces = list(spike())
    npt.assert_equal(_arr(traces | bfilt()), _arr(spike() | bfilt()))


def test_generator_collect_and_write(tmp_path):
    def gen():
        for tr in spike():
            yield tr

    mem = (gen() | bfilt()).to_memory()
    assert len(mem) == 32
    assert len(list(mem)) == 32

    path = tmp_path / "out.spy"
    (gen() | bfilt()).to_file(path)
    back = TraceCollection.from_file(path)
    assert len(back) == 32
    npt.assert_equal(_arr(back), _arr(mem))


def test_bad_items_raise():
    with pytest.raises(TypeError):
        list([1, 2, 3] | bfilt())


def test_or_with_non_stage():
    with pytest.raises(TypeError):
        bfilt() | 3

def test_stage_repr_uses_su_name():
    assert repr(bfilt(f_pass_low=10.0) | bfilt()) == "bfilt(f_pass_low=10.0) | bfilt()"
