import pickle
import threading
import time

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.filters import bfilt
from seispy.parallel import group_by, pmap, prefetch
from seispy.stage import Pipeline, stage
from seispy.synthetics import spike, synlv


def _arr(it):
    return np.array([np.asarray(t) for t in it])


def _normalize_gathers(upstream, key='ensemble_number'):
    """Scale every gather by its own peak amplitude (needs the whole gather)."""

    def gen():
        for grp in group_by(upstream, key):
            peak = max(np.abs(np.asarray(t)).max() for t in grp) or 1.0
            for t in grp:
                np.asarray(t)[:] /= peak
                yield t

    return from_iterable(gen())


normalize_gathers = stage(_normalize_gathers, parallelism='ensemble')


@stage
def _serial_stage(upstream):
    return from_iterable(upstream)


def _fail_stage_factory(upstream):
    def gen():
        for i, t in enumerate(upstream):
            if i == 5:
                raise RuntimeError("boom")
            yield t

    return from_iterable(gen())


fail_stage = stage(_fail_stage_factory, parallelism='trace')


def _wait_for_threads(n, timeout=5.0):
    t0 = time.time()
    while threading.active_count() > n and time.time() - t0 < timeout:
        time.sleep(0.01)
    return threading.active_count()


def test_trace_pickles():
    tr = list(spike())[8]
    tr2 = pickle.loads(pickle.dumps(tr))
    npt.assert_equal(np.asarray(tr), np.asarray(tr2))
    assert tr2.header == tr.header


def test_n_traces_property():
    assert spike().n_traces == 32
    assert from_iterable(t for t in spike()).n_traces is None
    assert from_iterable(list(spike())).n_traces == 32


@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("chunk", [1, 5, 100])
def test_pmap_thread_matches_sequential(workers, chunk):
    s = bfilt(f_pass_low=10.0, f_stop_low=5.0)
    expected = _arr(synlv() | s)
    out = synlv() | pmap(s, workers=workers, chunk=chunk)
    assert out.n_traces == expected.shape[0]
    npt.assert_equal(_arr(out), expected)


def test_pmap_process_matches_sequential():
    s = bfilt() | bfilt(zerophase=False)
    expected = _arr(spike() | s)
    out = _arr(spike() | pmap(s, workers=2, chunk=8, executor='process'))
    npt.assert_equal(out, expected)


def test_pmap_preserves_order_with_uneven_work():
    # make early chunks slower than late ones, results must still come out in order
    def slow_first(upstream):
        def gen():
            for t in upstream:
                if t.header['trace_id'] <= 4:
                    time.sleep(0.05)
                yield t

        return from_iterable(gen())

    s = stage(slow_first, parallelism='trace')()
    ids = [t.header['trace_id'] for t in spike() | pmap(s, workers=4, chunk=2)]
    assert ids == list(range(1, 33))


def test_pmap_is_lazy_and_bounded():
    consumed = []

    def source():
        for tr in spike(ntr=1000):
            consumed.append(1)
            yield tr

    chunk, workers, window = 4, 2, 3
    out = source() | pmap(bfilt(), workers=workers, chunk=chunk, prefetch_chunks=window)
    next(iter(out))
    # reading ahead is limited to the window of in-flight chunks
    assert len(consumed) <= chunk * (window + 1)
    assert len(consumed) < 1000
    out.close() if hasattr(out, 'close') else None


def test_pmap_early_exit_stops_workers():
    n_before = threading.active_count()
    out = spike(ntr=500) | pmap(bfilt(), workers=3, chunk=2)
    for i, _ in enumerate(out):
        if i == 3:
            break
    del out
    assert _wait_for_threads(n_before) <= n_before


def test_pmap_worker_error_propagates():
    with pytest.raises(RuntimeError, match="boom"):
        list(spike() | pmap(fail_stage(), workers=2, chunk=8))


def test_pmap_source_error_propagates():
    def source():
        yield from itertools_islice(spike(), 3)
        raise ValueError("bad source")

    with pytest.raises(ValueError, match="bad source"):
        list(source() | pmap(bfilt(), workers=2, chunk=2))


def itertools_islice(it, n):
    import itertools
    return itertools.islice(it, n)


def test_pmap_refuses_serial_stage():
    with pytest.raises(ValueError, match="parallel"):
        pmap(_serial_stage())
    # a pipeline is only as parallel as its least parallel part
    s = bfilt() | _serial_stage()
    assert isinstance(s, Pipeline)
    assert s.parallelism == 'serial'
    with pytest.raises(ValueError):
        pmap(s)


def test_ensemble_stage_needs_key_and_respects_gathers():
    s = normalize_gathers()
    assert s.parallelism == 'ensemble'
    with pytest.raises(ValueError, match="key"):
        pmap(s)

    n_gathers = len(list(group_by(synlv(), 'ensemble_number')))
    assert n_gathers > 1

    expected = _arr(synlv() | s)
    # a chunk size that doesn't divide the gathers: the chunks must still be cut between gathers
    for chunk in (1, 7, 1000):
        out = synlv() | pmap(s, workers=3, chunk=chunk, key='ensemble_number')
        npt.assert_equal(_arr(out), expected)
        assert out.n_traces is None  # an ensemble stage might change the count


def test_group_by():
    groups = list(group_by(synlv(), 'ensemble_number'))
    assert sum(len(g) for g in groups) == len(list(synlv()))
    for g in groups:
        assert len({t.header['ensemble_number'] for t in g}) == 1
    groups2 = list(group_by(spike(), lambda t: t.header['trace_id'] // 10))
    assert [len(g) for g in groups2] == [9, 10, 10, 3]


def test_threaded_matches_source():
    npt.assert_equal(_arr(spike() | prefetch()), _arr(spike()))
    npt.assert_equal(_arr(spike() | prefetch(buffer=1, batch=1)), _arr(spike()))
    assert (spike() | prefetch()).n_traces == 32


def test_prefetch_stage_in_pipeline():
    expected = _arr(spike() | bfilt() | bfilt(zerophase=False))
    out = spike() | prefetch(buffer=2, batch=4) | bfilt() | prefetch() | bfilt(zerophase=False)
    npt.assert_equal(_arr(out), expected)


def test_threaded_backpressure():
    produced = []

    def source():
        for tr in spike(ntr=1000):
            produced.append(1)
            yield tr

    out = source() | prefetch(buffer=2, batch=5)
    it = iter(out)
    next(it)
    time.sleep(0.3)  # the producer would run to the end by now if nothing held it back
    # buffer batches in the queue + one being filled + one the consumer holds
    assert len(produced) <= 5 * (2 + 2)
    assert len(produced) < 1000


def test_threaded_error_propagates():
    def source():
        yield from itertools_islice(spike(), 7)
        raise ValueError("producer failed")

    with pytest.raises(ValueError, match="producer failed"):
        list(source() | prefetch(batch=2))


def test_threaded_early_exit_stops_thread():
    n_before = threading.active_count()
    out = spike(ntr=1000) | prefetch(buffer=1, batch=2)
    for i, _ in enumerate(out):
        if i == 2:
            break
    del out
    assert _wait_for_threads(n_before) <= n_before


def test_pmap_composes_with_pipes():
    s = bfilt(f_pass_low=10.0, f_stop_low=5.0)
    expected = _arr(spike() | s | bfilt(zerophase=False))
    out = spike() | pmap(s, workers=2, chunk=4) | bfilt(zerophase=False)
    npt.assert_equal(_arr(out), expected)
