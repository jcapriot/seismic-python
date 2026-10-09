"""
The temporary files of the stages that read all of their traces (``tmpdir`` and ``memory``): they give the same traces as in
memory, say loudly which directory they use, and remove it.
"""
import logging

import numpy as np
import numpy.testing as npt
import pytest

from seispy import operations as op
from seispy import spool
from seispy.container import Trace, from_iterable
from seispy.windowing import sort

NT = 50
RNG = np.random.default_rng(21)


def make_traces(n=60, nt=NT):
    traces = []
    for i in range(n):
        x = RNG.normal(size=nt).astype(np.float32)
        traces.append(Trace(x, d_sample=0.004).replace(
            ensemble_number=int(RNG.integers(1, 6)), trace_id=i + 1, fold=int(RNG.integers(0, 3)), offset=float(RNG.integers(0, 8) * 100),
        ))
    return traces


def ids(out):
    return [t.header['trace_id'] for t in out]


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    monkeypatch.setattr(spool, 'tmpdir', None)
    monkeypatch.setattr(spool, 'memory', None)
    for name in ('SEISPY_TMPDIR', 'CWP_TMPDIR', 'SEISPY_MEMORY'):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------------------------------------------------ settings
def test_the_directory_comes_from_the_first_setting_that_is_made(tmp_path, monkeypatch):
    import tempfile

    assert spool.resolve_tmpdir() == tempfile.gettempdir()
    for name, expected in (('CWP_TMPDIR', 'cwp'), ('SEISPY_TMPDIR', 'seispy')):
        monkeypatch.setenv(name, expected)
        assert spool.resolve_tmpdir() == expected
    monkeypatch.setattr(spool, 'tmpdir', 'module')
    assert spool.resolve_tmpdir() == 'module'
    assert spool.resolve_tmpdir(tmp_path) == str(tmp_path)
    assert spool.resolve_tmpdir(True) == 'module'  # (True is the default one)
    assert spool.resolve_tmpdir(False) is None


def test_the_memory_comes_from_the_first_setting_that_is_made(monkeypatch):
    assert spool.resolve_memory() == spool.DEFAULT_MEMORY
    monkeypatch.setenv('SEISPY_MEMORY', '2e6')
    assert spool.resolve_memory() == 2_000_000
    monkeypatch.setattr(spool, 'memory', 1234)
    assert spool.resolve_memory() == 1234
    assert spool.resolve_memory(0) == 0
    with pytest.raises(ValueError):
        spool.resolve_memory(-1)
    monkeypatch.setattr(spool, 'memory', None)
    monkeypatch.setenv('SEISPY_MEMORY', 'lots')
    with pytest.raises(ValueError, match="SEISPY_MEMORY"):
        spool.resolve_memory()


def test_a_directory_that_can_not_be_used_is_an_error_when_the_stage_is_made(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        sort('offset', tmpdir=tmp_path / 'nothing')
    with pytest.raises(ValueError):
        op.flip(1, tmpdir=tmp_path / 'nothing')
    sort('offset', tmpdir=tmp_path)  # (a good one)
    sort('offset', tmpdir=False)


# ------------------------------------------------------------------------------------------------------------- sort
KEYS = [
    ('ensemble_number',),
    ('ensemble_number', 'offset'),
    ('ensemble_number', '-offset'),
    ('-fold', 'offset'),
]


@pytest.mark.parametrize('keys', KEYS)
@pytest.mark.parametrize('memory', [0, 4000])  # (everything on disk, and a few runs)
def test_sort_on_disk_gives_the_same_traces_as_in_memory(tmp_path, keys, memory):
    traces = make_traces()
    expected = list(from_iterable(traces) | sort(*keys))
    got = list(from_iterable(traces) | sort(*keys, tmpdir=tmp_path, memory=memory))
    assert ids(got) == ids(expected)  # (the same order, ties and all)
    for a, b in zip(got, expected):
        npt.assert_array_equal(np.asarray(a), np.asarray(b))
        assert a.header == b.header
    assert list(tmp_path.iterdir()) == []  # (the temporary directory is gone)


def test_sort_on_disk_keeps_equal_keys_in_the_order_they_came_in(tmp_path):
    traces = make_traces(40)
    got = list(from_iterable(traces) | sort('ensemble_number', tmpdir=tmp_path, memory=3000))
    for number in range(1, 6):
        mine = [t.header['trace_id'] for t in got if t.header['ensemble_number'] == number]
        assert mine == sorted(mine)


def test_sort_with_a_function_key_on_disk(tmp_path):
    traces = make_traces(30)
    key = (lambda t: -t.header['trace_id'], False)
    got = list(from_iterable(traces) | sort(key, tmpdir=tmp_path, memory=0))
    assert ids(got) == list(range(30, 0, -1))


def test_sort_says_which_directory_when_it_uses_the_disk(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger='seispy.spool'):
        list(from_iterable(make_traces()) | sort('offset', tmpdir=tmp_path, memory=0))
    messages = [r for r in caplog.records if r.name == 'seispy.spool']
    assert len(messages) == 1 and messages[0].levelno == logging.WARNING
    text = messages[0].getMessage()
    assert 'sort' in text and str(tmp_path) in text and 'temporary directory' in text


def test_sort_that_fits_in_memory_does_not_touch_the_disk(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG, logger='seispy.spool'):
        got = list(from_iterable(make_traces()) | sort('offset', tmpdir=tmp_path))
    assert len(got) == 60
    assert not [r for r in caplog.records if r.name == 'seispy.spool']
    assert list(tmp_path.iterdir()) == []


def test_sort_without_the_disk_does_not_use_it(tmp_path):
    with pytest.raises(MemoryError, match="tmpdir=False"):
        list(from_iterable(make_traces()) | sort('offset', tmpdir=False, memory=0))


def test_sort_removes_the_directory_if_it_is_stopped_early(tmp_path):
    stream = iter(from_iterable(make_traces()) | sort('offset', tmpdir=tmp_path, memory=0))
    next(stream)
    assert list(tmp_path.iterdir())  # (it is in use)
    stream.close()
    assert list(tmp_path.iterdir()) == []


def test_sort_uses_the_directory_from_the_environment(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv('SEISPY_TMPDIR', str(tmp_path))
    with caplog.at_level(logging.WARNING, logger='seispy.spool'):
        list(from_iterable(make_traces(20)) | sort('offset', memory=0))
    assert str(tmp_path) in caplog.text


def test_sort_of_nothing():
    assert list(from_iterable([]) | sort('offset', memory=0)) == []


# ------------------------------------------------------------------------------------------------------------- flip
@pytest.mark.parametrize('how', [-1, 0, 1, 2, 3])
def test_flip_on_disk_gives_the_same_traces_as_in_memory(tmp_path, how, caplog):
    traces = make_traces(12, nt=17)
    expected = list(from_iterable(traces) | op.flip(how))
    with caplog.at_level(logging.WARNING, logger='seispy.spool'):
        got = list(from_iterable(traces) | op.flip(how, tmpdir=tmp_path, memory=0))
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        npt.assert_array_equal(np.asarray(a), np.asarray(b))
        assert a.header == b.header
    assert str(tmp_path) in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_flip_on_disk_of_complex_traces(tmp_path):
    traces = [t.replace((np.asarray(t) * (1 + 2j)).astype(np.complex64)) for t in make_traces(6, nt=9)]
    expected = list(from_iterable(traces) | op.flip(1))
    got = list(from_iterable(traces) | op.flip(1, tmpdir=tmp_path, memory=0))
    for a, b in zip(got, expected):
        npt.assert_array_equal(np.asarray(a), np.asarray(b))
        assert np.asarray(a).dtype == np.complex64


def test_flip_checks_that_the_traces_are_the_same_length(tmp_path):
    traces = make_traces(3, nt=10) + make_traces(1, nt=11)
    for kwargs in ({}, dict(tmpdir=tmp_path, memory=0)):
        with pytest.raises(ValueError, match="same number of samples"):
            list(from_iterable(traces) | op.flip(1, **kwargs))
    assert list(tmp_path.iterdir()) == []  # (and it cleans up after the failure)


def test_flip_of_nothing(tmp_path):
    assert list(from_iterable([]) | op.flip(1, tmpdir=tmp_path, memory=0)) == []
