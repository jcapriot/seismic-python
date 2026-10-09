"""
SUSPLIT, SUCLEAVE, SUPUTGTHR, SUGETGTHR and SUSORTY: traces to files by a header word, files back to traces, and the data set for
looking at sorting.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.io.tracefile import read_trace_file
from seispy.windowing import cleave, getgthr, putgthr, sorty, sort, split

NT = 20


def make_traces(cdps, offsets=None):
    offsets = offsets if offsets is not None else [100.0 * (i + 1) for i in range(len(cdps))]
    return [Trace(np.full(NT, i, dtype=np.float32), d_sample=0.004).replace(
        ensemble_number=int(c), offset=float(o), trace_id=i + 1) for i, (c, o) in enumerate(zip(cdps, offsets))]


def run(stage, traces):
    return list(from_iterable(traces) | stage)


def ids(traces):
    return [t.header['trace_id'] for t in traces]


# ---------------------------------------------------------------------------------------------------------------- split
def test_split_writes_a_file_for_each_value_and_passes_the_traces_on(tmp_path):
    traces = make_traces([1, 1, 2, 2, 2, 5])
    out = run(split('ensemble_number', directory=tmp_path), traces)
    assert ids(out) == ids(traces)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ['split_ensemble_number0000001.spy', 'split_ensemble_number0000002.spy',
                     'split_ensemble_number0000005.spy']
    assert ids(read_trace_file(tmp_path / names[0])) == [1, 2]
    assert ids(read_trace_file(tmp_path / names[1])) == [3, 4, 5]
    assert ids(read_trace_file(tmp_path / names[2])) == [6]


def test_split_adds_traces_of_a_value_that_comes_again(tmp_path):
    traces = make_traces([1, 2, 1, 2, 1])
    run(split(directory=tmp_path), traces)
    assert ids(read_trace_file(tmp_path / 'split_ensemble_number0000001.spy')) == [1, 3, 5]
    assert ids(read_trace_file(tmp_path / 'split_ensemble_number0000002.spy')) == [2, 4]
    run(split(directory=tmp_path, close=False), make_traces([1, 2, 1]))  # (not closing in between changes nothing)
    assert ids(read_trace_file(tmp_path / 'split_ensemble_number0000001.spy')) == [1, 3]  # (and the old files are replaced)


def test_split_can_add_to_the_files_that_are_there(tmp_path):
    run(split(directory=tmp_path), make_traces([1, 1]))
    run(split(directory=tmp_path, append=True), make_traces([1]))
    assert read_trace_file(tmp_path / 'split_ensemble_number0000001.spy').n_traces == 3


def test_split_names_and_keys(tmp_path):
    traces = make_traces([3, 3], offsets=[250.0, 250.0])
    run(split('offset', directory=tmp_path, stem='a_', middle='off', suffix='.x', numlength=4), traces)
    assert [p.name for p in tmp_path.iterdir()] == ['a_off0250.x']
    run(split(lambda t: 7, directory=tmp_path / 'sub'), traces)  # (a function, and a directory that is made)
    assert [p.name for p in (tmp_path / 'sub').iterdir()] == ['split_key0000007.spy']


def test_split_writes_what_the_traces_are(tmp_path):
    traces = make_traces([4, 4, 9])
    run(split(directory=tmp_path), traces)
    back = list(read_trace_file(tmp_path / 'split_ensemble_number0000004.spy'))
    for a, b in zip(back, traces[:2]):
        npt.assert_array_equal(np.asarray(a), np.asarray(b))
        assert a.header == b.header


def test_files_are_closed_when_the_traces_stop_being_taken(tmp_path):
    stream = iter(from_iterable(make_traces([1, 1, 1, 2])) | split(directory=tmp_path))
    next(stream)
    next(stream)
    stream.close()
    assert read_trace_file(tmp_path / 'split_ensemble_number0000001.spy').n_traces == 2


# -------------------------------------------------------------------------------------------------- putgthr and getgthr
def test_putgthr_and_getgthr_round_trip(tmp_path):
    traces = make_traces([3, 3, 1, 1, 1, 2])
    out = run(putgthr(tmp_path / 'gathers'), traces)
    assert ids(out) == ids(traces)
    names = sorted(p.name for p in (tmp_path / 'gathers').iterdir())
    assert names == ['ensemble_number=0000001.spy', 'ensemble_number=0000002.spy', 'ensemble_number=0000003.spy']
    back = list(getgthr(tmp_path / 'gathers'))
    assert ids(back) == [3, 4, 5, 6, 1, 2]  # (the files in the order of their names)
    for a in back:
        npt.assert_array_equal(np.asarray(a), np.asarray(traces[a.header['trace_id'] - 1]))


def test_putgthr_keeps_both_gathers_of_a_key_that_comes_again(tmp_path):
    run(putgthr(tmp_path, numlength=3), make_traces([1, 2, 1]))
    assert ids(read_trace_file(tmp_path / 'ensemble_number=001.spy')) == [1, 3]


def test_getgthr_pattern_and_errors(tmp_path):
    run(putgthr(tmp_path), make_traces([1, 2]))
    (tmp_path / 'notes.txt').write_text('not a file of traces')
    assert ids(list(getgthr(tmp_path, pattern='*.spy'))) == [1, 2]
    with pytest.raises(ValueError, match="no files"):
        getgthr(tmp_path, pattern='*.none')
    with pytest.raises(ValueError, match="does not exist"):
        getgthr(tmp_path / 'nothing')


def test_a_sorted_data_set_can_be_taken_apart_and_put_together_again(tmp_path):
    shuffled = make_traces([2, 1, 3, 1, 2, 3, 1])
    run(sort('ensemble_number') | putgthr(tmp_path), shuffled)
    back = list(getgthr(tmp_path))
    assert ids(back) == ids(sorted(shuffled, key=lambda t: t.header['ensemble_number']))


# --------------------------------------------------------------------------------------------------------------- cleave
def file_names(directory):
    return sorted(p.name for p in directory.iterdir())


def test_cleave_puts_the_traces_in_the_files_of_their_ranges(tmp_path):
    # size 100 and low 50: the ranges are 0 to 100, 100 to 200, ... and their files are named for their middles
    offsets = [10.0, 60.0, 140.0, 190.0, 260.0, -60.0]  # (the key is the absolute value: -60 is 60)
    traces = make_traces([1] * 6, offsets)
    out = run(cleave('offset', size=100.0, high=300.0, directory=tmp_path), traces)
    assert ids(out) == ids(traces)
    assert file_names(tmp_path) == ['cleave_offset_150.spy', 'cleave_offset_250.spy', 'cleave_offset_50.spy']
    # (the traces of a file are in the order they came in)
    assert ids(read_trace_file(tmp_path / 'cleave_offset_50.spy')) == [1, 2, 6]
    assert ids(read_trace_file(tmp_path / 'cleave_offset_150.spy')) == [3, 4]
    assert ids(read_trace_file(tmp_path / 'cleave_offset_250.spy')) == [5]


def test_cleave_ranges_include_their_lower_boundary(tmp_path):
    traces = make_traces([1] * 4, [0.0, 99.999, 100.0, 199.0])
    run(cleave(size=100.0, low=50.0, high=250.0, directory=tmp_path), traces)
    assert ids(read_trace_file(tmp_path / 'cleave_offset_50.spy')) == [1, 2]
    assert ids(read_trace_file(tmp_path / 'cleave_offset_150.spy')) == [3, 4]  # (100 is in the range that it starts)


def test_cleave_without_the_absolute_value_has_ranges_on_both_sides(tmp_path):
    traces = make_traces([1] * 4, [-160.0, -40.0, 40.0, 160.0])
    run(cleave(abs=False, size=100.0, low=50.0, high=150.0, directory=tmp_path), traces)
    assert file_names(tmp_path) == ['cleave_offset_-150.spy', 'cleave_offset_-50.spy', 'cleave_offset_150.spy',
                                    'cleave_offset_50.spy']
    assert ids(read_trace_file(tmp_path / 'cleave_offset_-50.spy')) == [2]


def test_cleave_loses_no_traces_and_warns_about_the_extra_files(tmp_path):
    traces = make_traces([1] * 4, [10.0, 5000.0, 150.0, 9000.0])
    with pytest.warns(UserWarning, match="key > high range"):
        run(cleave(size=100.0, high=200.0, directory=tmp_path), traces)
    names = file_names(tmp_path)
    assert sum(read_trace_file(tmp_path / n).n_traces for n in names) == 4
    assert ids(read_trace_file(tmp_path / 'cleave_offset_350.spy')) == [2, 4]  # (the extra file for the high values)
    with pytest.warns(UserWarning, match="key < low range"):
        run(cleave(abs=False, size=100.0, low=50.0, high=150.0, directory=tmp_path / 'b'),
            make_traces([1, 1], [-1000.0, 40.0]))
    assert ids(read_trace_file(tmp_path / 'b' / 'cleave_offset_-250.spy')) == [1]


def test_cleave_names_and_checks(tmp_path):
    with pytest.warns(UserWarning):
        run(cleave('offset', size=100.0, high=200.0, outbase='part_', directory=tmp_path), make_traces([1], [5000.0]))
    assert all(n.startswith('part_') for n in file_names(tmp_path))
    for bad in (dict(size=0.0), dict(size=100.0, low=60.0), dict(low=-1.0), dict(size=100.0, low=50.0, high=10.0)):
        with pytest.raises(ValueError):
            cleave(**bad)


# --------------------------------------------------------------------------------------------------------------- sorty
def test_sorty_shows_the_geometry_in_the_traces():
    traces = list(sorty(nt=40, nshot=3, noff=4, dshot=10.0, doff=20.0))
    assert len(traces) == 12
    t = traces[5]  # shot 1 (sx = 10), offset index 1 (offset 40, gx = 50, midpoint 30)
    x = np.asarray(t)
    assert x.shape == (40,)
    assert set(x[:10]) == {10.0} and set(x[10:20]) == {50.0} and set(x[20:30]) == {40.0} and set(x[30:]) == {30.0}
    assert t.header['offset'] == pytest.approx(40.0) and t.header['ensemble_number'] == 30
    assert ids(traces) == list(range(1, 13))


def test_sorty_sorted_by_cdp_and_offset():
    out = list(sorty(nshot=4, noff=5) | sort('ensemble_number', 'offset'))
    cdps = [t.header['ensemble_number'] for t in out]
    assert cdps == sorted(cdps)
    for number in set(cdps):
        offsets = [t.header['offset'] for t in out if t.header['ensemble_number'] == number]
        assert offsets == sorted(offsets)


def test_sorty_defaults_and_checks():
    assert len(list(sorty())) == 10 * 20
    with pytest.raises(ValueError):
        sorty(nt=2)
