"""
The programs that work on a whole data set, or report on one: SUFLIP, SUVCAT, SUGAUSSTAPER, and SUMEAN, SUMAX, SUQUANTILE,
SUHISTOGRAM and SUCMP.
"""
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from seispy import attributes
from seispy import operations as op
from seispy.container import TraceCollection
from seispy.tapering import gausstaper

DT = 0.004
RNG = np.random.default_rng(11)
DATA = RNG.normal(size=(5, 12)).astype(np.float32)


def traces(data=DATA, dt=DT):
    return TraceCollection(data, d_sample=dt)


def run(stage, data=DATA):
    return np.array([np.asarray(t) for t in traces(data) | stage])


# ------------------------------------------------------------------------------------------------------ gausstaper
def test_gausstaper_weights_by_offset():
    source = [t.replace(offset=100.0 * (i + 1)) for i, t in enumerate(traces())]
    got = np.array([np.asarray(t) for t in iter(source) | gausstaper(key='offset', x0=300.0, xw=100.0)])
    weights = np.exp(-(((np.arange(1, 6) * 100.0) - 300.0) / 100.0) ** 2)
    npt.assert_allclose(got, DATA * weights[:, None], rtol=1e-6)
    # the center of the taper is not changed
    npt.assert_allclose(got[2], DATA[2], rtol=1e-6)


def test_gausstaper_key_can_be_a_function():
    got = run(gausstaper(key=lambda t: 3.0, x0=3.0, xw=1.0))
    npt.assert_array_equal(got, DATA)


def test_gausstaper_needs_a_width():
    with pytest.raises(ValueError):
        gausstaper(xw=0.0)


# ------------------------------------------------------------------------------------------------------------ flip
def flipper(indata, flip):
    """The loops of flipper() of suflip.c, with nrow traces of ncol samples"""
    nrow, ncol = indata.shape
    shape = (ncol, nrow) if flip in (-1, 0, 1) else (nrow, ncol)
    out = np.zeros(shape, dtype=indata.dtype)
    for irow in range(nrow):
        for icol in range(ncol):
            if flip == -1:
                out[icol, nrow - 1 - irow] = indata[irow, icol]
            elif flip == 0:
                out[icol, irow] = indata[irow, icol]
            elif flip == 1:
                out[ncol - 1 - icol, irow] = indata[irow, icol]
            elif flip == 2:
                out[nrow - 1 - irow, icol] = indata[irow, icol]
            else:
                out[irow, ncol - 1 - icol] = indata[irow, icol]
    return out


@pytest.mark.parametrize('how', [-1, 0, 1, 2, 3])
def test_flip_matches_the_loops_of_suflip(how):
    out = list(traces() | op.flip(how))
    got = np.array([np.asarray(t) for t in out])
    npt.assert_array_equal(got, flipper(DATA, how))
    assert [t.header['trace_id'] for t in out] == list(range(1, len(out) + 1))
    assert all(t.n_sample == got.shape[1] for t in out)


def test_flip_twice_is_the_identity():
    npt.assert_array_equal(run(op.flip(2) | op.flip(2)), DATA)
    npt.assert_array_equal(run(op.flip(1) | op.flip(-1)), DATA)
    npt.assert_array_equal(run(op.flip(0) | op.flip(0)), DATA)


def test_flip_rejects_unknown_flips():
    with pytest.raises(ValueError):
        op.flip(4)


def test_flip_headers_stay_in_order():
    source = [t.replace(ensemble_number=i + 1) for i, t in enumerate(traces())]
    out = list(iter(source) | op.flip(2))
    assert [t.header['ensemble_number'] for t in out] == [1, 2, 3, 4, 5]
    npt.assert_array_equal(np.asarray(out[0]), DATA[-1])


# ------------------------------------------------------------------------------------------------------------ vcat
def vcat_run(taplen, taptype, top=DATA, bottom=None):
    bottom = top + 10.0 if bottom is None else bottom
    return np.array([np.asarray(t) for t in traces(top) | op.vcat(traces(bottom), taplen=taplen, taptype=taptype)])


def test_vcat_without_overlap_joins_the_traces():
    got = vcat_run(0, 0)
    npt.assert_array_equal(got, np.hstack([DATA, DATA + 10.0]))


@pytest.mark.parametrize('taptype', [0, 1, 2, 3])
def test_vcat_overlap_methods(taptype):
    a, b = DATA, DATA + 10.0
    n = 4
    got = vcat_run(n, taptype)
    assert got.shape == (5, 12 + 12 - n)
    npt.assert_array_equal(got[:, :8], a[:, :8])
    npt.assert_array_equal(got[:, 12:], b[:, n:])
    top, bottom = a[:, -n:], b[:, :n]
    if taptype == 0:
        expected = (top + bottom) / 2
    elif taptype == 1:
        expected = np.where(np.abs(top) > np.abs(bottom), top, bottom)
    elif taptype == 2:
        s1 = np.cos(np.pi / 2 * np.arange(n) / (n - 1))
        expected = s1 * top + (1 - s1) * bottom
    else:
        expected = top + bottom
    npt.assert_allclose(got[:, 8:12], expected, rtol=1e-6, atol=1e-6)


def test_vcat_cosine_goes_from_the_top_to_the_bottom():
    top = np.ones((1, 6), dtype=np.float32)
    bottom = np.full((1, 6), 3.0, dtype=np.float32)
    got = vcat_run(5, 2, top=top, bottom=bottom)[0]
    assert got[1] == pytest.approx(1.0)  # first sample of the overlap: all of the top
    assert got[5] == pytest.approx(3.0, abs=1e-6)  # last: all of the bottom
    assert np.all(np.diff(got[1:6]) > 0)


def test_vcat_warns_about_unequal_data_sets():
    with pytest.warns(UserWarning):
        got = run(op.vcat(traces(DATA[:3])))
    assert got.shape[0] == 3
    with pytest.warns(UserWarning):
        run(op.vcat(traces(np.vstack([DATA, DATA]))))


def test_vcat_checks_its_parameters():
    with pytest.raises(ValueError):
        op.vcat(traces(), taptype=4)
    with pytest.raises(ValueError):
        run(op.vcat(traces(), taplen=13))


# ------------------------------------------------------------------------------------------------------------ mean
def test_mean_is_the_rms_by_default():
    result = attributes.mean(traces())
    expected = np.sqrt((DATA.astype(np.float64) ** 2).mean(axis=1))
    npt.assert_allclose(result.trace, expected, rtol=1e-6)
    assert result.section == pytest.approx(expected.mean(), rel=1e-6)


def test_mean_power_one_with_and_without_abs():
    result = attributes.mean(traces(), power=1.0, abs=True)
    npt.assert_allclose(result.trace, np.abs(DATA).mean(axis=1), rtol=1e-6)
    signed = attributes.mean(traces(), power=1.0)
    npt.assert_allclose(signed.trace, DATA.mean(axis=1), rtol=1e-5, atol=1e-7)


def test_mean_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        attributes.mean([])


# ------------------------------------------------------------------------------------------------------------- max
def test_max_maxmin():
    result = attributes.max(traces())
    npt.assert_array_equal(result.max.value, DATA.max(axis=1))
    npt.assert_array_equal(result.max.sample, DATA.argmax(axis=1))
    npt.assert_array_equal(result.min.value, DATA.min(axis=1))
    npt.assert_array_equal(result.min.sample, DATA.argmin(axis=1))
    npt.assert_allclose(result.max.time, DATA.argmax(axis=1) * DT)
    flat = DATA.argmax()
    assert result.max.global_value == DATA.max()
    assert (result.max.global_trace, result.max.global_sample) == divmod(flat, DATA.shape[1])
    flat = DATA.argmin()
    assert result.min.global_value == DATA.min()
    assert (result.min.global_trace, result.min.global_sample) == divmod(flat, DATA.shape[1])
    assert result.abs is None and result.rms is None


def test_max_does_not_remember_the_sample_of_the_trace_before():
    data = np.array([[1.0, 5.0, 2.0], [9.0, 1.0, 2.0]], dtype=np.float32)  # the max of the 2nd trace is at sample 0
    result = attributes.max(traces(data), mode='max')
    npt.assert_array_equal(result.max.sample, [1, 0])
    assert result.min is None


def test_max_abs():
    result = attributes.max(traces(), mode='abs')
    npt.assert_array_equal(result.abs.value, np.abs(DATA).max(axis=1))
    npt.assert_array_equal(result.abs.sample, np.abs(DATA).argmax(axis=1))
    assert result.abs.global_value == np.abs(DATA).max()


def test_max_rms():
    result = attributes.max(traces(), mode='rms')
    npt.assert_allclose(result.rms, np.sqrt((DATA.astype(np.float64) ** 2).mean(axis=1)), rtol=1e-6)
    assert result.global_rms == pytest.approx(np.sqrt((DATA.astype(np.float64) ** 2).mean()), rel=1e-6)


def test_max_threshold_takes_the_first_run_above_it():
    data = np.array([[0.0, 0.2, 3.0, 4.0, 0.1, 9.0, 0.0],  # the 9 is in the second run
                    [0.0, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0]],  # nothing above the threshold
                   dtype=np.float32)
    result = attributes.max(traces(data), mode='thd', threshamp=1.0)
    npt.assert_array_equal(result.thd.value, [4.0, 0.0])
    npt.assert_array_equal(result.thd.sample, [3, 0])
    # searching from the time of sample 5 on skips the first run
    late = attributes.max(traces(data), mode='thd', threshamp=1.0, threshtime=5 * DT)
    assert late.thd.value[0] == 9.0 and late.thd.sample[0] == 5


def test_max_rejects_bad_input():
    with pytest.raises(ValueError):
        attributes.max(traces(), mode='median')
    with pytest.raises(TypeError):
        attributes.max(traces(DATA + 1j * DATA))


# --------------------------------------------------------------------------------------------------------- quantile
def test_quantile_of_a_ramp():
    ramp = np.arange(1, 201, dtype=np.float32).reshape(2, 100)
    result = attributes.quantile(traces(ramp))
    assert result.n_samples == 200
    # the qth quantile is the sample at int(q n - 0.5) of the sorted values
    for q, value in result.percentiles.items():
        assert value == int(0.01 * q * 200 - 0.5) + 1
    assert (result.min, result.max) == (1.0, 200.0)
    assert (result.min_position, result.max_position) == (0, 199)


def test_quantile_ranks():
    ramp = np.arange(1, 101, dtype=np.float32).reshape(1, 100)
    result = attributes.quantile(traces(ramp), quantiles=False)
    assert result.percentiles == {}
    assert [r for r, _ in result.ranks] == [1, 6, 46, 55, 95, 100]
    assert [v for _, v in result.ranks] == [1.0, 6.0, 46.0, 55.0, 95.0, 100.0]


def test_quantile_trace_by_trace():
    per_trace = attributes.quantile(traces(), panel=False)
    assert len(per_trace) == 5
    for row, got in zip(DATA, per_trace):
        assert got.min == row.min() and got.max == row.max()
        assert got.n_samples == 12


# ------------------------------------------------------------------------------------------------------- histogram
def test_histogram_one_dimensional():
    edges, fractions = attributes.histogram(traces(), min=-2.0, max=2.0, bins=4)
    npt.assert_allclose(edges, [-2, -1, 0, 1])
    assert fractions.sum() == pytest.approx(1.0)
    # (the samples that are outside of the range are in the end bins)
    expected = np.bincount(np.clip(np.trunc((DATA + 2.0) / 1.0).astype(int), 0, 3).ravel(), minlength=4) / DATA.size
    npt.assert_allclose(fractions, expected, rtol=1e-6)


def test_histogram_counts_extremes_in_the_end_bins():
    data = np.array([[-50.0, 0.5, 50.0, 0.5]], dtype=np.float32)
    _, fractions = attributes.histogram(traces(data), min=0.0, max=1.0, bins=2)
    npt.assert_allclose(fractions, [0.25, 0.75])


def test_histogram_trend_two_is_a_panel_of_traces_by_bin():
    data = RNG.normal(size=(40, 8)).astype(np.float32)
    out = attributes.histogram(traces(data), min=-4.0, max=4.0, bins=8, trend=2)
    assert len(out) == 8
    stacked = np.array([np.asarray(t) for t in out])
    npt.assert_allclose(stacked.sum(axis=0), 1.0, rtol=1e-5)  # at each time, the fractions add up to 1
    assert stacked.shape == (8, 8)


def test_histogram_trend_one_picks_the_median():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(2000, 6)).astype(np.float32)
    rows = attributes.histogram(traces(data), min=-5.0, max=5.0, bins=200, trend=1)
    assert rows.shape == (6, 7)
    npt.assert_allclose(rows[:, 0], np.arange(6) * DT * 1000.0)
    assert np.all(np.abs(rows[:, 3]) < 0.2)  # the median of normal numbers
    assert np.all(rows[:, 2] < rows[:, 3]) and np.all(rows[:, 3] < rows[:, 4])
    npt.assert_allclose(rows[:, 6], data.std(axis=0, ddof=1), rtol=1e-3)


def test_histogram_needs_bins():
    with pytest.raises(ValueError):
        attributes.histogram(traces(), min=0.0, max=1.0, bins=0)
    with pytest.raises(ValueError):
        attributes.histogram(traces(), min=0.0, max=1.0, bins=2, trend=3)


# ------------------------------------------------------------------------------------------------------------ cmp
def test_cmp_same_data_sets():
    assert attributes.cmp(traces(), traces()) is None
    # (within the limit)
    assert attributes.cmp(traces(), traces(DATA * np.float32(1.00001))) is None


def test_cmp_finds_a_sample_that_differs():
    other = DATA.copy()
    other[3, 7] *= 1.1
    message = attributes.cmp(traces(), traces(other))
    assert message is not None and 'trace 3' in message and 'sample 7' in message


def test_cmp_limit_is_adjustable():
    other = DATA * np.float32(1.01)
    assert attributes.cmp(traces(), traces(other)) is not None
    assert attributes.cmp(traces(), traces(other), limit=0.1) is None


def test_cmp_finds_different_headers_and_lengths():
    assert 'headers' in attributes.cmp(traces(), traces(dt=0.002))
    assert 'number of traces' in attributes.cmp(traces(), traces(DATA[:4]))
    assert 'number of traces' in attributes.cmp(traces(DATA[:4]), traces())
