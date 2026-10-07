import numpy as np
import numpy.testing as npt
import pytest

from seispy import operations as op
from seispy.container import Trace, from_iterable
from seispy.filters import median, medmix
from seispy.parallel import pmap
from seispy.stacking import divstack, pws, stack, stackup
from seispy.windowing import mixgathers, sort

DT = 0.004


def make(data, dt=DT, **per_trace):
    out = []
    for i, row in enumerate(np.atleast_2d(data)):
        t = Trace(np.asarray(row, dtype=np.float32), d_sample=dt)
        header = {k: (v[i] if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in per_trace.items()}
        out.append(t.replace(**header) if header else t)
    return out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(41)
DATA = RNG.normal(size=(8, 40)).astype(np.float32)
GATHERS = [1, 1, 1, 2, 2, 3, 3, 3]


# ----------------------------------------------------------------------------------------------------------- stack
def test_stack_is_the_mean_of_each_gather():
    out = run(stack(), make(DATA, ensemble_number=GATHERS))
    assert len(out) == 3
    npt.assert_allclose(np.asarray(out[0]), DATA[:3].mean(axis=0), rtol=1e-5, atol=1e-6)
    npt.assert_allclose(np.asarray(out[1]), DATA[3:5].mean(axis=0), rtol=1e-5, atol=1e-6)
    assert [t.header['fold'] for t in out] == [3, 2, 3]
    assert [t.header['ensemble_number'] for t in out] == [1, 2, 3]


def test_stack_normpow_and_zeros():
    data = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0], [5.0, 4.0, 0.0]], dtype=np.float32)
    (mean,) = run(stack(), make(data, ensemble_number=1))
    npt.assert_allclose(np.asarray(mean), [3.0, 3.0, 0.0])  # (the mean of the values that are not zero)
    (total,) = run(stack(normpow=0.0), make(data, ensemble_number=1))
    npt.assert_allclose(np.asarray(total), [9.0, 6.0, 0.0])
    (root,) = run(stack(normpow=0.5), make(data, ensemble_number=1))
    npt.assert_allclose(np.asarray(root), [9.0 / np.sqrt(3), 6.0 / np.sqrt(2), 0.0], rtol=1e-6)


def test_stack_moves_the_traces_to_the_midpoint_and_keeps_offset_for_an_offset_stack():
    trs = make(DATA[:3], ensemble_number=1)
    trs = [
        t.replace(tx_loc=[100.0 * i, 0.0, 0.0], rx_loc=[100.0 * i + 50.0 * (i + 1), 0.0, 0.0])
        for i, t in enumerate(trs)
    ]
    (out,) = run(stack(), trs)
    assert out.header['offset'] == 0.0
    assert out.header['tx_loc'][0] == out.header['rx_loc'][0] == pytest.approx(25.0)  # (the midpoint of the first trace)
    same = run(stack(key='offset'), [t.replace(ensemble_number=1) for t in trs[:1]])
    assert same[0].header['offset'] == 50.0


def test_stack_of_stacks_adds_the_folds_and_repeats():
    out = run(stack(nrepeat=3), make(DATA[:2], ensemble_number=1, fold=[4, 6]))
    assert len(out) == 3 and all(t.header['fold'] == 10 for t in out)


def test_stack_in_parallel_matches_serial():
    serial = arr(run(stack(), make(DATA, ensemble_number=GATHERS)))
    stream = from_iterable(make(DATA, ensemble_number=GATHERS))
    parallel = arr(list(stream | pmap(stack(), workers=2, chunk=2, key='ensemble_number')))
    npt.assert_allclose(parallel, serial)
    assert stack().parallelism == 'ensemble'


def test_stack_needs_the_same_length_and_works_for_complex():
    with pytest.raises(ValueError):
        run(stack(), make(DATA[:1], ensemble_number=1) + make(DATA[1:2, :20], ensemble_number=1))
    z = (DATA + 1j * DATA[::-1]).astype(np.complex64)
    trs = [Trace(row, d_sample=DT).replace(ensemble_number=1) for row in z[:3]]
    (out,) = run(stack(), trs)
    npt.assert_allclose(np.asarray(out), z[:3].mean(axis=0), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        stack(nrepeat=0)


# ------------------------------------------------------------------------------------------------------- divstack
def test_divstack_down_weights_the_noisy_traces():
    clean = np.sin(np.linspace(0, 20, 64)).astype(np.float32)
    noisy = clean + np.random.default_rng(0).normal(scale=5.0, size=64).astype(np.float32)
    (out,) = run(divstack(key='ensemble_number'), make(np.vstack([clean, noisy]), ensemble_number=1))
    plain = (clean + noisy) / 2
    assert np.abs(np.asarray(out) - clean).mean() < np.abs(plain - clean).mean()


def test_divstack_one_window_is_the_power_weighted_mean():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32)
    (out,) = run(divstack(key='ensemble_number'), make(np.vstack([a, b]), ensemble_number=1))
    pa, pb = (a ** 2).mean(), (b ** 2).mean()
    expected = (a / pa + b / pb) / (1 / pa + 1 / pb)
    npt.assert_allclose(np.asarray(out), expected, rtol=1e-5)
    assert out.header['fold'] == 2


def test_divstack_windows_and_peak():
    x = np.concatenate([np.ones(32), 10 * np.ones(32)]).astype(np.float32)
    y = np.concatenate([5 * np.ones(32), np.ones(32)]).astype(np.float32)
    both = make(np.vstack([x, y]), ensemble_number=1)
    (windowed,) = run(divstack(key='ensemble_number', winlen=31 * DT), both)
    # in each half the stronger trace has the weaker weight: the result is pulled to the weaker trace's value
    assert abs(np.asarray(windowed)[10] - 1.0) < abs(np.asarray(windowed)[10] - 5.0)
    assert abs(np.asarray(windowed)[50] - 1.0) < abs(np.asarray(windowed)[50] - 10.0)
    (peak,) = run(divstack(key='ensemble_number', winlen=31 * DT, peak=True), both)
    assert np.isfinite(np.asarray(peak)).all()
    with pytest.raises(ValueError):
        divstack(winlen=0.0)


def test_divstack_zero_traces_stay_zero():
    (out,) = run(divstack(key='ensemble_number'), make(np.zeros((2, 8)), ensemble_number=1))
    assert not np.asarray(out).any()


# --------------------------------------------------------------------------------------------------------------- pws
def test_pws_keeps_the_coherent_signal_and_drops_noise():
    # a burst of signal in the middle of random noise
    t = np.arange(400) * DT
    signal = np.where((t > 0.6) & (t < 1.0), np.sin(2 * np.pi * 12 * t), 0.0)
    noise = np.random.default_rng(3).normal(scale=1.0, size=(20, 400))
    data = (signal + noise).astype(np.float32)
    (linear,) = run(stack(), make(data, ensemble_number=1))
    (phase,) = run(pws(), make(data, ensemble_number=1))
    quiet = slice(0, 100)  # (noise only)
    assert np.asarray(phase)[quiet].std() < 0.5 * np.asarray(linear)[quiet].std()
    # and the signal is still there
    inside = (t > 0.65) & (t < 0.95)
    assert np.abs(np.asarray(phase)[inside]).max() > 0.3
    assert phase.header['fold'] == 20 and phase.header['offset'] == 0.0


def test_phase_stack_is_between_0_and_1_and_is_one_for_equal_traces():
    t = np.arange(100) * DT
    same = np.vstack([np.sin(2 * np.pi * 10 * t)] * 5)
    (weights,) = run(pws(ps=True), make(same, ensemble_number=1))
    inside = np.asarray(weights)[10:-10]
    assert (inside <= 1.0 + 1e-5).all() and inside.min() > 0.95
    random = np.random.default_rng(1).normal(size=(30, 100)).astype(np.float32)
    (incoherent,) = run(pws(ps=True), make(random, ensemble_number=1))
    assert np.asarray(incoherent).mean() < 0.4
    # the phase stack is the modulus of a mean of unit phasors: never more than 1
    assert np.asarray(incoherent).max() <= 1.0 + 1e-5


def test_pws_smoothing_and_parameters():
    random = np.random.default_rng(2).normal(size=(10, 64)).astype(np.float32)
    (rough,) = run(pws(ps=True), make(random, ensemble_number=1))
    (smooth,) = run(pws(ps=True, sl=0.04), make(random, ensemble_number=1))
    assert np.abs(np.diff(np.asarray(smooth))).mean() < np.abs(np.diff(np.asarray(rough))).mean()
    (power,) = run(pws(ps=True, pwr=2.0), make(random, ensemble_number=1))
    npt.assert_allclose(np.asarray(power), np.asarray(rough) ** 2, rtol=1e-5)
    with pytest.raises(ValueError):
        run(pws(sl=10.0), make(random, ensemble_number=1))
    with pytest.raises(TypeError):
        run(pws(), [Trace((random[0] + 1j).astype(np.complex64), d_sample=DT)])


# --------------------------------------------------------------------------------------------------------- stackup
def test_stackup_any_order_and_sorted_output():
    order = [3, 1, 2, 1, 3, 2, 1, 3]
    out = run(stackup(), make(DATA, ensemble_number=order))
    assert [t.header['ensemble_number'] for t in out] == [1, 2, 3]
    ones = DATA[[i for i, g in enumerate(order) if g == 1]]
    npt.assert_allclose(np.asarray(out[0]), ones.mean(axis=0), rtol=1e-5, atol=1e-6)
    assert [t.header['fold'] for t in out] == [3, 2, 3]


def test_stackup_two_keys_and_the_header_of_the_nearest_trace():
    trs = make(DATA[:4], ensemble_number=[1, 1, 1, 1], trace_id=[1, 1, 2, 2])
    trs = [t.replace(rx_loc=[o, 0.0, 0.0]) for t, o in zip(trs, [300.0, 100.0, 200.0, 50.0])]
    out = run(stackup(keyloc=['ensemble_number', 'trace_id']), trs)
    assert len(out) == 2
    assert out[0].header['offset'] == 100.0 and out[1].header['offset'] == 50.0  # (the smallest offsets)


def test_stackup_range_limits_and_keep():
    trs = make(DATA[:6], ensemble_number=[1, 1, 1, 2, 2, 2])
    trs = [t.replace(rx_loc=[o, 0.0, 0.0]) for t, o in zip(trs, [100.0, 200.0, 300.0, 400.0, 500.0, 600.0])]
    out = run(stackup(keyabs='offset', minabs=150.0, maxabs=350.0), trs)
    assert [t.header['fold'] for t in out] == [2, 0]  # (the second location has nothing in range)
    assert not np.asarray(out[1]).any()
    npt.assert_allclose(np.asarray(out[0]), DATA[1:3].mean(axis=0), rtol=1e-5, atol=1e-6)
    dropped = run(stackup(keyabs='offset', minabs=150.0, maxabs=350.0, keep=False), trs)
    assert [t.header['ensemble_number'] for t in dropped] == [1]
    # a range that is the wrong way round wraps: outside of [350, 150)
    wrapped = run(stackup(keyabs='offset', minabs=350.0, maxabs=150.0), trs)
    assert [t.header['fold'] for t in wrapped] == [1, 3]


def test_stackup_signed_range_and_errors():
    trs = make(DATA[:4], ensemble_number=1)
    trs = [t.replace(rx_loc=[o, 0.0, 0.0]) for t, o in zip(trs, [-300.0, -100.0, 100.0, 300.0])]
    (out,) = run(stackup(keysign='offset', minsign=-150.0, maxsign=150.0), trs)
    assert out.header['fold'] == 2
    with pytest.raises(ValueError):
        stackup(minabs=1.0)
    with pytest.raises(ValueError):
        stackup(keyloc=[])


# ----------------------------------------------------------------------------------------------------------- sort
def test_sort_by_several_keys():
    gathers = [2, 1, 2, 1, 2, 1]
    offsets = [30.0, 20.0, 10.0, 40.0, 20.0, 10.0]
    trs = [t.replace(rx_loc=[o, 0.0, 0.0]) for t, o in zip(make(DATA[:6], ensemble_number=gathers), offsets)]
    out = run(sort('ensemble_number', 'offset'), trs)
    assert [(t.header['ensemble_number'], t.header['offset']) for t in out] == [
        (1, 10.0), (1, 20.0), (1, 40.0), (2, 10.0), (2, 20.0), (2, 30.0),
    ]
    descending = run(sort('ensemble_number', '-offset'), trs)
    assert [t.header['offset'] for t in descending] == [40.0, 20.0, 10.0, 30.0, 20.0, 10.0]
    # ties stay in the order that they came in
    ties = run(sort('ensemble_number'), make(DATA[:4], ensemble_number=[1, 1, 1, 1], trace_id=[4, 3, 2, 1]))
    assert [t.header['trace_id'] for t in ties] == [4, 3, 2, 1]
    # a function of a trace, and the default
    by_function = run(sort(lambda t: -t.header['trace_id']), make(DATA[:3], trace_id=[1, 3, 2]))
    assert [t.header['trace_id'] for t in by_function] == [3, 2, 1]
    assert [t.header['ensemble_number'] for t in run(sort(), make(DATA[:3], ensemble_number=[3, 1, 2]))] == [1, 2, 3]


# ----------------------------------------------------------------------------------------------------------- mix
def test_mix_is_a_trailing_weighted_average():
    out = arr(run(op.mix([1.0, 2.0, 1.0]), make(DATA[:6])))
    w = np.array([1.0, 2.0, 1.0]) / 3
    npt.assert_array_equal(out[:2], DATA[:2])  # (the first traces are as they were)
    for i in range(2, 6):
        expected = w[0] * DATA[i] + w[1] * DATA[i - 1] + w[2] * DATA[i - 2]
        npt.assert_allclose(out[i], expected, rtol=1e-5, atol=1e-6)


def test_mix_default_weights_and_gathers():
    out = arr(run(op.mix(), make(DATA)))
    w = np.array([0.6, 1, 1, 1, 0.6]) / 5
    npt.assert_allclose(out[6], sum(w[k] * DATA[6 - k] for k in range(5)), rtol=1e-5, atol=1e-6)
    # with a key the mix starts again in each gather
    gathered = arr(run(op.mix([1.0, 1.0], key='ensemble_number'), make(DATA, ensemble_number=GATHERS)))
    npt.assert_array_equal(gathered[0], DATA[0])  # (the first of a gather)
    npt.assert_array_equal(gathered[3], DATA[3])
    npt.assert_allclose(gathered[4], 0.5 * (DATA[4] + DATA[3]), rtol=1e-5, atol=1e-6)
    assert op.mix().parallelism == 'serial' and op.mix(key='ensemble_number').parallelism == 'ensemble'
    with pytest.raises(ValueError):
        op.mix([])


# --------------------------------------------------------------------------------------------- binary operations
def test_panel_operations():
    a, b = make(DATA[:4]), make(DATA[4:] + 5.0)
    npt.assert_allclose(arr(run(op.sum2(b), a)), DATA[:4] + DATA[4:] + 5.0, rtol=1e-5)
    npt.assert_allclose(arr(run(op.diff2(b), a)), DATA[:4] - (DATA[4:] + 5.0), rtol=1e-5)
    npt.assert_allclose(arr(run(op.prod2(b), a)), DATA[:4] * (DATA[4:] + 5.0), rtol=1e-5)
    npt.assert_allclose(arr(run(op.quo2(b), a)), DATA[:4] / (DATA[4:] + 5.0), rtol=1e-5)
    # the weights, and a trace of the second panel that is 0
    npt.assert_allclose(arr(run(op.sum2(b, w1=2.0, w2=-1.0), a)), 2 * DATA[:4] - (DATA[4:] + 5.0), rtol=1e-5)
    zero = make(np.vstack([np.zeros(40), np.ones(40)]))
    quotient = arr(run(op.quo2(zero), make(DATA[:2])))
    assert not quotient[0].any()
    # the header is that of the first, and the shorter one ends it
    out = run(op.sum2(make(DATA[4:6])), make(DATA[:4], trace_id=[7, 8, 9, 10]))
    assert [t.header['trace_id'] for t in out] == [7, 8]


def test_panel_with_a_trace_operations():
    single = make(DATA[7] + 1.0)
    panel = make(DATA[:3])
    npt.assert_allclose(arr(run(op.ptsum(single), panel)), DATA[:3] + DATA[7] + 1.0, rtol=1e-5)
    npt.assert_allclose(arr(run(op.ptdiff(single), panel)), DATA[:3] - (DATA[7] + 1.0), rtol=1e-5)
    npt.assert_allclose(arr(run(op.ptprod(single[0]), panel)), DATA[:3] * (DATA[7] + 1.0), rtol=1e-5)
    npt.assert_allclose(arr(run(op.ptquo(single), panel)), DATA[:3] / (DATA[7] + 1.0), rtol=1e-5)
    with pytest.raises(ValueError):
        run(op.ptsum([]), panel)


def test_zipper_and_polar_zipper_make_complex_traces():
    out = run(op.zipper(make(DATA[4:6])), make(DATA[:2]))
    assert all(t.dtype == np.complex64 for t in out)
    npt.assert_allclose(np.asarray(out[0]), DATA[0] + 1j * DATA[4], rtol=1e-6)
    polar = run(op.zippol(make(DATA[4:5])), make(np.abs(DATA[:1])))
    npt.assert_allclose(np.asarray(polar[0]), np.abs(DATA[0]) * np.exp(1j * DATA[4]), rtol=1e-5)
    with pytest.raises(TypeError):
        run(op.zipper(make(DATA[4:5])), [Trace((DATA[0] + 1j).astype(np.complex64), d_sample=DT)])
    with pytest.raises(ValueError):
        run(op.sum2(make(DATA[4:5, :10])), make(DATA[:1]))


def test_binary_operations_on_complex_traces():
    z = (DATA[:3] + 1j * DATA[3:6]).astype(np.complex64)
    w = (DATA[1:4] - 2j * DATA[4:7]).astype(np.complex64)
    first = [Trace(row, d_sample=DT) for row in z]
    second = [Trace(row, d_sample=DT) for row in w]
    npt.assert_allclose(arr(run(op.prod2(second), first)), z * w, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(arr(run(op.quo2(second), first)), z / w, rtol=1e-4, atol=1e-5)


# ------------------------------------------------------------------------------------------------------ mixgathers
def test_mixgathers_fills_in_missing_offsets():
    def gather(offsets, data):
        return [t.replace(rx_loc=[o, 0.0, 0.0]) for t, o in zip(make(data), offsets)]

    first = gather([100.0, 300.0, 500.0], DATA[:3])
    second = gather([100.0, 200.0, 300.0, 400.0, 500.0], DATA[3:8] + 100.0)
    out = run(mixgathers(second), first)
    assert [t.header['offset'] for t in out] == [100.0, 200.0, 300.0, 400.0, 500.0]
    # the traces of the first are kept, the new ones are from the second
    npt.assert_array_equal(np.asarray(out[0]), DATA[0])
    npt.assert_array_equal(np.asarray(out[1]), DATA[4] + 100.0)
    npt.assert_array_equal(np.asarray(out[2]), DATA[1])
    # a tenth of a percent is the same offset
    near = run(mixgathers(gather([100.05], DATA[3:4])), gather([100.0], DATA[:1]))
    assert len(near) == 1
    # the added traces can be boosted
    boosted = run(mixgathers(gather([1000.0], DATA[3:4]), scaling=True), gather([100.0], DATA[:1]))
    npt.assert_allclose(np.asarray(boosted[1]), DATA[3] * 1.03, rtol=1e-5)


# ------------------------------------------------------------------------------------------------------- median
def moveout_panel(n_traces=11, n=100, slope=2, event=40):
    """A flat event and an event with moveout (slope samples per trace)"""
    data = np.zeros((n_traces, n), dtype=np.float32)
    data[:, 60] = 1.0  # flat
    for i in range(n_traces):
        data[i, event + slope * i] += 1.0  # dipping
    return data


def test_median_removes_the_events_along_the_curve():
    data = moveout_panel()
    # the curve of the dipping event: 40 samples at trace 0, 2 samples later for each trace
    xs, ts = [0.0, 10.0], [40 * DT, 60 * DT]
    out = arr(run(median(xs, ts, nmed=5, sign=-1, key='trace_id'), make(data, trace_id=list(range(11)))))
    # the dipping event is flat after the shift, so the median keeps it and it is subtracted
    for i in range(2, 9):
        assert abs(out[i, 40 + 2 * i]) < 0.1
        assert out[i, 60] == pytest.approx(1.0, abs=0.1)  # the flat event stays: the median does not keep it


def test_median_without_subtracting_keeps_the_events_along_the_curve():
    data = moveout_panel()
    xs, ts = [0.0, 10.0], [40 * DT, 60 * DT]
    out = arr(run(median(xs, ts, nmed=5, subtract=False, key='trace_id'), make(data, trace_id=list(range(11)))))
    for i in range(2, 9):
        assert out[i, 40 + 2 * i] == pytest.approx(1.0, abs=0.05)  # (kept, in its place)
        assert abs(out[i, 60]) < 0.05


def test_medmix_attenuates_the_events_along_the_curve():
    data = moveout_panel()
    xs, ts = [0.0, 10.0], [40 * DT, 60 * DT]
    out = arr(run(medmix(xs, ts, mix=[1.0, 1.0, 1.0], key='trace_id'), make(data, trace_id=list(range(11)))))
    dipping = np.array([out[i, 40 + 2 * i] for i in range(2, 9)])
    assert np.abs(dipping).max() < 0.1  # (a mix of 3 of the same event, over 3, is the event: it is subtracted)
    assert (out[2:9, 60] > 0.5).all()


def test_median_checks_and_edges():
    with pytest.raises(ValueError):
        median([1.0, 0.0], [0.1, 0.2])
    with pytest.raises(ValueError):
        median([0.0, 1.0], [0.1])
    with pytest.raises(ValueError):
        median([0.0], [0.1], sign=2)
    with pytest.raises(ValueError):
        medmix([0.0], [0.1], mix=[1.0, 1.0])  # (an even number of weights)
    with pytest.warns(UserWarning):
        out = run(median([0.0, 10.0], [0.0, 0.0], nmed=4), make(DATA[:5], trace_id=list(range(5))))
    assert len(out) == 5
    assert len(run(median([0.0], [0.0]), make(DATA[:1], trace_id=[1]))) == 1  # (a panel of one trace)
    assert run(median([0.0], [0.0]), []) == []
