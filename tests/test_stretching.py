import numpy as np
import numpy.testing as npt
import pytest

from seispy.amplitudes import ai2r, r2ai, weight
from seispy.container import Trace, from_iterable
from seispy.parallel import pmap
from seispy.stretching import reduce, resamp, shift

DT = 0.004


def traces(data, dt=DT, sample_start=0.0, **per_trace):
    out = []
    for i, row in enumerate(np.atleast_2d(data)):
        header = {}
        for name, values in per_trace.items():
            header[name] = values[i] if isinstance(values, (list, tuple, np.ndarray)) else values
        start = sample_start[i] if isinstance(sample_start, (list, tuple, np.ndarray)) else sample_start
        out.append(Trace(np.asarray(row, dtype=np.float32), d_sample=dt, sample_start=start).replace(**header))
    return out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(5)
DATA = RNG.normal(size=(4, 20)).astype(np.float32) + 3.0


# ----------------------------------------------------------------------------------------------------------- shift
def test_shift_into_a_bigger_window():
    # the trace is 0.1 to 0.18 s, the window 0.08 to 0.2 s (30 samples), so it goes in at sample 5
    (out,) = run(shift(tmin=0.08, tmax=0.2, fill=-1.0), traces(DATA[:1], sample_start=0.1))
    expected = np.full(30, -1.0)
    expected[5:25] = DATA[0]
    npt.assert_array_equal(np.asarray(out), expected)
    assert out.n_sample == 30
    assert out.header['sample_start'] == pytest.approx(0.08)
    assert out.d_sample == DT


def test_shift_into_a_smaller_window():
    # 0.11 s is not on a sample: sushift starts at the sample before it (sample 2, at 0.108) and calls it 0.11
    (out,) = run(shift(tmin=0.11, tmax=0.15), traces(DATA[:1], sample_start=0.1))
    npt.assert_array_equal(np.asarray(out), DATA[0, 2:12])
    assert out.header['sample_start'] == pytest.approx(0.11)
    # a window on the samples
    (out,) = run(shift(tmin=0.108, tmax=0.148), traces(DATA[:1], sample_start=0.1))
    npt.assert_array_equal(np.asarray(out), DATA[0, 2:12])
    # a window that starts inside the trace and ends past it
    (out,) = run(shift(tmin=0.12, tmax=0.2), traces(DATA[:1], sample_start=0.1))
    expected = np.zeros(20)
    expected[:15] = DATA[0, 5:]
    npt.assert_array_equal(np.asarray(out), expected)


def test_shift_a_window_that_misses_the_trace():
    for tmin, tmax in ((0.5, 0.6), (-0.1, 0.05)):
        (out,) = run(shift(tmin=tmin, tmax=tmax, fill=7.0), traces(DATA[:1], sample_start=0.1))
        npt.assert_array_equal(np.asarray(out), 7.0)


def test_shift_defaults_come_from_the_first_trace():
    # the window is the first trace's, 0.1 to 0.18 s. A trace that starts 8 ms later is moved back.
    first = traces(DATA[:1], sample_start=0.1)[0]
    second = traces(DATA[1:2], sample_start=0.108)[0]
    a, b = run(shift(), [first, second])
    npt.assert_array_equal(np.asarray(a), DATA[0])
    expected = np.zeros(20)
    expected[2:] = DATA[1, :18]
    npt.assert_array_equal(np.asarray(b), expected)
    assert b.header['sample_start'] == pytest.approx(0.1)
    # only tmin: the window is as long as the first trace
    a, b = run(shift(tmin=0.108), [first, second])
    assert a.header['sample_start'] == pytest.approx(0.108) and a.n_sample == 20
    npt.assert_array_equal(np.asarray(b), DATA[1])


def test_shift_with_dt_and_errors():
    # dt (in seconds here) is for traces that do not say what theirs is
    (out,) = run(shift(tmin=0.0, tmax=0.1, dt=0.005), traces(DATA[:1], dt=0.0))
    assert out.n_sample == 20
    with pytest.raises(ValueError, match="sample interval"):
        run(shift(tmin=0.0), traces(DATA[:1], dt=0.0))
    with pytest.raises(ValueError):
        shift(tmin=0.2, tmax=0.1)


def test_shift_parallelism():
    assert shift().parallelism == 'serial'
    assert shift(tmin=0.0, tmax=0.1).parallelism == 'serial'
    full = shift(tmin=0.08, tmax=0.2, dt=DT)
    assert full.parallelism == 'trace'
    trs = traces(DATA, sample_start=[0.1, 0.1, 0.108, 0.092])
    npt.assert_array_equal(arr(run(full, trs)), arr(list(from_iterable(list(trs)) | pmap(full, workers=2, chunk=1))))
    with pytest.raises(ValueError):
        pmap(shift())


# ---------------------------------------------------------------------------------------------------------- resamp
def sine(n, dt, freq=20.0, t0=0.0):
    return np.sin(2 * np.pi * freq * (t0 + np.arange(n) * dt)).astype(np.float32)


def test_resamp_defaults_do_nothing():
    (out,) = run(resamp(), traces(DATA[:1]))
    npt.assert_allclose(np.asarray(out), DATA[0], rtol=1e-6)
    assert out.n_sample == 20 and out.d_sample == DT


def test_resamp_up_and_down():
    x = sine(128, DT)
    (up,) = run(resamp(rf=2), traces(x))
    assert up.n_sample == 256 and up.d_sample == pytest.approx(DT / 2)
    # the input samples are on every other output sample, and between them the sine
    npt.assert_allclose(np.asarray(up)[::2], x, atol=1e-5)
    t = np.arange(256) * DT / 2
    npt.assert_allclose(np.asarray(up)[10:-10], np.sin(2 * np.pi * 20 * t)[10:-10], atol=3e-3)
    (down,) = run(resamp(rf=0.5), traces(x))
    assert down.n_sample == 64 and down.d_sample == pytest.approx(2 * DT)
    npt.assert_allclose(np.asarray(down), x[::2], atol=1e-4)


def test_resamp_explicit_axis_and_header():
    x = sine(128, DT)
    (out,) = run(resamp(nt=100, dt=0.003, tmin=0.01), traces(x, sample_start=0.0, offset=300.0, trace_id=7))
    assert out.n_sample == 100 and out.d_sample == pytest.approx(0.003)
    assert out.header['sample_start'] == pytest.approx(0.01)
    assert out.header['offset'] == 300.0 and out.header['trace_id'] == 7
    t = 0.01 + np.arange(100) * 0.003
    npt.assert_allclose(np.asarray(out)[5:-5], np.sin(2 * np.pi * 20 * t)[5:-5], atol=3e-3)
    # before and after the input trace there is nothing
    (out,) = run(resamp(tmin=-0.1, nt=60), traces(x[:30]))
    npt.assert_array_equal(np.asarray(out)[:25], 0.0)
    (out,) = run(resamp(nt=50, dt=DT, tmin=0.0), traces(x[:30]))
    npt.assert_array_equal(np.asarray(out)[31:], 0.0)


def test_resamp_input_dt_and_errors():
    x = sine(64, DT)
    (out,) = run(resamp(rf=2, dt_in=DT), traces(x, dt=0.0))
    assert out.n_sample == 128
    with pytest.raises(ValueError, match="sample interval"):
        run(resamp(), traces(x, dt=0.0))
    for bad in (dict(rf=-1.0), dict(nt=0), dict(dt=0.0)):
        with pytest.raises(ValueError):
            resamp(**bad)


def test_resamp_in_threads():
    x = np.stack([sine(256, DT, f) for f in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)])
    trs = traces(x)
    stage = resamp(rf=3)
    expected = arr(run(stage, trs))
    got = arr(list(from_iterable(list(trs)) | pmap(stage, workers=4, chunk=1)))
    npt.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------------------------------------- reduce
def test_reduce():
    offsets = [0.0, 200.0, -400.0, 800.0]
    out = arr(run(reduce(rv=2.0), traces(DATA, offset=offsets)))
    # 200 m at 2 km/s is 0.1 s, which is 25 samples: more than the 20 of these traces, as are the others.
    # So all that is left is the trace with no offset.
    npt.assert_array_equal(out[0], DATA[0])
    npt.assert_array_equal(out[1:], 0.0)
    # a trace that is long enough
    x = np.arange(200, dtype=np.float32)[None, :]
    out = arr(run(reduce(rv=2.0), traces(x, offset=[-200.0])))[0]  # 25 samples, and the sign of the offset does not matter
    expected = np.zeros(200)
    expected[:175] = np.arange(25, 200)
    npt.assert_array_equal(out, expected)


def test_reduce_default_velocity_and_dt():
    x = np.arange(100, dtype=np.float32)[None, :]
    # 8 km/s: 8000 m is 1 s = 250 samples; 400 m is 0.05 s = 12.5 samples -> 13 (rounded)
    out = arr(run(reduce(), traces(x, offset=[400.0])))[0]
    assert out[0] == 13 and out[86] == 99 and out[87] == 0
    out = arr(run(reduce(rv=8.0, dt=DT), traces(x, dt=0.0, offset=[400.0])))[0]
    assert out[0] == 13
    with pytest.raises(ValueError, match="sample interval"):
        run(reduce(), traces(x, dt=0.0))
    with pytest.raises(ValueError):
        reduce(rv=0.0)


# ---------------------------------------------------------------------------------------------------------- weight
def test_weight_by_offset():
    offsets = [0.0, 1000.0, -2000.0, 500.0]
    out = arr(run(weight(), traces(DATA, offset=offsets)))
    factors = 1.0 + 0.0005 * np.array(offsets)
    npt.assert_allclose(out, DATA * factors[:, None], rtol=1e-6)
    out = arr(run(weight(a=2.0, b=0.0004, inv=True), traces(DATA, offset=offsets)))
    npt.assert_allclose(out, DATA / (2.0 + 0.0004 * np.array(offsets))[:, None], rtol=1e-6)


def test_weight_by_another_header_value():
    ids = [10000, 5000, 20000, 15000]
    out = arr(run(weight(key2='trace_id'), traces(DATA, trace_id=ids)))
    npt.assert_allclose(out, DATA * (np.array(ids) * 0.0001)[:, None], rtol=1e-6)
    out = arr(run(weight(key2=lambda t: 3.0, scale=0.5, inv=True), traces(DATA)))
    npt.assert_allclose(out, DATA / 1.5, rtol=1e-6)
    npt.assert_array_equal(arr(run(weight(key=lambda t: 7.0, a=1.0, b=0.0), traces(DATA))), DATA)
    with pytest.raises(KeyError):
        run(weight(key='nope'), traces(DATA))


def test_weight_by_zero_does_not_stop_anything():
    out = arr(run(weight(a=0.0, b=0.0, inv=True), traces(DATA)))
    assert np.isinf(out).all()


# ----------------------------------------------------------------------------------------------- ai2r and r2ai
AI = np.array([[1.0, 1.5, 2.0, 1.2, 1.2, 3.0, 0.5, 2.0]], dtype=np.float32)


def test_ai2r():
    (out,) = run(ai2r(), traces(AI))
    r = np.asarray(out)
    expected = np.zeros(8)
    for i in range(7):
        expected[i] = -(AI[0, i + 1] - AI[0, i]) / (AI[0, i + 1] + AI[0, i])
    npt.assert_allclose(r, expected, rtol=1e-6)
    assert r[-1] == 0.0
    assert (np.abs(r) < 1).all()


def test_r2ai_inverts_ai2r_and_keeps_every_header():
    trs = traces(np.vstack([AI, AI * 3]), trace_id=[11, 12], offset=[5.0, 6.0])
    out = run(ai2r() | r2ai(), trs)
    # r2ai starts from 1
    npt.assert_allclose(np.asarray(out[0]), AI[0] / AI[0, 0], rtol=1e-5)
    npt.assert_allclose(np.asarray(out[1]), (AI[0] * 3) / (AI[0, 0] * 3), rtol=1e-5)
    # (sur2ai gives every trace the header of the first)
    assert [t.header['trace_id'] for t in out] == [11, 12]
    assert [t.header['offset'] for t in out] == [5.0, 6.0]


def test_r2ai_by_hand_and_illegal_values():
    r = np.array([[0.5, -0.5, 0.0, 0.25, 9.0]], dtype=np.float32)  # (the last value is not used)
    (out,) = run(r2ai(), traces(r))
    npt.assert_allclose(np.asarray(out), [1.0, 1.0 / 3.0, 1.0, 1.0, 0.6], rtol=1e-6)
    with pytest.raises(ValueError, match="Illegal Reflectivity"):
        run(r2ai(), traces(np.array([[0.1, 1.5, 0.0]])))
    (empty,) = run(r2ai(), [Trace(np.zeros(0, dtype=np.float32), d_sample=DT)])
    assert empty.n_sample == 0


def test_parallelism_of_the_trace_wise_stages():
    trs = traces(DATA, offset=[0.0, 100.0, 200.0, 300.0])
    for stage in (reduce(rv=4.0), weight(), ai2r(), resamp(rf=2.0)):
        assert stage.parallelism == 'trace'
        npt.assert_array_equal(
            arr(run(stage, trs)), arr(list(from_iterable(list(trs)) | pmap(stage, workers=2, chunk=1)))
        )


# ------------------------------------------------------------------------------------------------------------ nmo
from seispy.stretching import nmo  # noqa: E402


def ricker(t, tc, freq=25.0):
    a = (np.pi * freq * (t - tc)) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


NT_N = 400
TIMES = np.arange(NT_N) * DT


def gather(velocity, t0, offsets, cdp=None):
    """A reflection at zero offset time t0, with the given velocity, at each offset: hyperbolas"""
    rows = []
    for x in offsets:
        v = velocity(t0) if callable(velocity) else velocity
        rows.append(ricker(TIMES, np.sqrt(t0 ** 2 + (x / v) ** 2)))
    kw = {'offset': list(offsets)}
    if cdp is not None:
        kw['ensemble_number'] = cdp
    return traces(np.array(rows), **kw)


OFFSETS = [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0]


def test_nmo_flattens_a_hyperbola():
    out = arr(run(nmo(vnmo=2000.0, sscale=False), gather(2000.0, 0.8, OFFSETS)))
    peaks = np.argmax(np.abs(out), axis=1)
    npt.assert_array_equal(peaks, 200)  # 0.8 s / 4 ms
    # all with the same peak (the far offsets are stretched, so the pulse gets wider, but not higher)
    npt.assert_allclose(out[:, 200], 1.0, atol=0.05)
    npt.assert_allclose(out[0], np.asarray(ricker(TIMES, 0.8)), atol=1e-5)
    # with the wrong velocity they are not flat
    wrong = arr(run(nmo(vnmo=3000.0, sscale=False), gather(2000.0, 0.8, OFFSETS)))
    assert np.argmax(np.abs(wrong), axis=1)[-1] != 200


def test_nmo_at_zero_offset_only_ramps_the_start():
    x = np.random.default_rng(1).normal(size=(1, NT_N)).astype(np.float32)
    (out,) = run(nmo(vnmo=2000.0), traces(x, offset=0.0))
    out = np.asarray(out)
    npt.assert_allclose(out[25:], x[0, 25:], atol=1e-5)
    # the first lmute samples are ramped up (as sunmo does, even with nothing muted)
    npt.assert_allclose(out[:25], x[0, :25] * (np.arange(25) + 1) / 25, atol=1e-5)
    (out,) = run(nmo(vnmo=2000.0, lmute=0), traces(x, offset=0.0))
    npt.assert_allclose(np.asarray(out), x[0], atol=1e-5)


def test_nmo_mutes_what_is_stretched_too_much():
    x = np.ones((1, NT_N), dtype=np.float32)
    # 1000 m at 1500 m/s is 167 samples. The stretch (how much the time axis is compressed) is tn / t(tn), which is
    # below 1 / smute = 0.667 until tn is 149 samples
    (out,) = run(nmo(vnmo=1500.0, lmute=0, sscale=False), traces(x, offset=1000.0))
    out = np.asarray(out)
    first = np.argmax(out != 0)
    assert first == pytest.approx(149, abs=2)
    npt.assert_array_equal(out[:first], 0.0)
    # a bigger limit on the stretch mutes less (until tn is 17 samples)
    (loose,) = run(nmo(vnmo=1500.0, lmute=0, sscale=False, smute=10.0), traces(x, offset=1000.0))
    assert np.argmax(np.asarray(loose) != 0) == pytest.approx(17, abs=2)
    # the ramp after the mute
    (ramped,) = run(nmo(vnmo=1500.0, lmute=10, sscale=False), traces(x, offset=1000.0))
    r = np.asarray(ramped)
    npt.assert_allclose(r[first:first + 10], out[first:first + 10] * (np.arange(10) + 1) / 10, atol=1e-5)


def test_nmo_stretch_scaling():
    x = np.ones((1, NT_N), dtype=np.float32)
    (plain,) = run(nmo(vnmo=2000.0, sscale=False, lmute=0), traces(x, offset=1000.0))
    (scaled,) = run(nmo(vnmo=2000.0, sscale=True, lmute=0), traces(x, offset=1000.0))
    plain, scaled = np.asarray(plain), np.asarray(scaled)
    # the scale is how much the time axis is compressed: t(tn)[i] - t(tn)[i-1], for t(tn) = sqrt(tn^2 + (x / v / dt)^2)
    tn = np.arange(NT_N, dtype=np.float64)
    ttn = np.sqrt(tn ** 2 + (1000.0 / 2000.0 / DT) ** 2)
    factor = np.diff(ttn, prepend=ttn[0] - 1 + (ttn[1] - ttn[0]))  # (the first is the same as the second)
    factor[0] = ttn[1] - ttn[0]
    live = plain != 0
    assert live.sum() > 200
    # (the interpolation of a constant is not quite that constant, so compare with the unscaled trace)
    npt.assert_allclose(scaled[live] / plain[live], factor[live], rtol=1e-4)
    assert (factor[live] < 1.0).all()


def test_nmo_time_varying_velocity():
    def v(t):
        return 1800.0 + 500.0 * t  # a velocity that increases with time

    trs = gather(v, 0.8, OFFSETS)
    tnmo = [0.0, 1.6]
    vnmo = [1800.0, 1800.0 + 500.0 * 1.6]
    out = arr(run(nmo(tnmo, vnmo, sscale=False), trs))
    npt.assert_array_equal(np.argmax(np.abs(out), axis=1), 200)


def test_nmo_inverse():
    # (exact inverse NMO is not possible, but an event's time comes back)
    flat = traces(np.tile(ricker(TIMES, 0.8), (len(OFFSETS), 1)), offset=OFFSETS)
    forward = nmo(vnmo=2000.0, sscale=False)
    out = arr(run(nmo(vnmo=2000.0, sscale=False, invert=True), flat))
    expected_peaks = np.round(np.sqrt(0.8 ** 2 + (np.array(OFFSETS) / 2000.0) ** 2) / DT).astype(int)
    npt.assert_allclose(np.argmax(np.abs(out), axis=1), expected_peaks, atol=1)
    # and back again
    back = arr(run(forward, [t.replace(offset=o) for t, o in zip(traces(out, offset=OFFSETS), OFFSETS)]))
    npt.assert_array_equal(np.argmax(np.abs(back), axis=1), 200)


def test_nmo_with_velocities_for_several_cdps():
    cdps = [10.0, 30.0]
    vnmo = [[1500.0], [2500.0]]
    tnmo = [[0.0], [0.0]]
    stage = nmo(tnmo, vnmo, cdp=cdps, sscale=False)
    for cdp, v in ((10, 1500.0), (30, 2500.0), (5, 1500.0), (40, 2500.0)):
        out = arr(run(stage, gather(v, 0.8, OFFSETS, cdp=cdp)))
        npt.assert_array_equal(np.argmax(np.abs(out), axis=1), 200)
    # between the cdps it is 1/v^2 that is interpolated
    v_mid = 1.0 / np.sqrt(0.5 * (1 / 1500.0 ** 2 + 1 / 2500.0 ** 2))
    out = arr(run(stage, gather(v_mid, 0.8, OFFSETS, cdp=20)))
    npt.assert_allclose(np.argmax(np.abs(out), axis=1), 200, atol=1)
    # cdps are sorted
    shuffled = nmo([[0.0], [0.0]], [[2500.0], [1500.0]], cdp=[30.0, 10.0], sscale=False)
    out = arr(run(shuffled, gather(1500.0, 0.8, OFFSETS, cdp=10)))
    npt.assert_array_equal(np.argmax(np.abs(out), axis=1), 200)


def test_nmo_traces_with_different_offsets_and_cdps_in_a_row():
    # the tables are kept while nothing changes, and made again when something does
    stage = nmo(vnmo=2000.0, sscale=False)
    trs = gather(2000.0, 0.8, [800.0, 800.0, 200.0, 800.0, 0.0])
    out = arr(run(stage, trs))
    npt.assert_array_equal(np.argmax(np.abs(out), axis=1), 200)
    separately = np.vstack([arr(run(stage, [t])) for t in trs])
    npt.assert_array_equal(out, separately)


def test_nmo_errors_and_properties():
    for bad in (
        dict(tnmo=[0.0, 1.0, 0.5], vnmo=[1.0, 2.0, 3.0]),
        dict(tnmo=[0.0, 1.0], vnmo=[1500.0]),
        dict(vnmo=2000.0, smute=0.0),
        dict(vnmo=2000.0, lmute=-1),
        dict(tnmo=[0.0], vnmo=[1500.0], cdp=[1.0, 2.0]),
        dict(tnmo=[[0.0]], vnmo=[[1500.0], [2000.0]], cdp=[1.0, 2.0]),
    ):
        with pytest.raises(ValueError):
            nmo(**bad)
    with pytest.raises(ValueError, match="sample interval"):
        run(nmo(vnmo=2000.0), traces(np.ones((1, 10)), dt=0.0))
    with pytest.raises(ValueError, match="at least 2"):
        run(nmo(vnmo=2000.0), traces(np.ones((1, 1))))
    with pytest.raises(ValueError, match="at least 4"):
        run(nmo(vnmo=2000.0, invert=True), traces(np.ones((1, 3))))
    assert nmo(vnmo=2000.0).parallelism == 'trace'


def test_nmo_in_threads_and_pickled():
    import pickle

    stage = nmo([0.0, 1.0], [1700.0, 2300.0], cdp=None, smute=2.0)
    trs = gather(2000.0, 0.6, OFFSETS * 3)
    expected = arr(run(stage, trs))
    got = arr(list(from_iterable(list(trs)) | pmap(stage, workers=4, chunk=3)))
    npt.assert_array_equal(got, expected)
    again = pickle.loads(pickle.dumps(stage))
    npt.assert_array_equal(arr(run(again, trs)), expected)
