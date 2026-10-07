import math

import numpy as np
import numpy.testing as npt
import pytest

from seispy.amplitudes import normalize, nan, zero
from seispy.container import Trace, from_iterable
from seispy.parallel import pmap
from seispy.tapering import ramp, taper
from seispy.windowing import kill, mute, wind

DT = 0.004


def traces(data, dt=DT, sample_start=0.0, **per_trace):
    """Traces of the rows of data. Header values are lists with one value per trace (or one value for all)."""
    out = []
    for i, row in enumerate(np.atleast_2d(data)):
        header = {}
        for name, values in per_trace.items():
            header[name] = values[i] if isinstance(values, (list, tuple, np.ndarray)) else values
        out.append(Trace(np.asarray(row, dtype=np.float32), d_sample=dt, sample_start=sample_start).replace(**header))
    return out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(99)
DATA = RNG.normal(size=(6, 50)).astype(np.float32) + 3.0  # (away from 0, so that nothing is zero by accident)


# ------------------------------------------------------------------------------------------------------ replace
def test_trace_replace():
    t = Trace(np.arange(5, dtype=np.float32), d_sample=DT, sample_start=0.1)
    u = t.replace(np.arange(3), offset=250.0, sample_start=0.2, tx_loc=[1, 2, 3])
    assert u.n_sample == 3
    assert u.header['offset'] == 250.0 and u.header['sample_start'] == 0.2 and u.header['tx_loc'] == [1.0, 2.0, 3.0]
    assert u.d_sample == t.d_sample
    npt.assert_array_equal(np.asarray(u), [0, 1, 2])
    # a copy, that does not share anything with the original
    v = t.replace()
    np.asarray(v)[:] = -1
    npt.assert_array_equal(np.asarray(t), np.arange(5))
    assert v.header == t.header | {}
    with pytest.raises(TypeError, match="Unknown header"):
        t.replace(not_a_header_value=1)


# ---------------------------------------------------------------------------------------------------------- zero
def test_zero():
    out = arr(run(zero(10, itmin=4, value=-1.0), traces(DATA)))
    expected = DATA.copy()
    expected[:, 4:11] = -1.0
    npt.assert_array_equal(out, expected)
    npt.assert_array_equal(arr(run(zero(3), traces(DATA)))[:, :4], 0.0)


def test_zero_errors():
    for kwargs in (dict(itmax=5, itmin=-1), dict(itmax=2, itmin=5)):
        with pytest.raises(ValueError):
            zero(**kwargs)
    with pytest.raises(ValueError, match="< nt"):
        run(zero(50), traces(DATA))  # one past the end of the trace


# ----------------------------------------------------------------------------------------------------------- nan
def test_nan_with_a_value():
    x = DATA[:1].copy()
    x[0, [0, 5, 6, 49]] = [np.nan, np.inf, -np.inf, np.nan]
    out = arr(run(nan(value=7.0), traces(x)))[0]
    assert np.isfinite(out).all()
    npt.assert_array_equal(out[[0, 5, 6, 49]], 7.0)
    keep = np.ones(50, bool)
    keep[[0, 5, 6, 49]] = False
    npt.assert_array_equal(out[keep], x[0, keep])


def test_nan_interpolation():
    x = np.array([[np.nan, 1, 2, np.nan, 4, np.nan, np.nan, 8, np.inf]], dtype=np.float32)
    out = arr(run(nan(interp=True, value=-1.0), traces(x)))[0]
    # first takes its neighbor, the middle one the average of its neighbors, the last takes the one before it.
    # Two in a row are replaced one at a time: the first has a neighbor that is not finite so it gets the value, and
    # that is then a neighbor of the second.
    npt.assert_array_equal(out, [1, 1, 2, 3, 4, -1, 3.5, 8, 8])
    # the default is no interpolation
    npt.assert_array_equal(arr(run(nan(), traces(x)))[0], [0, 1, 2, 0, 4, 0, 0, 8, 0])


def test_nan_leaves_clean_traces_alone():
    npt.assert_array_equal(arr(run(nan(interp=True), traces(DATA))), DATA)


# ------------------------------------------------------------------------------------------------------ normalize
def test_normalize_levels():
    rms = np.sqrt((DATA.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    npt.assert_allclose(arr(run(normalize(), traces(DATA))), DATA / rms, rtol=1e-5)
    npt.assert_allclose(arr(run(normalize('max'), traces(DATA))), DATA / np.abs(DATA).max(axis=1, keepdims=True), rtol=1e-6)
    med = np.median(DATA, axis=1, keepdims=True)
    for name in ('med', 'median'):
        npt.assert_allclose(arr(run(normalize(name), traces(DATA))), DATA / med, rtol=1e-5)
    npt.assert_allclose(arr(run(normalize('balmed'), traces(DATA))), DATA - med, atol=1e-6)


def test_normalize_window_and_independent_traces():
    # the window is in time: samples 5 to 19 here
    out = arr(run(normalize('max', t0=5 * DT, t1=20 * DT), traces(DATA)))
    expected = DATA / np.abs(DATA[:, 5:20]).max(axis=1, keepdims=True)
    npt.assert_allclose(out, expected, rtol=1e-6)
    # every trace on its own: a loud trace before a quiet one does not change the quiet one
    loud_then_quiet = np.stack([DATA[0] * 1000, DATA[1]])
    alone = arr(run(normalize('max'), traces(DATA[1:2])))[0]
    npt.assert_array_equal(arr(run(normalize('max'), traces(loud_then_quiet)))[1], alone)
    rms_alone = arr(run(normalize('rms'), traces(DATA[1:2])))[0]
    npt.assert_array_equal(arr(run(normalize('rms'), traces(loud_then_quiet)))[1], rms_alone)


def test_normalize_zero_level_and_errors():
    z = np.zeros((1, 10), dtype=np.float32)
    npt.assert_array_equal(arr(run(normalize(), traces(z))), z)
    with pytest.raises(ValueError, match="unknown norm"):
        normalize('nope')
    with pytest.raises(ValueError, match="window"):
        run(normalize(t0=1.0), traces(DATA))
    with pytest.raises(ValueError, match="sample interval"):
        run(normalize(), traces(DATA, dt=0.0))
    assert arr(run(normalize(dt=DT), traces(DATA, dt=0.0))).shape == DATA.shape


# ---------------------------------------------------------------------------------------------------- time taper
def envelope(kind, f):
    return {
        1: f,
        2: np.sin(np.pi * f / 2),
        3: 0.5 * (1 - np.cos(np.pi * f)),
        4: np.exp(-((3.8090232 * (1 - f)) ** 2)),
        5: np.exp(-((2.0 * (1 - f)) ** 2)),
    }[kind]


@pytest.mark.parametrize('kind', [1, 2, 3, 4, 5])
def test_time_taper(kind):
    # 20 ms at the start is 6 samples (the taper has 1 + t/dt of them), 12 ms at the end is 4
    out = arr(run(taper(20.0, 12.0, type=kind), traces(DATA)))
    expected = DATA.astype(np.float64).copy()
    f1 = np.arange(6) / 6
    expected[:, :6] *= envelope(kind, f1)
    f2 = np.arange(4) / 4
    expected[:, ::-1][:, :4] *= envelope(kind, f2)  # from the last sample inwards
    npt.assert_allclose(out, expected, rtol=1e-5)


def test_time_taper_one_end_and_none():
    out = arr(run(taper(tend=8.0), traces(DATA)))
    # 8 ms is 1 + 8/4 = 3 samples
    expected = DATA.copy()
    expected[:, -1] *= 0.0
    expected[:, -2] *= 1 / 3
    expected[:, -3] *= 2 / 3
    npt.assert_allclose(out, expected, rtol=1e-6, atol=1e-7)
    npt.assert_array_equal(arr(run(taper(), traces(DATA))), DATA)


# --------------------------------------------------------------------------------------------------- trace taper
def test_trace_taper():
    data = np.ones((10, 8), dtype=np.float32)
    out = arr(run(taper(tr1=3, tr2=2, min=0.1), traces(data)))[:, 0]
    # the first traces go from the minimum up to (not including) 1, the last down to the minimum on the last trace
    expected = np.ones(10)
    expected[:3] = [0.1, 0.1 + 0.9 / 3, 0.1 + 0.9 * 2 / 3]
    expected[8:] = [0.1 + 0.9 * 0.5, 0.1]
    npt.assert_allclose(out, expected, rtol=1e-6)
    # tr2 is tr1 unless it is given
    out = arr(run(taper(tr1=2), traces(data)))[:, 0]
    expected = np.ones(10)
    expected[[0, 1, 8, 9]] = [0.0, 0.5, 0.5, 0.0]
    npt.assert_allclose(out, expected, atol=1e-7)
    # other types ignore the minimum
    out = arr(run(taper(tr1=3, tr2=0, type=2, min=0.5), traces(data)))[:, 0]
    npt.assert_allclose(out[:3], np.sin(np.pi * np.arange(3) / 3 / 2), atol=1e-6)


def test_trace_taper_needs_the_number_of_traces():
    data = np.ones((10, 8), dtype=np.float32)
    # a generator does not say how many traces it has
    with pytest.raises(ValueError, match="ntr"):
        list((t for t in traces(data)) | taper(tr1=2))
    out = arr(list((t for t in traces(data)) | taper(tr1=2, ntr=10)))[:, 0]
    assert out[0] == 0 and out[-1] == 0


def test_taper_parameters_and_parallelism():
    assert taper(10.0, 10.0).parallelism == 'trace'
    assert taper(tr1=2).parallelism == 'serial'
    assert taper(tr2=2).parallelism == 'serial'
    for kwargs in (dict(min=1.5), dict(type=6), dict(tr1=-1)):
        with pytest.raises(ValueError):
            taper(**kwargs)
    with pytest.raises(ValueError, match="exceeds"):
        run(taper(100.0, 100.0), traces(DATA))  # more than the 196 ms of the trace
    with pytest.raises(ValueError):
        pmap(taper(tr1=2))


# ----------------------------------------------------------------------------------------------------------- ramp
def test_ramp():
    # 50 samples: starts at 0, ends at 0.196 s
    out = arr(run(ramp(tmin=0.02, tmax=0.18), traces(DATA)))
    expected = DATA.copy()
    expected[:, :5] *= (np.arange(5) + 1) / 5
    expected[:, -4:] *= [1.0, 0.75, 0.5, 0.25]
    npt.assert_allclose(out, expected, rtol=1e-6)


def test_ramp_default_is_a_no_op():
    npt.assert_array_equal(arr(run(ramp(), traces(DATA))), DATA)
    # (also if the trace does not start at 0)
    npt.assert_array_equal(arr(run(ramp(), traces(DATA, sample_start=0.1))), DATA)
    # the start of the ramp is measured from the start of the trace
    out = arr(run(ramp(tmin=0.12), traces(DATA, sample_start=0.1)))
    expected = DATA.copy()
    expected[:, :5] *= (np.arange(5) + 1) / 5
    npt.assert_allclose(out, expected, rtol=1e-6)
    with pytest.raises(ValueError, match="sample interval"):
        run(ramp(), traces(DATA, dt=0.0))


# ---------------------------------------------------------------------------------------------------------- mute
NT = 100
FLAT = np.ones((1, NT), dtype=np.float32)
XM, TM = [0.0, 1000.0], [0.0, 0.4]


def mute_one(offset, stage_fn=mute, data=FLAT, **kw):
    """The mute of one trace at the given offset"""
    return arr(run(stage_fn(**kw) if stage_fn is not mute else mute(kw.pop('xmute', XM), kw.pop('tmute', TM), **kw),
                   traces(data, offset=offset)))[0]


def test_mute_above_and_below():
    out = mute_one(500.0)  # t = 0.2 s: 50 samples
    npt.assert_array_equal(out, [0.0] * 50 + [1.0] * 50)
    out = mute_one(500.0, mode=1)
    npt.assert_array_equal(out, [1.0] * 50 + [0.0] * 50)
    # past the end of xmute, it is the last time (0.4 s is the whole trace), before the start it is the first time
    # of the trace
    npt.assert_array_equal(mute_one(5000.0), 0.0)
    npt.assert_array_equal(mute_one(-50.0), 1.0)
    # (mode 1 mutes below the curve, which is at the end of the trace by then)
    npt.assert_array_equal(mute_one(5000.0, mode=1), 1.0)
    npt.assert_array_equal(mute_one(-50.0, mode=1), 0.0)


def test_mute_taper():
    taper_w = np.sin((np.arange(4) + 1) * np.pi / 8) ** 2
    out = mute_one(500.0, ntaper=4)
    npt.assert_allclose(out, [0.0] * 50 + list(taper_w) + [1.0] * 46, rtol=1e-6)
    out = mute_one(500.0, mode=1, ntaper=4)
    npt.assert_allclose(out, [1.0] * 46 + list(taper_w[::-1]) + [0.0] * 50, rtol=1e-6)
    # a taper that runs off the end of the trace is cut off, it does not write past it
    assert mute_one(500.0, xmute=[0, 1000], tmute=[0.0, 0.396], ntaper=8).shape == (NT,)


def test_mute_uses_the_start_time_of_the_trace():
    tr = traces(FLAT, sample_start=0.1, offset=500.0)
    out = arr(run(mute(XM, [0.1, 0.5]), tr))[0]  # t = 0.3 s, the trace starts at 0.1: 50 samples
    npt.assert_array_equal(out, [0.0] * 50 + [1.0] * 50)


def test_mute_zone_modes():
    # mode 2, the zone (0.04 s wide, 10 samples) is around the line t = offset / linvel: 0.1 s, sample 25
    out = mute_one(330.0, mode=2, xmute=XM, tmute=[0.04, 0.04], linvel=3300.0)
    expected = np.ones(NT)
    expected[20:30] = 0
    npt.assert_array_equal(out, expected)
    # with a time shift
    out = mute_one(330.0, mode=2, xmute=XM, tmute=[0.04, 0.04], linvel=3300.0, tm0=0.02)
    expected = np.ones(NT)
    expected[25:35] = 0  # sample 30 now
    npt.assert_array_equal(out, expected)
    # mode 3, a hyperbola: sqrt((tm0/dt)^2 + (offset/linvel/dt)^2) = sqrt(12.5^2 + 25^2) = 27.95: sample 28
    out = mute_one(330.0, mode=3, xmute=XM, tmute=[0.04, 0.04], linvel=3300.0, tm0=0.05)
    expected = np.ones(NT)
    expected[23:33] = 0
    npt.assert_array_equal(out, expected)
    # the offset is made positive for these, unless asked not to
    npt.assert_array_equal(mute_one(-330.0, mode=2, xmute=[-1000, 1000], tmute=[0.04, 0.04], linvel=3300.0), expected * 0 + (
        np.where((np.arange(NT) >= 20) & (np.arange(NT) < 30), 0.0, 1.0)))
    out = mute_one(-330.0, mode=2, xmute=[-1000, 1000], tmute=[0.04, 0.04], linvel=3300.0, absolute=False)
    expected = np.ones(NT)
    expected[-30 + 25 - 5 + 0:0] = 1  # (the zone is at sample -25, which is off the trace: nothing is muted)
    npt.assert_array_equal(out, 1.0)


def test_mute_zone_taper():
    taper_w = np.sin((np.arange(3) + 1) * np.pi / 6) ** 2
    out = mute_one(330.0, mode=2, xmute=XM, tmute=[0.04, 0.04], linvel=3300.0, ntaper=3)
    expected = np.ones(NT)
    expected[20:30] = 0
    # (sumute starts the taper before the zone at its first muted sample, so there it has one sample fewer than after)
    expected[19] = taper_w[1]
    expected[18] = taper_w[2]
    expected[30:33] = taper_w
    npt.assert_allclose(out, expected, rtol=1e-6)


def test_mute_polygon():
    # t is the time of the center, twindow its width
    out = mute_one(500.0, mode=4, xmute=XM, tmute=[0.1, 0.2], twindow=[0.04, 0.08])
    expected = np.ones(NT)
    expected[31:45] = 0  # center 0.15 s: sample 38 (NINT of 37.5), the width 0.06 s: 15 samples, 7 on each side
    npt.assert_array_equal(out, expected)


def test_mute_key_and_errors():
    tr = traces(FLAT, offset=0.0, trace_id=500)
    out = arr(run(mute(XM, TM, key='trace_id'), tr))[0]
    npt.assert_array_equal(out, [0.0] * 50 + [1.0] * 50)
    out = arr(run(mute(XM, TM, key=lambda t: 2 * t.header['trace_id']), tr))[0]
    npt.assert_array_equal(out, 0.0)
    for bad in (
        dict(xmute=[0, 1], tmute=[0]),
        dict(xmute=[0, 1], tmute=[0, 1], mode=7),
        dict(xmute=[0, 1], tmute=[0, 1], mode=4),
        dict(xmute=[0, 1], tmute=[0, 1], mode=4, twindow=[1]),
        dict(xmute=[0, 1], tmute=[0, 1], linvel=0),
        dict(xmute=[0, 1], tmute=[0, 1], ntaper=-1),
    ):
        with pytest.raises(ValueError):
            mute(**bad)
    with pytest.raises(ValueError, match="sample interval"):
        run(mute(XM, TM), traces(FLAT, dt=0.0))
    with pytest.raises(KeyError):
        run(mute(XM, TM, key='not_a_key'), traces(FLAT))


# ---------------------------------------------------------------------------------------------------------- wind
def wind_ids(trs, **kw):
    return [t.header['trace_id'] for t in run(wind(**kw), trs)]


@pytest.fixture
def ten():
    return traces(np.tile(np.arange(20, dtype=np.float32), (10, 1)), sample_start=0.1, trace_id=list(range(1, 11)))


def test_wind_by_key(ten):
    assert wind_ids(ten) == list(range(1, 11))
    assert wind_ids(ten, min=3, max=7) == [3, 4, 5, 6, 7]
    assert wind_ids(ten, min=8) == [8, 9, 10]
    assert wind_ids(ten, j=3) == [3, 6, 9]
    assert wind_ids(ten, j=3, s=1) == [1, 4, 7, 10]
    assert wind_ids(ten, reject=[2, 5]) == [1, 3, 4, 6, 7, 8, 9, 10]
    # accept overrides everything but not the order the traces come in
    assert wind_ids(ten, max=0, accept=[4, 5, 6]) == [4, 5, 6]
    assert wind_ids(ten, min=3, max=5, reject=[4], accept=[9]) == [3, 5, 9]


def test_wind_count_skip_order(ten):
    assert wind_ids(ten, count=3) == [1, 2, 3]
    assert wind_ids(ten, skip=4) == [5, 6, 7, 8, 9, 10]
    assert wind_ids(ten, skip=4, count=2, j=2) == [6, 8]
    # ordered=1 says the key only goes up, so when it passes max everything after it is dropped, even an accept
    assert wind_ids(ten, max=5, ordered=1, accept=[9]) == [1, 2, 3, 4, 5]
    assert wind_ids(ten, max=5, ordered=0, accept=[9]) == [1, 2, 3, 4, 5, 9]
    rev = list(reversed(ten))
    assert wind_ids(rev, min=6, ordered=-1) == [10, 9, 8, 7, 6]
    assert wind_ids(ten, skip=20) == []


def test_wind_abs_and_function_keys(ten):
    signed = [t.replace(offset=(-1.0) ** i * 100 * (i + 1)) for i, t in enumerate(ten)]
    ids = [t.header['trace_id'] for t in run(wind(key='offset', abs=True, min=500, max=700), signed)]
    assert ids == [5, 6, 7]
    ids = [t.header['trace_id'] for t in run(wind(key='offset', min=500, max=700), signed)]
    assert ids == [5, 7]  # (-600 is outside)
    assert wind_ids(ten, key=lambda t: t.header['trace_id'] * 10, min=30, max=50) == [3, 4, 5]
    # (the key is cut to an integer)
    frac = [t.replace(offset=v) for t, v in zip(ten, np.arange(10) + 0.9)]
    assert [t.header['trace_id'] for t in run(wind(key='offset', min=3, max=5), frac)] == [4, 5, 6]


def test_wind_in_time(ten):
    out = run(wind(itmin=5, nt=8), ten)
    assert len(out) == 10
    for t in out:
        npt.assert_array_equal(np.asarray(t), np.arange(5, 13))
        assert t.n_sample == 8
        assert t.header['sample_start'] == pytest.approx(0.1 + 5 * DT)
    # by time: tmin, tmax are times, f1 is the time of the first sample
    a = run(wind(tmin=0.12, tmax=0.148), ten)[0]
    npt.assert_array_equal(np.asarray(a), np.arange(5, 13))
    b = run(wind(itmin=2, itmax=4), ten)[0]
    npt.assert_array_equal(np.asarray(b), [2, 3, 4])
    # itmin beats tmin, itmax beats tmax beats nt
    c = run(wind(itmin=2, tmin=0.5, itmax=4, tmax=0.5, nt=100), ten)[0]
    npt.assert_array_equal(np.asarray(c), [2, 3, 4])
    d = run(wind(tmax=0.116), ten)[0]
    npt.assert_array_equal(np.asarray(d), np.arange(0, 5))
    # everything the window sticks out of the trace by is zeros
    e = run(wind(itmin=15, nt=10), ten)[0]
    npt.assert_array_equal(np.asarray(e), [15, 16, 17, 18, 19, 0, 0, 0, 0, 0])
    # the header and the other traces' headers are kept
    assert [t.header['trace_id'] for t in out] == list(range(1, 11))
    # no window: the same traces
    same = run(wind(), ten)
    for t, s in zip(same, ten):
        npt.assert_array_equal(np.asarray(t), np.asarray(s))
        assert t.header == s.header


def test_wind_time_and_key_together(ten):
    out = run(wind(min=4, max=5, itmin=10), ten)
    assert [t.header['trace_id'] for t in out] == [4, 5]
    npt.assert_array_equal(np.asarray(out[0]), np.arange(10, 20))


def test_wind_errors_and_properties(ten):
    for bad in (dict(j=0), dict(ordered=2), dict(skip=-1), dict(count=0)):
        with pytest.raises(ValueError):
            wind(**bad)
    with pytest.raises(ValueError, match="conflict"):
        run(wind(itmin=5, itmax=2), ten)
    with pytest.raises(ValueError, match="positive"):
        run(wind(itmin=-1), ten)
    with pytest.raises(ValueError, match="sample interval"):
        run(wind(), traces(DATA, dt=0.0))
    assert wind().parallelism == 'serial'
    assert (from_iterable(ten) | wind()).n_traces is None
    assert run(wind(), []) == []


# ---------------------------------------------------------------------------------------------------------- kill
def test_kill_by_key(ten):
    out = run(kill('trace_id', 3), ten)
    assert [bool(np.asarray(t).any()) for t in out] == [True, True, False] + [True] * 7
    assert all(t.n_sample == 20 for t in out)
    # nothing selected: the traces are not touched
    for t, s in zip(run(kill('trace_id', 99), ten), ten):
        npt.assert_array_equal(np.asarray(t), np.asarray(s))
    assert kill('trace_id', 3).parallelism == 'trace'


def test_kill_by_position(ten):
    out = run(kill(min=2, count=3), ten)
    assert [bool(np.asarray(t).any()) for t in out] == [True, False, False, False] + [True] * 6
    out = run(kill(min=10), ten)
    assert not np.asarray(out[-1]).any() and np.asarray(out[-2]).any()
    # the position wins over the key
    out = run(kill('trace_id', 3, min=1), ten)
    assert [bool(np.asarray(t).any()) for t in out][:3] == [False, True, True]
    assert kill(min=2).parallelism == 'serial'
    assert (from_iterable(ten) | kill(min=2)).n_traces == 10
    with pytest.raises(ValueError, match="failed to get"):
        run(kill(min=9, count=3), ten)


def test_kill_errors():
    for kwargs in (dict(), dict(key='trace_id'), dict(min=0), dict(min=1, count=0)):
        with pytest.raises(ValueError):
            kill(**kwargs)


# --------------------------------------------------------------------------------------------------- parallelism
def test_pmap_matches_sequential_for_the_trace_wise_stages():
    trs = traces(DATA, offset=[0, 100, 200, 300, 400, 500])
    stages = [
        zero(10, itmin=3),
        nan(interp=True),
        normalize('rms'),
        taper(20.0, 20.0, type=3),
        ramp(tmin=0.03, tmax=0.15),
        mute(XM, TM, ntaper=3),
        kill('trace_id', 0),
    ]
    for stage in stages:
        assert stage.parallelism == 'trace'
        expected = arr(run(stage, trs))
        got = arr(list(from_iterable(list(trs)) | pmap(stage, workers=3, chunk=2)))
        npt.assert_array_equal(got, expected)
    for serial in (taper(tr1=1), kill(min=1), wind()):
        with pytest.raises(ValueError):
            pmap(serial)


def test_stages_pickle():
    import pickle

    for stage in (zero(5), nan(), normalize('max'), taper(10, 10), ramp(tmin=0.1), mute(XM, TM), wind(count=2), kill(min=2)):
        again = pickle.loads(pickle.dumps(stage))
        assert repr(again) == repr(stage)
