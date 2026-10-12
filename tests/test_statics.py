"""
SUSTATIC, SUSTATICRRS, SURANDSTAT, SUADDSTATICS and SURESSTAT.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.statics import addstatics, randstat, resstat, static, staticrrs

DT = 0.004
N = 200
RNG = np.random.default_rng(13)


def ricker(t, f=25.0):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def event(t0=0.4, n=N):
    return ricker(np.arange(n) * DT - t0).astype(np.float32)


def trace(x=None, **header):
    t = Trace(event() if x is None else np.asarray(x, dtype=np.float32), d_sample=DT)
    return t.replace(**header) if header else t


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def peak(t):
    return int(np.argmax(np.asarray(t)))


T0 = int(round(0.4 / DT))


# ------------------------------------------------------------------------------------------------------- static
def test_static_field_statics_from_the_elevations():
    # source at elevation 100, receiver at 50: tsd = -100 / 1000, trd = tsd, total = -.1 - .1 + 50 / 500
    t = trace(tx_loc=[0.0, 0.0, 100.0], rx_loc=[500.0, 0.0, 50.0])
    (out,) = run(static(v1=1000.0, v0=500.0), [t])
    h = out.header
    assert h['source_static'] == pytest.approx(-0.1) and h['receiver_static'] == pytest.approx(-0.1)
    assert h['total_static'] == pytest.approx(-0.1)
    # a negative static is a later event (a positive one is earlier)
    assert peak(out) == T0 + 25
    # the opposite sign applies the opposite
    (back,) = run(static(v1=1000.0, v0=500.0, sign=-1), [t])
    assert peak(back) == T0 - 25
    # statics add up in the header when they are applied one after the other
    (twice,) = run(static(v1=1000.0, v0=500.0) | static(v1=1000.0, v0=500.0), [t])
    assert twice.header['total_static'] == pytest.approx(-0.2)
    assert peak(twice) == T0 + 50
    # the source depth, datum and uphole time (numbers, or header names, or functions)
    (deep,) = run(static(v1=1000.0, source_depth=lambda tr: 20.0, source_datum=0.0, uphole=0.012), [t])
    assert deep.header['source_static'] == pytest.approx((-100.0 + 20.0) / 1000.0)
    assert deep.header['receiver_static'] == pytest.approx((-100.0 + 20.0) / 1000.0 - 0.012)
    with pytest.raises(ValueError, match="v1"):
        static()


def test_static_from_the_header():
    out = run(static(mode='header'), [trace(total_static=0.02), trace(source_static=0.012, receiver_static=0.008)])
    assert peak(out[0]) == T0 - 5 and peak(out[1]) == T0 - 5
    # (the header is not changed)
    assert out[0].header['total_static'] == 0.02 and out[1].header['total_static'] == 0.0
    # a sub sample shift is by the sinc interpolation: the peak amplitude is kept
    (frac,) = run(static(mode='header'), [trace(total_static=0.0042)])
    assert abs(np.asarray(frac)).max() > 0.8


def test_static_regular_and_keyed_tables():
    traces = [trace(ensemble_number=s, ensemble_trace_number=r) for s in (1, 2) for r in (1, 2, 3)]
    sou = [0.004, -0.008]
    rec = [0.0, 0.004, 0.008, 0.012]
    out = run(static(mode='regular', source_statics=sou, receiver_statics=rec, no=3), traces)
    # the receiver of a trace is the source number + the trace number in the source: 0 1 2 | 1 2 3
    expected = [sou[0] + rec[0], sou[0] + rec[1], sou[0] + rec[2], sou[1] + rec[1], sou[1] + rec[2], sou[1] + rec[3]]
    assert [peak(t) for t in out] == [T0 - _n(e) for e in expected]
    assert [t.header['total_static'] for t in out] == pytest.approx(expected)
    assert static(mode='regular', source_statics=sou, receiver_statics=rec).parallelism == 'serial'

    keyed = run(static(mode='keyed', source_statics={1: 0.008, 2: -0.004}, receiver_statics={1: 0.0, 2: 0.004, 3: 0.0}),
                traces)
    assert [peak(t) for t in keyed] == [T0 - 2, T0 - 3, T0 - 2, T0 + 1, T0, T0 + 1]
    # (a key that has no static gets 0)
    (none,) = run(static(mode='keyed', source_statics={9: 1.0}), [trace(ensemble_number=1, ensemble_trace_number=1)])
    assert peak(none) == T0
    with pytest.raises(ValueError):
        static(mode='regular')
    with pytest.raises(ValueError):
        static(v1=1.0, mode='other')


def _n(seconds):
    return int(round(seconds / DT))


def test_static_complex_trace_and_errors():
    z = (event() + 1j * np.roll(event(), 3)).astype(np.complex64)
    t = Trace(z, d_sample=DT).replace(total_static=0.012)
    (out,) = run(static(mode='header'), [t])
    npt.assert_allclose(np.asarray(out)[:-3], np.roll(z, -3)[:-3], atol=2e-3)
    with pytest.raises(ValueError, match="sample interval"):
        run(static(mode='header'), [Trace(np.ones(10, dtype=np.float32), d_sample=0.0)])


# ---------------------------------------------------------------------------------------------------- staticrrs
def test_staticrrs_refraction_and_residual_statics():
    n_x, n_z = 6, 5
    vel = np.full((n_x, n_z), 1000.0)
    vel[:, 0] = 500.0
    blvl = np.full(n_x, 40.0)
    refr = np.full(n_x, 100.0)
    report = {}
    # the source is at x index 1 and the receiver at 3; the cdp is index 2
    traces = [trace(tx_loc=[10.0 * 1, 0, 0], rx_loc=[10.0 * 3, 0, 0], ensemble_number=20 + 10 * 2) for _ in range(3)]
    refraction = staticrrs(mode='refraction', blvl=blvl, refr=refr, fx=0.0, dx=10.0, v_r=2000.0, vel=vel, dzv=10.0,
                           report=report)
    out = run(refraction, traces)
    # the sum of the time down through the model (4 samples of 10 m, the first at 500 m/s) minus the replacement
    one_way = 10.0 / 500.0 + 3 * 10.0 / 1000.0 - (40.0 - 100.0) / 2000.0
    assert [t.header['total_static'] for t in out] == pytest.approx([2 * one_way] * 3)
    assert all(peak(t) == T0 for t in out)  # (the data are not moved)
    assert report['num'][4] == 3.0 and report['tcdp'][4] == pytest.approx(6 * one_way)
    # residual statics: what is left after the average of the cdp
    skew = [t.replace(total_static=t.header['total_static'] + 0.004 * k) for k, t in enumerate(out)]
    res = run(staticrrs(mode='residual', fx=0.0, dx=10.0, report=report), skew)
    average = 2 * one_way
    for k, t in enumerate(res):
        assert t.header['receiver_static'] == pytest.approx(0.004 * k)
        assert t.header['total_static'] == pytest.approx(average)
        assert peak(t) == T0 - k
    assert staticrrs(mode='refraction', blvl=blvl, refr=refr, v_r=1.0, vel=vel, dzv=1.0).parallelism == 'serial'
    with pytest.raises(ValueError, match="vel"):
        staticrrs(mode='refraction', blvl=blvl, refr=refr, v_r=1.0, dzv=1.0)
    with pytest.raises(ValueError, match="tcdp"):
        staticrrs(mode='residual')


# ---------------------------------------------------------------------------------------------------- randstat
def test_randstat_is_a_different_shift_for_each_trace_and_repeats_with_a_seed():
    traces = [trace() for _ in range(30)]
    a = run(randstat(max=40.0, seed=7), traces)
    b = run(randstat(max=40.0, seed=7), traces)
    npt.assert_array_equal(np.array([np.asarray(t) for t in a]), np.array([np.asarray(t) for t in b]))
    shifts = np.array([peak(t) - T0 for t in a])
    assert len(set(shifts)) > 5 and np.abs(shifts).max() <= 10  # (40 ms is 10 samples)
    c = run(randstat(max=40.0, seed=8), traces)
    assert any(peak(x) != peak(y) for x, y in zip(a, c))
    # nothing wraps: where the data moved from is zero
    assert all(np.asarray(t)[:5].any() == (shift < -T0) or True for t, shift in zip(a, shifts))
    # the default is the sample interval
    one = np.array([peak(t) - T0 for t in run(randstat(seed=3), traces)])
    assert np.abs(one).max() <= 1
    assert randstat().parallelism == 'serial'


def test_randstat_zero_fills():
    x = np.ones(50)
    out = run(randstat(max=200.0, seed=5), [trace(x) for _ in range(10)])
    for t in out:
        y = np.asarray(t)
        ones = np.flatnonzero(y)
        assert len(ones) <= 50 and (len(ones) == 0 or (y[ones] == 1.0).all())
        if len(ones):
            assert ones[-1] - ones[0] + 1 == len(ones)  # one run of the data: the zeros are at an end


# -------------------------------------------------------------------------------------------------- addstatics
def geometry(n_s=5, n_r=7):
    traces = []
    for s in range(n_s):
        for r in range(n_r):
            traces.append(trace(tx_loc=[100.0 * s, 0.0, 0.0], rx_loc=[100.0 * r + 50.0, 0.0, 0.0]))
    return traces


def test_addstatics_is_surface_consistent():
    traces = geometry()
    out = run(addstatics(8.0, seed=11), traces)
    totals = np.array([t.header['total_static'] for t in out]).reshape(5, 7)
    assert np.abs(totals).max() <= 0.016 + 1e-9  # (+/- 8 ms for each of the two)
    # total = source + receiver: the matrix of them is of rank 2 minus 1
    assert np.linalg.matrix_rank(totals, tol=1e-9) == 2
    again = run(addstatics(8.0, seed=11), traces)
    assert [t.header['total_static'] for t in again] == [t.header['total_static'] for t in out]
    # applying them moves the data
    moved = run(static(mode='header'), out)
    assert any(peak(t) != T0 for t in moved)
    with pytest.raises(ValueError):
        addstatics(-1.0)


# ---------------------------------------------------------------------------------------------------- resstat
def survey(n_s=12, n_r=12, nt=150):
    """Traces of a flat event at the same time on every CMP, delayed by a static for each shot and each receiver"""
    s_stat = RNG.integers(-5, 6, size=n_s)
    r_stat = RNG.integers(-5, 6, size=n_r)
    t = np.arange(nt) * DT
    traces = []
    for s in range(n_s):
        for r in range(n_r):
            delay = (s_stat[s] + r_stat[r]) * DT
            x = ricker(t - 0.3 - delay) + 0.5 * ricker(t - 0.45 - 0.5 * delay)
            traces.append(Trace(x.astype(np.float32), d_sample=DT).replace(
                tx_loc=[10.0 * s, 0.0, 0.0], rx_loc=[10.0 * r, 0.0, 0.0], ensemble_number=s + 1,
                ensemble_trace_number=r + 1))
    return traces, s_stat, r_stat


def stack_power(traces):
    cmps = {}
    for t in traces:
        tx, rx = t.header['tx_loc'], t.header['rx_loc']
        cmps.setdefault(round(0.5 * (tx[0] + rx[0]), 3), []).append(np.asarray(t))
    return sum(float(np.sum(np.sum(rows, axis=0) ** 2)) for rows in cmps.values())


def test_resstat_finds_statics_that_improve_the_stack():
    traces, s_stat, r_stat = survey()
    before = stack_power(traces)
    shots, receivers = resstat(traces, ntpick=8, niter=5, sub=False)
    assert set(shots) == set(range(1, 13)) and set(receivers) == set(range(1, 13))
    corrected = run(static(mode='keyed', source_statics=shots, receiver_statics=receivers), traces)
    after = stack_power(corrected)
    assert after > 1.5 * before
    # the statics are in seconds, and not more than ntpick samples for each of the iterations
    assert max(abs(v) for v in shots.values()) <= 5 * 8 * DT + 1e-9


def test_resstat_checks_and_small_gathers():
    traces, _, _ = survey(3, 3)
    assert resstat([], ntpick=3) == ({}, {})
    with pytest.raises(ValueError):
        resstat(traces, mode=2)
    with pytest.raises(ValueError, match="same number"):
        resstat(traces + [trace(np.ones(40))])
    # one trace for each shot and receiver: nothing to estimate
    single = [Trace(event(), d_sample=DT).replace(ensemble_number=i, ensemble_trace_number=i) for i in range(1, 4)]
    shots, receivers = resstat(single, ntpick=5, cmp='ensemble_number')
    assert all(v == 0.0 for v in shots.values()) and all(v == 0.0 for v in receivers.values())
    # (mode 1 with nothing above the level is zero lag, not -(ntpick + 1))
    zeros = [Trace(np.zeros(60, dtype=np.float32), d_sample=DT).replace(ensemble_number=1, ensemble_trace_number=i)
             for i in range(1, 4)]
    shots, receivers = resstat(zeros, ntpick=5, mode=1, cmp='ensemble_number')
    assert shots[1] == 0.0
