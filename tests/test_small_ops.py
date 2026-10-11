"""The small programs of operations, stacking, windowing, picking, noise and velocity: fwmix, groll, permute, recmo, simmat,
recip, cvs4fowler, mutecsv, windpoly, padmissing, braid, gprfb, fbpickw, fnzero, pickamp, intvel and stkvel."""
import numpy as np
import numpy.testing as npt
import pytest

from seispy import operations, picking, stacking, velocity, windowing
from seispy.container import Trace, from_iterable
from seispy.filters import fwmix, groll, svdfilt


def make(data, dt=0.004, **header):
    return Trace(np.asarray(data, dtype=np.float32), d_sample=dt).replace(**header)


def stream(rows, dt=0.004, **header):
    return from_iterable([make(r, dt, **header) for r in rows])


def samples(s):
    return np.array([np.asarray(t) for t in s])


# ------------------------------------------------------------------------------------------------------------- fwmix
def test_fwmix_is_a_weighted_mean_of_a_gather_and_has_the_header_of_the_marked_trace():
    rng = np.random.default_rng(1)
    base = np.sin(np.arange(128) * 0.2) * np.hanning(128)
    data = base[None, :] + 0.3 * rng.standard_normal((9, 128))
    traces = [make(row, ensemble_number=7, ensemble_trace_number=i + 1) for i, row in enumerate(data)]
    key = lambda t: float(t.header['ensemble_trace_number'])  # noqa: E731
    out = list(from_iterable(traces) | fwmix(key=key, dx=1.0, mark=4, fac=100.0))
    assert len(out) == 1 and out[0].header['ensemble_trace_number'] == 5
    # a weight of nearly 1 for all: the mean, which has less noise than a trace
    assert np.std(np.asarray(out[0]) - base) < 0.6 * np.std(data[4] - base)
    # a near weight is larger: with fac = 1 the weights fall off with the distance
    near = list(from_iterable(traces) | fwmix(key=key, dx=1.0, mark=4, fac=0.01))
    assert np.std(np.asarray(near[0]) - data[4]) < np.std(np.asarray(out[0]) - data[4])
    # one gather, one trace
    two = list(from_iterable([t.replace(ensemble_number=1 + (i > 4)) for i, t in enumerate(traces)]) | fwmix(key=key, dx=1.0))
    assert len(two) == 2
    # the frequency dependent mix leaves out the traces that are far for high frequencies
    vf = list(from_iterable(traces) | fwmix(key=key, dx=100.0, mark=4, vf=True, vmin=500.0))
    assert np.isfinite(np.asarray(vf[0])).all()
    with pytest.raises(ValueError, match="same number"):
        fwmix(key=[key, key], dx=[1.0])


# ------------------------------------------------------------------------------------------------------------- groll
def test_groll_removes_the_coherent_energy_that_dominates():
    rng = np.random.default_rng(2)
    roll = np.outer(np.linspace(1, 2, 12), np.sin(np.arange(100) * 0.15) * 10.0)  # (rank one)
    signal = 0.3 * rng.standard_normal((12, 100))
    data = roll + signal
    cleaned = samples(stream(data) | groll(nrot=1))
    assert np.std(cleaned - signal) < 0.4 * np.std(roll)
    only = samples(stream(data) | groll(nrot=1, sb=True))
    npt.assert_allclose(only + cleaned, data, atol=1e-4)
    # it is the first principal components of svdfilt
    npt.assert_allclose(cleaned, samples(stream(data) | svdfilt(numpp=1, subtract=True)), atol=1e-4)
    with pytest.raises(ValueError, match="nrot"):
        groll(nrot=-1)


# ---------------------------------------------------------------------------------------------------------- permute
def test_permute_transposes_a_cube():
    cube = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # (n3, n2, n1)
    traces = [make(cube[k, j]) for k in range(2) for j in range(3)]
    # the panels are transposed: the traces become samples
    out = samples(from_iterable(traces) | operations.permute(n2=3, n3=2, o=(2, 1, 3)))
    npt.assert_array_equal(out, np.transpose(cube, (0, 2, 1)).reshape(-1, 3))
    # a reversed and decimated dimension
    out = samples(from_iterable(traces) | operations.permute(n2=3, n3=2, d=(-2, 1, 1)))
    npt.assert_array_equal(out, cube[:, :, ::-1][:, :, ::2].reshape(-1, 2))
    # the header
    first = list(from_iterable(traces) | operations.permute(n2=3, n3=2))[0]
    assert first.d_sample == 1.0 and first.header['sample_start'] == 1.0
    with pytest.raises(ValueError, match="permutation"):
        operations.permute(o=(1, 1, 3))
    with pytest.raises(ValueError, match="wrong amount"):
        list(from_iterable(traces[:4]) | operations.permute(n2=3, n3=2))


# ------------------------------------------------------------------------------------------------------------ recmo
def test_recmo_mixes_with_the_traces_before_with_the_weights_of_the_movement():
    nt, dt, vb, dx = 100, 0.004, 2.5, 25.0
    rows = [np.full(nt, float(i + 1)) for i in range(4)]
    out = samples(stream(rows, dt) | operations.recmo(vb=vb, dx=dx))
    moved = vb * np.arange(nt) * dt / dx  # (always under 1 here)
    # the first trace is mixed with zeros before it, the others with the trace before
    npt.assert_allclose(out[0], (1 - moved) * 1.0, atol=1e-6)
    npt.assert_allclose(out[1], (1 - moved) * 2.0 + moved * 1.0, atol=1e-5)
    npt.assert_allclose(out[3], (1 - moved) * 4.0 + moved * 3.0, atol=1e-5)
    # a streamer that does not move
    npt.assert_allclose(samples(stream(rows, dt) | operations.recmo(vb=0.0, dx=dx)), np.array(rows), atol=1e-6)
    with pytest.warns(UserWarning, match="tmax"):
        list(stream(rows, 0.5) | operations.recmo(vb=2.5, dx=1.0))


# ----------------------------------------------------------------------------------------------------------- simmat
def test_simmat_methods():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(40)
    b = rng.standard_normal(40)
    other = [make(b)]
    # a window of one sample: the product of the samples
    out = samples(stream([a]) | operations.simmat(other=other, mt=1))
    npt.assert_allclose(out, np.outer(a, b), atol=1e-5)
    # three samples, in the middle
    out = samples(stream([a]) | operations.simmat(other=other, mt=3))
    i, j = 10, 25
    npt.assert_allclose(out[i, j], sum(a[i + s] * b[j + s] for s in (-1, 0, 1)), atol=1e-5)
    # (and at the ends the window is shorter, and does not leave the traces)
    npt.assert_allclose(out[0, 5], a[0] * b[5], atol=1e-5)
    npt.assert_allclose(out[39, 5], a[39] * b[5], atol=1e-5)
    # the same trace: the vector angle is 0 and the correlation coefficient 1 on the diagonal of the interior
    same = [make(a)]
    angle = samples(stream([a]) | operations.simmat(other=same, mt=11, method=4, eps=0.0))
    npt.assert_allclose(np.diag(angle)[5:-5], 0.0, atol=0.5)
    cc = samples(stream([a]) | operations.simmat(other=same, mt=11, method=3))
    npt.assert_allclose(np.diag(cc)[5:-5], 1.0, atol=1e-4)
    norm = samples(stream([a]) | operations.simmat(other=same, mt=11, method=2, eps=0.0))
    npt.assert_allclose(np.diag(norm)[5:-5], 1.0, atol=1e-4)
    # one output trace for each sample of the trace
    assert len(list(stream([a, a]) | operations.simmat(other=same, mt=3))) == 80
    # a panel of the others
    both = samples(stream([a, b]) | operations.simmat(other=[make(b), make(a)], mt=1, panel=True))
    npt.assert_allclose(both[40:], np.outer(b, a), atol=1e-5)
    with pytest.raises(ValueError, match="Bad method"):
        operations.simmat(other=other, method=9)
    with pytest.raises(ValueError, match="5 points"):
        operations.simmat(other=other, mt=3, method=3)


# ------------------------------------------------------------------------------------------------------------ recip
def test_recip_sums_traces_of_the_same_absolute_offset():
    def trace(value, offset, cdp, **kw):
        return make(np.full(8, value), tx_loc=[100.0 - offset / 2, 0, 0], rx_loc=[100.0 + offset / 2, 0, 0],
                    ensemble_number=cdp, **kw)

    traces = [trace(1.0, 200.0, 5), trace(3.0, -200.0, 5), trace(10.0, 400.0, 5), trace(5.0, 200.0, 6)]
    out = list(from_iterable(traces) | stacking.recip())
    assert [t.header['ensemble_number'] for t in out] == [5, 5, 6]
    assert [t.header['fold'] for t in out] == [2, 1, 1]
    npt.assert_allclose(np.asarray(out[0]), 2.0)  # (1 + 3) / 2
    raw = list(from_iterable(traces) | stacking.recip(norm=False))
    npt.assert_allclose(np.asarray(raw[0]), 4.0)
    # the source is at the smaller x
    assert all(t.header['tx_loc'][0] <= t.header['rx_loc'][0] for t in out)


# ------------------------------------------------------------------------------------------------------- cvs4fowler
def test_cvs4fowler_has_its_largest_stack_at_the_velocity_of_the_event():
    dt, nt, v, t0 = 0.004, 500, 2000.0, 0.5
    offsets = np.arange(0, 1001, 100.0)
    traces = []
    for h in offsets:
        d = np.zeros(nt)
        t = np.sqrt(t0 ** 2 + (h / v) ** 2)
        d[int(round(t / dt))] = 1.0
        traces.append(make(d, dt, tx_loc=[-h / 2, 0, 0], rx_loc=[h / 2, 0, 0], ensemble_number=3))
    out = list(from_iterable(traces) | stacking.cvs4fowler(nvstack=45, vminstack=1500.0, dtout=dt, nonhyp=False, salias=1.0))
    assert len(out) == 45
    assert [t.header['ensemble_trace_number'] for t in out[:3]] == [1, 2, 3]
    assert out[0].d_sample == pytest.approx(dt) and out[0].header['ensemble_number'] == 3
    peaks = np.array([np.abs(np.asarray(t)).max() for t in out])
    # iv = vminstack^2 (nvstack - 5) / v^2 = 22.5
    assert 20 <= int(peaks.argmax()) <= 25
    with pytest.raises(ValueError, match="mxfold"):
        list(from_iterable(traces) | stacking.cvs4fowler(nvstack=10, mxfold=3, dtout=dt))
    with pytest.raises(ValueError, match="nvstack"):
        stacking.cvs4fowler(nvstack=3)


# ----------------------------------------------------------------------------------------------------------- mutecsv
def offset_trace(offset, cdp, nt=200, dt=0.004):
    return make(np.ones(nt), dt, tx_loc=[0.0, 0, 0], rx_loc=[offset, 0, 0], ensemble_number=cdp)


def test_mutecsv_for_one_cdp_is_mute():
    mc = windowing.mutecsv(cdp=[1], offs=[[0.0, 1000.0]], tims=[[0.1, 0.5]])
    mu = windowing.mute([0.0, 1000.0], [0.1, 0.5])
    traces = [offset_trace(o, 1) for o in (0.0, 250.0, 500.0, 1000.0, 2000.0)]
    npt.assert_array_equal(samples(from_iterable(traces) | mc), samples(from_iterable(traces) | mu))


def test_mutecsv_interpolates_between_cdps_and_can_extrapolate():
    stage = windowing.mutecsv(cdp=[10, 20], offs=[0.0, 1000.0], tims=[[0.1, 0.1], [0.3, 0.3]])
    muted = lambda cdp: int(np.argmax(np.asarray(next(iter(from_iterable([offset_trace(500.0, cdp)]) | stage)))))  # noqa: E731
    dt = 0.004
    assert muted(10) == pytest.approx(0.1 / dt, abs=1)
    assert muted(20) == pytest.approx(0.3 / dt, abs=1)
    assert muted(15) == pytest.approx(0.2 / dt, abs=1)
    # beyond the cdps the end function is held, or extrapolated
    assert muted(30) == pytest.approx(0.3 / dt, abs=1)
    extra = windowing.mutecsv(cdp=[10, 20], offs=[0.0, 1000.0], tims=[[0.1, 0.1], [0.3, 0.3]], extrapi=1)
    assert int(np.argmax(np.asarray(next(iter(from_iterable([offset_trace(500.0, 30)]) | extra))))) == pytest.approx(
        0.5 / dt, abs=1)
    # times over 100 are milliseconds
    ms = windowing.mutecsv(cdp=[1], offs=[0.0, 1000.0], tims=[[120.0, 120.0]])
    assert int(np.argmax(np.asarray(next(iter(from_iterable([offset_trace(500.0, 1)]) | ms))))) == pytest.approx(30, abs=1)
    with pytest.raises(ValueError, match="monotonically"):
        windowing.mutecsv(cdp=[1], offs=[[0.0, 0.0]], tims=[[0.1, 0.2]])
    with pytest.raises(ValueError, match="same number"):
        windowing.mutecsv(cdp=[1, 2], offs=[0.0, 1.0], tims=[[0.1, 0.2]])
    with pytest.raises(ValueError, match="mode"):
        windowing.mutecsv(cdp=[1], offs=[0.0], tims=[[0.1]], mode=7)


# ---------------------------------------------------------------------------------------------------------- windpoly
def xy_traces(points):
    return [make(np.zeros(4), ensemble_number=int(x), ensemble_trace_number=int(y)) for x, y in points]


def test_windpoly_polygon_modes():
    points = [(5, 5), (1, 1), (15, 5), (5, 15), (10, 10), (0, 0)]
    square = dict(x=[0, 10, 10, 0], y=[0, 0, 10, 10], xkey='ensemble_number', ykey='ensemble_trace_number')
    inside = list(from_iterable(xy_traces(points)) | windowing.windpoly(pass_=1, **square))
    assert [(t.header['ensemble_number'], t.header['ensemble_trace_number']) for t in inside] == [(5, 5), (1, 1), (10, 10),
                                                                                                    (0, 0)]
    outside = list(from_iterable(xy_traces(points)) | windowing.windpoly(pass_=2, **square))
    assert [(t.header['ensemble_number'], t.header['ensemble_trace_number']) for t in outside] == [(15, 5), (5, 15)]
    # several polygons for the exterior
    both = list(from_iterable(xy_traces(points)) | windowing.windpoly(
        pass_=2, x=[[0, 10, 10, 0], [20, 30, 30]], y=[[0, 0, 10, 10], [0, 0, 5]], xkey='ensemble_number',
        ykey='ensemble_trace_number'))
    assert len(both) == 2


def test_windpoly_polyline_mode_gives_the_position_along_and_across_the_line():
    line = dict(x=[0, 10], y=[5, 5], xkey='ensemble_number', ykey='ensemble_trace_number', dw=2.0)
    traces = xy_traces([(4, 4), (6, 6), (7, 10), (20, 5)])
    out = list(from_iterable(traces) | windowing.windpoly(pass_=0, **line))
    assert len(out) == 2  # (the third is too far, the last is beyond the end)
    # along the line, and across it: the right of the direction of the line (below a line along +x) is positive
    assert [t.header['iline'] for t in out] == [4, 6]
    assert [t.header['xline'] for t in out] == [1, -1]
    # a vertical line has the same sign
    vertical = dict(x=[10, 10], y=[0, 10], xkey='ensemble_number', ykey='ensemble_trace_number', dw=2.0)
    out = list(from_iterable(xy_traces([(11, 4), (9, 4)])) | windowing.windpoly(pass_=0, **vertical))
    assert [t.header['xline'] for t in out] == [1, -1]  # (the right of a line along +y is +x)
    with pytest.raises(ValueError, match="pass"):
        windowing.windpoly(pass_=3, **line)
    with pytest.raises(ValueError, match="one polygon"):
        windowing.windpoly(pass_=1, x=[[0, 1, 1], [2, 3, 3]], y=[[0, 0, 1], [0, 0, 1]], xkey='ensemble_number',
                           ykey='ensemble_trace_number')


# -------------------------------------------------------------------------------------------------------- padmissing
def keyed(values):
    return [make(np.full(4, float(v)), ensemble_number=int(v), ensemble_trace_number=int(v) + 100, fold=1) for v in values]


def test_padmissing_adds_dead_traces_for_the_missing_values():
    out = list(from_iterable(keyed([1, 2, 5, 6, 9])) | windowing.padmissing(key='ensemble_number'))
    assert [t.header['ensemble_number'] for t in out] == list(range(1, 10))
    dead = [t for t in out if t.header['fold'] == 0]
    assert [t.header['ensemble_number'] for t in dead] == [3, 4, 7, 8]
    assert all((np.asarray(t) == 0).all() for t in dead)
    # a step, the minimum and the maximum
    stepped = list(from_iterable(keyed([4, 10, 16])) | windowing.padmissing(key='ensemble_number', incval=3.0, minval=1, maxval=19))
    assert [t.header['ensemble_number'] for t in stepped] == [1, 4, 7, 10, 13, 16, 19]
    # (the padding before the first trace has its header, and after the last trace that of the last one)
    assert stepped[0].header['ensemble_trace_number'] == 104 and stepped[-1].header['ensemble_trace_number'] == 116
    # traces outside of the range are not output
    cut = list(from_iterable(keyed([1, 2, 3, 4, 5])) | windowing.padmissing(key='ensemble_number', minval=2, maxval=4))
    assert [t.header['ensemble_number'] for t in cut] == [2, 3, 4]
    with pytest.raises(ValueError, match="not increasing"):
        list(from_iterable(keyed([1, 3, 2])) | windowing.padmissing(key='ensemble_number'))
    with pytest.raises(ValueError, match="incval"):
        windowing.padmissing(key='ensemble_number', incval=0.0)


# ------------------------------------------------------------------------------------------------------------ braid
def test_braid_merges_ordered_streams():
    a = keyed([1, 3, 3, 7])
    b = keyed([2, 3, 8])
    out = list(windowing.braid(a, b, key='ensemble_number'))
    assert [t.header['ensemble_number'] for t in out] == [1, 2, 3, 3, 3, 7, 8]
    # equal keys: the traces of the first input come first
    assert [np.asarray(t)[0] for t in out][2:5] == [3.0, 3.0, 3.0]
    assert [t.header['trace_id'] for t in out] == list(range(1, 8))
    kept = list(windowing.braid(keyed([1, 2]), keyed([1.0]), key='ensemble_number', renumber=False))
    assert len(kept) == 3
    # decreasing order
    down = list(windowing.braid(keyed([9, 5, 1]), keyed([8, 2]), key='-ensemble_number'))
    assert [t.header['ensemble_number'] for t in down] == [9, 8, 5, 2, 1]
    # two keys
    traces_a = [make(np.zeros(2), ensemble_number=1, ensemble_trace_number=n) for n in (1, 4)]
    traces_b = [make(np.zeros(2), ensemble_number=1, ensemble_trace_number=n) for n in (2, 3)]
    both = list(windowing.braid(traces_a, traces_b, key=['ensemble_number', 'ensemble_trace_number']))
    assert [t.header['ensemble_trace_number'] for t in both] == [1, 2, 3, 4]
    with pytest.raises(ValueError, match="not in the order"):
        list(windowing.braid(keyed([1, 5, 2]), keyed([3]), key='ensemble_number'))
    with pytest.raises(ValueError, match="at least 2"):
        windowing.braid(keyed([1]))


# ------------------------------------------------------------------------------------------------------------ gprfb
def test_gprfb_removes_the_first_break_that_changes_in_scale_and_time():
    rng = np.random.default_rng(4)
    nt, nx, fbt = 200, 21, 60
    wavelet = np.sin(np.arange(30) * 0.4) * np.hanning(30)
    rows, truth = [], []
    for i in range(nx):
        shift = i % 3
        scale = 1.0 + 0.05 * i
        d = np.zeros(nt)
        d[10 + shift:40 + shift] += scale * wavelet
        reflection = np.zeros(nt)
        reflection[120:140] = 0.5 * np.hanning(20)
        rows.append(d + reflection + 0.005 * rng.standard_normal(nt))
        truth.append(reflection)
    report = []
    out = samples(stream(rows) | windowing.gprfb(nx=11, fbt=fbt, report=report))
    assert out.shape == (nx, nt) and len(report) == nx
    before = np.abs(np.array(rows)[:, :fbt]).max(axis=1)
    after = np.abs(out[:, :fbt]).max(axis=1)
    assert (after < 0.5 * before).all()
    # the reflection is not touched
    npt.assert_allclose(out[:, 120:140], np.array(truth)[:, 120:140], atol=0.05)
    with pytest.raises(ValueError, match="needed"):
        list(stream(rows[:5]) | windowing.gprfb(nx=11))


# --------------------------------------------------------------------------------------------------------- picking
def test_fbpickw_picks_the_onset():
    rng = np.random.default_rng(5)
    nt, onset = 400, 150
    rows = []
    for _ in range(5):
        d = 0.005 * rng.standard_normal(nt)
        d[onset:] += np.cos(np.arange(nt - onset) * 0.5) * 1.0
        rows.append(d)
    result = picking.fbpickw(stream(rows), window=0.04, characteristic=True)
    assert result.times.shape == (5,)
    # (the characteristic function is smoothed over the window, which is 10 samples: the picks are about the onset)
    npt.assert_allclose(result.times / 0.004, onset, atol=12)
    assert abs(np.median(result.times / 0.004) - onset) < 4
    assert len(result.characteristic) == 5 and result.characteristic[0].shape == (nt,)
    # a template of the first break: the search is around it
    templ = picking.fbpickw(stream(rows).__class__ and stream(rows), window=0.04, offsets=[0.0, 1000.0], times=[0.6, 0.6],
                            tdv=0.1)
    assert templ.times.shape == (5,)
    with pytest.raises(ValueError, match="window"):
        picking.fbpickw(stream(rows), window=0.0)


def test_fnzero_and_pickamp():
    rows = [np.zeros(20), np.zeros(20), np.zeros(20)]
    rows[0][5:9] = 1.0
    rows[1][2] = -3.0
    rows[1][15] = 0.5
    result = picking.fnzero(stream(rows, 0.002), min=0.1)
    npt.assert_allclose(result.first[:2], [5 * 0.002, 2 * 0.002])
    npt.assert_allclose(result.last[:2], [8 * 0.002, 15 * 0.002])
    assert np.isnan(result.first[2]) and np.isnan(result.last[2])
    assert picking.fnzero(stream(rows, 0.002), min=1.0).last[1] == pytest.approx(2 * 0.002)
    # pickamp: a window across the trace from t_above to wl
    data = [np.zeros(100) for _ in range(3)]
    data[0][30], data[1][50], data[2][70] = 2.0, -5.0, 1.0
    amps = picking.pickamp(stream(data, 0.004), d2=10.0, x_above=[0.0, 20.0], t_above=[0.1, 0.1], wl=0.2)
    assert amps.max.shape == (3,)
    assert amps.max[0] == pytest.approx(2.0, abs=0.05) and amps.tmax[0] == pytest.approx(30 * 0.004, abs=0.002)
    assert amps.min[1] == pytest.approx(-5.0, abs=0.05)
    assert amps.abs[1] == pytest.approx(5.0, abs=0.05) and amps.global_abs[1] == 1
    assert amps.global_min[0] == pytest.approx(-5.0, abs=0.05)
    npt.assert_allclose(amps.x, [0.0, 10.0, 20.0])
    npt.assert_allclose(amps.t_below - amps.t_above, 0.2, atol=1e-6)
    # a trace of all negative samples: the global maximum is the largest of them
    neg = picking.pickamp(stream([-np.ones(20), -2 * np.ones(20)], 0.004), d2=1.0)
    assert neg.global_max[0] == pytest.approx(-1.0, abs=1e-5) and neg.global_max[1] == 0
    with pytest.raises(ValueError, match="d2"):
        picking.pickamp(stream(data))


# ----------------------------------------------------------------------------------------------------------- velocity
def test_intvel_and_stkvel_are_inverse_of_each_other():
    v = np.array([2000.0, 2500.0, 3200.0, 4000.0])
    h = np.array([500.0, 600.0, 700.0, 800.0])
    model = velocity.stkvel(v, h)
    # the first layer has the interval velocity as its stacking velocity, t0 = 2 h / v
    assert model.velocity[0] == pytest.approx(2000.0)
    assert model.times[0] == pytest.approx(2 * 500.0 / 2000.0)
    back = velocity.intvel(model.velocity, model.times)
    npt.assert_allclose(back.thickness, h, rtol=1e-9)
    npt.assert_allclose(back.velocity, v, rtol=1e-9)
    npt.assert_allclose(back.times, model.times)
    # the documentation's example
    example = velocity.intvel([5000, 5523, 6339, 7264], [0.4, 0.8, 1.125, 1.425])
    assert example.thickness[0] == pytest.approx(0.5 * 5000 * 0.4)
    assert example.velocity[1] == pytest.approx(np.sqrt((5523 ** 2 * 0.4 - 5000 ** 2 * 0.2) / 0.2))
    # a dip makes the times shorter and the velocities larger
    dipping = velocity.stkvel(v, h, dip=30.0)
    npt.assert_allclose(dipping.times, model.times * np.cos(np.radians(30.0)))
    npt.assert_allclose(dipping.velocity, model.velocity / np.cos(np.radians(30.0)))
    with pytest.raises(ValueError, match="positive"):
        velocity.intvel([-1.0, 2.0], [0.1, 0.2])
    with pytest.raises(ValueError, match="increase"):
        velocity.intvel([1000.0, 2000.0], [0.2, 0.1])
    with pytest.raises(ValueError, match="same length"):
        velocity.stkvel([1000.0], [1.0, 2.0])
