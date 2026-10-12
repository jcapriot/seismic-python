"""
SUTXTAPER, SUPAD, SUMATRIX, SUDLMO and SUNMO_A.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.stretching import dlmo, nmo, nmo_a
from seispy.tapering import txtaper
from seispy.windowing import matrix, pad

DT = 0.004
RNG = np.random.default_rng(97)


def trace(x, dt=DT, **header):
    t = Trace(np.asarray(x, dtype=np.float32), d_sample=dt)
    return t.replace(**header) if header else t


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


def ricker(t, f=25.0):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


# -------------------------------------------------------------------------------------------------------- txtaper
def test_txtaper_tapers_the_edge_traces_by_count():
    out = arr(run(txtaper(tr1=4, tr2=2), [trace(np.ones(30)) for _ in range(10)]))
    w = out[:, 15]
    npt.assert_allclose(w[:5], [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6)
    npt.assert_allclose(w[5:8], 1.0)
    npt.assert_allclose(w[-3:], [1.0, 0.5, 0.0], atol=1e-6)
    # the symmetrical one
    sym = arr(run(txtaper(tr1=3), [trace(np.ones(10)) for _ in range(8)]))[:, 0]
    npt.assert_allclose(sym, [0, 1 / 3, 2 / 3, 1, 1, 2 / 3, 1 / 3, 0], atol=1e-6)
    assert txtaper(tr1=3).parallelism == 'serial' and txtaper(tbeg=10.0).parallelism == 'trace'


def test_txtaper_minimum_amplitude_and_other_types():
    out = arr(run(txtaper(tr1=2, low=0.2), [trace(np.ones(5)) for _ in range(6)]))[:, 0]
    npt.assert_allclose(out[[0, 1, 2, 5]], [0.2, 0.6, 1.0, 0.2], atol=1e-6)
    sine = arr(run(txtaper(tr1=4, type=2), [trace(np.ones(5)) for _ in range(9)]))[:4, 0]
    npt.assert_allclose(sine, np.sin(np.pi * np.arange(4) / 8), atol=1e-6)


def test_txtaper_by_the_value_of_a_key():
    offsets = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    trs = [trace(np.ones(4), offset=o) for o in offsets]
    out = arr(run(txtaper(key='offset', min=200.0, max=400.0, dx=200.0), trs))[:, 0]
    # 1 from 200 to 400, falling to 0 over 200 on either side
    npt.assert_allclose(out, [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0], atol=1e-6)
    assert txtaper(key='offset').parallelism == 'trace'


def test_txtaper_time_taper_ends_on_the_last_sample():
    out = arr(run(txtaper(tbeg=20.0, tend=20.0), [trace(np.ones(40))]))[0]
    # (20 ms is 6 samples: weights i / 6)
    npt.assert_allclose(out[:6], np.arange(6) / 6.0, atol=1e-6)
    npt.assert_allclose(out[-6:], (np.arange(6) / 6.0)[::-1], atol=1e-6)
    assert out[20] == 1.0
    with pytest.raises(ValueError, match="exceeds"):
        run(txtaper(tbeg=100.0, tend=100.0), [trace(np.ones(40))])
    with pytest.raises(ValueError):
        txtaper(low=2.0)
    with pytest.raises(ValueError):
        txtaper(type=7)


# ---------------------------------------------------------------------------------------------------------- pad
def numbered(panel, number, n=6):
    return trace(np.full(n, float(number)), ensemble_number=panel, ensemble_trace_number=number)


def test_pad_fills_in_the_missing_traces_of_each_panel():
    trs = [numbered(1, 1), numbered(1, 2), numbered(1, 4), numbered(2, 3), numbered(2, 4), numbered(2, 9)]
    out = run(pad(min=1, max=5), trs)
    keys = [(t.header['ensemble_number'], t.header['ensemble_trace_number']) for t in out]
    # (panel 1: 3 and 5 are padded; panel 2: 1 and 2 before, and 5 after; 9 is out of range, and lost)
    assert keys == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]
    dead = [t.header['trace_type'] == 2 for t in out]
    assert dead == [False, False, True, False, True, True, True, False, False, True]
    padded = [np.asarray(t) for t, d in zip(out, dead) if d]
    assert all(not p.any() and p.shape == (6,) for p in padded)
    real = [t for t, d in zip(out, dead) if not d]
    assert [float(np.asarray(t)[0]) for t in real] == [1.0, 2.0, 4.0, 3.0, 4.0]


def test_pad_other_keys_spacing_and_errors():
    trs = [trace(np.ones(3), ensemble_number=1, trace_id=k) for k in (10, 20, 50)]
    out = run(pad(min=10, max=60, key1='ensemble_number', key2='trace_id', key3='fold', val3=7, d=10), trs)
    assert [t.header['trace_id'] for t in out] == [10, 20, 30, 40, 50, 60]
    assert [t.header['fold'] for t in out] == [0, 0, 7, 7, 0, 7]
    # a key that goes down (the series starts at max)
    down = [numbered(1, k) for k in (5, 4, 2)]
    out = run(pad(min=1, max=5, d=-1), down)
    assert [t.header['ensemble_trace_number'] for t in out] == [5, 4, 3, 2, 1]
    with pytest.raises(ValueError):
        pad(min=1, max=5, d=0)
    with pytest.raises(ValueError):
        pad(min=5, max=1)


# -------------------------------------------------------------------------------------------------------- matrix
def grid(n_s=3, n_r=4, n=100):
    traces = []
    for s in range(n_s):
        for r in range(n_r):
            t = trace(np.full(n, 10.0 * (s + 1) + r + 1))
            traces.append(t.replace(tx_loc=[float(s), 0.0, 0.0], rx_loc=[float(r), 5.0, 0.0]))
    return traces


def test_matrix_puts_the_zone_of_each_source_in_a_trace_of_each_receiver():
    out = run(matrix(lenzone=100.0, numzone=3), grid())
    # (100 ms is 25 samples at 4 ms)
    assert len(out) == 4 and out[0].n_sample == 75
    for k, t in enumerate(out):
        x = np.asarray(t)
        npt.assert_allclose(x[0:25], 10.0 + k + 1)
        npt.assert_allclose(x[25:50], 20.0 + k + 1)
        npt.assert_allclose(x[50:75], 30.0 + k + 1)
        assert t.header['fold'] == 3
        assert t.header['rx_loc'][0] == k and t.header['tx_loc'][0] == 0.0


def test_matrix_sets_of_sources_dividers_and_filling():
    out = run(matrix(lenzone=40.0, numzone=2), grid(5, 2))
    # 5 sources in sets of 2: 3 sets of 2 receivers
    assert len(out) == 6
    assert [t.header['fold'] for t in out] == [2, 2, 2, 2, 1, 1]
    # reversed sources
    rev = run(matrix(lenzone=40.0, numzone=5, sdivider=[1.0, 1.0], skeyloc='source'), grid(5, 1))
    assert np.asarray(rev[0])[0] == 11.0
    flipped = run(matrix(lenzone=40.0, numzone=5, sdivider=[-1.0, 1.0]), grid(5, 1))
    assert np.asarray(flipped[0])[0] == 51.0
    # a receiver without traces in a set, between ones that have them: zeros (rfill=1), or nothing (rfill=0)
    # (the receiver 1 has no trace for the sources 2 and 3: in their set it is between the receivers that have)
    sparse = [t for t in grid(4, 3) if not (t.header['tx_loc'][0] >= 2.0 and t.header['rx_loc'][0] == 1.0)]
    assert len(run(matrix(lenzone=40.0, numzone=2), sparse)) == 6
    assert len(run(matrix(lenzone=40.0, numzone=2, rfill=0), sparse)) == 5
    # and rfill=2 makes the trace in a set where the receiver has no trace at the end
    ends = [t for t in grid(4, 3) if not (t.header['tx_loc'][0] >= 2.0 and t.header['rx_loc'][0] == 2.0)]
    assert len(run(matrix(lenzone=40.0, numzone=2), ends)) == 5
    assert len(run(matrix(lenzone=40.0, numzone=2, rfill=2), ends)) == 6
    with pytest.raises(ValueError):
        matrix(rfill=3)
    with pytest.raises(ValueError):
        matrix(numzone=0)
    with pytest.raises(ValueError):
        matrix(sdivider=[0.0, 1.0])


# ---------------------------------------------------------------------------------------------------------- dlmo
def test_dlmo_removes_a_linear_moveout_of_the_velocity_of_the_frequencies():
    n = 250
    t = np.arange(n) * DT
    out = []
    for offset in (0.0, 100.0, 300.0):
        shifted = ricker(t - 0.2 - offset / 500.0)
        out.append(trace(shifted, offset=offset))
    corrected = run(dlmo(vnmo=500.0), out)
    for c in corrected:
        x = np.asarray(c)
        assert np.argmax(x) == pytest.approx(0.2 / DT, abs=1)
    # the inverse puts it back
    back = run(dlmo(vnmo=500.0, invert=True), corrected)
    for b, o in zip(back, out):
        npt.assert_allclose(np.asarray(b), np.asarray(o), atol=2e-3)


def test_dlmo_velocity_that_depends_on_frequency_and_options():
    n = 512
    t = np.arange(n) * DT
    x = trace(ricker(t - 1.0), offset=200.0)
    slow = run(dlmo(vnmo=[300.0, 600.0], fnmo=[0.0, 50.0]), [x])[0]
    fast = run(dlmo(vnmo=600.0), [x])[0]
    # the two are different for the frequencies that are slower in the first
    assert np.abs(np.asarray(slow) - np.asarray(fast)).max() > 0.05
    # cm: offsets in centimeters
    in_m = run(dlmo(vnmo=500.0), [x])[0]
    in_cm = run(dlmo(vnmo=500.0, cm=True), [x.replace(offset=20000.0)])[0]
    npt.assert_allclose(np.asarray(in_cm), np.asarray(in_m), atol=1e-5)
    # dt overrides a trace without one
    (no_dt,) = run(dlmo(vnmo=500.0, dt=DT), [trace(ricker(t - 1.0), dt=0.0, offset=200.0)])
    npt.assert_allclose(np.asarray(no_dt), np.asarray(in_m), atol=1e-5)
    with pytest.raises(ValueError, match="sample interval"):
        run(dlmo(vnmo=500.0), [trace(np.ones(20), dt=0.0, offset=10.0)])
    with pytest.raises(ValueError, match="monotonically"):
        dlmo(vnmo=[300.0, 400.0], fnmo=[10.0, 5.0])
    # a velocity function for each cdp
    a = trace(ricker(t - 1.0), offset=200.0, ensemble_number=1)
    b = trace(ricker(t - 1.0), offset=200.0, ensemble_number=3)
    out = arr(run(dlmo(vnmo=[[300.0], [900.0]], fnmo=[[0.0], [0.0]], cdp=[1, 3]), [a, b]))
    assert abs(np.argmax(out[0]) - (1.0 - 200.0 / 300.0) / DT) <= 2
    assert abs(np.argmax(out[1]) - (1.0 - 200.0 / 900.0) / DT) <= 2


# -------------------------------------------------------------------------------------------------------- nmo_a
def hyperbola(t0, offset, v, anis1=0.0, anis2=0.0, n=300):
    t = np.arange(n) * DT
    arrival = np.sqrt(t0 ** 2 + offset ** 2 / v ** 2 + anis1 * offset ** 4 / (1.0 + anis2 * offset ** 2))
    return ricker(t - arrival, 30.0)


def test_nmo_a_without_anisotropy_is_nmo():
    traces = [trace(hyperbola(0.5, o, 2000.0), offset=o) for o in (200.0, 600.0, 1000.0)]
    plain = arr(run(nmo(vnmo=2000.0), traces))
    same = arr(run(nmo_a(vnmo=2000.0), traces))
    # (the two sets of tables are compiled apart: where the compiler fuses multiply-adds the times differ in the last bits)
    npt.assert_allclose(same, plain, atol=1e-4)


def test_nmo_a_flattens_the_quartic_moveout():
    # (the program's coefficients are for the offset in units of dt: x^2 / dt^2, so they are those of the documented
    # formula times dt^2)
    v, a1, a2 = 2000.0, 2e-14, 1e-7
    offsets = (500.0, 1000.0, 1500.0)
    traces = [trace(hyperbola(0.6, o, v, a1, a2), offset=o) for o in offsets]
    flat = arr(run(nmo_a(vnmo=v, anis1=a1 * DT ** 2, anis2=a2 * DT ** 2, smute=5.0), traces))
    plain = arr(run(nmo(vnmo=v, smute=5.0), traces))
    t0_sample = int(round(0.6 / DT))
    for row in flat:
        assert abs(np.argmax(row) - t0_sample) <= 1
    # the hyperbolic NMO leaves the far offset late
    assert abs(np.argmax(plain[-1]) - t0_sample) > 1


def test_nmo_a_errors_and_cdps():
    traces = [trace(hyperbola(0.3, 3000.0, 2000.0), offset=3000.0)]
    with pytest.raises(ValueError, match="negative moveout"):
        run(nmo_a(vnmo=2000.0, anis1=-1e-10), traces)
    with pytest.raises(ValueError, match="anis2"):
        run(nmo_a(vnmo=2000.0, anis1=1e-12, anis2=-1e-6), traces)
    with pytest.raises(ValueError):
        nmo_a(vnmo=[2000.0], tnmo=[0.0], cdp=[1, 2])
    out = run(nmo_a(vnmo=[[2000.0], [2500.0]], tnmo=[[0.0], [0.0]], anis1=[[1e-14], [2e-14]], anis2=[[0.0], [0.0]],
                    cdp=[1, 3]),
              [trace(hyperbola(0.5, 800.0, 2200.0), offset=800.0, ensemble_number=2)])
    assert out[0].n_sample == 300
