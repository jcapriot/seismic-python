import numpy as np
import numpy.testing as npt
import pytest

from seispy import waveforms
from seispy.synthetics import null, randspike

DT = 0.004


def only(source):
    (trace,) = list(source)
    return trace


# ------------------------------------------------------------------------------------------------------ waveforms
def test_ricker1_matches_its_formula_and_peaks_at_time_zero():
    fpeak = 15.0
    tr = only(waveforms.ricker1(fpeak=fpeak, dt=DT))
    t0 = 1 / fpeak
    t = np.arange(tr.n_sample) * DT
    expected = np.exp(-np.pi**2 * fpeak**2 * (t - t0) ** 2) * (1 - 2 * np.pi**2 * fpeak**2 * (t - t0) ** 2)
    npt.assert_allclose(np.asarray(tr), expected, atol=1e-5)
    assert tr.n_sample == int(2 / (fpeak * DT) + 0.5) + 1
    assert tr.header['sample_start'] == pytest.approx(-t0)
    # the peak is at time 0: sample_start + index * dt
    peak = int(np.argmax(np.asarray(tr)))
    assert tr.header['sample_start'] + peak * DT == pytest.approx(0.0, abs=DT)
    assert tr.header['trace_id'] == 1 and tr.header['trace_type'] == 1 and tr.d_sample == DT


def test_gauss_and_gaussd():
    fpeak = 25.0
    g = only(waveforms.gauss(fpeak=fpeak, dt=0.002))
    t0, s = 1 / fpeak, 1 / (np.sqrt(2) * np.pi * fpeak)
    t = np.arange(g.n_sample) * 0.002
    npt.assert_allclose(np.asarray(g), np.exp(-((t - t0) ** 2) / (2 * s * s)) / (s * np.sqrt(2 * np.pi)), rtol=1e-4, atol=1e-4)
    d = only(waveforms.gaussd(fpeak=fpeak, dt=0.002))
    # the first derivative of the gauss, from the formula
    expected = -(t - t0) / (s**3 * np.sqrt(2 * np.pi)) * np.exp(-((t - t0) ** 2) / (2 * s * s))
    npt.assert_allclose(np.asarray(d), expected, rtol=1e-4, atol=1e-2)
    assert d.header['sample_start'] == 0.0


def test_akb_berlage_unit_and_spike():
    akb = only(waveforms.akb(fpeak=30.0, dt=0.002))
    t = np.arange(akb.n_sample) * 0.002
    t0 = 1.8 / 30.0
    npt.assert_allclose(np.asarray(akb), -60 * (t - t0) * np.exp(-2 * 30.0**2 * (t - t0) ** 2), rtol=1e-4, atol=1e-3)
    assert akb.n_sample == int(4 / (30.0 * 0.002) + 0.5) + 1

    b = only(waveforms.berlage(fpeak=20.0, dt=0.002, tn=2.0, decay=40.0, ampl=3.0, ipa=-90.0))
    t = np.arange(b.n_sample) * 0.002
    npt.assert_allclose(np.asarray(b), 3.0 * t**2 * np.exp(-40.0 * t) * np.cos(2 * np.pi * 20.0 * t - np.pi / 2), rtol=1e-3, atol=1e-5)

    u = only(waveforms.unit(dt=DT, ns=7))
    npt.assert_array_equal(np.asarray(u), np.ones(7))

    s = only(waveforms.spike(tspike=0.02, dt=DT))
    assert s.n_sample == 5 + 5 and np.argmax(np.asarray(s)) == 5 and np.asarray(s).sum() == 1.0


def test_ricker2_is_symmetric_around_its_peak():
    tr = only(waveforms.ricker2(half=0.1, dt=0.004, ampl=2.0))
    w = np.asarray(tr)
    hlw = 25
    assert tr.n_sample == 2 * hlw - 1
    assert np.argmax(w) == hlw - 1 and w[hlw - 1] == pytest.approx(2.0)
    npt.assert_allclose(w, w[::-1], atol=1e-6)
    assert tr.header['sample_start'] == pytest.approx(-(hlw - 1) * 0.004)
    # with a distortion it is not symmetric and the start is not moved
    distorted = only(waveforms.ricker2(half=0.1, dt=0.004, distort=0.5))
    assert not np.allclose(np.asarray(distorted), np.asarray(distorted)[::-1])
    assert distorted.header['sample_start'] == 0.0
    # and a length can be asked for
    assert only(waveforms.ricker2(half=0.1, dt=0.004, ns=80)).n_sample == 80


def test_dgauss_is_the_nth_derivative_of_a_gaussian():
    n, fpeak = 2, 35.0
    tr = only(waveforms.dgauss(n=n, fpeak=fpeak))
    dt = tr.d_sample
    assert dt == pytest.approx(0.5 / (n * n * fpeak))
    t0 = np.sqrt(n) / fpeak
    assert tr.n_sample == int(2 * t0 / dt + 1)
    w = np.asarray(tr)
    # symmetric (even order derivative of a gaussian, about t0), and the sign flips with sign=-1
    assert np.argmax(np.abs(w)) == pytest.approx(tr.n_sample // 2, abs=1)
    flipped = only(waveforms.dgauss(n=n, fpeak=fpeak, sign=-1))
    npt.assert_allclose(np.asarray(flipped), -w, atol=1e-6)
    # n=1 is the first derivative: zero at t0, odd about it
    d1 = np.asarray(only(waveforms.dgauss(n=1, fpeak=fpeak)))
    assert d1[len(d1) // 2] == pytest.approx(0.0, abs=1e-3 * np.abs(d1).max())


@pytest.mark.parametrize('make, kwargs', [
    (waveforms.ricker1, dict(fpeak=0.0)),
    (waveforms.ricker1, dict(dt=-1.0)),
    (waveforms.akb, dict(ns=0)),
    (waveforms.berlage, dict(tn=-1.0)),
    (waveforms.spike, dict(tspike=-1.0)),
    (waveforms.ricker2, dict(half=-1.0)),
    (waveforms.dgauss, dict(n=0)),
    (waveforms.dgauss, dict(sign=2)),
])
def test_waveform_parameters_are_checked(make, kwargs):
    with pytest.raises(ValueError):
        make(**kwargs)


def test_waveform_pipes_into_stages():
    from seispy.convolution import conv
    from seispy.synthetics import spike

    out = list(spike(nt=64, ntr=3, spikes=[(0, 10)]) | conv(only(waveforms.ricker1(fpeak=20.0, dt=DT))))
    assert len(out) == 3


# ----------------------------------------------------------------------------------------------- null, randspike
def test_null_traces():
    trs = list(null(30, ntr=3, dt=0.002))
    assert len(trs) == 3
    assert all(t.n_sample == 30 and t.d_sample == 0.002 and not np.asarray(t).any() for t in trs)
    assert [t.header['trace_id'] for t in trs] == [1, 2, 3]
    assert len(list(null(10))) == 5
    with pytest.raises(ValueError):
        null(0)


def test_randspike_modes_and_seed():
    trs = list(randspike(n1=200, n2=6, nspk=10, amax=0.5, seed=3))
    assert len(trs) == 6 and all(t.n_sample == 200 for t in trs)
    data = np.array([np.asarray(t) for t in trs])
    assert (np.abs(data) <= 0.5).all()
    assert all(0 < (row != 0).sum() <= 10 for row in data)
    assert not np.array_equal(data[0], data[1])  # (mode 1: different on every trace)
    again = np.array([np.asarray(t) for t in randspike(n1=200, n2=6, nspk=10, amax=0.5, seed=3)])
    npt.assert_array_equal(data, again)

    same = np.array([np.asarray(t) for t in randspike(n1=200, n2=4, mode=2, seed=1)])
    assert all(np.array_equal(same[0], row) for row in same)
    assert [t.header['trace_id'] for t in trs] == list(range(1, 7))
    assert {t.header['ensemble_number'] for t in trs} == {1}


def test_randspike_never_writes_past_the_trace():
    for seed in range(20):
        for t in randspike(n1=5, n2=3, nspk=30, seed=seed):
            assert t.n_sample == 5
    with pytest.raises(ValueError):
        randspike(mode=3)


# ------------------------------------------------------------------------------------------------------- sweeps
def test_linear_sweep_is_a_chirp():
    tr = only(waveforms.vibro_linear(f1=10.0, f2=40.0, tv=2.0, dt=0.002, t1=0.0, t2=0.0))
    t = np.arange(1001) * 0.002
    npt.assert_allclose(np.asarray(tr), np.cos(2 * np.pi * (10.0 + 15.0 / 2 * t) * t), atol=1e-4)
    assert tr.n_sample == 1001 and tr.d_sample == 0.002 and tr.header['trace_type'] == 1
    # the instantaneous frequency (from the zero crossings) goes from 10 to 40 Hz
    x = np.asarray(tr)
    crossings = np.flatnonzero(np.diff(np.sign(x)) != 0)
    early = np.diff(crossings[:6]).mean() * 0.002 * 2   # period
    late = np.diff(crossings[-6:]).mean() * 0.002 * 2
    assert 1 / early == pytest.approx(10.0, rel=0.25) and 1 / late == pytest.approx(40.0, rel=0.1)


def test_sweep_phase_and_degrees():
    a = np.asarray(only(waveforms.vibro_linear(tv=1.0, dt=0.004, phz=np.pi / 2, t1=0.0, t2=0.0)))
    b = np.asarray(only(waveforms.vibro_linear(tv=1.0, dt=0.004, phz=90.0, radians=False, t1=0.0, t2=0.0)))
    npt.assert_allclose(a, b, atol=1e-5)
    assert a[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize('taper, kind', [(1, 'linear'), (2, 'sine'), (3, 'cosine'), (4, 'gaussian'), (5, 'gaussian2')])
def test_sweep_tapers(taper, kind):
    plain = np.asarray(only(waveforms.vibro_linear(tv=4.0, dt=0.004, t1=0.0, t2=0.0)))
    tapered = np.asarray(only(waveforms.vibro_linear(tv=4.0, dt=0.004, t1=0.5, t2=0.5, taper=taper)))
    envelope = tapered / np.where(plain == 0, np.nan, plain)
    n1 = int(np.float32(0.5) / np.float32(0.004) + np.float32(1))  # (the number of samples is found in single precision)
    f = np.arange(n1) / n1
    expected = {1: f, 2: np.sin(np.pi * f / 2), 3: 0.5 * (1 - np.cos(np.pi * f)),
                4: np.exp(-((3.8090232 * (1 - f)) ** 2)), 5: np.exp(-((2.0 * (1 - f)) ** 2))}[taper]
    good = ~np.isnan(envelope[:n1])
    npt.assert_allclose(envelope[:n1][good], expected[good], atol=2e-3)
    npt.assert_allclose(tapered[n1:-n1], plain[n1:-n1], atol=1e-6)  # (the middle is not touched)
    assert abs(tapered[-1]) <= abs(plain[-1]) + 1e-6
    assert waveforms.vibro_linear(tv=4.0, taper=kind)  # (named)


def test_segment_sweep_matches_a_linear_one_in_one_segment():
    seg = np.asarray(only(waveforms.vibro_segments(fseg=[10.0, 40.0], tseg=[0.0, 2.0], dt=0.004, t1=0.0, t2=0.0)))
    lin = np.asarray(only(waveforms.vibro_linear(f1=10.0, f2=40.0, tv=2.0, dt=0.004, t1=0.0, t2=0.0)))
    assert seg.shape == lin.shape
    npt.assert_allclose(seg[:-2], lin[:-2], atol=2e-2)


def test_segment_sweep_has_the_end_of_the_last_segment_as_its_length():
    tr = only(waveforms.vibro_segments(fseg=[10.0, 20.0, 50.0], tseg=[0.0, 1.0, 3.0], dt=0.004, t1=0.0, t2=0.0))
    assert tr.n_sample == int(3.0 / 0.004 + 1)
    with pytest.raises(ValueError):
        waveforms.vibro_segments(fseg=[10.0, 20.0], tseg=[0.0, 0.0])
    with pytest.raises(ValueError):
        waveforms.vibro_segments(fseg=[10.0], tseg=[0.0])


def test_nonlinear_sweeps_run_and_differ():
    base = dict(f1=10.0, f2=60.0, tv=2.0, dt=0.004, t1=0.0, t2=0.0)
    linear = np.asarray(only(waveforms.vibro_linear(**base)))
    for make, swconst in [(waveforms.vibro_octave, 3.0), (waveforms.vibro_hertz, 0.5), (waveforms.vibro_tpower, 2.0)]:
        x = np.asarray(only(make(swconst=swconst, **base)))
        assert x.shape == linear.shape and np.isfinite(x).all() and np.abs(x).max() <= 1.0 + 1e-6
        assert not np.allclose(x, linear, atol=1e-2)
    # no boost is the linear sweep
    npt.assert_allclose(np.asarray(only(waveforms.vibro_hertz(swconst=0.0, **base))), linear, atol=1e-6)
    # the t-power sweep with the power 1 is the linear sweep
    npt.assert_allclose(np.asarray(only(waveforms.vibro_tpower(swconst=1.0, **base))), linear, atol=1e-4)
    # the octave sweep with no boost, from the formula: power = 1, s = 2, k1 = f1, k2 = (f2 - f1) / tv
    s = 2.0 / 1.0
    k1, k2 = 10.0, (60.0 - 10.0) / 2.0
    t = np.arange(501) * 0.004
    npt.assert_allclose(np.asarray(only(waveforms.vibro_octave(swconst=0.0, **base))),
                        np.cos(2 * np.pi / (s * k2) * (k1 + k2 * t) ** s), atol=1e-3)


def test_sweep_parameters_are_checked():
    with pytest.raises(ValueError):
        waveforms.vibro_linear(t1=6.0, t2=6.0, tv=10.0)
    with pytest.raises(ValueError):
        waveforms.vibro_linear(taper=9)
    with pytest.raises(ValueError):
        waveforms.vibro_linear(dt=0.0)
    with pytest.raises(ValueError):
        waveforms.vibro_tpower(swconst=-2.0)
