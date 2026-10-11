"""The filters of a panel or gather in the Fourier domain: kfilter, k1k2filter, kfrac, smgauss2, lfaf, ccfilt, deghost,
fwatrim and svdfilt (seispy/filters/_panel.py)."""
import math

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.filters import (ccfilt, deghost, fwatrim, k1k2filter, kfilter, kfrac, lfaf, smgauss2, svdfilt)


def traces_of(array, dt=0.004, **header):
    return [Trace(row.astype(np.float32), d_sample=dt).replace(**header) for row in np.atleast_2d(array)]


def panel(array, dt=1.0, **header):
    return from_iterable(traces_of(array, dt, **header))


def samples(stream):
    return np.array([np.asarray(t) for t in stream])


def plane_wave(nx2, nx1, k1, k2=0.0, phase=0.0, width=None):
    """cos(2 pi (k1 x1 + k2 x2)), (nx2, nx1), unit sampling. With a width, a wave packet: a Gaussian envelope that is near 0
    at the edges, so that the padding of the transforms (which is not whole cycles) makes no edge effects"""
    x1 = np.arange(nx1)[None, :]
    x2 = np.arange(nx2)[:, None]
    wave = np.cos(2 * np.pi * (k1 * x1 + k2 * x2) + phase)
    if width is not None:
        wave = wave * np.exp(-(((x1 - nx1 / 2) / width) ** 2 + ((x2 - nx2 / 2) / width) ** 2))
    return wave


# ------------------------------------------------------------------------------------------------------- kfilter
def test_kfilter_passes_low_wavenumbers_and_stops_high_ones():
    low = plane_wave(64, 64, 0.06, 0.03, width=8.0)
    high = plane_wave(64, 64, 0.31, 0.28, width=8.0)
    k = [0.10, 0.15, 0.30, 0.40]
    amps = [1, 1, 0, 0]
    out_low = samples(panel(low) | kfilter(k=k, amps=amps))
    out_high = samples(panel(high) | kfilter(k=k, amps=amps))
    npt.assert_allclose(out_low, low, atol=2e-3)
    assert np.abs(out_high).max() < 2e-3


def test_kfilter_is_radial_not_a_box():
    # a tone at (k1, k2) = (.2, .2) has |k| = .28: inside a lowpass of .3 although each of k1 and k2 is over .15
    tone = plane_wave(64, 64, 0.2, 0.2, width=8.0)
    k = [0.30, 0.35, 0.40, 0.45]
    out = samples(panel(tone) | kfilter(k=k, amps=[1, 1, 0, 0]))
    npt.assert_allclose(out, tone, atol=2e-3)
    box = samples(panel(tone) | k1k2filter(k1=[0.3, 0.35, 0.4, 0.45], amps1=[1, 1, 0, 0], k2=[0.3, 0.35, 0.4, 0.45],
                                           amps2=[1, 1, 0, 0]))
    npt.assert_allclose(box, tone, atol=2e-3)  # (this one is under .3 in each: both pass it)
    stopped = samples(panel(tone) | kfilter(k=[0.1, 0.12, 0.15, 0.2], amps=[1, 1, 0, 0]))
    assert np.abs(stopped).max() < 2e-3


def test_kfilter_uses_the_sample_intervals():
    # with d1 = 2 the same tone is at half the wavenumber in cycles per unit: .05 instead of .1
    tone = plane_wave(64, 64, 0.125, 0.0, width=8.0)  # .125 cycles per sample = .0625 per unit with d1 = 2
    passed = samples(panel(tone, dt=1.0) | kfilter(k=[0.1, 0.12, 0.15, 0.2], amps=[1, 1, 0, 0], d1=2.0))
    npt.assert_allclose(passed, tone, atol=2e-3)
    stopped = samples(panel(tone, dt=1.0) | kfilter(k=[0.1, 0.12, 0.15, 0.2], amps=[1, 1, 0, 0], d1=0.5))
    assert np.abs(stopped).max() < 0.03


def test_kfilter_checks_its_input():
    with pytest.raises(ValueError, match="increasing"):
        kfilter(k=[0.3, 0.1], amps=[0, 1])
    with pytest.raises(ValueError, match="positive"):
        kfilter(k=[0.1, 0.2], amps=[-1, 1])
    with pytest.raises(ValueError, match="zero"):
        kfilter(k=[0.1, 0.2], amps=[0, 0])
    with pytest.raises(ValueError, match="must ="):
        kfilter(k=[0.1, 0.2, 0.3], amps=[0, 1])
    with pytest.raises(ValueError, match="same number"):
        list(from_iterable(traces_of(np.zeros((3, 8)))
                           + traces_of(np.zeros((1, 6)))) | kfilter())


# ----------------------------------------------------------------------------------------------------- k1k2filter
def test_k1k2filter_is_a_product_of_filters_in_each_dimension():
    both = plane_wave(64, 64, 0.09, 0.06, width=8.0)
    only_k1 = samples(panel(both) | k1k2filter(k1=[0.01, 0.02, 0.2, 0.3], amps1=[0, 1, 1, 0],
                                               k2=[0.01, 0.02, 0.4, 0.45], amps2=[0, 1, 1, 0]))
    assert np.abs(only_k1 - both).max() < 0.1 * np.abs(both).max()
    stop_k2 = samples(panel(both) | k1k2filter(k2=[0.01, 0.02, 0.03, 0.04], amps2=[1, 1, 0, 0]))
    assert np.abs(stop_k2).max() < 0.15 * np.abs(both).max()


def test_k1k2filter_design_matches_the_polygon_of_sufilter():
    # a flat filter of 1 is the identity (the same one that sufilter makes)
    rng = np.random.default_rng(1)
    data = rng.standard_normal((20, 30))
    out = samples(panel(data) | k1k2filter(k1=[0.1, 0.2, 0.3, 0.4], amps1=[1, 1, 1, 1], k2=[0.1, 0.2, 0.3, 0.4],
                                           amps2=[1, 1, 1, 1]))
    npt.assert_allclose(out, data, atol=1e-4)


def test_k1k2filter_quadrants():
    # a tone with k2 > 0 moves one way across the traces, and one with k2 < 0 the other way: quad keeps one of them
    up = plane_wave(64, 64, 0.125, 0.1, width=8.0)
    down = plane_wave(64, 64, 0.125, -0.1, width=8.0)
    wide = dict(k1=[0.3, 0.35, 0.4, 0.45], amps1=[1, 1, 1, 1], k2=[0.3, 0.35, 0.4, 0.45], amps2=[1, 1, 1, 1])
    kept = {}
    for name, tone in (('up', up), ('down', down)):
        for quad in (1, 2):
            out = samples(panel(tone) | k1k2filter(quad=quad, **wide))
            kept[name, quad] = np.abs(out).max()
    # each tone is kept (about half of the amplitude of its real part) by one of the quadrants and removed by the other
    assert (kept['up', 1] > 0.3) != (kept['up', 2] > 0.3)
    assert (kept['down', 1] > 0.3) != (kept['down', 2] > 0.3)
    assert (kept['up', 1] > 0.3) != (kept['down', 1] > 0.3)
    with pytest.raises(ValueError, match="quad"):
        k1k2filter(quad=3)


# --------------------------------------------------------------------------------------------------------- kfrac
def test_kfrac_with_no_power_and_no_phase_is_the_identity():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((16, 24))
    out = samples(panel(data) | kfrac())
    npt.assert_allclose(out, data, atol=1e-4)


def test_kfrac_first_derivative_of_a_tone():
    n = 64
    k1, width = 0.1, 9.0
    x1 = np.arange(n)[None, :]
    x2 = np.arange(n)[:, None]
    envelope = np.exp(-(((x1 - n / 2) / width) ** 2 + ((x2 - n / 2) / width) ** 2))
    tone = envelope * np.cos(2 * np.pi * k1 * x1)
    out = samples(panel(tone) | kfrac(power=1.0))
    # d/dx1 [envelope cos(2 pi k1 x1)]: the derivative of the envelope is -2 (x1 - n / 2) / width^2 times itself
    expected = envelope * (-2.0 * (x1 - n / 2) / width ** 2 * np.cos(2 * np.pi * k1 * x1)
                           - 2 * np.pi * k1 * np.sin(2 * np.pi * k1 * x1))
    # (the operator is |k|, not k1: not the derivative of the envelope, which has k2 in it, but near it)
    npt.assert_allclose(out, expected, atol=0.05 * np.abs(expected).max())
    # a half cycle of phase turns cos into -cos
    npt.assert_allclose(samples(panel(tone) | kfrac(phasefac=1.0)), -tone, atol=1e-4)
    # with the angle at 90 degrees the wavenumber has no k1 in it: there is not much of a derivative of this tone
    zero = samples(panel(tone) | kfrac(power=1.0, angle=90.0))
    assert np.abs(zero).max() < 0.5 * np.abs(out).max()


def test_kfrac_integration_does_not_blow_up_at_k_equals_0():
    rng = np.random.default_rng(3)
    out = samples(panel(rng.standard_normal((16, 16))) | kfrac(power=-1.0))
    assert np.isfinite(out).all()


# ----------------------------------------------------------------------------------------------------- smgauss2
def test_smgauss2_does_nothing_without_correlation_lengths_and_smooths_otherwise():
    rng = np.random.default_rng(4)
    v = 2000.0 + 300.0 * rng.random((24, 30))
    same = samples(panel(v) | smgauss2())
    npt.assert_allclose(same, v, rtol=1e-4)
    for slowness in (True, False):
        smooth = samples(panel(v) | smgauss2(a1=3.0, a2=3.0, slowness=slowness))
        assert smooth.std() < 0.6 * v.std()
        # the maximum is kept
        if slowness:
            assert (1.0 / smooth).max() == pytest.approx((1.0 / v).max(), rel=1e-4) or smooth.min() == pytest.approx(
                v.min(), rel=1e-3)
        else:
            assert smooth.max() == pytest.approx(v.max(), rel=1e-4)
    with pytest.raises(ValueError, match="positive"):
        list(panel(np.zeros((4, 8))) | smgauss2(a1=1.0))


# --------------------------------------------------------------------------------------------------------- lfaf
def gathers(ntr=12, nt=128, seed=5, signal=True):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * 0.004
    clean = np.sin(2 * np.pi * 10.0 * t) * np.exp(-((t - 0.25) / 0.1) ** 2)
    noise = rng.standard_normal((ntr, nt))
    return clean, noise


def test_lfaf_does_nothing_with_no_add_back_and_mixes_coherent_energy():
    clean, noise = gathers()
    data = clean[None, :] + 0.5 * noise
    same = samples(from_iterable(traces_of(data, ensemble_number=1)) | lfaf(adb=0.0))
    npt.assert_allclose(same, data, atol=1e-5)
    mixed = samples(from_iterable(traces_of(data, ensemble_number=1)) | lfaf(f1=1.0, f2=100.0, fr=5.0, vel=1500.0, dx=10.0))
    # the mix of 1500 / (10 f) traces: the noise is averaged down, the signal (the same on every trace) stays
    assert (mixed - clean).std() < 0.8 * (data - clean).std()
    # a triangular window does the same
    tri = samples(from_iterable(traces_of(data, ensemble_number=1)) | lfaf(f1=1.0, f2=100.0, vel=1500.0, tri=True))
    assert (tri - clean).std() < 0.85 * (data - clean).std()


def test_lfaf_goes_gather_by_gather():
    clean, noise = gathers(ntr=8)
    data = np.concatenate([clean[None, :] + noise, 5.0 * (clean[None, :] + noise)])
    out = list(from_iterable(traces_of(data[:8], ensemble_number=1) + traces_of(data[8:], ensemble_number=2)) | lfaf(vel=300.0))
    assert len(out) == 16
    first = samples(out[:8])
    second = samples(out[8:])
    npt.assert_allclose(second, 5.0 * first, rtol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="positive"):
        lfaf(vel=0.0)


# ------------------------------------------------------------------------------------------------------- ccfilt
def test_ccfilt_keeps_what_correlates_between_traces():
    clean, noise = gathers(ntr=10, seed=6)
    signal = np.tile(clean, (10, 1))
    data = signal + 0.8 * noise
    out = samples(from_iterable(traces_of(data, ensemble_number=1)) | ccfilt(ccl=0.8))
    # (the last trace is not compared with a next one, and is not filtered)
    assert (out[:-1] - signal[:-1]).std() < (data[:-1] - signal[:-1]).std()
    # nothing is removed with a range that has all of the correlation coefficients
    same = samples(from_iterable(traces_of(data, ensemble_number=1)) | ccfilt(ccl=-1.0, cch=1.0))
    npt.assert_allclose(same, data, atol=1e-4)
    with pytest.raises(ValueError, match="2 traces"):
        list(from_iterable(traces_of(data[:1], ensemble_number=1)) | ccfilt())


# ------------------------------------------------------------------------------------------------------ deghost
def test_deghost_ghost_is_a_delayed_reflection_of_the_surface():
    nt = 64
    spike = np.zeros(nt)
    spike[20] = 1.0
    # 2 h / v = 4 ms = 1 sample
    ghost = samples(panel(spike, dt=0.004) | deghost(h=3.0, v=1500.0, r=0.5, deghost=False, lambert=False))
    assert ghost[0, 20] == pytest.approx(1.0, abs=1e-3)
    assert ghost[0, 21] == pytest.approx(-0.5, abs=1e-3)
    assert ghost[0, 22] == pytest.approx(0.0, abs=1e-3)


def test_deghost_undoes_the_ghost():
    rng = np.random.default_rng(7)
    nt = 128
    data = np.zeros((3, nt))
    data[:, 5:60] = rng.standard_normal((3, 55))
    with_ghost = panel(data, dt=0.004) | deghost(h=6.0, v=1500.0, r=0.6, deghost=False)
    recovered = samples(with_ghost | deghost(h=6.0, v=1500.0, r=0.6, deghost=True))
    npt.assert_allclose(recovered, data, atol=2e-3)


def test_deghost_in_tau_p_has_an_angle_dependent_delay():
    nt = 128
    spike = np.zeros((2, nt))
    spike[:, 30] = 1.0
    # p = 0 and p = .4 / v: sqrt(1 - .16) smaller delay for the second trace
    out = samples(panel(spike, dt=0.004) | deghost(h=60.0, v=1500.0, r=0.5, deghost=False, lambert=False, p0=0.0, dp=0.4 / 1500.0))
    delay0 = 2 * 60.0 / 1500.0 / 0.004  # 20 samples
    delay1 = delay0 * math.sqrt(1 - 0.16)
    peak0 = np.argmin(out[0])
    peak1 = np.argmin(out[1])
    assert peak0 - 30 == round(delay0)
    assert peak1 - 30 == round(delay1)
    assert peak0 != peak1


def test_deghost_needs_a_depth():
    with pytest.raises(TypeError):
        deghost()


# ------------------------------------------------------------------------------------------------------ fwatrim
def test_fwatrim_is_a_mean_that_leaves_out_the_extremes():
    nt = 64
    rng = np.random.default_rng(8)
    base = rng.standard_normal(nt)
    traces = np.tile(base, (10, 1))
    traces[3] = 1000.0 * rng.standard_normal(nt)  # one that is far from the rest
    out = list(from_iterable(traces_of(traces, ensemble_number=1)) | fwatrim())
    assert len(out) == 1
    npt.assert_allclose(np.asarray(out[0]), base, atol=1e-3)
    # a gather makes one trace: two gathers, two traces, and the header of the one that is marked
    both = traces_of(traces[:5], ensemble_number=1) + traces_of(traces[5:], ensemble_number=2)
    two = list(from_iterable(both) | fwatrim(mark=2))
    assert len(two) == 2 and two[0].header['ensemble_number'] == 1 and two[1].header['ensemble_number'] == 2
    with pytest.raises(ValueError, match="outside"):
        list(from_iterable(both) | fwatrim(mark=9))
    with pytest.raises(ValueError, match="trim"):
        fwatrim(trim=60.0)


# ------------------------------------------------------------------------------------------------------- svdfilt
def test_svdfilt_removes_the_first_principal_plane():
    rng = np.random.default_rng(9)
    nt, ntr = 80, 12
    wavelet = np.sin(np.linspace(0, 6 * np.pi, nt)) * np.hanning(nt)
    flat = np.tile(wavelet, (ntr, 1))
    noise = 0.05 * rng.standard_normal((ntr, nt))
    data = flat + noise
    kept = samples(panel(data) | svdfilt(numpp=1, subtract=False))
    removed = samples(panel(data) | svdfilt(numpp=1, subtract=True))
    npt.assert_allclose(kept + removed, data, atol=1e-5)
    assert np.abs(kept - flat).max() < 0.1
    assert np.abs(removed).max() < 0.3 * np.abs(flat).max()


def test_svdfilt_windows_of_traces_are_independent():
    rng = np.random.default_rng(10)
    data = rng.standard_normal((9, 20))
    whole = samples(panel(data) | svdfilt(ntrw=3, numpp=1))
    for start in (0, 3, 6):
        block = samples(panel(data[start:start + 3]) | svdfilt(numpp=1))
        npt.assert_allclose(whole[start:start + 3], block, atol=1e-5)
    with pytest.raises(ValueError, match="at least 3"):
        svdfilt(ntrw=2)
    with pytest.raises(ValueError, match="larger"):
        list(panel(data) | svdfilt(ntrw=20))


def test_these_filters_work_with_no_traces():
    for stage in (kfilter(), k1k2filter(), kfrac(), svdfilt(), lfaf(), ccfilt(), fwatrim()):
        assert list(from_iterable([]) | stage) == []
