import pickle

import numpy as np
import numpy.testing as npt
import pytest

from seispy import attributes as at
from seispy.container import Trace, from_iterable
from seispy.convolution import acor, conv, xcor
from seispy.filters import frac, phase
from seispy.parallel import pmap
from seispy.transforms import hilb, zerophase

DT = 0.004


def traces(data, dt=DT, sample_start=0.0):
    return [
        Trace(np.asarray(row, dtype=np.float32), d_sample=dt, sample_start=sample_start)
        for row in np.atleast_2d(data)
    ]


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(11)
NT = 400
T = np.arange(NT) * DT
COS25 = np.cos(2 * np.pi * 25 * T).astype(np.float32)
SIN25 = np.sin(2 * np.pi * 25 * T).astype(np.float32)
INTERIOR = slice(50, -50)


# ---------------------------------------------------------------------------------------------------------- hilb
def test_hilb_of_a_cosine():
    (out,) = run(hilb(), traces(COS25))
    # (SU's Hilbert transform has the opposite sign to the usual one: H{cos} = -sin)
    npt.assert_allclose(np.asarray(out)[INTERIOR], -SIN25[INTERIOR], atol=4e-3)
    (back,) = run(hilb(), [out])
    npt.assert_allclose(np.asarray(back)[INTERIOR], -COS25[INTERIOR], atol=8e-3)


def test_hilb_is_linear_and_keeps_the_trace_shape():
    x = RNG.normal(size=(3, 120)).astype(np.float32)
    out = arr(run(hilb(), traces(x, sample_start=0.2)))
    assert out.shape == x.shape
    both = arr(run(hilb(), traces(x[0] + 2 * x[1])))[0]
    npt.assert_allclose(both, out[0] + 2 * out[1], atol=1e-5)
    # short traces, and none
    assert arr(run(hilb(), traces(np.ones((1, 7))))).shape == (1, 7)
    (empty,) = run(hilb(), [Trace(np.zeros(0, dtype=np.float32), d_sample=DT)])
    assert empty.n_sample == 0


def test_hilb_in_threads():
    x = RNG.normal(size=(12, 500)).astype(np.float32)
    trs = traces(x)
    expected = arr(run(hilb(), trs))
    got = arr(list(from_iterable(list(trs)) | pmap(hilb(), workers=4, chunk=2)))
    npt.assert_array_equal(got, expected)


# ------------------------------------------------------------------------------------------------------ zerophase
def causal_wavelet(n=256, start=10):
    t = np.arange(n - start) * DT
    w = np.zeros(n, dtype=np.float32)
    w[start:] = np.exp(-t / 0.03) * np.sin(2 * np.pi * 30 * t)
    return w


def test_zerophase():
    w = causal_wavelet()
    (out,) = run(zerophase(t0=0.1), traces(w))
    y = np.asarray(out)
    # the amplitude spectrum is the same
    npt.assert_allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(w)), rtol=1e-3, atol=1e-4 * np.abs(np.fft.rfft(w)).max())
    # and the wavelet is symmetric about sample t0 / dt = 25, which is time 0
    assert np.argmax(np.abs(y)) == 25
    for k in range(1, 15):
        assert y[25 + k] == pytest.approx(y[25 - k], abs=1e-4 * np.abs(y).max())
    assert out.header['sample_start'] == pytest.approx(-0.1)
    assert out.n_sample == 256


def test_zerophase_dt_and_errors():
    w = causal_wavelet()
    (a,) = run(zerophase(t0=0.1), traces(w))
    (b,) = run(zerophase(t0=0.1, dt=DT), traces(w, dt=0.0))
    npt.assert_array_equal(np.asarray(a), np.asarray(b))
    # (.004, if there is nothing at all)
    (c,) = run(zerophase(t0=0.1), traces(w, dt=0.0))
    npt.assert_array_equal(np.asarray(a), np.asarray(c))


# ---------------------------------------------------------------------------------------------------------- frac
NF, DF = 200, 0.005  # 1 second, so a 5 Hz sine fits exactly in a whole number of periods
TF = np.arange(NF) * DF
SINE = np.sin(2 * np.pi * 5 * TF).astype(np.float32)
COSINE = np.cos(2 * np.pi * 5 * TF).astype(np.float32)
OMEGA = 2 * np.pi * 5


def test_frac_differentiates_and_integrates():
    (d,) = run(frac(power=1), traces(SINE, dt=DF))
    npt.assert_allclose(np.asarray(d), OMEGA * COSINE, atol=2e-3)
    (i,) = run(frac(power=-1), traces(SINE, dt=DF))
    npt.assert_allclose(np.asarray(i), -COSINE / OMEGA, atol=1e-5)
    (second,) = run(frac(power=2), traces(SINE, dt=DF))
    npt.assert_allclose(np.asarray(second), -(OMEGA ** 2) * SINE, atol=5e-3)
    # the sign flips the derivative
    (flipped,) = run(frac(power=1, sign=1.0), traces(SINE, dt=DF))
    npt.assert_allclose(np.asarray(flipped), -OMEGA * COSINE, atol=2e-3)


def test_frac_phase_shift_only():
    (same,) = run(frac(), traces(COSINE, dt=DF))
    npt.assert_allclose(np.asarray(same), COSINE, atol=1e-5)
    (shifted,) = run(frac(phasefac=0.5), traces(COSINE, dt=DF))  # a quarter of a period
    npt.assert_allclose(np.asarray(shifted), SINE, atol=1e-5)
    (flipped,) = run(frac(phasefac=1.0), traces(COSINE, dt=DF))
    npt.assert_allclose(np.asarray(flipped), -COSINE, atol=1e-5)


def test_frac_fractional_power_composes():
    # half a derivative twice is a derivative
    twice = run(frac(power=0.5) | frac(power=0.5), traces(SINE, dt=DF))
    npt.assert_allclose(np.asarray(twice[0]), OMEGA * COSINE, atol=3e-3)


def test_frac_errors():
    with pytest.raises(ValueError, match="sample interval"):
        run(frac(power=1), traces(SINE, dt=0.0))
    (ok,) = run(frac(power=1, dt=DF), traces(SINE, dt=0.0))
    assert ok.n_sample == NF


# --------------------------------------------------------------------------------------------------------- phase
def test_phase_rotation():
    (same,) = run(phase(), traces(COSINE, dt=DF))
    npt.assert_allclose(np.asarray(same), COSINE, atol=1e-5)
    (ninety,) = run(phase(a=90.0), traces(COSINE, dt=DF))
    npt.assert_allclose(np.asarray(ninety), SINE, atol=1e-5)
    (flipped,) = run(phase(a=180.0), traces(COSINE, dt=DF))
    npt.assert_allclose(np.asarray(flipped), -COSINE, atol=1e-5)
    # a rotation is the same as the frac one
    (frac_out,) = run(frac(phasefac=0.5), traces(COSINE, dt=DF))
    npt.assert_allclose(np.asarray(ninety), np.asarray(frac_out), atol=1e-5)


def test_phase_linear_term_is_a_delay():
    # a signal with only frequencies in the middle, so a circular shift is exact
    spectrum = np.zeros(NF // 2 + 1, dtype=complex)
    spectrum[3:30] = RNG.normal(size=27) + 1j * RNG.normal(size=27)
    x = np.fft.irfft(spectrum, NF).astype(np.float32)
    (out,) = run(phase(c=2 * np.pi * 5 / NF), traces(x, dt=DF))
    npt.assert_allclose(np.asarray(out), np.roll(x, 5), atol=1e-5)


def test_phase_slope():
    # the phase is multiplied by b: -1 reverses it, which turns the signal around in time
    spectrum = np.zeros(NF // 2 + 1, dtype=complex)
    spectrum[3:30] = RNG.normal(size=27) + 1j * RNG.normal(size=27)
    x = np.fft.irfft(spectrum, NF).astype(np.float32)
    (out,) = run(phase(b=-180.0 / np.pi), traces(x, dt=DF))
    expected = np.fft.irfft(np.conj(spectrum), NF)
    npt.assert_allclose(np.asarray(out), expected, atol=1e-5)


# ------------------------------------------------------------------------------------------------------------ acor
def test_acor_symmetric():
    x = RNG.normal(size=60).astype(np.float32)
    (out,) = run(acor(ntout=11), traces(x))
    z = np.asarray(out)
    expected = np.array([np.sum(x[:60 - abs(k)] * x[abs(k):]) for k in range(-5, 6)])
    npt.assert_allclose(z, expected / expected[5], rtol=1e-5)
    assert z[5] == pytest.approx(1.0)
    npt.assert_allclose(z, z[::-1], rtol=1e-5)
    assert out.n_sample == 11 and out.header['sample_start'] == pytest.approx(-5 * DT)


def test_acor_one_sided_unnormalized():
    x = RNG.normal(size=40).astype(np.float32)
    (out,) = run(acor(ntout=6, sym=False, norm=False), traces(x))
    expected = np.array([np.sum(x[:40 - k] * x[k:]) for k in range(6)])
    npt.assert_allclose(np.asarray(out), expected, rtol=1e-5)
    assert out.header['sample_start'] == 0.0
    # more lags than there are samples is zeros
    (long,) = run(acor(ntout=21, sym=False), traces(x[:4]))
    npt.assert_array_equal(np.asarray(long)[4:], 0.0)
    # a trace of zeros has no maximum to divide by
    (zero,) = run(acor(), traces(np.zeros(30)))
    npt.assert_array_equal(np.asarray(zero), 0.0)
    with pytest.raises(ValueError):
        acor(ntout=0)


# ------------------------------------------------------------------------------------------------------------ conv
def test_conv_with_an_array():
    x = RNG.normal(size=(3, 30)).astype(np.float32)
    out = run(conv([1.0, 2.0, 1.0]), traces(x, sample_start=0.1))
    for row, o in zip(x, out):
        npt.assert_allclose(np.asarray(o), np.convolve(row, [1, 2, 1]), rtol=1e-5, atol=1e-6)
        assert o.n_sample == 32
        assert o.header['sample_start'] == pytest.approx(0.1)


def test_conv_with_a_trace_adds_its_start_time():
    wavelet = Trace(np.array([1.0, -1.0, 0.5], dtype=np.float32), d_sample=DT, sample_start=0.2)
    x = RNG.normal(size=(1, 20)).astype(np.float32)
    (out,) = run(conv(wavelet), traces(x, sample_start=0.1))
    npt.assert_allclose(np.asarray(out), np.convolve(x[0], [1, -1, 0.5]), rtol=1e-5, atol=1e-6)
    assert out.header['sample_start'] == pytest.approx(0.3)


def test_conv_panel_pairs_traces_with_filters():
    x = RNG.normal(size=(3, 12)).astype(np.float32)
    filters = [np.array([1.0, 1.0]), np.array([1.0, 0.0, -1.0]), traces(np.array([[2.0]]))[0]]
    out = run(conv(filters, panel=True), traces(x))
    for row, f, o in zip(x, filters, out):
        f = np.asarray(f)
        npt.assert_allclose(np.asarray(o), np.convolve(row, f), rtol=1e-5, atol=1e-6)
    assert [o.n_sample for o in out] == [13, 14, 12]
    with pytest.warns(UserWarning, match="fewer filters"):
        out = run(conv(filters[:1], panel=True), traces(x))
    npt.assert_array_equal(np.asarray(out[2]), x[2])
    assert conv(filters, panel=True).parallelism == 'serial'
    assert conv([1.0]).parallelism == 'trace'


def test_conv_errors():
    for bad in ([], [[1.0, 2.0]], 5.0 * np.ones((2, 2))):
        with pytest.raises(ValueError):
            conv(bad)


# ------------------------------------------------------------------------------------------------------------ xcor
def reference_xcor(x, y, first_lag, n_lags):
    """z[i] = sum_j x[j] y[i+j] for i = first_lag, ..."""
    z = np.zeros(n_lags)
    for k, i in enumerate(range(first_lag, first_lag + n_lags)):
        for j in range(len(x)):
            if 0 <= i + j < len(y):
                z[k] += x[j] * y[i + j]
    return z


def test_xcor_all_lags():
    x = RNG.normal(size=25).astype(np.float32)
    f = np.array([1.0, 2.0, -1.0, 0.5], dtype=np.float32)
    (out,) = run(xcor(f), traces(x))
    assert out.n_sample == 28
    npt.assert_allclose(np.asarray(out), reference_xcor(f, x, -3, 28), rtol=1e-5, atol=1e-6)
    assert out.header['sample_start'] == pytest.approx(-3 * DT)
    # the filter second
    (out2,) = run(xcor(f, first=False), traces(x))
    npt.assert_allclose(np.asarray(out2), reference_xcor(x, f, -24, 28), rtol=1e-5, atol=1e-6)
    assert out2.header['sample_start'] == pytest.approx(-24 * DT)
    # it is the same correlation, turned around
    npt.assert_allclose(np.asarray(out2), np.asarray(out)[::-1], rtol=1e-5, atol=1e-6)


def test_xcor_vibroseis_and_start_times():
    sweep = Trace(np.sin(np.arange(10) * 0.7).astype(np.float32), d_sample=DT, sample_start=0.0)
    x = RNG.normal(size=40).astype(np.float32)
    (out,) = run(xcor(sweep, vibroseis=15), traces(x))
    npt.assert_allclose(np.asarray(out), reference_xcor(np.asarray(sweep), x, 0, 15), rtol=1e-5, atol=1e-6)
    assert out.n_sample == 15 and out.header['sample_start'] == 0.0
    # the start times of the trace and the filter make a difference
    (late,) = run(xcor(sweep), traces(x, sample_start=0.1))
    assert late.header['sample_start'] == pytest.approx(-9 * DT + 0.1)


def test_acor_is_xcor_with_itself():
    x = RNG.normal(size=50).astype(np.float32)
    (full,) = run(xcor(x), traces(x))
    (a,) = run(acor(ntout=99, norm=False), traces(x))
    npt.assert_allclose(np.asarray(a), np.asarray(full), rtol=1e-5, atol=1e-6)


def test_xcor_errors():
    with pytest.raises(ValueError):
        xcor([])


# ------------------------------------------------------------------------------------------------------ attributes
def test_envelope_of_a_tone():
    (out,) = run(at.amp(), traces(COS25))
    npt.assert_allclose(np.asarray(out)[INTERIOR], 1.0, atol=5e-3)
    modulated = (np.exp(-2.0 * T) * COS25).astype(np.float32)
    (env,) = run(at.amp(), traces(modulated))
    npt.assert_allclose(np.asarray(env)[INTERIOR], np.exp(-2.0 * T)[INTERIOR], atol=5e-3, rtol=2e-2)


def test_phase_and_normalized_amplitude_of_a_tone():
    (ph,) = run(at.phase(), traces(COS25))
    p = np.asarray(ph)
    # (with SU's Hilbert transform the analytic trace of cos is cos - i sin, so the phase goes backwards)
    npt.assert_allclose(np.cos(p)[INTERIOR], COS25[INTERIOR], atol=5e-3)
    npt.assert_allclose(np.sin(p)[INTERIOR], -SIN25[INTERIOR], atol=5e-3)
    assert (np.abs(p) <= np.pi + 1e-6).all()
    (na,) = run(at.normamp(), traces(COS25))
    npt.assert_allclose(np.asarray(na)[INTERIOR], COS25[INTERIOR], atol=5e-3)
    # unwrapped, it just keeps growing: 2 pi per period
    (up,) = run(at.phase(unwrap=1), traces(COS25))
    u = np.asarray(up)
    assert (np.diff(u) >= -1e-6).all()
    npt.assert_allclose(np.diff(u)[100:300], 2 * np.pi * 25 * DT, atol=0.05)


def test_instantaneous_frequency():
    (out,) = run(at.freq(), traces(COS25))
    npt.assert_allclose(np.asarray(out)[INTERIOR], 25.0, atol=1.0)
    # a chirp: 10 Hz going up by 40 Hz per second
    chirp = np.cos(2 * np.pi * (10 * T + 20 * T ** 2)).astype(np.float32)
    (c,) = run(at.freq(), traces(chirp))
    expected = 10 + 40 * T
    npt.assert_allclose(np.asarray(c)[100:300], expected[100:300], atol=2.5)
    # nothing is above the Nyquist frequency
    assert np.asarray(c).max() <= 0.5 / DT + 1e-3


def test_envelope_derivatives_bandwidth_and_q():
    modulated = (np.exp(-2.0 * T) * COS25).astype(np.float32)
    (fd,) = run(at.fdenv(), traces(modulated))
    # the derivative of the envelope with respect to 2 pi t (SU has an extra 2 pi)
    d_env = np.gradient(np.asarray(run(at.amp(), traces(modulated))[0]), 2 * np.pi * DT)
    npt.assert_allclose(np.asarray(fd)[INTERIOR], d_env[INTERIOR], atol=2e-3)
    (sd,) = run(at.sdenv(), traces(modulated))
    dd = np.gradient(d_env, 2 * np.pi * DT)
    npt.assert_allclose(np.asarray(sd)[INTERIOR], dd[INTERIOR], atol=5e-3)
    # |d envelope / dt| / (2 pi envelope) = a / (2 pi) for exp(-a t). (The 61 point Hilbert transformer makes the
    # envelope ripple a little, so look at the average.)
    (bw,) = run(at.bandwidth(), traces(modulated))
    assert np.mean(np.asarray(bw)[INTERIOR]) == pytest.approx(2.0 / (2 * np.pi), rel=0.05)
    assert np.abs(np.asarray(bw)[INTERIOR] - 2.0 / (2 * np.pi)).max() < 0.06
    # Q = pi f / a
    (qq,) = run(at.q(), traces(modulated))
    assert np.median(np.asarray(qq)[100:300]) == pytest.approx(np.pi * 25 / 2.0, rel=0.1)
    assert np.asarray(qq).shape == (NT,)


def test_attribute_errors_and_properties():
    with pytest.raises(ValueError, match="sample interval"):
        run(at.freq(), traces(COS25, dt=0.0))
    with pytest.raises(ValueError, match="at least 2"):
        run(at.amp(), traces(np.ones((1, 1))))
    with pytest.raises(ValueError):
        at.freq(unwrap=-1)
    with pytest.raises(TypeError):
        at.amp(unwrap=1)
    assert repr(at.freq()) == "freq()" and repr(at.freq(unwrap=2)) == "freq(unwrap=2)"
    assert repr(at.phase(unwrap=0)) == "phase()" and repr(at.phase(unwrap=1)) == "phase(unwrap=1)"
    for name in at.__all__:
        assert getattr(at, name)().parallelism == 'trace'
        assert getattr(at, name).__doc__


def test_parallel_and_pickled():
    x = RNG.normal(size=(12, 300)).astype(np.float32)
    trs = traces(x)
    stages = [
        zerophase(t0=0.1), frac(power=0.5), phase(a=30.0), acor(ntout=21), conv([1.0, 2.0, 1.0]), xcor([1.0, -1.0]),
        at.amp(), at.phase(unwrap=1), at.freq(), at.q(),
    ]
    for stage in stages:
        assert stage.parallelism == 'trace'
        expected = arr(run(stage, trs))
        got = arr(list(from_iterable(list(trs)) | pmap(stage, workers=3, chunk=2)))
        npt.assert_array_equal(got, expected)
        again = pickle.loads(pickle.dumps(stage))
        assert repr(again) == repr(stage)
        npt.assert_array_equal(arr(run(again, trs)), expected)
