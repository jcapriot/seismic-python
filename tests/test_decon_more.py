"""
SUCDDECON, SUPHIDECON and SUFXDECON.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy import _fft
from seispy.container import Trace, from_iterable
from seispy.convolution import conv
from seispy.decon import cddecon, fxdecon, phidecon

DT = 0.004
RNG = np.random.default_rng(23)


def traces(rows, dt=DT, **header):
    out = [Trace(np.asarray(r, dtype=np.float32), d_sample=dt) for r in np.atleast_2d(rows)]
    return [t.replace(**header) for t in out] if header else out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


# ------------------------------------------------------------------------------------------------------ cddecon
def test_cddecon_undoes_a_convolution():
    x = RNG.normal(size=100).astype(np.float32)
    f = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    (blurred,) = run(conv(f), traces(x))
    (out,) = run(cddecon(f, pnoise=0.0), [blurred])
    z = np.asarray(out)
    assert out.n_sample == 102
    npt.assert_allclose(z[:100], x, atol=1e-5)
    npt.assert_allclose(z[100:], 0.0, atol=1e-5)


def test_cddecon_noise_term_shrinks_the_result_and_is_the_same_for_every_trace():
    f = np.array([1.0, -0.4, 0.1], dtype=np.float32)
    x = RNG.normal(size=60).astype(np.float32)
    first = arr(run(cddecon(f, pnoise=0.1), traces(np.tile(x, (5, 1)))))
    # (the program adds up the noise term over the traces: every trace here is the same)
    npt.assert_allclose(first, np.tile(first[0], (5, 1)), atol=1e-6)
    quiet = arr(run(cddecon(f, pnoise=0.0), traces(x)))[0]
    assert np.abs(first[0]).sum() < np.abs(quiet).sum()


def test_cddecon_start_time_and_trace_filter():
    f = Trace(np.array([1.0, 0.3], dtype=np.float32), d_sample=DT, sample_start=0.1)
    x = RNG.normal(size=40).astype(np.float32)
    (blurred,) = run(conv(f), traces(x, sample_start=0.05))
    assert blurred.header['sample_start'] == pytest.approx(0.15)
    (out,) = run(cddecon(f, pnoise=0.0), [blurred])
    assert out.header['sample_start'] == pytest.approx(0.05)
    npt.assert_allclose(np.asarray(out)[:40], x, atol=1e-5)


def test_cddecon_sym_is_a_shift_by_half_the_transform():
    x = RNG.normal(size=50).astype(np.float32)
    n_fft = _fft.good_length_within(50, 250)
    padded = np.zeros(n_fft, dtype=np.float32)
    padded[:50] = x
    (out,) = run(cddecon([1.0], pnoise=0.0, sym=True), traces(x))
    npt.assert_allclose(np.asarray(out), np.roll(padded, n_fft // 2)[:50], atol=1e-5)


def test_cddecon_panel_and_errors():
    x = RNG.normal(size=(2, 30)).astype(np.float32)
    filters = [np.array([1.0, 0.5]), np.array([1.0, -0.5, 0.25])]
    out = arr(run(cddecon(filters, panel=True, pnoise=0.0), traces(x)))
    for row, f, o in zip(x, filters, out):
        (ref,) = run(cddecon(f, pnoise=0.0), traces(row))
        npt.assert_allclose(o, np.asarray(ref), atol=1e-6)
    with pytest.warns(UserWarning, match="fewer filters"):
        run(cddecon(filters[:1], panel=True), traces(x))
    with pytest.raises(ValueError):
        cddecon([1.0], pnoise=-1.0)
    assert cddecon([1.0]).parallelism == 'trace' and cddecon([[1.0]], panel=True).parallelism == 'serial'


# ---------------------------------------------------------------------------------------------------- phidecon
def wavelet():
    t = np.arange(40) * DT
    return (np.exp(-t / 0.02) * np.sin(2 * np.pi * 30 * t)).astype(np.float32)


def test_phidecon_finds_the_reflectivity_of_a_wavelet_trace():
    n = 500
    refl = np.zeros(n, dtype=np.float32)
    refl[100], refl[250] = 1.0, -0.5
    x = np.convolve(refl, wavelet())[:n]
    (out,) = run(phidecon(), traces(x))
    y = np.asarray(out)
    assert out.n_sample == n and np.isfinite(y).all()
    # the events are where the spikes are (the wavelet starts at zero and peaks at a sample or two), with their ratio
    first = 100 + np.argmax(np.abs(y[100:110]))
    second = 250 + np.argmax(np.abs(y[250:260]))
    assert first - 100 <= 3 and second - 250 <= 3
    assert y[second] / y[first] == pytest.approx(-0.5, rel=0.15)
    # and nothing else is as big
    away = np.ones(n, dtype=bool)
    away[95:115] = False
    away[245:265] = False
    assert np.abs(y[away]).max() < 0.5 * np.abs(y[first])


def test_phidecon_errors():
    with pytest.raises(ValueError, match="sample interval"):
        run(phidecon(), traces(np.ones(100), dt=0.0))
    (ok,) = run(phidecon(dt=DT), traces(np.random.default_rng(1).normal(size=100), dt=0.0))
    assert ok.n_sample == 100
    with pytest.raises(ValueError):
        run(phidecon(tm=0.5, tp=0.1), traces(np.ones(100)))
    with pytest.raises(ValueError):
        phidecon(pnoise=-1.0)


# ---------------------------------------------------------------------------------------------------- fxdecon
def events(nx=40, nt=300):
    tt = np.arange(nt) * DT
    clean = np.zeros((nx, nt), dtype=np.float32)
    for t0, slope in [(0.3, 0.001), (0.7, -0.0008), (0.9, 0.0005)]:
        for ix in range(nx):
            tau = tt - (t0 + slope * ix * 10)
            clean[ix] += np.exp(-(tau / 0.015) ** 2) * np.cos(2 * np.pi * 25 * tau)
    return clean


def test_fxdecon_attenuates_random_noise():
    clean = events()
    noisy = clean + 0.5 * RNG.normal(size=clean.shape).astype(np.float32)
    out = arr(run(fxdecon(), traces(noisy)))
    assert out.shape == noisy.shape and np.isfinite(out).all()
    rms = lambda a: np.sqrt((a ** 2).mean())
    assert rms(out - clean) < 0.5 * rms(noisy - clean)


def test_fxdecon_zeros_few_traces_and_windows():
    assert not arr(run(fxdecon(), traces(np.zeros((12, 100))))).any()
    # fewer traces than the window: the program gives zeros, this does not
    clean = events(6)
    out = arr(run(fxdecon(), traces(clean)))
    assert np.abs(out).max() > 0.1
    # more than one time window
    noisy = events() + 0.3 * RNG.normal(size=(40, 300)).astype(np.float32)
    windows = arr(run(fxdecon(twlen=0.6, taper=0.04), traces(noisy)))
    assert windows.shape == noisy.shape and np.isfinite(windows).all()
    assert np.abs(windows).max() > 0.1


def test_fxdecon_parameters_and_headers():
    with pytest.raises(ValueError):
        fxdecon(ntrw=-1)
    with pytest.raises(TypeError):
        run(fxdecon(), [Trace(np.ones(50, dtype=np.complex64), d_sample=DT)])
    with pytest.raises(ValueError, match="same number"):
        run(fxdecon(), traces(np.ones(50)) + traces(np.ones(60)))
    trs = traces(events(12), trace_id=7)
    out = run(fxdecon(ntrw=6, ntrf=2), trs)
    assert len(out) == 12 and all(t.header['trace_id'] == 7 for t in out)
    assert fxdecon().parallelism == 'serial'
