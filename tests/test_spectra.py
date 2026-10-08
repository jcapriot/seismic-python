"""
SUSPECFX, SUSPECFK and SUSPECK1K2: amplitude spectra, checked with waves of known frequency and wavenumber.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.transforms import specfk, specfx, speck1k2

DT = 0.004
FOURIER = 2


def traces(data, dt=DT):
    return [Trace(np.asarray(row, dtype=np.float32), d_sample=dt) for row in data]


def run(stage, data, dt=DT):
    return list(from_iterable(traces(data, dt)) | stage)


def amplitudes(out):
    return np.array([np.asarray(t) for t in out])


# ------------------------------------------------------------------------------------------------------------ specfx
def test_specfx_finds_a_sine_at_its_frequency():
    nt = 250  # df = 1 Hz
    t = np.arange(nt) * DT
    (out,) = run(specfx(), [np.cos(2 * np.pi * 25.0 * t)])
    a = np.asarray(out)
    assert a.shape == (nt // 2 + 1,)
    assert np.argmax(a) == 25
    assert a[25] == pytest.approx(nt / 2, rel=1e-4)
    assert out.d_sample == pytest.approx(1.0 / (nt * DT))
    assert out.header['sample_start'] == 0.0 and out.header['sampling_domain'] == FOURIER


def test_specfx_halves_the_zero_frequency():
    (out,) = run(specfx(), [np.ones(100)])
    assert np.asarray(out)[0] == pytest.approx(50.0)  # (the sum is 100)


def test_specfx_makes_odd_traces_even_first():
    (out,) = run(specfx(), [np.random.default_rng(1).normal(size=101)])
    assert out.n_sample == 52 and out.d_sample == pytest.approx(1.0 / (102 * DT))


def test_specfx_uses_dt_for_traces_without_one():
    x = np.random.default_rng(1).normal(size=100)
    (out,) = run(specfx(dt=0.002), [x], dt=0.0)
    assert out.d_sample == pytest.approx(1.0 / (100 * 0.002))
    with pytest.warns(UserWarning):
        (out,) = run(specfx(), [x], dt=0.0)
    assert out.d_sample == pytest.approx(1.0 / (100 * 0.004))


def test_specfx_rejects_complex_traces():
    with pytest.raises(TypeError):
        list(from_iterable([Trace((np.ones(8) + 1j).astype(np.complex64), d_sample=DT)]) | specfx())


# ------------------------------------------------------------------------------------------------------------ specfk
def plane_wave(nx, nt, dx, f0, k0):
    t = np.arange(nt) * DT
    x = np.arange(nx) * dx
    return np.cos(2 * np.pi * (f0 * t[None, :] - k0 * x[:, None]))


def test_specfk_peak_is_at_the_frequency_and_wavenumber():
    nx, nt, dx = 16, 200, 10.0  # dk = 1 / (16 10), df = 1 / (200 .004) = 1.25 Hz
    f0, k0 = 20.0, 3.0 / (nx * dx)
    out = run(specfk(dx=dx), plane_wave(nx, nt, dx, f0, k0))
    a = amplitudes(out)
    assert a.shape == (nx, nt // 2 + 1)
    row, col = np.unravel_index(np.argmax(a), a.shape)
    # the wavenumber of trace i is -1 / (2 dx) + i / (nx dx)
    assert col == round(f0 / (1.25)) and row == nx // 2 + 3
    assert out[0].d_sample == pytest.approx(1.25) and out[0].header['sampling_domain'] == FOURIER
    assert [t.header['trace_id'] for t in out] == list(range(1, nx + 1))
    # the opposite direction of travel is at the opposite wavenumber
    back = amplitudes(run(specfk(dx=dx), plane_wave(nx, nt, dx, f0, -k0)))
    assert np.unravel_index(np.argmax(back), back.shape) == (nx // 2 - 3, col)


def test_specfk_of_a_flat_event_is_at_zero_wavenumber():
    a = amplitudes(run(specfk(dx=1.0), np.tile(np.random.default_rng(2).normal(size=64), (8, 1))))
    assert np.argmax(a.sum(axis=1)) == 4  # nx / 2


def test_specfk_warns_without_dx():
    with pytest.warns(UserWarning):
        run(specfk(), np.ones((4, 8)))


# -------------------------------------------------------------------------------------------------------- speck1k2
def test_speck1k2_finds_the_wavenumbers():
    n1, n2, d1, d2 = 32, 16, 5.0, 10.0
    k1, k2 = 4.0 / (n1 * d1), 3.0 / (n2 * d2)
    x1 = np.arange(n1) * d1
    x2 = np.arange(n2) * d2
    data = np.cos(2 * np.pi * (k1 * x1[None, :] + k2 * x2[:, None]))
    out = run(speck1k2(d1=d1, d2=d2), data, dt=0.0)
    a = amplitudes(out)
    assert a.shape == (n2, n1)
    first = out[0]
    f1, step1 = first.header['sample_start'], first.d_sample
    assert f1 == pytest.approx(-1.0 / (2 * d1)) and step1 == pytest.approx(1.0 / (n1 * d1))
    # the two peaks of a real wave are at the opposite wavenumbers
    peaks = sorted(zip(*np.unravel_index(np.argsort(a, axis=None)[-2:], a.shape)))
    step2, f2 = 1.0 / (n2 * d2), -1.0 / (2 * d2) + 1.0 / (n2 * d2)
    found = {(round(f2 + row * step2, 9), round(f1 + col * step1, 9)) for row, col in peaks}
    expected = {(round(s * k2, 9), round(s * k1, 9)) for s in (1, -1)}
    assert found == expected
    # the zero wavenumber of k1 is on a sample
    flat = amplitudes(run(speck1k2(d1=1.0, d2=1.0), np.ones((8, 16)), dt=0.0))
    assert np.argmax(flat.sum(axis=0)) == 8 and flat[:, 8].sum() > 0.99 * flat.sum()


def test_speck1k2_is_symmetric_through_the_origin():
    a = amplitudes(run(speck1k2(d1=1.0, d2=1.0), np.random.default_rng(3).normal(size=(12, 20)), dt=0.0))
    # the point k1 = k2 = 0 is at sample 10 (n1 / 2) of trace 5 (n2 / 2 - 1): the spectrum of real data is symmetric about it
    inner = a[:11, 1:]
    npt.assert_allclose(inner, inner[::-1, ::-1], rtol=1e-4, atol=1e-4 * a.max())


def test_speck1k2_uses_the_sample_interval_of_the_traces_for_d1():
    out = run(speck1k2(d2=1.0), np.ones((4, 8)), dt=0.5)
    assert out[0].d_sample == pytest.approx(1.0 / (8 * 0.5))


def test_the_panel_stages_check_their_input():
    with pytest.raises(ValueError):
        list(from_iterable([Trace(np.ones(8, dtype=np.float32), d_sample=DT), Trace(np.ones(9, dtype=np.float32), d_sample=DT)])
             | specfk(dx=1.0))
    assert list(from_iterable([]) | speck1k2()) == []
    assert specfk().parallelism == 'serial' and speck1k2().parallelism == 'serial' and specfx().parallelism == 'trace'
