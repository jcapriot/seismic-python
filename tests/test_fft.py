"""seispy._fft: the one Fourier transform of seispy, SU's prime factor FFT."""
import concurrent.futures

import numpy as np
import numpy.testing as npt
import pytest

from seispy import _fft

RNG = np.random.default_rng(0)
# (products of 2, 3, 4, 5, 7, 8, 9, 11, 13 and 16 that are mutually prime: 64, 100 and 1000 are not)
REAL_LENGTHS = [2, 4, 6, 8, 16, 30, 60, 66, 104, 120, 208, 252, 504, 1008, 2002, 4004]
COMPLEX_LENGTHS = [1, 2, 3, 5, 7, 9, 11, 13, 15, 63, 104, 105, 252, 1001, 5040]


def tolerance(n):
    return 3e-6 * np.sqrt(n)


@pytest.mark.parametrize('n', REAL_LENGTHS)
def test_rfft_is_the_dft_of_numpy_with_the_opposite_default_sign(n):
    x = RNG.standard_normal(n).astype(np.float32)
    ref = np.fft.rfft(x.astype(np.float64))
    # sign -1 is numpy's kernel exp(-i w t); the default, +1, is SU's exp(+i w t): the conjugate
    npt.assert_allclose(_fft.rfft(x, sign=-1), ref, atol=tolerance(n), rtol=0)
    got = _fft.rfft(x)
    assert got.dtype == np.complex64 and got.shape == (n // 2 + 1,)
    npt.assert_allclose(got, np.conj(ref), atol=tolerance(n), rtol=0)


@pytest.mark.parametrize('n', REAL_LENGTHS)
def test_irfft_inverts_rfft_with_the_other_sign(n):
    x = RNG.standard_normal(n).astype(np.float32)
    for forward, inverse in ((1, -1), (-1, 1)):
        back = _fft.irfft(_fft.rfft(x, sign=forward), n, sign=inverse)
        assert back.dtype == np.float32 and back.shape == (n,)
        npt.assert_allclose(back, x, atol=tolerance(n), rtol=0)
    # unscaled: n times
    npt.assert_allclose(_fft.irfft(_fft.rfft(x), n, scale=False), n * x, atol=tolerance(n) * n, rtol=0)


@pytest.mark.parametrize('n', COMPLEX_LENGTHS)
def test_fft_and_ifft(n):
    x = (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(np.complex64)
    ref = np.fft.fft(x.astype(np.complex128))
    npt.assert_allclose(_fft.fft(x, sign=-1), ref, atol=tolerance(n), rtol=0)
    npt.assert_allclose(_fft.fft(x), np.conj(np.fft.fft(np.conj(x).astype(np.complex128))), atol=tolerance(n), rtol=0)
    npt.assert_allclose(_fft.ifft(_fft.fft(x)), x, atol=tolerance(n), rtol=0)
    npt.assert_allclose(_fft.ifft(_fft.fft(x, sign=-1), sign=1), x, atol=tolerance(n), rtol=0)


def test_axes_and_shapes():
    x = RNG.standard_normal((6, 40, 8)).astype(np.float32)
    # the real transform along each axis (40 and 8 are valid real lengths, 6 is too)
    for axis in (0, 1, 2):
        n = x.shape[axis]
        got = _fft.rfft(x, axis=axis, sign=-1)
        ref = np.fft.rfft(x.astype(np.float64), axis=axis)
        assert got.shape == ref.shape
        npt.assert_allclose(got, ref, atol=tolerance(n), rtol=0)
        back = _fft.irfft(got, n, axis=axis, sign=1)
        npt.assert_allclose(back, x, atol=tolerance(n), rtol=0)
    c = (RNG.standard_normal((5, 7, 9)) + 1j * RNG.standard_normal((5, 7, 9))).astype(np.complex64)
    for axis in (0, 1, 2):
        npt.assert_allclose(_fft.fft(c, axis=axis, sign=-1), np.fft.fft(c, axis=axis), atol=tolerance(9), rtol=0)
    # an axis that is not the last one, and the result has its axis in the same place
    assert _fft.rfft(x, axis=0).shape == (4, 40, 8)


def test_padding_and_cutting():
    x = RNG.standard_normal(100).astype(np.float32)
    padded = _fft.rfft(x, 120, sign=-1)
    npt.assert_allclose(padded, np.fft.rfft(x.astype(np.float64), 120), atol=tolerance(120), rtol=0)
    cut = _fft.rfft(x, 60, sign=-1)
    npt.assert_allclose(cut, np.fft.rfft(x[:60].astype(np.float64)), atol=tolerance(60), rtol=0)
    # the inverse takes the frequencies that it needs, zero padded
    few = _fft.irfft(padded[:20], 120, sign=1)
    full = np.zeros(61, dtype=np.complex64)
    full[:20] = padded[:20]
    npt.assert_allclose(few, _fft.irfft(full, 120, sign=1))
    c = RNG.standard_normal(30).astype(np.complex64)
    npt.assert_allclose(_fft.fft(c, 60, sign=-1), np.fft.fft(c.astype(np.complex128), 60), atol=tolerance(60), rtol=0)


def test_lengths():
    assert _fft.good_length(1000) == 1008 and _fft.good_length(1008) == 1008
    assert _fft.good_length(1009, real=False) == 1010 or _fft.good_length(1009, real=False) > 1009
    assert _fft.good_length_within(1001, 2002) >= 1001
    # real lengths are even, and half of them is valid
    for n in range(1, 400):
        m = _fft.good_length(n)
        assert m >= n and m % 2 == 0 and _fft.good_length(m // 2, real=False) == m // 2
    with pytest.raises(ValueError, match="not a valid length"):
        _fft.rfft(np.zeros(1000, dtype=np.float32))
    with pytest.raises(ValueError, match="not a valid length"):
        _fft.fft(np.zeros(17, dtype=np.complex64))
    with pytest.raises(ValueError, match="not a valid length"):
        _fft.rfft(np.zeros(9, dtype=np.float32))  # (odd)
    with pytest.raises(ValueError, match="from 1 to"):
        _fft.fft(np.zeros(2, dtype=np.complex64), n=800000)
    with pytest.raises(ValueError, match="at least 1"):
        _fft.good_length(0)


def test_arguments_are_checked():
    with pytest.raises(ValueError, match="sign"):
        _fft.rfft(np.zeros(8, dtype=np.float32), sign=0)
    with pytest.raises(TypeError, match="real"):
        _fft.rfft(np.zeros(8, dtype=np.complex64))
    with pytest.raises(ValueError, match="array"):
        _fft.rfft(np.float32(1.0))


def test_spikes_and_dc():
    n = 60
    spike = np.zeros(n, dtype=np.float32)
    spike[0] = 1.0
    npt.assert_allclose(_fft.rfft(spike), np.ones(n // 2 + 1), atol=1e-6)
    ones = np.ones(n, dtype=np.float32)
    f = _fft.rfft(ones)
    assert f[0] == pytest.approx(n)
    npt.assert_allclose(f[1:], 0.0, atol=1e-4)
    # a shifted spike: a phase that goes the way of the sign
    spike[:] = 0
    spike[1] = 1.0
    plus = _fft.rfft(spike, sign=1)
    npt.assert_allclose(plus, np.exp(2j * np.pi * np.arange(n // 2 + 1) / n), atol=1e-6)
    minus = _fft.rfft(spike, sign=-1)
    npt.assert_allclose(minus, np.exp(-2j * np.pi * np.arange(n // 2 + 1) / n), atol=1e-6)


def test_threads_get_the_same_answers():
    x = RNG.standard_normal((20, 1008)).astype(np.float32)
    expected = _fft.rfft(x)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for got in pool.map(lambda _: _fft.rfft(x), range(64)):
            npt.assert_array_equal(got, expected)


def test_inputs_are_not_changed_and_results_do_not_alias_them():
    c = (RNG.standard_normal((4, 60)) + 1j * RNG.standard_normal((4, 60))).astype(np.complex64)
    for x in (c, c.T.copy().T, c.astype(np.complex128), np.asfortranarray(c)):
        before = np.array(x, copy=True)
        out = _fft.fft(x)
        npt.assert_array_equal(x, before)
        assert not np.shares_memory(out, x)
        npt.assert_allclose(out, _fft.fft(c), atol=1e-6)
    r = RNG.standard_normal((6, 60)).astype(np.float32)
    f = _fft.rfft(r)
    before = f.copy()
    back = _fft.irfft(f, 60)
    npt.assert_array_equal(f, before)
    npt.assert_allclose(back, r, atol=1e-5)
    assert not np.shares_memory(back, f)
