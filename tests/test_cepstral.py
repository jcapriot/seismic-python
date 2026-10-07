import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.transforms import (
    amp, cepstrum, clogfft, fft, icepstrum, iclogfft, ifft, imag, real, wfft,
)

DT = 0.004
N = 64


def traces(data, dt=DT):
    return [Trace(np.asarray(row, dtype=np.float32), d_sample=dt) for row in np.atleast_2d(data)]


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(31)
# (the DC and the Nyquist components are made positive: a real spectrum that is negative has a phase of pi there, which
# is not kept by the real transform)
DATA = (RNG.normal(size=(3, N)) + 3.0 + 2.0 * (-1.0) ** np.arange(N)).astype(np.float32)


# ------------------------------------------------------------------------------------------------------- clogfft
def test_clogfft_real_part_is_the_log_amplitude():
    (out,) = run(clogfft(unwrap=0), traces(DATA[:1]))
    spectrum = np.fft.rfft(DATA[0].astype(np.float64))
    assert out.dtype == np.complex64 and out.n_sample == N // 2 + 1
    npt.assert_allclose(np.asarray(out).real, np.log(np.abs(spectrum)), rtol=1e-4, atol=1e-4)
    # (the SU transform with sign 1 is the conjugate of numpy's, so is its phase)
    npt.assert_allclose(np.asarray(out).imag, -np.angle(spectrum), atol=1e-4)
    assert out.d_sample == pytest.approx(1.0 / (N * DT))
    assert out.header['sample_start'] == 0.0 and out.header['sampling_domain'] == 2


def test_clogfft_sign_conjugates_the_phase():
    (a,) = run(clogfft(unwrap=0, sign=1), traces(DATA[:1]))
    (b,) = run(clogfft(unwrap=0, sign=-1), traces(DATA[:1]))
    npt.assert_allclose(np.asarray(a).imag, -np.asarray(b).imag, atol=1e-5)


def test_clogfft_unwraps_the_phase_of_a_delay():
    delay = 5
    spike = np.zeros(N, dtype=np.float32)
    spike[delay] = 1.0
    (out,) = run(clogfft(trend=False), traces(spike))
    phase = np.asarray(out).imag
    assert np.abs(np.asarray(out).real).max() < 1e-5  # (the amplitude is one)
    # exp(+i w t0) in the SU kernel: the phase increases linearly, well past pi
    expected = 2 * np.pi * np.arange(N // 2 + 1) * delay / N
    assert phase[-1] > 10
    # (the derivative of the phase is a central difference, which makes the slope sin(x) / x too small, x = 2 pi 5 / 64)
    npt.assert_allclose(phase, expected, rtol=0.05, atol=0.05)
    assert np.all(np.diff(phase) > 0)
    # and with the trend removed (the default) the delay is taken out
    (flat,) = run(clogfft(), traces(spike))
    assert np.abs(np.asarray(flat).imag).max() < 0.5


def test_clogfft_simple_unwrap_has_no_jumps():
    delay = 5
    spike = np.zeros(N, dtype=np.float32)
    spike[delay] = 1.0
    (out,) = run(clogfft(mode='suphase', unwrap=1.0, trend=False), traces(spike))
    steps = np.abs(np.diff(np.asarray(out).imag))
    assert steps.max() < np.pi


def test_clogfft_odd_lengths_and_zero_traces():
    (out,) = run(clogfft(unwrap=0), traces(DATA[0, :63]))
    assert out.n_sample == 64 // 2 + 1  # (made even)
    (zero,) = run(clogfft(), traces(np.zeros(N)))
    assert not np.asarray(zero).any()


def test_clogfft_parameters_are_checked():
    with pytest.raises(ValueError):
        clogfft(sign=0)
    with pytest.raises(ValueError):
        clogfft(mode='other')
    with pytest.raises(ValueError):
        clogfft(unwrap=-1)


# ------------------------------------------------------------------------------------------------------ iclogfft
def test_iclogfft_undoes_clogfft():
    out = arr(run(clogfft(unwrap=0) | iclogfft(), traces(DATA)))
    npt.assert_allclose(out, DATA, rtol=1e-4, atol=1e-4)


def test_iclogfft_sym_centers_the_output():
    x = DATA[:1]
    (out,) = run(clogfft(unwrap=0) | iclogfft(sym=True), traces(x))
    npt.assert_allclose(np.asarray(out), np.roll(x[0], N // 2), rtol=1e-4, atol=1e-4)
    assert out.header['sample_start'] == pytest.approx(-N * DT / 2)


def test_iclogfft_needs_complex_traces():
    with pytest.raises(TypeError):
        run(iclogfft(), traces(DATA[:1]))


def test_iclogfft_sets_the_sample_interval_and_domain():
    (out,) = run(clogfft(unwrap=0) | iclogfft(), traces(DATA[:1]))
    assert out.d_sample == pytest.approx(DT) and out.header['sampling_domain'] == 1 and out.n_sample == N


# ------------------------------------------------------------------------------------------------- cepstrum
def test_cepstrum_is_the_inverse_transform_of_the_complex_log():
    (c,) = run(cepstrum(unwrap=0), traces(DATA[:1]))
    spectrum = np.fft.rfft(DATA[0].astype(np.float64))
    log_spectrum = np.log(np.abs(spectrum)) - 1j * np.angle(spectrum)  # (the SU kernel is the conjugate)
    expected = np.fft.irfft(np.conj(log_spectrum), N)
    assert c.n_sample == N and c.d_sample == pytest.approx(DT)
    npt.assert_allclose(np.asarray(c), expected, rtol=1e-3, atol=1e-4)


def test_cepstrum_of_a_spike_and_its_echo():
    # an echo shows up in the cepstrum at the time of the echo: x = d(t) + 0.5 d(t - 10 dt)
    x = np.zeros(N, dtype=np.float32)
    x[0], x[10] = 1.0, 0.5
    (c,) = run(cepstrum(unwrap=0), traces(x))
    values = np.asarray(c)
    assert np.argmax(np.abs(values[1:N // 2])) + 1 == 10


def test_icepstrum_undoes_cepstrum():
    out = arr(run(cepstrum(unwrap=0) | icepstrum(), traces(DATA)))
    npt.assert_allclose(out, DATA, rtol=1e-3, atol=1e-3)


def test_icepstrum_sym_and_header():
    (c,) = run(cepstrum(unwrap=0), traces(DATA[:1]))
    (out,) = run(icepstrum(sym=True), [c])
    npt.assert_allclose(np.asarray(out), np.roll(DATA[0], N // 2), rtol=1e-3, atol=1e-3)
    assert out.header['sample_start'] == pytest.approx(-N * DT / 2)


# ------------------------------------------------------------------------------------------------------ wfft
def test_wfft_with_one_zero_zero_is_flat():
    (out,) = run(wfft(w0=0.0, w1=1.0, w2=0.0), traces(DATA[:1]))
    npt.assert_allclose(np.abs(np.asarray(out)), 1.0, rtol=1e-5)
    # the phase is that of the transform (the SU kernel)
    npt.assert_allclose(np.angle(np.asarray(out)), -np.angle(np.fft.rfft(DATA[0].astype(np.float64))), atol=1e-4)
    assert out.header['sampling_domain'] == 2 and out.d_sample == pytest.approx(1 / (N * DT))


def test_wfft_weights_the_amplitudes_of_the_neighbours():
    (out,) = run(wfft(), traces(DATA[:1]))
    spectrum = np.conj(np.fft.rfft(DATA[0].astype(np.float64)))
    a = np.abs(spectrum)
    c = a.copy()
    c[1:] += 0.75 * a[:-1]
    c[:-1] += 0.75 * a[1:]
    c[0], c[-1] = a[0], a[-1]
    npt.assert_allclose(np.asarray(out), spectrum / c, rtol=1e-4, atol=1e-6)
    # an inverse transform gets a time trace out of it
    (back,) = run(wfft() | ifft(), traces(DATA[:1]))
    assert back.n_sample == N and back.dtype == np.float32


def test_wfft_zero_trace_and_checks():
    (out,) = run(wfft(), traces(np.zeros(N)))
    assert not np.asarray(out).any()
    with pytest.raises(ValueError):
        wfft(sign=2)
