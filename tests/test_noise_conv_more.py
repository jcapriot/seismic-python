import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.convolution import acorfrac, refcon
from seispy.noise import addflatnoise, addnoise, jitter

DT = 0.004


def traces(data, dt=DT, **per_trace):
    out = []
    for i, row in enumerate(np.atleast_2d(data)):
        t = Trace(np.asarray(row, dtype=np.float32), d_sample=dt)
        header = {k: (v[i] if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in per_trace.items()}
        out.append(t.replace(**header) if header else t)
    return out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(21)
DATA = RNG.normal(size=(6, 128)).astype(np.float32)


# -------------------------------------------------------------------------------------------------------- addnoise
@pytest.mark.parametrize('make', [addnoise, addflatnoise])
def test_signal_to_noise_ratio(make):
    sn = 10.0
    out = arr(run(make(sn=sn, seed=1), traces(DATA)))
    noise = out - DATA
    # the rms value of the signal's largest value over the noise rms is sn
    expected = np.abs(DATA).max() / np.sqrt(2.0) / sn
    assert np.sqrt((noise ** 2).mean()) == pytest.approx(expected, rel=1e-3)


def test_noise_is_repeatable_and_the_kinds_differ():
    a = arr(run(addnoise(seed=5), traces(DATA)))
    b = arr(run(addnoise(seed=5), traces(DATA)))
    c = arr(run(addnoise(seed=6), traces(DATA)))
    npt.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    # flat noise is bounded (uniform in [-1, 1] has an rms of sqrt(1/3), and is scaled to the rms that is wanted)
    flat = arr(run(addflatnoise(seed=1, sn=1.0), traces(DATA))) - DATA
    scale = np.abs(DATA).max() / np.sqrt(2.0) / np.sqrt(1.0 / 3.0)
    assert np.abs(flat).max() <= scale * 1.01
    gauss = arr(run(addnoise(seed=1, sn=1.0), traces(DATA))) - DATA
    assert np.abs(gauss).max() > 2.5 * gauss.std()


def test_noise_of_zero_signal_and_headers():
    out = run(addnoise(seed=1), traces(np.zeros((2, 20)), trace_id=[3, 4]))
    assert [t.header['trace_id'] for t in out] == [3, 4]
    assert np.asarray(out[0]).std() > 0  # (the scale is 1)


def test_bandlimited_noise():
    quiet = np.zeros((4, 256), dtype=np.float32)
    quiet[:, 0] = 1.0
    out = arr(run(addnoise(sn=1.0, seed=2, f=[10, 15, 30, 40], amps=[0, 1, 1, 0]), traces(quiet)))
    noise = out - quiet
    spectrum = np.abs(np.fft.rfft(noise, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(256, DT)
    assert spectrum[(freqs > 15) & (freqs < 30)].mean() > 10 * spectrum[freqs > 60].mean()


def test_noise_checks():
    with pytest.raises(ValueError):
        addnoise(sn=0)
    with pytest.raises(ValueError):
        run(addnoise(), traces(DATA[:1]) + traces(DATA[1:2, :50]))
    with pytest.raises(TypeError):
        run(addnoise(), [Trace((DATA[0] + 1j).astype(np.complex64), d_sample=DT)])
    assert run(addnoise(), []) == []


# --------------------------------------------------------------------------------------------------------- jitter
def test_jitter_shifts_by_whole_samples_with_zero_fill():
    x = np.arange(1, 21, dtype=np.float32)
    outs = arr(run(jitter(min=3, max=3, seed=1), traces(np.tile(x, (8, 1)))))
    for out in outs:
        if out[0] == 0:  # later: 3 zeros then the start of the trace
            npt.assert_array_equal(out, np.concatenate([np.zeros(3), x[:-3]]))
        else:  # earlier
            npt.assert_array_equal(out, np.concatenate([x[3:], np.zeros(3)]))
    assert {bool(o[0] == 0) for o in outs} == {True, False}  # (both directions, with 8 traces)


def test_jitter_positive_only_and_range():
    x = np.ones((50, 30), dtype=np.float32)
    outs = arr(run(jitter(min=1, max=4, pon=False, seed=3), traces(x)))
    shifts = (outs == 0).sum(axis=1)
    assert shifts.min() >= 1 and shifts.max() <= 3  # (int(1 + 3 u) is 1 to 3, the top is never reached)
    assert len(set(shifts)) > 1


def test_jitter_new_shift_only_when_the_key_changes():
    x = np.tile(np.arange(1, 31, dtype=np.float32), (8, 1))
    outs = arr(run(jitter(min=1, max=5, key='ensemble_number', seed=4), traces(x, ensemble_number=[1, 1, 1, 2, 2, 3, 3, 3])))
    zeros_front = (outs == 0).sum(axis=1)
    assert len(set(zeros_front[:3])) == 1 and len(set(zeros_front[3:5])) == 1 and len(set(zeros_front[5:])) == 1


def test_jitter_checks_and_big_shifts():
    with pytest.raises(ValueError):
        jitter(min=3, max=1)
    out = arr(run(jitter(min=50, max=50, pon=False), traces(np.ones((1, 10)))))
    assert not out.any()  # (shifted out of the trace)


# ------------------------------------------------------------------------------------------------------- acorfrac
def test_acorfrac_zero_is_the_identity():
    out = arr(run(acorfrac(), traces(DATA)))
    npt.assert_allclose(out, DATA, atol=1e-5)


def test_acorfrac_one_one_is_the_autocorrelation():
    x = DATA[:2, :40]
    out = arr(run(acorfrac(a=1.0, b=1.0, ntout=40), traces(x)))
    for trace, row in zip(x, out):
        full = np.correlate(trace, trace, mode='full')  # lags -39 to 39
        npt.assert_allclose(row, full[39:], rtol=1e-3, atol=1e-3)
    sym = arr(run(acorfrac(a=1.0, b=1.0, ntout=21, sym=True), traces(x)))
    full = np.correlate(x[0], x[0], mode='full')
    npt.assert_allclose(sym[0], full[39 - 10:39 + 11], rtol=1e-3, atol=1e-3)


def test_acorfrac_one_minus_one_is_the_autoconvolution():
    x = DATA[:1, :30]
    out = arr(run(acorfrac(a=1.0, b=-1.0, ntout=59), traces(x)))[0]
    npt.assert_allclose(out, np.convolve(x[0], x[0]), rtol=1e-3, atol=1e-3)


def test_acorfrac_phase_only_with_a_zero():
    # a = 0 and b = 1 takes the phase out: it is the zero phase wavelet with the amplitude spectrum of the trace
    x = DATA[:1, :32]
    out = arr(run(acorfrac(a=0.0, b=1.0, ntout=64), traces(x)))[0]
    spectrum = np.fft.rfft(out)
    original = np.fft.rfft(x[0], 64)
    npt.assert_allclose(np.abs(spectrum), np.abs(original), rtol=1e-3, atol=1e-3)
    npt.assert_allclose(spectrum.imag, 0.0, atol=1e-3)


def test_acorfrac_checks():
    with pytest.raises(ValueError):
        acorfrac(ntout=0)
    with pytest.raises(ValueError):
        run(acorfrac(ntout=1000), traces(DATA[:1]))


# --------------------------------------------------------------------------------------------------------- refcon
def test_refcon_convolves_with_the_offset_forward_trace():
    forward = traces(DATA[:, :20])
    reverse = traces(DATA[:, 20:50])
    out = run(refcon(forward, xy=2), reverse)
    assert len(out) == 4  # (the forward shot runs out: traces 2 to 5)
    for n, trace in enumerate(out):
        npt.assert_allclose(np.asarray(trace), np.convolve(DATA[n + 2, :20], DATA[n, 20:50]), rtol=1e-4, atol=1e-4)
        assert trace.n_sample == 20 + 30 - 1
        assert trace.d_sample == pytest.approx(DT / 2)


def test_refcon_checks():
    with pytest.raises(ValueError):
        refcon([], xy=-1)
    with pytest.raises(ValueError):
        run(refcon(traces(DATA[:2]), xy=5), traces(DATA))
