"""
SUSLOWFT, SUSLOWIFT, SUPHASEVEL, SUCCWT and SURADON.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.transforms import ccwt, phasevel, radon, slowft, slowift

DT = 0.004
RNG = np.random.default_rng(41)


def trace(x, dt=DT, **header):
    t = Trace(np.asarray(x, dtype=np.float32), d_sample=dt)
    return t.replace(**header) if header else t


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def ricker(t, f=25.0):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


# ---------------------------------------------------------------------------------------------------- slowft
@pytest.mark.parametrize('n', [37, 60, 101])
def test_slowft_is_the_dft_without_padding(n):
    x = RNG.normal(size=n).astype(np.float32)
    (minus,) = run(slowft(sign=-1), [trace(x)])
    (plus,) = run(slowft(), [trace(x)])
    assert minus.n_sample == n and minus.dtype == np.complex64
    npt.assert_allclose(np.asarray(minus), np.fft.fft(x.astype(np.float64)), atol=1e-4 * np.sqrt(n))
    npt.assert_allclose(np.asarray(plus), np.conj(np.fft.fft(x.astype(np.float64))), atol=1e-4 * np.sqrt(n))
    assert minus.d_sample == pytest.approx(1.0 / (n * DT))
    assert minus.header['sampling_domain'] == 2 and minus.header['sample_start'] == 0.0


def test_slowift_inverts_slowft():
    x = RNG.normal(size=45).astype(np.float32)
    (back,) = run(slowft() | slowift(), [trace(x)])
    npt.assert_allclose(np.asarray(back), x, atol=1e-5)
    assert back.d_sample == pytest.approx(DT) and back.header['sampling_domain'] == 1
    with pytest.raises(TypeError, match="complex"):
        run(slowift(), [trace(x)])
    with pytest.raises(ValueError):
        slowft(sign=0)
    (no_dt,) = run(slowft(dt=0.002), [trace(x, dt=0.0)])
    assert no_dt.d_sample == pytest.approx(1.0 / (45 * 0.002))


# -------------------------------------------------------------------------------------------------- phasevel
def shot(v, n_x=40, nt=250, dx=10.0):
    t = np.arange(nt) * DT
    traces = []
    for ix in range(n_x):
        offset = (ix + 1) * dx
        traces.append(trace(ricker(t - 0.1 - offset / v, 20.0), offset=offset, ensemble_number=1))
    return traces


def test_phasevel_peaks_at_the_velocity_of_the_event():
    v0 = 1000.0
    out = run(phasevel(fv=500.0, nv=60, dv=25.0, fmax=40.0), shot(v0))
    assert len(out) == 60
    energy = np.array([np.abs(np.asarray(t)).sum() for t in out])
    best = np.argmax(energy)
    assert abs((500.0 + best * 25.0) - v0) <= 50.0
    assert [t.header['ensemble_trace_number'] for t in out] == list(range(1, 61))
    # complex frequencies, 0 to just under fmax, in the Fourier domain
    assert out[0].dtype == np.complex64
    n_fft = 252  # (the length that the program's FFT has for 250 samples)
    assert out[0].d_sample == pytest.approx(1.0 / (n_fft * DT))
    assert out[0].header['sampling_domain'] == 2
    assert out[0].n_sample == int(40.0 * n_fft * DT)


def test_phasevel_in_threads_gives_the_same_images():
    import concurrent.futures

    traces = shot(1100.0)
    expected = np.array([np.asarray(t) for t in run(phasevel(nv=12, fmax=40.0), traces)])
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: np.array([np.asarray(t) for t in run(phasevel(nv=12, fmax=40.0), traces)]),
                                range(16)))
    for got in results:
        npt.assert_array_equal(got, expected)


def test_phasevel_options_and_errors():
    traces = shot(1200.0)
    normalized = run(phasevel(nv=5, norm=True, fmax=0), traces)
    assert normalized[0].n_sample == 252 // 2 + 1
    with pytest.raises(ValueError, match="offsets"):
        run(phasevel(), [trace(np.ones(50), ensemble_number=1) for _ in range(3)])
    with pytest.raises(ValueError, match="sample interval"):
        run(phasevel(), [trace(np.ones(50), dt=0.0, offset=10.0)])
    with pytest.raises(ValueError):
        phasevel(nv=0)
    # one gather for each ensemble
    two = shot(1000.0, 10) + [t.replace(ensemble_number=2) for t in shot(1500.0, 10)]
    assert len(run(phasevel(nv=7), two)) == 14


# -------------------------------------------------------------------------------------------------------- ccwt
def test_ccwt_localizes_a_tone_in_scale():
    n = 480
    t = np.arange(n) * DT
    tone = np.cos(2 * np.pi * 25.0 * t)
    out = run(ccwt(), [trace(tone, trace_id=3)])
    assert len(out) == 50 and out[0].dtype == np.complex64 and out[0].n_sample == n
    assert {o.header['ensemble_number'] for o in out} == {3}
    assert [o.header['ensemble_trace_number'] for o in out] == list(range(1, 51))
    amplitude = np.array([np.abs(np.asarray(o))[120:360].mean() for o in out])  # (away from the ends)
    best = np.argmax(amplitude)
    # the scale k has the frequency fmax / 2 ** (k / nv): 25 Hz is at k = 10 log2(125 / 25) = 23.2
    assert abs(best - 23.2) <= 1.5
    # and the output is complex: a real part and a quadrature part of the tone
    row = np.asarray(out[best])[120:360]
    assert np.abs(row.imag).max() > 0.1 * np.abs(row).max()


def test_ccwt_linear_sampling_and_fourier_kernel():
    n = 400
    t = np.arange(n) * DT
    tone = np.cos(2 * np.pi * 20.0 * t)
    out = run(ccwt(fs=2, k=2, fmax=50.0, p=0.0), [trace(tone)])
    assert len(out) == 50
    # the Fourier kernel is a filter that passes the one frequency of its row (the scale of row k is 50 / k Hz, from 1)
    amplitude = np.array([np.abs(np.asarray(o)).mean() for o in out])
    assert np.argmax(amplitude) in (19, 20)  # (20 Hz)
    assert ccwt().parallelism == 'trace'


def test_ccwt_errors():
    with pytest.raises(ValueError):
        ccwt(noct=0)
    with pytest.raises(ValueError):
        ccwt(fs=3)
    with pytest.raises(TypeError):
        run(ccwt(), [Trace(np.ones(30, dtype=np.complex64), d_sample=DT)])
    with pytest.raises(ValueError, match="sample interval"):
        run(ccwt(), [trace(np.ones(30), dt=0.0)])


# ------------------------------------------------------------------------------------------------------- radon
OFFREF = 2000.0


def gather(events, n_x=40, nt=250):
    """events: (tau0 s, moveout in ms at 2000 m, amplitude): parabolic tau0 + q x^2"""
    t = np.arange(nt) * DT
    offsets = np.arange(n_x) * 50.0
    data = np.zeros((n_x, nt))
    for tau0, moveout, amp in events:
        q = moveout * 0.001 / OFFREF ** 2
        for ix, x in enumerate(offsets):
            data[ix] += amp * ricker(t - tau0 - q * x * x, 25.0)
    return [trace(row, offset=float(x), ensemble_number=1) for row, x in zip(data, offsets)], data


def test_radon_forward_puts_a_parabolic_event_at_its_moveout():
    traces, _ = gather([(0.4, 100.0, 1.0)])
    panel = run(radon(), traces)
    n_p = 1 + int((400.0 + 200.0) / 16.0)
    assert len(panel) == n_p and panel[0].n_sample == 250
    assert [p.header['ensemble_trace_number'] for p in panel] == list(range(1, n_p + 1))
    values = np.array([np.asarray(p) for p in panel])
    ip, it = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    assert abs(ip - (100.0 + 200.0) / 16.0) <= 1.5  # (the trace k is for -200 + 16 k ms)
    assert abs(it * DT - 0.4) <= 0.02
    # a flat event is at p = 0
    flat, _ = gather([(0.5, 0.0, 1.0)])
    values = np.array([np.asarray(p) for p in run(radon(), flat)])
    ip, it = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    assert abs(ip - 200.0 / 16.0) <= 1.5 and abs(it * DT - 0.5) <= 0.02


def test_radon_forward_and_back_gives_the_data():
    traces, data = gather([(0.4, 100.0, 1.0), (0.7, 20.0, -0.6)])
    out = np.array([np.asarray(t) for t in run(radon(choose=3, prewhite=0.01), traces)])
    assert out.shape == data.shape
    assert np.corrcoef(out.ravel(), data.ravel())[0, 1] > 0.97
    # the inverse of the panel of the forward transform is the same thing
    panel = run(radon(prewhite=0.01), traces)
    offsets = [t.header['offset'] for t in traces]
    back = np.array([np.asarray(t) for t in run(radon(choose=4, offsets=offsets), panel)])
    npt.assert_allclose(back, out, atol=2e-2 * np.abs(out).max())
    assert [t.header['offset'] for t in run(radon(choose=4, offsets=offsets), panel)] == pytest.approx(offsets)


def test_radon_removes_multiples():
    # a flat primary and a multiple that has a moveout of 300 ms at 2000 m (past pmulb = 200 ms)
    traces, data = gather([(0.5, 0.0, 1.0), (0.9, 300.0, 0.8)])
    cleaned = np.array([np.asarray(t) for t in run(radon(choose=1, prewhite=0.01), traces)])
    multiples = np.array([np.asarray(t) for t in run(radon(choose=2, prewhite=0.01), traces)])
    primary, multiple = gather([(0.5, 0.0, 1.0)])[1], gather([(0.9, 300.0, 0.8)])[1]
    # the multiple is mostly gone and the primary mostly kept
    assert np.linalg.norm(cleaned - primary) < 0.4 * np.linalg.norm(multiple)
    assert np.linalg.norm(multiples - multiple) < 0.4 * np.linalg.norm(multiple)
    # (choose=1 mutes the primaries from the transform before it estimates the multiples, so it is not data - choose 2)
    assert np.linalg.norm(cleaned - (data - multiples)) < 0.5 * np.linalg.norm(multiple)


def test_radon_windows_and_interpolation_and_errors():
    traces, data = gather([(0.4, 100.0, 1.0)])
    mutes = np.linspace(0.0, 400.0, len(traces))
    windowed = np.array([np.asarray(t) for t in run(radon(choose=3, nwin=3, muts=mutes, prewhite=0.01), traces)])
    assert np.corrcoef(windowed.ravel(), data.ravel())[0, 1] > 0.9
    with pytest.raises(ValueError, match="muts"):
        run(radon(nwin=3), traces)
    interpolated = run(radon(ninterp=1), traces)
    assert len(interpolated) == 1 + int(600.0 / 16.0)
    # a gather of one trace is passed as it is
    (one,) = run(radon(), traces[:1])
    npt.assert_array_equal(np.asarray(one), np.asarray(traces[0]))
    for bad in (dict(choose=7), dict(igopt=9), dict(dp=0.0), dict(choose=4), dict(f1=90.0, f2=80.0)):
        with pytest.raises(ValueError):
            radon(**bad)
    assert radon().parallelism == 'ensemble'
