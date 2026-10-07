import numpy as np
import numpy.testing as npt
import pytest

from seispy.amplitudes import gain
from seispy.container import TraceCollection, Trace
from seispy.parallel import pmap

DT = 0.004
NT = 64
RNG = np.random.default_rng(1234)
DATA = RNG.normal(size=(5, NT)).astype(np.float32)
T = np.arange(NT) * DT


def run(stage, data=DATA, dt=DT):
    coll = TraceCollection(data, d_sample=dt)
    return np.array([np.asarray(t) for t in coll | stage])


def check(stage, expected, **kw):
    npt.assert_allclose(run(stage), expected, rtol=kw.get('rtol', 2e-5), atol=kw.get('atol', 1e-6))


def test_no_op():
    npt.assert_array_equal(run(gain()), DATA)


def test_tpow():
    fac = T ** 2.5
    fac[0] = 0.0  # as SU, so that a negative tpow does not blow up
    check(gain(tpow=2.5), DATA * fac)


def test_tpow_with_start_time():
    tmin = 0.1
    coll = TraceCollection(DATA, d_sample=DT, sample_start=tmin)
    out = np.array([np.asarray(t) for t in coll | gain(tpow=1.0)])
    t = tmin + np.arange(NT) * DT
    npt.assert_allclose(out, DATA * t, rtol=2e-5)


def test_epow():
    check(gain(epow=3.0), DATA * np.exp(3.0 * T))
    check(gain(epow=-1.0, etpow=2.0), DATA * np.exp(-1.0 * T ** 2))


@pytest.mark.parametrize('gpow', [0.5, 2.0, 3.0, 0.3])
def test_gpow(gpow):
    if gpow == 2.0:
        expected = DATA * np.abs(DATA)
    else:
        expected = np.sign(DATA) * np.abs(DATA) ** gpow
    check(gain(gpow=gpow), expected)


def test_bias_scale_norm():
    check(gain(bias=0.5), DATA + 0.5)
    check(gain(scale=3.0), DATA * 3.0)
    check(gain(scale=3.0, norm=2.0), DATA * 1.5)
    # the bias goes in before the power, and the scale comes at the very end
    check(gain(bias=1.0, tpow=1.0, scale=2.0), 2.0 * (DATA + 1.0) * np.where(T == 0, 0, T))


def test_clipping():
    check(gain(clip=0.5), np.clip(DATA, -0.5, 0.5))
    check(gain(trap=0.5), np.where(np.abs(DATA) > 0.5, 0.0, DATA))
    check(gain(pclip=0.5), np.minimum(DATA, 0.5))
    check(gain(nclip=-0.5), np.maximum(DATA, -0.5))


def test_quantile_clip_and_balance():
    q = 0.9
    iq = int(q * NT - 0.5)
    clip_vals = np.sort(np.abs(DATA), axis=1)[:, iq][:, None]
    check(gain(qclip=q), np.clip(DATA, -clip_vals, clip_vals))
    # qbal=1 with qclip=1 balances by the maximum magnitude
    check(gain(qbal=True), DATA / np.abs(DATA).max(axis=1, keepdims=True))
    check(gain(qbal=True, qclip=q), np.clip(DATA / clip_vals, -1, 1))


def test_balances():
    rms = np.sqrt((DATA.astype(np.float64) ** 2).mean(axis=1, keepdims=True))
    check(gain(pbal=True), DATA / rms)
    check(gain(mbal=True), DATA - DATA.mean(axis=1, keepdims=True), atol=1e-6)
    check(gain(maxbal=True), DATA - DATA.max(axis=1, keepdims=True))


def test_jon():
    explicit = run(gain(tpow=2.0, gpow=0.5, qclip=0.95))
    npt.assert_array_equal(run(gain(jon=True)), explicit)


def test_vred():
    # the synthetics have an offset (the offset can't be set from a plain array)
    from seispy.synthetics import spike
    coll = spike(nt=NT, ntr=5, dt=DT, offset=300.0, spikes=[(i, 10) for i in range(5)]).to_memory()
    vred = 1500.0
    out = np.array([np.asarray(t) for t in coll | gain(tpow=1.0, vred=vred)])
    tred = 300.0 / vred
    fac = np.where(np.arange(NT) == 0, 0.0, tred + np.arange(NT) * DT)
    expected = np.array([np.asarray(t) for t in coll]) * fac
    npt.assert_allclose(out, expected, rtol=2e-5)


def agc_reference(x, iwagc):
    nt = len(x)
    d2 = x.astype(np.float64) ** 2
    out = np.zeros(nt)

    def g(i, mean_sq):
        return 0.0 if mean_sq == 0 else x[i] / np.sqrt(mean_sq)

    total = d2[:iwagc].sum()
    nwin = iwagc
    out[0] = g(0, total / nwin)
    for i in range(1, iwagc):
        total += d2[i + iwagc - 1]
        nwin += 1
        out[i] = g(i, total / nwin)
    nwin += 1
    for i in range(iwagc, nt - iwagc):
        out[i] = g(i, d2[i - iwagc:i + iwagc].sum() / nwin)
    for i in range(nt - iwagc, nt):
        nwin -= 1
        out[i] = 0.0 if x[i] == 0 else x[i] / np.sqrt(d2[i - iwagc:].sum() / nwin)
    return out


def test_agc():
    wagc = 0.1  # 25 samples, so a half window of 12
    expected = np.array([agc_reference(row, 12) for row in DATA])
    check(gain(agc=True, wagc=wagc), expected, rtol=1e-4)


def test_agc_evens_out_amplitudes():
    t = np.arange(512) * DT
    sig = (np.sin(2 * np.pi * 20 * t) * np.exp(-2 * t)).astype(np.float32)[None, :] * 100
    out = run(gain(agc=True, wagc=0.1), sig)[0]
    assert np.abs(out[40:-40]).max() / np.abs(out[40:-40]).mean() < 3
    # without it the amplitude decays by a lot
    assert np.abs(sig[0, :50]).max() / np.abs(sig[0, -50:]).max() > 10


def gagc_reference(x, iwagc):
    nt = len(x)
    d2 = x.astype(np.float64) ** 2
    u = 3.8090232 / iwagc
    s = d2.copy()
    for j in range(1, iwagc):
        w = np.exp(-(u * j) ** 2)
        s[j:] += w * d2[:nt - j]
        s[:nt - j] += w * d2[j:]
    return np.where(s == 0, 0.0, x / np.sqrt(np.where(s == 0, 1.0, s)))


def test_gagc():
    expected = np.array([gagc_reference(row, 12) for row in DATA])
    check(gain(gagc=True, wagc=0.1), expected, rtol=1e-4)


def test_agc_window_errors():
    with pytest.raises(ValueError, match="too long"):
        run(gain(agc=True, wagc=10.0))
    with pytest.raises(ValueError):
        run(gain(agc=True, wagc=0.001))


def test_bad_parameters_fail_early():
    for kw in (dict(qclip=1.5), dict(clip=-1.0), dict(trap=-1.0), dict(vred=-1.0)):
        with pytest.raises(ValueError):
            gain(**kw)
    with pytest.raises(TypeError):
        gain(not_a_parameter=3)


def test_dt():
    data = DATA.copy()
    coll = TraceCollection(data, d_sample=0.0)
    with pytest.raises(ValueError, match="sample interval"):
        list(coll | gain(tpow=1.0))
    out = np.array([np.asarray(t) for t in coll | gain(tpow=1.0, dt=DT)])
    expected = DATA * np.where(T == 0, 0.0, T)
    npt.assert_allclose(out, expected, rtol=2e-5)


def test_changing_trace_length():
    # SU's static lookup tables could not cope with this
    from seispy.container import from_iterable
    short = Trace(np.ones(16, dtype=np.float32), d_sample=DT)
    long = Trace(np.ones(40, dtype=np.float32), d_sample=DT)
    out = list(from_iterable([short, long, short]) | gain(tpow=1.0))
    for tr in out:
        n = tr.n_sample
        expected = np.arange(n) * DT
        npt.assert_allclose(np.asarray(tr), expected, rtol=1e-5, atol=1e-7)


def test_inplace():
    coll = TraceCollection(DATA.copy(), d_sample=DT).to_memory()
    list(coll | gain(scale=2.0))
    npt.assert_array_equal(np.array([np.asarray(t) for t in coll]), DATA)
    list(coll | gain(scale=2.0, inplace=True))
    npt.assert_allclose(np.array([np.asarray(t) for t in coll]), DATA * 2.0)


def test_panel_treats_the_data_as_one_long_trace():
    flat = DATA.reshape(-1)
    rms = np.sqrt((flat.astype(np.float64) ** 2).mean())
    check(gain(pbal=True, panel=True), DATA / rms)
    # whereas trace by trace each trace gets its own
    assert not np.allclose(run(gain(pbal=True)), DATA / rms)

    t_long = np.arange(flat.size) * DT
    fac = t_long.copy()
    fac[0] = 0.0
    check(gain(tpow=1.0, panel=True), (flat * fac).reshape(DATA.shape), rtol=1e-4)


def test_panel_needs_equal_traces():
    from seispy.container import from_iterable
    traces = [Trace(np.ones(8, dtype=np.float32), d_sample=DT), Trace(np.ones(9, dtype=np.float32), d_sample=DT)]
    with pytest.raises(ValueError, match="same number of samples"):
        list(from_iterable(traces) | gain(panel=True))


def test_panel_empty_stream():
    from seispy.container import from_iterable
    assert list(from_iterable([]) | gain(panel=True)) == []


def test_parallelism_levels_and_pmap():
    assert gain().parallelism == 'trace'
    assert gain(panel=True).parallelism == 'serial'
    with pytest.raises(ValueError):
        pmap(gain(panel=True))

    s = gain(tpow=1.5, agc=True, wagc=0.1, qbal=True, qclip=0.95)
    from seispy.synthetics import synlv
    expected = np.array([np.asarray(t) for t in synlv() | s])
    got = np.array([np.asarray(t) for t in synlv() | pmap(s, workers=3, chunk=4)])
    npt.assert_array_equal(got, expected)


def test_repr():
    assert repr(gain(tpow=2.0)) == "gain(panel=False, tpow=2.0)"
