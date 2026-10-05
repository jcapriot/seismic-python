import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import TraceCollection, Trace, from_iterable
from seispy.filters import filter as sufilter
from seispy.parallel import pmap

DT = 0.002
NT = 2048
DF = 1.0 / (NT * DT)  # frequency spacing of the unpadded spectrum


def tone(freq_hz, nt=NT, dt=DT):
    # use a frequency that sits on a bin
    k = round(freq_hz / (1.0 / (nt * dt)))
    return np.sin(2 * np.pi * k / nt * np.arange(nt)).astype(np.float32), k / (nt * dt)


def amp(x, freq, nt=NT, dt=DT):
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64)))
    return spec[round(freq * nt * dt)]


def run(stage, data, dt=DT):
    coll = TraceCollection(np.atleast_2d(data), d_sample=dt)
    return np.array([np.asarray(t) for t in coll | stage])


@pytest.mark.parametrize(
    'amps, expected',
    [
        # the gain expected at tones of 5, 15, 30, 45 and 90 Hz, for f = 10, 20, 40, 50. The filter is a polygon
        # through (f, amps), with sin^2 tapers, so the middle of a taper has a gain half way between its ends.
        (None, [0.0, 0.5, 1.0, 0.5, 0.0]),  # bandpass, the default amps are 0, 1, 1, 0
        ([1, 1, 0, 0], [1.0, 1.0, 0.5, 0.0, 0.0]),  # lowpass
        ([0, 0, 1, 1], [0.0, 0.0, 0.5, 1.0, 1.0]),  # highpass
        ([1, 0, 0, 1], [1.0, 0.5, 0.0, 0.5, 1.0]),  # bandreject
    ],
)
def test_filter_shapes(amps, expected):
    for freq, want in zip([5, 15, 30, 45, 90], expected):
        x, freq = tone(freq)
        kw = dict(f=[10, 20, 40, 50])
        if amps is not None:
            kw['amps'] = amps
        out = run(sufilter(**kw), x)[0]
        got = amp(out, freq) / amp(x, freq)
        assert got == pytest.approx(want, abs=0.03), (freq, got)


def test_default_filter_is_a_trapezoid_in_terms_of_nyquist():
    nyq = 0.5 / DT  # 250 Hz, so the passband is 37.5 to 112.5 Hz
    low, _ = tone(0.05 * nyq)
    mid, f_mid = tone(0.3 * nyq)
    high, _ = tone(0.8 * nyq)
    out = run(sufilter(), np.stack([low, mid, high]))
    assert amp(out[0], 0.05 * nyq) / amp(low, 0.05 * nyq) < 0.03
    assert amp(out[1], f_mid) / amp(mid, f_mid) == pytest.approx(1.0, abs=0.03)
    assert amp(out[2], 0.8 * nyq) / amp(high, 0.8 * nyq) < 0.03


def test_zero_phase():
    x, _ = tone(30)
    out = run(sufilter(f=[10, 20, 40, 50]), x)[0]
    corr = np.dot(out, x) / (np.linalg.norm(out) * np.linalg.norm(x))
    assert corr > 0.999


def test_notch_example_from_the_su_docs():
    stage = sufilter(f=[10, 12.5, 35, 50, 60], amps=[1, 0.5, 0, 0.5, 1])
    for freq, want in [(5, 1.0), (11.25, 0.75), (35, 0.0), (55, 0.75), (90, 1.0)]:
        x, f0 = tone(freq)
        out = run(stage, x)[0]
        assert amp(out, f0) / amp(x, f0) == pytest.approx(want, abs=0.04), freq


@pytest.mark.parametrize('nt', [100, 777, 1000, 1024, 4001])
def test_awkward_trace_lengths(nt):
    x = np.random.default_rng(nt).normal(size=nt).astype(np.float32)
    out = run(sufilter(f=[10, 20, 100, 200]), x)[0]
    assert out.shape == (nt,)
    assert np.isfinite(out).all()


def test_dt_handling():
    x, f0 = tone(40, dt=0.004)
    # no sample interval on the trace: use the one given
    coll = TraceCollection(x[None, :], d_sample=0.0)
    out = np.array([np.asarray(t) for t in coll | sufilter(f=[10, 20, 50, 60], dt=0.004)])[0]
    assert amp(out, f0, dt=0.004) / amp(x, f0, dt=0.004) == pytest.approx(1.0, abs=0.03)
    # and as in SU, .004 if there is nothing at all
    out2 = np.array([np.asarray(t) for t in coll | sufilter(f=[10, 20, 50, 60])])[0]
    npt.assert_array_equal(out, out2)


def test_changing_trace_length_and_dt():
    short, f1 = tone(30, nt=512)
    long, f2 = tone(30, nt=2048)
    traces = [Trace(short, d_sample=DT), Trace(long, d_sample=DT), Trace(short, d_sample=DT)]
    out = list(from_iterable(traces) | sufilter(f=[10, 20, 40, 50]))
    assert [t.n_sample for t in out] == [512, 2048, 512]
    ref_short = run(sufilter(f=[10, 20, 40, 50]), short)[0]
    npt.assert_array_equal(np.asarray(out[0]), ref_short)
    npt.assert_array_equal(np.asarray(out[2]), ref_short)


def test_bad_parameters_fail_early():
    with pytest.raises(ValueError, match="Bad filter"):
        sufilter(f=[10, 5, 40, 50])
    with pytest.raises(ValueError, match="Bad filter"):
        sufilter(f=[-1, 5, 40, 50])
    with pytest.raises(ValueError, match="positive"):
        sufilter(f=[10, 20, 40, 50], amps=[0, -1, 1, 0])
    with pytest.raises(ValueError, match="zero"):
        sufilter(f=[10, 20, 40, 50], amps=[0, 0, 0, 0])
    with pytest.raises(ValueError, match="number of f"):
        sufilter(f=[10, 20, 40, 50], amps=[0, 1, 0])
    with pytest.raises(ValueError, match="number of f"):
        sufilter(amps=[0, 1, 0])  # the default f has 4 values
    with pytest.warns(UserWarning, match="same"):
        sufilter(f=[10, 20], amps=[1, 1])


def test_inplace():
    x, _ = tone(30)
    coll = TraceCollection(x[None, :].copy(), d_sample=DT).to_memory()
    list(coll | sufilter(f=[10, 20, 40, 50]))
    npt.assert_array_equal(np.asarray(next(iter(coll))), x)
    list(coll | sufilter(f=[10, 20, 40, 50], inplace=True))
    assert not np.array_equal(np.asarray(next(iter(coll))), x)


def test_pmap_matches_sequential():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(24, 1500)).astype(np.float32)
    s = sufilter(f=[10, 20, 60, 90])
    expected = run(s, data)
    coll = TraceCollection(data, d_sample=DT)
    got = np.array([np.asarray(t) for t in coll | pmap(s, workers=4, chunk=3)])
    npt.assert_array_equal(got, expected)
    assert s.parallelism == 'trace'
    assert repr(s) == "filter(f=[10, 20, 60, 90])"
