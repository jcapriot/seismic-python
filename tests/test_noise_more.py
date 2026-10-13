"""SUHARLAN, SUSEMBASE and SUSEMGROUP."""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.noise import harlan, sembase, semgroup

DT = 0.002


def ricker(t, f=30.0):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def run(stage, traces):
    return list(from_iterable(list(traces)) | stage)


def envelope_peak(x):
    x = np.asarray(x, dtype=float)
    h = np.zeros(len(x))
    h[0] = 1
    h[1:len(x) // 2] = 2
    h[len(x) // 2] = 1
    return int(np.argmax(np.abs(np.fft.ifft(np.fft.fft(x) * h))))


def cdp_line(n=9, slope=0.001, t0=0.2, nt=200, noise=0.0, seed=0):
    """A stack: an event that dips `slope` s per CDP, one trace per CDP"""
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * DT
    out = []
    for c in range(1, n + 1):
        data = ricker(t - t0 - slope * c) + noise * rng.normal(size=nt)
        out.append(Trace(data.astype(np.float32), d_sample=DT).replace(ensemble_number=c))
    return out


def gather(n=11, slope_ms_per_m=0.5, t0=0.2, nt=200, noise=0.0, seed=0, cdp=1):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * DT
    out = []
    for i in range(n):
        offset = 10.0 * i
        data = ricker(t - t0 - slope_ms_per_m * 1e-3 * offset) + noise * rng.normal(size=nt)
        out.append(Trace(data.astype(np.float32), d_sample=DT).replace(
            tx_loc=[0.0, 0.0, 0.0], rx_loc=[offset, 0.0, 0.0], ensemble_number=cdp))
    return out


# ------------------------------------------------------------------------------------------------- semgroup
def test_semgroup_finds_the_dip_of_the_event():
    report = {}
    run(semgroup(locmaxscan=50.0, report=report), gather())
    dip = np.asarray(report['dip'][5])  # ms per m
    event = int((0.2 + 0.0005 * 50.0) / DT)
    assert abs(dip[event] - 0.5) < 0.1
    assert np.asarray(report['sem'][5])[event] > 0.8
    assert len(report['dip']) == 11


def test_semgroup_smooths_noise_without_moving_the_event():
    clean = gather()
    noisy = gather(noise=0.3, seed=2)
    out = run(semgroup(locmaxscan=50.0, locmaxsum=30.0), noisy)
    assert len(out) == len(noisy)
    err_in = np.sqrt(np.mean([(np.asarray(a) - np.asarray(b)) ** 2 for a, b in zip(noisy, clean)]))
    err_out = np.sqrt(np.mean([(np.asarray(a) - np.asarray(b)) ** 2 for a, b in zip(out, clean)]))
    assert err_out < 0.7 * err_in
    for a, b in zip(out[2:-2], clean[2:-2]):
        lags = np.arange(-5, 6)
        corr = [np.dot(np.roll(np.asarray(a), -lag), np.asarray(b)) for lag in lags]
        assert abs(lags[int(np.argmax(corr))]) <= 1
    # the headers are those of the input traces
    assert out[3].header['rx_loc'][0] == 30.0


def test_semgroup_groups_and_indip():
    two = gather(cdp=1) + gather(n=9, cdp=2, slope_ms_per_m=-0.4)
    report = {}
    out = run(semgroup(locmaxscan=40.0, report=report), two)
    assert len(out) == 20
    dips = [np.asarray(d) for d in report['dip']]
    assert abs(dips[11 + 4][int((0.2 - 0.0004 * 40.0) / DT)] + 0.4) < 0.1
    # the dips of the first run, used by a second one, give the same traces
    again = run(semgroup(locmaxsum=40.0, indip=report['dip']), two)
    for a, b in zip(out, again):
        npt.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_semgroup_errors():
    with pytest.raises(ValueError):
        semgroup(locmaxscan=0.0)  # locmaxscan=0 needs locmaxsum
    with pytest.raises(ValueError):
        semgroup(indip=[], locmaxscan=3.0, locmaxsum=3.0)
    with pytest.raises(ValueError):
        semgroup(maxfold=1)
    with pytest.raises(ValueError):
        run(semgroup(maxfold=5), gather(n=8))


# ------------------------------------------------------------------------------------------------- sembase
def test_sembase_interpolates_along_the_dip():
    out = run(sembase(outputinc=0.5, locmaxscan=3.0, scanstep=0.5, scanmax=3.0), cdp_line())
    lockeys = [t.header['ensemble_number'] for t in out]
    assert lockeys == sorted(lockeys)
    # (the lockey of the output is the index of the output location when outputinc is below 1)
    by_key = {k: t for k, t in zip(lockeys, out)}
    for k in (6, 8, 10, 12):  # the locations 3, 4, 5, 6
        loc = k * 0.5
        expected = (0.2 + 0.001 * loc) / DT
        assert abs(envelope_peak(by_key[k]) - expected) <= 1.0


def test_sembase_bypass_and_sumopts():
    line = cdp_line()
    only = run(sembase(outputinc=1.0, bypass=-1), line)
    assert [t.header['ensemble_number'] for t in only] == list(range(1, 10))
    copied = run(sembase(outputinc=0.5, bypass=1), line)
    on_grid = {t.header['ensemble_number']: t for t in copied}
    npt.assert_array_equal(np.asarray(on_grid[4]), np.asarray(line[1]))  # k=4 is location 2.0: the input
    for opt in ('stack', 'sum', 'one', 'two', 'twolinear', 'dadstack', 'dadsum'):
        out = run(sembase(sumopt=opt, outputinc=0.5, locmaxscan=3.0), line)
        assert len(out) > len(line)
        assert all(np.isfinite(np.asarray(t)).all() for t in out)


def test_sembase_dipuse_and_indip_roundtrip():
    line = cdp_line(noise=0.1, seed=3)
    report = {}
    out = run(sembase(locmaxscan=3.0, report=report), line)
    assert len(report['dip']) == len(line)
    again = run(sembase(locmaxsum=3.0, indip=report['dip']), line)
    for a, b in zip(out, again):
        npt.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-4)
    at_output = run(sembase(dipuse=0, locmaxscan=3.0), line)
    assert len(at_output) == len(out)


def test_sembase_errors():
    with pytest.raises(ValueError):
        sembase(sumopt='nonsense')
    with pytest.raises(ValueError):
        sembase(sumopt='stack', dadscale=1.0)
    with pytest.raises(ValueError):
        sembase(outputinc=0.0)
    with pytest.raises(ValueError):
        sembase(locmaxscan=0.0, scanstep=1.0)
    shuffled = cdp_line()
    shuffled[3], shuffled[5] = shuffled[5], shuffled[3]
    with pytest.raises(ValueError):
        run(sembase(), shuffled)


# ------------------------------------------------------------------------------------------------- harlan
def linear_gather(nx=24, nt=200, dt=0.004, noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * dt
    out = []
    for i in range(nx):
        x = 20.0 * i
        data = ricker(t - 0.3 - 1e-4 * x, f=20.0) + 0.8 * ricker(t - 0.5 + 0.5e-4 * x, f=20.0)
        data = data + noise * rng.normal(size=nt)
        out.append(Trace(data.astype(np.float32), d_sample=dt))
    return out


def test_harlan_attenuates_random_noise():
    clean = np.array([np.asarray(t) for t in linear_gather()])
    noisy = linear_gather(noise=0.4)
    report = {}
    out = run(harlan(gopt=3, pmin1=-200.0, pmax1=200.0, nslopes=60, rel1=0.1, seed=7, report=report), noisy)
    signal = np.array([np.asarray(t) for t in out])
    noise = np.array([np.asarray(t) for t in report['noise']])
    inp = np.array([np.asarray(t) for t in noisy])
    assert signal.shape == inp.shape
    # the separation adds up, and the signal is closer to the clean data than the input is
    npt.assert_allclose(signal + noise, inp, atol=1e-4)
    assert np.linalg.norm(signal - clean) < np.linalg.norm(inp - clean)


def test_harlan_is_repeatable_and_has_outputs():
    noisy = linear_gather(noise=0.3)
    kw = dict(gopt=3, pmin1=-200.0, pmax1=200.0, nslopes=40, rel1=0.1, seed=3)
    a = run(harlan(**kw), noisy)
    b = run(harlan(**kw), noisy)
    for x, y in zip(a, b):
        npt.assert_array_equal(np.asarray(x), np.asarray(y))
    both = run(harlan(output='both', **kw), noisy)
    assert len(both) == 2 * len(noisy)
    noise_only = run(harlan(output='noise', **kw), noisy)
    for x, y in zip(both[len(noisy):], noise_only):
        npt.assert_array_equal(np.asarray(x), np.asarray(y))
    two = run(harlan(niter=2, **kw), noisy)
    assert len(two) == len(noisy)


def test_harlan_errors():
    with pytest.raises(ValueError):
        harlan(output='x')
    with pytest.raises(ValueError):
        harlan(gopt=9)
    with pytest.raises(ValueError):
        harlan(rgt=3)
    with pytest.raises(ValueError):
        run(harlan(), linear_gather(nx=1))
