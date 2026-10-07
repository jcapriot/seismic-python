import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.decon import pef, shape
from seispy.filters import minphase, tvband
from seispy.parallel import pmap

DT = 0.004
RNG = np.random.default_rng(2024)


def traces(data, dt=DT, sample_start=0.0):
    return [
        Trace(np.asarray(row, dtype=np.float32), d_sample=dt, sample_start=sample_start) for row in np.atleast_2d(data)
    ]


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


def toeplitz(r):
    n = len(r)
    return np.array([[r[abs(i - j)] for j in range(n)] for i in range(n)])


def acorr(x, lags):
    return np.array([np.sum(x[:len(x) - k] * x[k:]) for k in range(lags)])


# ---------------------------------------------------------------------------------------------------------- shape
def shape_reference(x, w, d, nshape, pnoise):
    """sushape: the least squares filter, and its (centered) convolution with the trace"""
    nw, nd = len(w), len(d)
    r = np.array([np.sum(w[:nw - k] * w[k:]) if k < nw else 0.0 for k in range(nshape)])
    g = np.array([sum(w[j] * d[k + j] for j in range(nw) if 0 <= k + j < nd) for k in range(nshape)])
    r[0] *= 1 + pnoise
    f = np.linalg.solve(toeplitz(r), g)
    shift = int((nw - nd) / 2)
    out = np.zeros(len(x))
    for i in range(len(x)):
        for k in range(nshape):
            j = i - k - shift
            if 0 <= j < len(x):
                out[i] += f[k] * x[j]
    return out


@pytest.mark.parametrize('w, d, nshape', [
    ([1.0, -0.5, 0.2], [1.0, 0.0, 0.0], 12),
    ([1.0, 0.6, -0.3, 0.1], [0.0, 1.0], 20),   # (the lengths are not the same, so the output is shifted)
    ([0.5, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0], 8),
])
def test_shape_matches_the_normal_equations(w, d, nshape):
    x = RNG.normal(size=(2, 60)).astype(np.float32)
    out = arr(run(shape(w, d, nshape=nshape, pnoise=0.01), traces(x)))
    for row, o in zip(x, out):
        expected = shape_reference(row, np.array(w), np.array(d), nshape, 0.01)
        npt.assert_allclose(o, expected, rtol=2e-3, atol=2e-4)


def test_shape_turns_the_wavelet_into_the_desired_one():
    w = np.array([1.0, -0.5, 0.25, 0.1], dtype=np.float32)
    d = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    x = np.zeros((1, 80), dtype=np.float32)
    x[0, 10:14] = w
    (out,) = run(shape(w, d, nshape=40, pnoise=1e-6), traces(x))
    y = np.asarray(out)
    # a spike (the desired wavelet is a spike at its 2nd sample, and the wavelet starts at 10)
    assert np.argmax(np.abs(y)) == 11
    assert np.abs(y).max() == pytest.approx(1.0, abs=0.03)
    off_peak = np.delete(np.abs(y), 11).max()
    assert off_peak < 0.05


def test_shape_wavelets_can_be_traces_and_the_length_follows_the_trace():
    w = Trace(np.array([1.0, 0.5], dtype=np.float32), d_sample=DT)
    d = Trace(np.array([1.0, 0.0], dtype=np.float32), d_sample=DT)
    x = RNG.normal(size=(2, 30)).astype(np.float32)
    a = arr(run(shape(w, d), traces(x)))
    b = arr(run(shape([1.0, 0.5], [1.0, 0.0], nshape=30), traces(x)))
    npt.assert_array_equal(a, b)
    # traces of different lengths get filters of their own lengths
    mixed = [traces(x[:1, :20])[0], traces(x[1:, :30])[0]]
    out = run(shape(w, d), mixed)
    assert [o.n_sample for o in out] == [20, 30]


def test_shape_errors():
    with pytest.raises(ValueError, match="zero wavelet"):
        run(shape([0.0, 0.0], [1.0]), traces(np.ones((1, 10))))
    for bad in (dict(w=[], d=[1.0]), dict(w=[[1.0]], d=[1.0]), dict(w=[1.0], d=[], nshape=3), dict(w=[1.0], d=[1.0], nshape=0)):
        with pytest.raises(ValueError):
            shape(**bad)
    assert shape([1.0], [1.0]).parallelism == 'trace'


# ------------------------------------------------------------------------------------------------------------ pef
def pef_reference(traces_in, dt, minlag, maxlag, pnoise, mix, window=None):
    """supef, written out as the C does it, for one trace at a time"""
    def nint(v):
        return int(v + 0.5) if v > 0 else int(v - 0.5)

    nmix = len(mix)
    mix = [m / nmix for m in mix]
    mixacorr = np.zeros((nmix, 0))
    outs = []
    for x in traces_in:
        nt = len(x)
        imin = 1 if minlag is None else nint(minlag / dt)
        imax = nint(0.05 * nt) if maxlag is None else nint(maxlag / dt)
        nlag, lcorr = imax - imin + 1, imax + 1
        if mixacorr.shape[1] != lcorr:
            mixacorr = np.zeros((nmix, lcorr))
        w = x[:nt + 1]  # (the window is one sample longer than the correlation window, as in the program)
        auto = np.array([np.sum(w[:len(w) - k] * w[k:]) for k in range(lcorr)])
        if auto[0] == 0:
            outs.append(x.copy())
            continue
        auto[0] *= 1 + pnoise
        mixacorr[0] = auto
        temp = np.zeros(lcorr)
        for ilag in range(lcorr):
            for imix in range(nmix):
                temp[ilag] += mixacorr[imix][ilag] * mix[imix]
        for imix in range(nmix - 1, 0, -1):
            mixacorr[imix] = mixacorr[imix - 1]
        wiener = np.linalg.solve(toeplitz(temp[:nlag]), temp[imin:imin + nlag])
        out = np.zeros(nt)
        for i in range(nt):
            n = min(i, imax)
            s = x[i]
            for j in range(imin, n + 1):
                s -= wiener[j - imin] * x[i - j]
            out[i] = s
        outs.append(out)
    return np.array(outs)


def test_pef_matches_the_program():
    x = RNG.normal(size=(1, 200)).astype(np.float32)
    got = arr(run(pef(), traces(x)))
    npt.assert_allclose(got, pef_reference(x.astype(np.float64), DT, None, None, 0.001, [1.0]), rtol=3e-3, atol=3e-4)
    got = arr(run(pef(minlag=0.012, maxlag=0.06, pnoise=0.01), traces(x)))
    expected = pef_reference(x.astype(np.float64), DT, 0.012, 0.06, 0.01, [1.0])
    npt.assert_allclose(got, expected, rtol=3e-3, atol=3e-4)


def test_pef_mixing_autocorrelations_of_neighbouring_traces():
    x = RNG.normal(size=(5, 150)).astype(np.float32)
    mix = [1.0, 2.0, 1.0]
    got = arr(run(pef(maxlag=0.04, mix=mix), traces(x)))
    expected = pef_reference(x.astype(np.float64), DT, None, 0.04, 0.001, mix)
    npt.assert_allclose(got, expected, rtol=4e-3, atol=4e-4)
    assert pef(mix=mix).parallelism == 'serial'
    assert pef(mix=[1.0]).parallelism == 'trace'
    assert pef().parallelism == 'trace'


def test_pef_leaves_a_dead_trace_alone_and_does_not_count_it_as_a_neighbour():
    x = RNG.normal(size=(3, 100)).astype(np.float32)
    with_dead = np.vstack([x[0], np.zeros(100, dtype=np.float32), x[1]])
    out = arr(run(pef(maxlag=0.04, mix=[1.0, 1.0]), traces(with_dead)))
    npt.assert_array_equal(out[1], 0.0)
    without = arr(run(pef(maxlag=0.04, mix=[1.0, 1.0]), traces(x[:2])))
    npt.assert_allclose(out[[0, 2]], without, rtol=1e-5, atol=1e-6)


def test_pef_spikes_the_wavelet():
    # a minimum phase wavelet and sparse reflections: deconvolution takes out most of the wavelet
    wavelet = np.array([1.0, 0.8, 0.5, 0.2, -0.1], dtype=np.float32)
    reflectivity = np.zeros(400, dtype=np.float32)
    idx = RNG.choice(np.arange(5, 395), size=25, replace=False)
    reflectivity[idx] = RNG.normal(size=25)
    x = np.convolve(reflectivity, wavelet)[:400].astype(np.float32)
    (out,) = run(pef(maxlag=0.02, pnoise=1e-4), traces(x))
    y = np.asarray(out)
    ac_in = acorr(x, 6) / acorr(x, 1)[0]
    ac_out = acorr(y, 6) / acorr(y, 1)[0]
    assert np.abs(ac_in[1:]).max() > 0.5
    assert np.abs(ac_out[1:]).max() < 0.15
    # and it is a gapped filter when it has a minimum lag: the near lags are left alone
    (gapped,) = run(pef(minlag=0.02, maxlag=0.06), traces(x))
    ac_gap = acorr(np.asarray(gapped), 6) / acorr(np.asarray(gapped), 1)[0]
    npt.assert_allclose(ac_gap[1:3], ac_in[1:3], atol=0.25)


def test_pef_correlation_window_and_errors():
    x = RNG.normal(size=(1, 300)).astype(np.float32)
    whole = arr(run(pef(maxlag=0.04), traces(x)))
    windowed = arr(run(pef(maxlag=0.04, mincorr=0.2, maxcorr=0.8), traces(x)))
    assert not np.allclose(whole, windowed)
    # (the window changes the filter, which is then applied to the whole trace)
    assert windowed.shape == whole.shape
    for bad in (dict(minlag=-1.0), dict(minlag=0.1, maxlag=0.05), dict(mincorr=0.5, maxcorr=0.5), dict(mix=[]),
                dict(mix=[[1.0]])):
        with pytest.raises(ValueError):
            pef(**bad)
    with pytest.raises(ValueError, match="maxcorr"):
        run(pef(maxcorr=100.0), traces(x))
    with pytest.raises(ValueError, match="sample interval"):
        run(pef(), traces(x, dt=0.0))
    # (5% of 8 samples is less than the one sample that the filter has to start at)
    with pytest.raises(ValueError, match="no samples"):
        run(pef(), traces(np.ones((1, 8))))


# ------------------------------------------------------------------------------------------------------ minphase
def test_minphase_of_a_maximum_phase_wavelet():
    x = np.zeros((1, 64), dtype=np.float32)
    x[0, :2] = [-0.5, 1.0]  # zero outside the unit circle: maximum phase
    (out,) = run(minphase(), traces(x))
    y = np.asarray(out)
    npt.assert_allclose(y[:2], [1.0, -0.5], atol=2e-3)
    npt.assert_allclose(y[2:], 0.0, atol=2e-3)
    npt.assert_allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x[0])), rtol=2e-3, atol=1e-3)


def test_minphase_puts_the_energy_as_early_as_possible():
    x = np.zeros((1, 128), dtype=np.float32)
    x[0, 20:28] = RNG.normal(size=8)  # a wavelet that has its energy in the middle
    (out,) = run(minphase(), traces(x))
    y = np.asarray(out)
    # the same amplitude spectrum (so the same energy), and never less of it by any time
    npt.assert_allclose(np.abs(np.fft.rfft(y)), np.abs(np.fft.rfft(x[0])), rtol=5e-3, atol=2e-3)
    assert (y ** 2).sum() == pytest.approx((x[0] ** 2).sum(), rel=1e-2)
    assert (np.cumsum(y ** 2) >= np.cumsum(x[0] ** 2) - 1e-3).all()
    # and it is minimum phase for the first sample being the biggest, for a wavelet like that
    assert np.argmax(np.abs(y)) < 20


def test_minphase_odd_lengths_zeros_and_signs():
    x = np.zeros((1, 63), dtype=np.float32)
    x[0, :2] = [-0.5, 1.0]
    (out,) = run(minphase(), traces(x))
    assert out.n_sample == 63
    npt.assert_allclose(np.asarray(out)[:2], [1.0, -0.5], atol=5e-3)
    zero = traces(np.zeros((1, 16)))
    npt.assert_array_equal(arr(run(minphase(), zero)), 0.0)
    # the other pair of signs gives a wavelet with the same amplitudes
    (other,) = run(minphase(sign1=-1, sign2=1), traces(x))
    npt.assert_allclose(np.abs(np.fft.rfft(np.asarray(other))), np.abs(np.fft.rfft(x[0])), rtol=5e-3, atol=2e-3)
    for bad in (dict(sign1=0), dict(sign2=0)):
        with pytest.raises(ValueError):
            minphase(**bad)


# --------------------------------------------------------------------------------------------------------- tvband
def band_filter_reference(corners, n, dt):
    """sutvband's makefilter"""
    n_freq = n // 2 + 1
    df = 1.0 / (n * dt)

    def nint(v):
        return int(v + 0.5) if v > 0 else int(v - 0.5)

    last = n_freq - 1
    if1, if2 = nint(corners[0] / df), nint(corners[1] / df)
    if3, if4 = min(nint(corners[2] / df), last), min(nint(corners[3] / df), last)
    filt = np.zeros(n_freq)
    c = (np.pi / 2) / (if2 - if1 + 2)
    for i in range(if1, if2 + 1):
        filt[i] = np.sin(c * (i - if1 + 1)) ** 2
    c = (np.pi / 2) / (if4 - if3 + 2)
    for i in range(if3, if4 + 1):
        filt[i] = np.sin(c * (if4 - i + 1)) ** 2
    for i in range(if2 + 1, if3):
        filt[i] = 1.0
    return filt


def test_tvband_with_one_filter_is_that_filter():
    n = 400
    x = RNG.normal(size=(1, n)).astype(np.float32)
    corners = [10.0, 15.0, 40.0, 60.0]
    # (at the start of the trace, so that the filter is for the whole trace. The part that a filter is applied to goes
    # from the center of the one before to the center of the one after, so a filter in the middle of a trace is applied
    # to a shorter part of it.)
    (out,) = run(tvband([0.0], [corners]), traces(x))
    expected = np.fft.irfft(np.fft.rfft(x[0]) * band_filter_reference(corners, n, DT), n)
    npt.assert_allclose(np.asarray(out), expected, rtol=1e-3, atol=1e-4)
    # one 4-tuple can be given as it is
    (same,) = run(tvband([0.0], corners), traces(x))
    npt.assert_array_equal(np.asarray(same), np.asarray(out))


def test_tvband_changes_with_time():
    n = 800
    t = np.arange(n) * DT
    both = (np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    low, high = [5.0, 8.0, 20.0, 25.0], [40.0, 45.0, 60.0, 70.0]
    (out,) = run(tvband([0.5, 2.5], [low, high]), traces(both))
    y = np.asarray(out)

    def amplitude(segment, freq):
        spectrum = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
        return spectrum[int(round(freq * len(segment) * DT))] / (len(segment) / 4)

    early, late = y[20:120], y[-120:-20]  # (the filters are at 0.5 s and 2.5 s, the trace is 3.2 s long)
    assert amplitude(early, 15) > 0.8 and amplitude(early, 50) < 0.1
    assert amplitude(late, 50) > 0.8 and amplitude(late, 15) < 0.1
    # in between it fades from one to the other
    middle = y[375:425]
    assert 0.05 < amplitude(middle, 15) < 0.95


def tvband_reference(x, tf, filters, tmin, dt):
    """The program's way of putting it together"""
    n = len(x)

    def nint(v):
        return int(v + 0.5) if v > 0 else int(v - 0.5)

    itf = [nint((t - tmin) / dt) for t in tf]
    filt = list(filters)
    if itf[0] > 0:
        itf.insert(0, 0)
        filt.insert(0, filt[0])
    if itf[-1] < n - 1:
        itf.append(n - 1)
        filt.append(filt[-1])
    ext = [0] + itf + [n - 1]
    sub = []
    for j in range(len(itf)):
        data = np.zeros(n)
        for i in range(ext[j], ext[j + 2] + 1):
            data[i] = x[i]
        sub.append(np.fft.irfft(np.fft.rfft(data) * filt[j], n))
    out = np.zeros(n)
    for j in range(len(itf) - 1):
        for i in range(itf[j], itf[j + 1] + 1):
            a = (i - itf[j]) / (itf[j + 1] - itf[j])
            out[i] = (1 - a) * sub[j][i] + a * sub[j + 1][i]
    return out


def test_tvband_matches_the_programs_composition():
    n = 300
    x = RNG.normal(size=(1, n)).astype(np.float32)
    tf = [0.3, 0.7]
    corners = [[5.0, 10.0, 30.0, 40.0], [20.0, 25.0, 60.0, 80.0]]
    (out,) = run(tvband(tf, corners), traces(x, sample_start=0.1))
    filters = [band_filter_reference(c, n, DT) for c in corners]
    expected = tvband_reference(x[0].astype(np.float64), tf, filters, 0.1, DT)
    npt.assert_allclose(np.asarray(out), expected, rtol=1e-3, atol=2e-4)


def test_tvband_errors_and_properties():
    for bad in (
        dict(tf=[], f=[]),
        dict(tf=[0.1, 0.2], f=[[1, 2, 3, 4]]),
        dict(tf=[0.2, 0.1], f=[[1, 2, 3, 4], [1, 2, 3, 4]]),
        dict(tf=[0.1], f=[[10, 5, 30, 40]]),
        dict(tf=[0.1], f=[[1, 2, 2, 4]]),
        dict(tf=[0.1], f=[[1, 2, 3]]),
    ):
        with pytest.raises(ValueError):
            tvband(**bad)
    with pytest.raises(ValueError, match="sample interval"):
        run(tvband([0.1], [[1, 2, 3, 4]]), traces(np.ones((1, 50)), dt=0.0))
    (ok,) = run(tvband([0.1], [[1, 2, 30, 40]], dt=DT), traces(np.ones((1, 50)), dt=0.0))
    assert ok.n_sample == 50
    assert tvband([0.1], [[1, 2, 3, 4]]).parallelism == 'trace'


def test_parallel_and_pickled():
    import pickle

    x = RNG.normal(size=(8, 200)).astype(np.float32)
    trs = traces(x)
    for stage in (shape([1.0, 0.5], [1.0, 0.0], nshape=30), pef(maxlag=0.03), minphase(),
                  tvband([0.3], [[5.0, 10.0, 40.0, 60.0]])):
        assert stage.parallelism == 'trace'
        expected = arr(run(stage, trs))
        got = arr(list(from_iterable(list(trs)) | pmap(stage, workers=3, chunk=2)))
        npt.assert_array_equal(got, expected)
        again = pickle.loads(pickle.dumps(stage))
        npt.assert_array_equal(arr(run(again, trs)), expected)
    with pytest.raises(ValueError):
        pmap(pef(mix=[1.0, 1.0]))
