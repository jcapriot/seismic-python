import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import TraceCollection
from seispy import operations as op
from seispy.parallel import pmap

DT = 0.004
RNG = np.random.default_rng(7)
DATA = RNG.normal(size=(4, 40)).astype(np.float32)
DATA[:, ::7] = 0.0  # some exact zeros, which several operations treat specially
DATA[0, 3] = 25.0


def run(stage, data=DATA, dt=DT):
    coll = TraceCollection(data, d_sample=dt)
    return np.array([np.asarray(t) for t in coll | stage])


def sgn(x):
    # SU's SGN macro: 1 for zero
    return np.where(x < 0, -1.0, 1.0)


def slog(x, log):
    safe = np.where(x == 0, 1.0, np.abs(x))
    return np.where(x == 0, 0.0, sgn(x) * log(safe))


POINTWISE = {
    'abs': np.abs,
    'ssqrt': lambda x: sgn(x) * np.sqrt(np.abs(x)),
    'sqr': lambda x: x * x,
    'ssqr': lambda x: sgn(x) * x * x,
    'sgn': sgn,
    'exp': lambda x: np.exp(np.clip(x, None, 40)),
    'sexp': lambda x: sgn(x) * np.exp(np.abs(x)),
    'slog': lambda x: slog(x, np.log),
    'slog2': lambda x: slog(x, np.log2),
    'slog10': lambda x: slog(x, np.log10),
    'cos': np.cos,
    'sin': np.sin,
    'tan': np.tan,
    'cosh': lambda x: np.cosh(np.clip(x, -40, 40)),
    'sinh': lambda x: np.sinh(np.clip(x, -40, 40)),
    'tanh': np.tanh,
    'norm': lambda x: x / np.abs(x).max(axis=1, keepdims=True),
    'db': lambda x: 20.0 * slog(x, np.log10),
    'neg': lambda x: -x,
    'nop': lambda x: x,
    'posonly': lambda x: np.where(x > 0, x, 0.0),
    'negonly': lambda x: np.where(x < 0, x, 0.0),
    'sum': lambda x: np.cumsum(x.astype(np.float64), axis=1),
    'integ': lambda x: np.cumsum(x.astype(np.float64), axis=1),
    'avg': lambda x: x - x.mean(axis=1, keepdims=True),
    'inv': lambda x: np.where(x == 0, 0.0, 1.0 / np.where(x == 0, 1.0, x)),
    'd2m': lambda x: 1000.0 * x,
}


@pytest.mark.parametrize('name', sorted(POINTWISE))
def test_pointwise_operations(name):
    data = DATA
    if name in ('exp', 'cosh', 'sinh'):
        data = np.clip(DATA, -5, 5)
    expected = POINTWISE[name](data.astype(np.float64))
    got = run(getattr(op, name)(), data)
    npt.assert_allclose(got, expected, rtol=2e-4, atol=2e-5)


def test_default_is_abs():
    npt.assert_array_equal(run(op.abs()), np.abs(DATA))


def test_sgn_of_zero_is_one():
    # that is what SU's SGN macro does
    out = run(op.sgn())
    assert (out[DATA == 0] == 1.0).all()


def test_slog2_is_really_log_base_2():
    # (in SU it falls through into slog10)
    x = np.array([[1.0, 2.0, 8.0, -4.0, 0.0, 0.5]], dtype=np.float32)
    npt.assert_allclose(run(op.slog2(), x), [[0.0, 1.0, 3.0, -2.0, 0.0, -1.0]], atol=1e-6)


def test_s2v():
    x = np.array([[1.0, 2.0, 4.0]], dtype=np.float32)
    npt.assert_allclose(run(op.s2v(), x), 1e6 / x)
    npt.assert_allclose(run(op.s2vm(), x), 304800.0 / x)


def test_refl():
    x = DATA.astype(np.float64)
    expected = np.zeros_like(x)
    for i in range(1, x.shape[1]):
        num = x[:, i] - x[:, i - 1]
        den = x[:, i] + x[:, i - 1]
        den = np.where(den == 0, 1.0, den)
        expected[:, i] = num / den
    npt.assert_allclose(run(op.refl()), expected, rtol=2e-4, atol=1e-6)


def test_diff():
    x = DATA.astype(np.float64)
    n = x.shape[1]
    expected = np.zeros_like(x)
    expected[:, 2:n - 2] = (x[:, 3:n - 1] - x[:, 1:n - 3]) / (2 * DT)
    expected[:, 0] = (x[:, 1] - x[:, 0]) / DT
    expected[:, n - 1] = (x[:, n - 1] - x[:, n - 2]) / DT
    expected[:, 1] = (x[:, 2] - x[:, 0]) / (2 * DT)
    expected[:, n - 2] = (x[:, n - 1] - x[:, n - 3]) / (2 * DT)
    npt.assert_allclose(run(op.diff()), expected, rtol=2e-4, atol=1e-3)

    # the derivative of a ramp is its slope
    ramp = (3.0 * np.arange(20) * DT).astype(np.float32)[None, :]
    npt.assert_allclose(run(op.diff(), ramp), 3.0, rtol=1e-4)


def test_diff_needs_dt_and_samples():
    with pytest.raises(ValueError, match="sample interval"):
        run(op.diff(), dt=0.0)
    with pytest.raises(ValueError, match="3 samples"):
        run(op.diff(), np.ones((1, 2), dtype=np.float32))


def test_rmsamp():
    rms = np.sqrt((DATA.astype(np.float64) ** 2).mean(axis=1))
    out = run(op.rmsamp())
    npt.assert_allclose(out[:, 0], rms, rtol=1e-5)
    assert (out[:, 1:] == 0).all()


def test_mod2pi():
    x = np.array([[-1.0, 0.0, 7.0, 6.2, 100.0, -20.0]], dtype=np.float32)
    got = run(op.mod2pi(), x)
    npt.assert_allclose(got, np.mod(x.astype(np.float64), 2 * np.pi), atol=1e-5)
    assert ((got >= 0) & (got < 2 * np.pi + 1e-6)).all()


def test_mean_window():
    nw = 5
    mt = 2
    x = DATA.astype(np.float64)
    n = x.shape[1]
    expected = np.zeros_like(x)
    for i in range(n):
        if i - mt > 0 and i + mt < n:
            expected[:, i] = x[:, i - mt:i + mt + 1].mean(axis=1)
    npt.assert_allclose(run(op.mean(nw=nw)), expected, rtol=1e-4, atol=1e-6)
    # an even window is made odd
    npt.assert_array_equal(run(op.mean(nw=4)), run(op.mean(nw=5)))


def window_stat(x, nw, fn):
    """SU's window operations: zero where the window sticks out of the trace (or touches its start)."""
    mt = (nw - 1) // 2
    n = x.shape[1]
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(n):
        if i - mt > 0 and i + mt < n:
            out[:, i] = fn(x[:, i - mt:i + mt + 1].astype(np.float64))
    return out


def test_std_and_var():
    # (in suop.c these also add up the samples themselves, which is a bug)
    npt.assert_allclose(run(op.var(nw=7)), window_stat(DATA, 7, lambda w: w.var(axis=1)), rtol=2e-4, atol=1e-5)
    npt.assert_allclose(run(op.std(nw=7)), window_stat(DATA, 7, lambda w: w.std(axis=1)), rtol=2e-4, atol=1e-5)
    # a constant has no variance
    npt.assert_allclose(run(op.var(), np.full((1, 60), 3.0, dtype=np.float32)), 0.0, atol=1e-5)


def test_despike():
    nw = 5
    mt = 2
    x = DATA.astype(np.float64)
    n = x.shape[1]
    expected = np.zeros_like(x)
    for i in range(n):
        if i - mt >= 0 and i + mt < n:
            w = x[:, i - mt:i + mt + 1]
        elif i - mt < 0:
            w = x[:, :nw]
        else:
            w = x[:, n - nw:]
        expected[:, i] = np.median(w, axis=1)
    npt.assert_allclose(run(op.despike(nw=nw)), expected, rtol=1e-6)
    # a lone spike is removed
    x = np.zeros((1, 30), dtype=np.float32)
    x[0, 12] = 100.0
    npt.assert_array_equal(run(op.despike(nw=5), x), 0.0)
    with pytest.raises(ValueError, match="at least"):
        run(op.despike(nw=5), np.ones((1, 4), dtype=np.float32))


def test_derivatives():
    # SU's drv2 and drv4 are the negative of the derivative, and leave the ends of the trace alone
    ramp = (3.0 * np.arange(30) * DT).astype(np.float32)[None, :]
    got4 = run(op.drv4(), ramp)[0]
    npt.assert_allclose(got4[2:-2], -3.0, rtol=1e-4)
    npt.assert_array_equal(got4[:2], ramp[0, :2])
    npt.assert_array_equal(got4[-2:], ramp[0, -2:])
    got2 = run(op.drv2(), ramp)[0]
    npt.assert_allclose(got2[1:-1], -1.5 * np.ones(28) / 1.0 * (1.0), rtol=1e-4)  # (x[i-1]-x[i])/(2dt)
    npt.assert_array_equal(got2[[0, -1]], ramp[0, [0, -1]])
    with pytest.raises(ValueError, match="at least"):
        run(op.drv4(), np.ones((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="sample interval"):
        run(op.drv2(), dt=0.0)


def test_spike_and_saf():
    x = np.array([[0, 1, 3, 2, 2.5, 4, 1, 1, 0.5, 2, 3]], dtype=np.float32)
    # local extrema kept as spikes (the ends are zero)
    npt.assert_array_equal(run(op.spike(), x), [[0, 0, 3, 2, 0, 4, 0, 0, 0.5, 0, 0]])

    n = x.shape[1]
    tmp = np.zeros(n)
    iold, vold = 0, 0.0
    for i in range(1, n - 1):
        x1, x2, x3 = x[0, i - 1], x[0, i], x[0, i + 1]
        if (x1 < x2 and x3 < x2) or (x1 > x2 and x3 > x2):
            tmp[i] = x2
            tmp[iold:i] = vold
            iold, vold = i, x2
        else:
            tmp[i] = 0.0
    tmp[0] = 0.0
    npt.assert_array_equal(run(op.saf(), x)[0], tmp)


def test_lnza():
    def sg(v):
        return -1.0 if v < 0 else 1.0

    x = DATA[:, :]
    expected = x.astype(np.float64).copy()
    for r in range(x.shape[0]):
        for i in range(1, x.shape[1] - 1):
            a, b, c = x[r, i - 1], x[r, i], x[r, i + 1]
            if sg(a) == sg(b) == sg(c):
                expected[r, i] = 0.0
            elif sg(a) == sg(b) and sg(b) != sg(c):
                expected[r, i] = b if abs(b) < abs(c) else c
            elif sg(a) != sg(b) and sg(b) == sg(c):
                expected[r, i] = a if abs(a) < abs(b) else b
        # the first and last samples are set to zero
        expected[r, 0] = 0.0
        expected[r, -1] = 0.0
    npt.assert_allclose(run(op.lnza()), expected, rtol=1e-6, atol=1e-7)


def test_freq():
    # a sine of 23.37 Hz. (SU finds minima by strict comparison, so one that falls exactly between two samples
    # is skipped now and then, which is why this looks at the median.)
    t = np.arange(400) * DT
    x = np.sin(2 * np.pi * 23.37 * t).astype(np.float32)[None, :]
    got = run(op.freq(), x)[0]
    assert np.median(got[20:300]) == pytest.approx(23.37, rel=0.1)
    # nothing after the last minimum is left unset
    assert np.isfinite(got).all()

    # and the same as the algorithm written out
    n = x.shape[1]
    tmp = np.zeros(n)
    for i in range(1, n - 1):
        if x[0, i] > x[0, i - 1] and x[0, i] > x[0, i + 1]:
            tmp[i] = 1
        elif x[0, i] < x[0, i - 1] and x[0, i] < x[0, i + 1]:
            tmp[i] = -1
    out = np.zeros(n)
    iold = 0
    for i in range(5, n):
        if tmp[i] == -1:
            out[iold:i] = 1.0 / ((i - iold) * DT)
            iold = i
    npt.assert_allclose(got, out, rtol=1e-5)


def test_cnorm():
    x = np.array([[3.0, 4.0, 0.0, 2.0, -5.0, 12.0]], dtype=np.float32)
    npt.assert_allclose(run(op.cnorm(), x), [[0.6, 0.8, 0.0, 1.0, -5 / 13, 12 / 13]], rtol=1e-6)


def test_each_operation_is_its_own_stage():
    # (the panel operations, `mix` and the binary ones, are in _panels.py and take a second data set, and the operations on
    # a whole data set are in _dataset.py)
    panels = {'mix', 'sum2', 'diff2', 'prod2', 'quo2', 'ptsum', 'ptdiff', 'ptprod', 'ptquo', 'zipper', 'zippol', 'flip', 'vcat'}
    assert panels <= set(op.__all__)
    names = [name for name in op.__all__ if name not in panels]
    assert len(names) == len(set(names)) == 44
    for name in names:
        assert callable(getattr(op, name)) and getattr(op, name).__doc__
        assert repr(getattr(op, name)()) == f"{name}()"
    # there is no catch-all, and parameters are checked straight away
    assert not hasattr(op, 'not_an_op')
    with pytest.raises(ValueError):
        op.mean(nw=0)
    with pytest.raises(TypeError):
        op.abs(nw=5)  # (only the window operations have a window)
    with pytest.raises(TypeError):
        op.mean(not_a_parameter=1)


def test_inplace():
    coll = TraceCollection(DATA.copy(), d_sample=DT).to_memory()
    list(coll | op.neg())
    npt.assert_array_equal(np.array([np.asarray(t) for t in coll]), DATA)
    list(coll | op.neg(inplace=True))
    npt.assert_array_equal(np.array([np.asarray(t) for t in coll]), -DATA)


def test_pmap_matches_sequential():
    s = op.ssqrt()
    expected = run(s)
    coll = TraceCollection(DATA, d_sample=DT)
    got = np.array([np.asarray(t) for t in coll | pmap(s, workers=2, chunk=1)])
    npt.assert_array_equal(got, expected)
    assert s.parallelism == 'trace'
    assert repr(op.mean(nw=7)) == "mean(nw=7)"


def test_stages_pickle_and_run_in_processes():
    import pickle

    # one from numpy, one from the SU sources, and a window operation
    stage = op.slog10() | op.saf() | op.mean(nw=5)
    again = pickle.loads(pickle.dumps(stage))
    assert repr(again) == repr(stage)
    coll = TraceCollection(DATA, d_sample=DT)
    expected = np.array([np.asarray(t) for t in coll | stage])
    got = np.array([np.asarray(t) for t in coll | pmap(again, workers=2, chunk=1, executor='process')])
    npt.assert_array_equal(got, expected)


def test_c_operations_match_their_references_on_changing_lengths():
    # (the scratch arrays are made for every trace)
    from seispy.container import Trace, from_iterable
    traces = [Trace(DATA[0, :n], d_sample=DT) for n in (40, 12, 30, 12)]
    got = [np.asarray(t) for t in from_iterable(traces) | op.saf()]
    for n, g in zip((40, 12, 30, 12), got):
        npt.assert_array_equal(g, run(op.saf(), DATA[:1, :n])[0])
