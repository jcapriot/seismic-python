import numpy as np
import numpy.testing as npt
import pytest

from seispy.amplitudes import centsamp, divcor, pgc
from seispy.container import Trace, from_iterable

DT = 0.004


def traces(data, dt=DT, sample_start=0.0):
    return [Trace(np.asarray(row, dtype=np.float32), d_sample=dt, sample_start=sample_start) for row in np.atleast_2d(data)]


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


RNG = np.random.default_rng(11)
DATA = RNG.normal(size=(5, 60)).astype(np.float32)


# --------------------------------------------------------------------------------------------------------- divcor
def test_divcor_default_is_time_over_dt():
    out = arr(run(divcor(), traces(DATA)))
    npt.assert_allclose(out, DATA * (np.arange(60) * DT / DT), rtol=1e-5, atol=1e-6)


def test_divcor_follows_the_velocity_function():
    trms, vrms = [0.0, 0.1, 0.2], [1500.0, 2000.0, 2500.0]
    out = arr(run(divcor(trms=trms, vrms=vrms), traces(DATA)))
    t = np.arange(60) * DT
    v = np.interp(t, trms, vrms)
    denom = 0.1 * 2000.0**2  # (the second pick)
    npt.assert_allclose(out, DATA * (t * v * v / denom), rtol=1e-5, atol=1e-6)


def test_divcor_uses_the_start_time_and_a_velocity_per_sample():
    vt = np.linspace(1500, 3000, 60)
    out = arr(run(divcor(vt=vt), traces(DATA, sample_start=0.1)))
    t = 0.1 + np.arange(60) * DT
    npt.assert_allclose(out, DATA * (t * vt**2 / ((0.1 + DT) * vt[1] ** 2)), rtol=1e-5)
    with pytest.raises(ValueError):
        run(divcor(vt=vt[:10]), traces(DATA))


def test_divcor_one_pair_is_a_constant_velocity():
    out = arr(run(divcor(trms=[0.05], vrms=[2000.0]), traces(DATA)))
    t = np.arange(60) * DT
    npt.assert_allclose(out, DATA * (t / (0.05 + DT)), rtol=1e-5, atol=1e-6)


def test_divcor_checks_the_function():
    with pytest.raises(ValueError):
        divcor(trms=[0.0, 1.0], vrms=[1500.0])
    with pytest.raises(ValueError):
        divcor(trms=[1.0, 0.5], vrms=[1500.0, 1600.0])


def test_divcor_works_on_complex_traces():
    z = (DATA + 1j * DATA[::-1]).astype(np.complex64)
    out = arr(run(divcor(), [Trace(z[0], d_sample=DT)]))[0]
    npt.assert_allclose(out, z[0] * np.arange(60), rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------------------------------------------------ pgc
def test_pgc_applies_one_gain_to_all_the_traces():
    out = arr(run(pgc(ntrscan=3, lwindow=0.2), traces(DATA)))
    lw = int(0.5 * 0.2 / DT + 0.5)  # 25 samples, the window of 50
    total = np.abs(DATA[:3]).sum(axis=0)
    g = np.empty(60)
    for j in range(60):
        window = total[max(j - lw, 0):min(j + lw, 60)]
        g[j] = len(window) * 3 / window.sum()
    npt.assert_allclose(out, DATA * g, rtol=1e-4)


def test_pgc_keeps_the_relative_amplitudes():
    big = DATA * np.array([[1.0], [10.0], [100.0], [1.0], [1.0]])
    out = arr(run(pgc(ntrscan=5, lwindow=0.5), traces(big)))
    ratio = out / big
    npt.assert_allclose(ratio, np.broadcast_to(ratio[0], ratio.shape), rtol=1e-4)


def test_pgc_scans_what_there_is_and_streams_the_rest():
    from seispy.amplitudes import pgc as make

    few = arr(run(make(ntrscan=100), traces(DATA)))
    all_five = arr(run(make(ntrscan=5), traces(DATA)))
    npt.assert_allclose(few, all_five)
    # the traces after the scan are scaled with the same function
    short = arr(run(make(ntrscan=2), traces(DATA)))
    assert short.shape == DATA.shape
    npt.assert_allclose(short[3] / DATA[3], short[2] / DATA[2] * (DATA[3] / DATA[3]) , rtol=1e-4)
    assert list(run(make(), [])) == []


def test_pgc_zero_data_and_errors():
    out = arr(run(pgc(), traces(np.zeros((2, 20)))))
    assert not out.any()
    with pytest.raises(ValueError):
        pgc(ntrscan=0)
    with pytest.raises(ValueError):
        run(pgc(), traces(DATA[:1]) + traces(DATA[1:2, :30]))


# ------------------------------------------------------------------------------------------------------ centsamp
LOBES = np.array([1, 3, 4, 3, 1, -1, -3, -4, -3, -1, 2, 5, 2, -1, -2, -1, 3, 3], dtype=np.float32)


def test_centsamp_puts_a_spike_at_the_centre_of_each_lobe():
    (out,) = run(centsamp(), traces(LOBES))
    spikes = np.flatnonzero(np.asarray(out))
    # the lobes end at a sign change, so the last one is not given a spike
    assert list(spikes) == [2, 7, 11, 14]
    values = np.asarray(out)[spikes]
    assert (np.sign(values) == [1, -1, 1, -1]).all()
    assert out.n_sample == len(LOBES)


def test_centsamp_is_independent_of_the_other_traces():
    one = arr(run(centsamp(), traces(LOBES)))
    two = arr(run(centsamp(), traces(np.vstack([LOBES, DATA[0][:18]]))))
    npt.assert_array_equal(one[0], two[0])  # (no state is carried from one trace to the next)


def test_centsamp_nvals_min_drops_short_lobes():
    kept = np.flatnonzero(np.asarray(run(centsamp(nvals_min=4), traces(LOBES))[0]))
    assert list(kept) == [2, 7]  # (the lobes of 3 samples are gone)


def test_centsamp_zeros_and_errors():
    assert not arr(run(centsamp(), traces(np.zeros((1, 10))))).any()
    with pytest.raises(ValueError):
        centsamp(nvals_min=0)
    z = Trace((LOBES + 1j).astype(np.complex64), d_sample=DT)
    with pytest.raises(TypeError):
        run(centsamp(), [z])


# ---------------------------------------------------------------------------------------------------- impedance
def test_impedance_recursion():
    from seispy.amplitudes import impedance

    r = np.array([0.1, -0.2, 0.05, 0.5, -0.999999, 0.99999999, 0.0], dtype=np.float32)
    (out,) = run(impedance(v0=2000.0, rho0=2.0e6), traces(r))
    z = [2000.0 * 2.0e6]
    for rc in np.clip(r.astype(np.float64), -0.9999, 0.9999)[:-1]:
        z.append(z[-1] * (1 + rc) / (1 - rc))
    npt.assert_allclose(np.asarray(out), z, rtol=1e-5)
    # the defaults
    (first,) = run(impedance(), traces([0.0, 0.0]))
    assert np.asarray(first)[0] == pytest.approx(1500.0 * 1.0e6, rel=1e-6)


def test_impedance_is_the_inverse_of_the_reflectivity_of_the_impedance():
    from seispy.amplitudes import impedance

    z = np.array([3.0e6, 4.0e6, 3.5e6, 6.0e6, 6.0e6], dtype=np.float32)
    r = (z[1:] - z[:-1]) / (z[1:] + z[:-1])
    (out,) = run(impedance(v0=1.0, rho0=float(z[0])), traces(np.append(r, 0.0)))
    npt.assert_allclose(np.asarray(out), z, rtol=1e-4)


# ------------------------------------------------------------------------------------------------------ vlength
def test_vlength_pads_and_cuts():
    from seispy.windowing import vlength

    short, middle, long = traces(DATA[0, :10])[0], traces(DATA[1, :20])[0], traces(DATA[2, :30])[0]
    out = run(vlength(), [middle, short, long])
    assert [t.n_sample for t in out] == [20, 20, 20]  # (the length of the first)
    npt.assert_array_equal(np.asarray(out[1])[:10], DATA[0, :10])
    assert not np.asarray(out[1])[10:].any()
    npt.assert_array_equal(np.asarray(out[2]), DATA[2, :20])
    out = run(vlength(ns=25), [short, long])
    assert [t.n_sample for t in out] == [25, 25]
    with pytest.raises(ValueError):
        vlength(ns=0)
    assert vlength(ns=5).parallelism == 'trace' and vlength().parallelism == 'serial'
