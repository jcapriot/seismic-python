"""
SUTIVEL, SUVEL2DF, SURELANAN and SUVELAN_UCCS, _NCCS, _USEL and _NSEL.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.velocity import relanan, tivel, vel2df, velan_nccs, velan_nsel, velan_uccs, velan_usel

DT = 0.004


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def ricker(t, f=30.0):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def gather(velocity, t0=0.6, offsets=None, nt=250, cdp=1, dt=DT, eta_vh=None):
    offsets = np.arange(0.0, 1500.0, 100.0) if offsets is None else offsets
    t = np.arange(nt) * dt
    out = []
    for h in offsets:
        arrival = np.sqrt(t0 ** 2 + h ** 2 / velocity ** 2)
        x = ricker(t - arrival)
        out.append(Trace(x.astype(np.float32), d_sample=dt).replace(offset=float(h), ensemble_number=cdp))
    return out


# ------------------------------------------------------------------------------------------------------- tivel
def test_tivel_isotropic_medium_has_the_vertical_velocity_everywhere():
    for option in (4, 5, 6):
        increment, values = tivel(a=2000.0, b=1000.0, e=0.0, d=0.0, option=option, nangle=91, np_values=21)
        npt.assert_allclose(values[:-3], 2000.0, rtol=1e-4)
    # (the NMO velocity of an isotropic medium is the velocity over cos(theta), the angle of the phase)
    _, vnmo = tivel(a=2000.0, b=1000.0, e=0.0, d=0.0, option=2, nangle=91)
    npt.assert_allclose(vnmo[:-1], 2000.0 / np.cos(np.radians(np.arange(90.0))), rtol=1e-4)
    # (the derivatives of the phase velocity are 0, and so is the test of option 13)
    for option in (7, 8, 10, 11, 13):
        _, values = tivel(a=2000.0, b=1000.0, e=0.0, d=0.0, option=option, nangle=91, np_values=21)
        npt.assert_allclose(values, 0.0, atol=1e-3 if option != 13 else 1e-9)


def test_tivel_increments_and_tables_as_a_function_of_p():
    deg, values = tivel(option=2, nangle=19, maxangle=90.0)
    assert deg == pytest.approx(5.0) and values.shape == (19,)
    rad, _ = tivel(option=3, nangle=19, maxangle=90.0)
    assert rad == pytest.approx(5.0 * np.pi / 180.0)
    # the NMO velocity at 0 is a sqrt(1 + 2 d)
    assert values[0] == pytest.approx(2500.0 * np.sqrt(1.2), rel=1e-3)
    dp, vnmo = tivel(option=1, np_values=41, nangle=901)
    assert vnmo.shape == (41,) and dp > 0 and vnmo[0] == pytest.approx(values[0], rel=1e-4)
    # options 7 and 10 (derivatives as a function of p): the program leaves them out
    _, dv = tivel(option=7, np_values=41, nangle=901)
    assert dv.shape == (41,) and np.abs(dv).max() > 0
    # weak anisotropy: the NMO velocity at p = 0, and a table up to the evanescent region
    dp_w, weak = tivel(option=14, np_values=21)
    assert weak[0] == pytest.approx(2500.0 * np.sqrt(1.2), rel=1e-4)
    assert (weak > 0).all() and dp_w > 0
    with pytest.raises(ValueError):
        tivel(option=15)


def test_tivel_test_of_option_13_is_zero_when_delta_is_epsilon():
    _, values = tivel(e=0.15, d=0.15, option=13, nangle=181, maxangle=60.0)
    assert np.abs(values).max() < 1e-9
    _, off = tivel(e=0.25, d=0.05, option=13, nangle=181, maxangle=60.0)
    assert np.abs(off).max() > 1e-9


# ------------------------------------------------------------------------------------------------------- vel2df
def test_vel2df_peaks_at_the_velocity_of_an_isotropic_event():
    v0 = 2200.0
    out = run(vel2df(tn=0.6, nv=21, dv=50.0, fv=1700.0, nvh=21, dvh=50.0, fvh=1700.0, offsetm=1500.0),
              gather(v0))
    assert len(out) == 21 and out[0].n_sample == 21
    panel = np.array([np.asarray(t) for t in out])
    iv, ivh = np.unravel_index(np.argmax(panel), panel.shape)
    assert abs((1700.0 + 50.0 * iv) - v0) <= 50.0
    # an isotropic event is where the two velocities are the same
    assert abs((1700.0 + 50.0 * ivh) - v0) <= 150.0
    assert [t.header['ensemble_trace_number'] for t in out] == list(range(1, 22))
    assert out[0].d_sample == pytest.approx(50.0) and out[0].header['offset'] == 0.0
    # the semblance is a ratio of energies: from 0 to 1
    assert panel.min() >= 0.0 and panel.max() <= 1.0 + 1e-5


def test_vel2df_options_and_gathers():
    traces = gather(2000.0, cdp=1) + gather(2500.0, cdp=2)
    out = run(vel2df(tn=0.6, nv=5, nvh=4), traces)
    assert len(out) == 10
    # offsetm drops the far traces: nothing is left from 1.5 km beyond a short one
    none = run(vel2df(tn=0.6, nv=3, nvh=3, offsetm=-1.0), gather(2000.0))
    assert not np.asarray(none[0]).any()
    with pytest.raises(ValueError):
        vel2df(tn=0.5, nv=0)
    with pytest.raises(TypeError):
        vel2df()
    with pytest.raises(ValueError, match="sample interval"):
        run(vel2df(tn=0.5), [Trace(np.ones(50, dtype=np.float32), d_sample=0.0).replace(offset=100.0)])


# ------------------------------------------------------------------------------------------------------ relanan
def test_relanan_finds_the_residual_moveout_parameters():
    z0, r1_true, r2_true, dz = 1000.0, 0.06, -0.03, 10.0
    nz = 200
    depths = np.arange(nz) * dz
    traces = []
    for offset in np.arange(0.0, 2000.0, 200.0):
        h = offset / 2.0
        z = np.sqrt(z0 ** 2 + r1_true * h ** 2 + r2_true * h ** 4 / (h ** 2 + z0 ** 2))
        x = ricker((depths - z) / 1000.0, 50.0)  # (a wavelet of about 30 m)
        traces.append(Trace(x.astype(np.float32), d_sample=dz).replace(offset=float(offset), ensemble_number=3))
    out = run(relanan(x=[1.0, 10.0], z=[z0, z0], nr1=11, dr1=0.02, fr1=-0.05, nr2=11, dr2=0.02, fr2=-0.15,
                      dzratio=2),
              traces)
    assert len(out) == 11 and out[0].n_sample == 11
    panel = np.array([np.asarray(t) for t in out])
    i1, i2 = np.unravel_index(np.argmax(panel), panel.shape)
    r1, r2 = -0.05 + 0.02 * i1, -0.15 + 0.02 * i2
    # (the two parameters trade off: what is found is a residual moveout that is close to the true one at all offsets)
    for h in (500.0, 1000.0):
        found = np.sqrt(z0 ** 2 + r1 * h ** 2 + r2 * h ** 4 / (h ** 2 + z0 ** 2))
        true = np.sqrt(z0 ** 2 + r1_true * h ** 2 + r2_true * h ** 4 / (h ** 2 + z0 ** 2))
        assert abs(found - true) <= 12.0  # (a depth sample is 10 m)
    assert out[0].d_sample == pytest.approx(0.02) and out[0].header['sample_start'] == pytest.approx(-0.15)


def test_relanan_reflector_methods_and_errors():
    traces = [Trace(np.ones(40, dtype=np.float32), d_sample=10.0).replace(offset=100.0, ensemble_number=c)
              for c in (1, 2, 3)]
    # the depth is found for each gather (the program has the one of the first)
    for method in ('linear', 'mono', 'akima', 'spline'):
        out = run(relanan(x=[1.0, 2.0, 3.0, 4.0], z=[100.0, 150.0, 120.0, 90.0], nr1=2, nr2=2, method=method), traces)
        assert len(out) == 6
    with pytest.raises(ValueError, match="unknown"):
        relanan(x=[1.0, 2.0], z=[1.0, 2.0], method='cubic')
    with pytest.raises(ValueError):
        relanan(x=[1.0, 2.0], z=[1.0])
    with pytest.raises(ValueError):
        relanan(x=[1.0], z=[1.0], nr1=0)


# ------------------------------------------------------------------------------------------- velan cc sums
@pytest.mark.parametrize('stage', [velan_uccs, velan_nccs, velan_usel, velan_nsel])
def test_cross_correlation_sums_peak_at_the_stacking_velocity(stage):
    v0 = 2400.0
    out = run(stage(nv=20, dv=50.0, fv=1900.0), gather(v0))
    assert len(out) == 20
    panel = np.array([np.asarray(t) for t in out])
    assert panel.shape[1] == 1 + (250 - 1) // 5
    best_v, best_t = np.unravel_index(np.argmax(panel), panel.shape)
    assert abs((1900.0 + 50.0 * best_v) - v0) <= 100.0
    assert abs(best_t * 5 * DT - 0.6) <= 0.04
    assert out[0].d_sample == pytest.approx(5 * DT) and out[0].header['offset'] == 0.0
    assert [t.header['ensemble_trace_number'] for t in out] == list(range(1, 21))


def test_cross_correlation_sums_normalization_and_selection():
    traces = gather(2400.0)
    unnormalized = np.array([np.asarray(t) for t in run(velan_uccs(nv=10, fv=2100.0, dv=100.0), traces)])
    normalized = np.array([np.asarray(t) for t in run(velan_nccs(nv=10, fv=2100.0, dv=100.0), traces)])
    # the normalized sum is a coherence of at most 1 (the amplitudes are divided out), the other is not
    assert normalized.max() <= 1.0 + 1e-5 and normalized.min() >= 0.0
    assert unnormalized.max() > 5 * normalized.max()
    # a high threshold keeps fewer pairs (the sum of fewer terms of a coherent event), tau=0 keeps all of them
    all_pairs = np.array([np.asarray(t) for t in run(velan_usel(nv=10, fv=2100.0, dv=100.0, tau=0.0), traces)])
    npt.assert_allclose(all_pairs, unnormalized, rtol=1e-4, atol=1e-5)
    fewer = np.array([np.asarray(t) for t in run(velan_usel(nv=10, fv=2100.0, dv=100.0, tau=0.7), traces)])
    assert np.abs(fewer).max() < np.abs(unnormalized).max()
    # pwr
    squared = np.array([np.asarray(t) for t in run(velan_nccs(nv=10, fv=2100.0, dv=100.0, pwr=2.0), traces)])
    npt.assert_allclose(squared, normalized ** 2, rtol=1e-4, atol=1e-6)


def test_cross_correlation_sums_gathers_and_errors():
    two = gather(2000.0, cdp=1) + gather(2600.0, cdp=2)
    assert len(run(velan_nsel(nv=4), two)) == 8
    # a gather of one trace has no pairs
    (one,) = run(velan_uccs(nv=1), gather(2000.0, offsets=[100.0]))
    assert not np.asarray(one).any()
    for bad in (dict(smute=1.0), dict(pwr=0.0), dict(pwr=-1.0), dict(nv=0), dict(dtratio=0)):
        with pytest.raises(ValueError):
            velan_uccs(**bad)
    with pytest.raises(ValueError):
        velan_usel(tau=1.5)
    assert velan_nccs().parallelism == 'ensemble'
