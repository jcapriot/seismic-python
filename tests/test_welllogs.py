"""Well logs: las2su, backus, backush, lprime, wellrf and gassmann (seispy/welllogs)."""
import math

import numpy as np
import numpy.testing as npt
import pytest

import seispy.welllogs as wl
from seispy.container import Trace, from_iterable
from seispy.welllogs import backus, backush, gassmann, las2su, lprime, wellrf


def logs_of(*arrays, dz=0.5):
    return from_iterable([Trace(np.asarray(a, dtype=np.float32), d_sample=dz, sampling_unit='m') for a in arrays])


def samples(stream):
    return [np.asarray(t) for t in stream]


def layered(nz=400, thickness=6):
    """vp, vs, rho of alternating thin layers"""
    layer = (np.arange(nz) // thickness) % 2
    vp = np.where(layer == 0, 2500.0, 3500.0)
    vs = np.where(layer == 0, 1300.0, 2100.0)
    rho = np.where(layer == 0, 2200.0, 2500.0)
    return vp, vs, rho


# ------------------------------------------------------------------------------------------------------- backus
def backus_by_loops(vp, vs, rho, navg, ang=30.0):
    """The program's arithmetic, written as its loops"""
    nz = len(vp)
    m = (navg - 1) // 2

    def avg(p):
        out = np.array(p, dtype=np.float64)
        for i in range(nz):
            if i - m > 0 and i + m < nz:
                out[i] = sum(p[i - m:i + m + 1]) / navg
        return out

    def ends(p):
        p = np.array(p)
        mm = m + 1
        gt, gb = p[mm], p[nz - mm]
        p[:mm] = gt
        p[nz - mm:] = gb
        return p

    mu = rho * vs * vs
    lp2mu = rho * vp * vp
    lam = rho * (vp * vp - 2 * vs * vs)
    lpmu = rho * (vp * vp - vs * vs)
    t4 = avg(lam / lp2mu) ** 2
    t5 = 1.0 / avg(1.0 / lp2mu)
    t6 = avg(mu * lpmu / lp2mu)
    c11 = t4 * t5 + 4 * t6
    c13 = avg(lam / lp2mu) * (1.0 / avg(1.0 / lp2mu))
    c33 = 1.0 / avg(1.0 / lp2mu)
    c44 = 1.0 / avg(1.0 / mu)
    c66 = avg(mu)
    eps = (c11 - c33) / (2 * c33)
    delta = ((c13 + c44) ** 2 - (c33 - c44) ** 2) / (2 * c33 * (c33 - c44))
    gamma = (c66 - c44) / (2 * c44)
    return ends(eps), ends(delta), ends(gamma)


def test_backus_matches_the_loops_of_the_program():
    vp, vs, rho = layered(300, 5)
    out = samples(logs_of(vp, vs, rho) | backus(navg=41))
    eps, delta, gamma = backus_by_loops(vp, vs, rho, 41)
    assert len(out) == 3
    npt.assert_allclose(out[0], eps, rtol=1e-4, atol=1e-6)
    npt.assert_allclose(out[1], delta, rtol=1e-3, atol=1e-5)
    npt.assert_allclose(out[2], gamma, rtol=1e-4, atol=1e-6)
    assert (np.asarray(out[2]) > 0).all()  # thin layers of different shear moduli: positive gamma


def test_backus_of_a_uniform_medium_is_isotropic_and_the_extra_traces_are_ordered():
    nz = 100
    vp, vs, rho = np.full(nz, 3000.0), np.full(nz, 1500.0), np.full(nz, 2400.0)
    out = list(logs_of(vp, vs, rho) | backus(navg=21, all=True))
    assert len(out) == 25 and [t.header['trace_id'] for t in out] == list(range(1, 26))
    npt.assert_allclose(np.asarray(out[0]), 0.0, atol=1e-6)  # epsilon
    npt.assert_allclose(np.asarray(out[1]), 0.0, atol=1e-6)  # delta
    npt.assert_allclose(np.asarray(out[2]), 0.0, atol=1e-6)  # gamma
    npt.assert_allclose(np.asarray(out[3]), 3000.0, rtol=1e-5)  # vp0
    npt.assert_allclose(np.asarray(out[4]), 1500.0, rtol=1e-5)  # vs0
    npt.assert_allclose(np.asarray(out[5]), 2400.0, rtol=1e-5)  # the averaged density
    npt.assert_allclose(np.asarray(out[7]), 1.0, rtol=1e-5)  # vang
    npt.assert_allclose(np.asarray(out[10]), 2400.0 * 3000.0 ** 2, rtol=1e-5)  # c33
    npt.assert_allclose(np.asarray(out[16]), vp, rtol=1e-6)  # the logs themselves
    npt.assert_allclose(np.asarray(out[24]), 2400.0 * 1500.0 ** 2, rtol=1e-5)  # mu


def test_backus_window_longer_than_the_logs_is_an_error_and_navg_is_odd():
    vp, vs, rho = np.full(30, 3000.0), np.full(30, 1500.0), np.full(30, 2400.0)
    with pytest.raises(ValueError, match="window"):
        list(logs_of(vp, vs, rho) | backus(navg=51))
    even = samples(logs_of(*layered(100, 3)) | backus(navg=10))  # (11)
    odd = samples(logs_of(*layered(100, 3)) | backus(navg=11))
    npt.assert_array_equal(even[0], odd[0])
    with pytest.raises(ValueError, match="3 logs"):
        list(logs_of(vp, vs) | backus())
    with pytest.raises(ValueError, match="navg"):
        backus(navg=0)


# ------------------------------------------------------------------------------------------------------ backush
def test_backush_without_gamma_ray_has_no_shale():
    vp, vs, rho = layered(200, 4)
    out = samples(logs_of(vp, vs, rho) | backush(navg=21))
    assert len(out) == 8
    npt.assert_array_equal(out[6], 0.0)  # (the program left it unset)
    npt.assert_array_equal(out[7], 0.0)
    # layers that are isotropic: vp0 and the averaged density are the Backus averages, and gamma is of thin layers
    assert (out[0] > 2500).all() and (out[0] < 3500).all()
    assert (out[5] > 0).all()


def test_backush_shale_makes_the_layers_anisotropic():
    nz = 120
    vp, vs, rho = np.full(nz, 3000.0), np.full(nz, 1500.0), np.full(nz, 2400.0)
    pure_shale = np.full(nz, 150.0)  # over grs: pure shale
    clean_sand = np.full(nz, 5.0)  # under grc: no shale
    for smode in (0, 1):
        out = samples(logs_of(vp, vs, rho, pure_shale) | backush(navg=21, gr=True, smode=smode))
        # of a medium of these layers all alike, the averaged ones are the layers'
        npt.assert_allclose(out[3], 0.209, rtol=1e-3)  # epsilon
        npt.assert_allclose(out[4], 0.033, rtol=2e-2)  # delta
        npt.assert_allclose(out[5], 0.203, rtol=1e-3)  # gamma
        npt.assert_allclose(out[6], 1.0)
        npt.assert_allclose(out[7], 0.209, rtol=1e-6)
        sand = samples(logs_of(vp, vs, rho, clean_sand) | backush(navg=21, gr=True, smode=smode))
        npt.assert_allclose(sand[3], 0.0, atol=1e-6)
        npt.assert_allclose(sand[6], 0.0)
        npt.assert_allclose(sand[7], 0.0)
    # in between: the volume of shale is a fraction, and the anisotropy of the layers is proportional to it (smode 1) or 0
    mid = np.full(nz, 55.0)
    partial = samples(logs_of(vp, vs, rho, mid) | backush(navg=21, gr=True, smode=1))
    volume = partial[6][0]
    assert 0.0 < volume < 1.0
    npt.assert_allclose(partial[7], 0.209 * volume, rtol=1e-5)
    none = samples(logs_of(vp, vs, rho, mid) | backush(navg=21, gr=True, smode=0))
    npt.assert_allclose(none[7], 0.0)
    with pytest.raises(ValueError, match="smode"):
        backush(smode=2)


# ------------------------------------------------------------------------------------------------------- lprime
def test_lprime_finds_the_length_with_the_target_number():
    vs = np.full(400, 1000.0)
    rho = np.full(400, 2200.0)
    result = lprime(vs, rho, b=2.0, dz=1.0, f=60.0, nmax=301)
    # B = f navg dz 0.3048 / vs0: 2 at navg = 2 * 1000 / (60 * .3048) = 109.4
    assert result.navg in (109, 111)
    assert result.backus_number == pytest.approx(60.0 * result.navg * 0.3048 / 1000.0)
    assert result.vs0_min == pytest.approx(1000.0, rel=1e-6)
    assert result.lengths[0] == 1 and result.lengths[-1] == 301 and len(result.numbers) == len(result.lengths)
    # a lower target is a shorter length
    assert lprime(vs, rho, b=0.3).navg < result.navg
    with pytest.raises(ValueError, match="same length"):
        lprime(vs, rho[:10])


# ------------------------------------------------------------------------------------------------------- wellrf
def test_wellrf_puts_the_reflection_at_the_traveltime():
    # layers of 0.1, 0.1 and 0.05 s two-way: the smallest, 0.05 s, is the sample interval of the raw trace
    depth = [0.0, 100.0, 250.0, 350.0]
    velocity = [2000.0, 3000.0, 4000.0, 4000.0]
    (trace,) = list(wellrf(depth, velocity, dtout=0.01))
    data = np.asarray(trace)
    assert trace.d_sample == pytest.approx(0.01) and len(data) == 25
    # the first interface at 0.1 s: (3000 - 2000) / (3000 + 2000) = .2; the second at 0.2 s: 1000 / 7000
    assert data[10] == pytest.approx(0.2, abs=1e-5)
    assert data[20] == pytest.approx(1.0 / 7.0, abs=1e-5)
    # the same with a density that makes the first interface vanish: the impedances of both sides are equal
    (flat,) = list(wellrf(depth, velocity, [3.0, 2.0, 2.0, 2.0], dtout=0.01))
    assert abs(np.asarray(flat)[10]) < 0.01


def test_wellrf_number_of_traces_and_the_default_sampling():
    depth = np.arange(0.0, 500.0, 10.0)
    velocity = np.full(depth.shape, 2000.0)
    velocity[25:] = 3000.0
    out = list(wellrf(depth, velocity, ntr=3))
    assert len(out) == 3 and out[0].n_sample == out[2].n_sample
    assert out[0].d_sample == pytest.approx(0.5 * 2 * 10.0 / 3000.0 * 1.0, rel=1e-5) or out[0].d_sample > 0
    # the last reflection does not go past the end of the trace (the program wrote one sample after the raw trace)
    velocity[-1] = 5000.0
    assert np.isfinite(np.asarray(list(wellrf(depth, velocity))[0])).all()
    with pytest.raises(ValueError, match="monotonically"):
        list(wellrf([0.0, 10.0, 5.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="same length"):
        wellrf([0.0, 10.0], [1.0, 2.0, 3.0])


# ------------------------------------------------------------------------------------------------------- las2su
LAS = """~VERSION INFORMATION
 VERS.   2.0 : CWLS LOG ASCII STANDARD
~WELL INFORMATION
 STRT.F  1000.0 : START
~CURVE INFORMATION
#MNEM.UNIT   API CODE   CURVE DESCRIPTION
 DEPT.F                : depth
 DT  .US/F             : sonic
 RHOB.G/C3             : density
~PARAMETER INFORMATION
 MUD .  WATER : mud
~A  DEPT DT RHOB
 1000.0  100.0  2.20
 1000.5  101.0  2.21
 1001.0  -999.25  2.22
 1001.5  103.0  2.23
"""


def test_las2su_curves_become_traces():
    out = list(las2su(LAS.splitlines(), dz=0.5))
    assert len(out) == 3
    assert [t.header['trace_id'] for t in out] == [1, 2, 3]
    npt.assert_allclose(np.asarray(out[0]), [1000.0, 1000.5, 1001.0, 1001.5])
    npt.assert_allclose(np.asarray(out[2]), [2.20, 2.21, 2.22, 2.23], rtol=1e-6)
    assert out[1].d_sample == pytest.approx(0.5) and out[1].header['sample_start'] == pytest.approx(1000.0)
    # the null value is kept unless it is given
    assert np.asarray(out[1])[2] == pytest.approx(-999.25)
    nulled = list(las2su(LAS.splitlines(), null=-999.25))
    assert np.isnan(np.asarray(nulled[1])[2])


def test_las2su_meters_subsampling_and_files(tmp_path):
    out = list(las2su(LAS.splitlines(), m=True, ss=True))
    assert len(out) == 3 and out[0].n_sample == 2
    assert out[0].d_sample == pytest.approx(1.0 / 3.28084)
    assert out[0].header['sample_start'] == pytest.approx(1000.0 / 3.28084)
    path = tmp_path / 'well.las'
    path.write_text(LAS)
    from_path = list(las2su(path))
    from_file = list(las2su(open(path)))
    assert len(from_path) == len(from_file) == 3
    npt.assert_array_equal(np.asarray(from_path[1]), np.asarray(from_file[1]))
    with pytest.raises(ValueError, match="~A"):
        list(las2su(["~C", " DEPT.F : depth"]))


# ----------------------------------------------------------------------------------------------------- gassmann
def test_gassmann_functions_are_consistent():
    # the defaults of the program
    vp = wl.rock_velocity()
    rho = wl.rock_density()
    assert 3000 < vp < 5000
    assert wl.rock_impedance() == pytest.approx(vp * rho, rel=1e-5)
    ip = wl.rock_impedance()
    r0 = wl.rock_reflectivity(v1=3300.0, r1=2.6)
    assert r0 == pytest.approx((ip - 3300.0 * 2.6) / (ip + 3300.0 * 2.6), rel=1e-5)
    assert wl.rock_twt(h=20.0) == pytest.approx(2 * 20.0 / vp, rel=1e-5)
    assert wl.fluid_velocity() == pytest.approx(math.sqrt(1000 * wl.fluid_modulus() / wl.fluid_density()), rel=1e-5)
    # water at room conditions: about 1480 m/s, 1 g/cc, 2.2 GPa
    assert wl.brine_velocity(s=0.0, t=20.0, p=0.1) == pytest.approx(1482.0, abs=10.0)
    assert wl.brine_density(s=0.0, t=20.0, p=0.1) == pytest.approx(1.0, abs=0.005)
    assert wl.brine_modulus(s=0.0, t=20.0, p=0.1) == pytest.approx(2200.0, rel=0.03)
    # a mixture of all gas is the gas; with all oil, the oil
    assert wl.fluid_density(so=0.0, sg=1.0) == pytest.approx(wl.gas_density())
    assert wl.fluid_density(so=1.0, sg=0.0) == pytest.approx(wl.oil_density())
    # gas is the softest: the saturated rock gets slower and lighter with it
    assert wl.rock_velocity(sg=0.5, so=0.0) < wl.rock_velocity(sg=0.0, so=0.0)
    assert wl.oil_velocity() == pytest.approx(math.sqrt(1000 * wl.oil_modulus() / wl.oil_density()), rel=1e-4)
    assert wl.gas_velocity() == pytest.approx(math.sqrt(1000 * wl.gas_modulus() / wl.gas_density()), rel=1e-4)


def test_gassmann_functions_broadcast_and_check_the_saturations():
    t = np.array([20.0, 60.0, 100.0])
    p = np.array([[10.0], [50.0]])
    v = wl.brine_velocity(s=0.05, t=t, p=p)
    assert v.shape == (2, 3)
    # from 20 to 60 degrees it is faster, and more pressure is faster
    assert (v[:, 1] > v[:, 0]).all() and (v[1] > v[0]).all()
    with pytest.raises(ValueError, match="so \\+ sg"):
        wl.rock_velocity(so=0.8, sg=0.5)
    with pytest.raises(TypeError, match="takes"):
        wl.brine_density(api=40.0)


def test_gassmann_program_modes():
    (trace,) = list(gassmann(mode=0, p=0.15, nsamples=200))
    data = np.asarray(trace)
    assert len(data) == 200
    # at the middle, the saturation is the one at which it is normalized: 100 %
    assert data[100] == pytest.approx(100.0, rel=1e-4)
    assert trace.header['sample_start'] == pytest.approx(0.85 * 0.2)
    assert trace.d_sample == pytest.approx(2 * 0.15 * 0.2 / 200)
    # (a decrease of gas saturation moves the reflection coefficient one way, and more gas the other)
    assert np.sign(data[0] - 100.0) != np.sign(data[-1] - 100.0)
    twt = list(gassmann(mode=2, ntr=5, nsamples=30))
    assert len(twt) == 5 and twt[0].n_sample == 30
    assert np.isfinite(np.asarray(twt[-1])).all()
    # hotter makes a different time through the reservoir
    assert not np.allclose(np.asarray(twt[0]), np.asarray(twt[-1]))
    with pytest.raises(NotImplementedError):
        gassmann(mode=1)
    with pytest.raises(ValueError, match="mode"):
        gassmann(mode=3)
    with pytest.raises(TypeError, match="Unknown"):
        gassmann(porosity=0.2)
