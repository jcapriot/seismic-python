"""
SUHROT, SUPOLAR, SUPOFILT, SUEIPOFI, SULTT and SUALFORD.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.multicomponent import alford, eipofi, hrot, ltt, pofilt, polar

DT = 0.004
RNG = np.random.default_rng(71)


def ricker(t, f=25.0):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def trace(x, **header):
    t = Trace(np.asarray(x, dtype=np.float32), d_sample=DT)
    return t.replace(**header) if header else t


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def station(z, n, e, **header):
    return [trace(z, **header), trace(n, **header), trace(e, **header)]


# ------------------------------------------------------------------------------------------------------------ hrot
def test_hrot_rotates_north_east_to_radial_transverse():
    t = np.arange(200) * DT
    wavelet = ricker(t - 0.4)
    # a wave that is along the azimuth 30 degrees (clockwise from north): N = cos, E = sin
    az = np.radians(30.0)
    st = station(0.3 * wavelet, np.cos(az) * wavelet, np.sin(az) * wavelet)
    out = run(hrot(a=[30.0], angle='deg'), st)
    assert len(out) == 3
    z, r, tr = (np.asarray(o) for o in out)
    npt.assert_allclose(z, 0.3 * wavelet, atol=1e-6)
    npt.assert_allclose(r, wavelet, atol=1e-6)
    npt.assert_allclose(tr, 0.0, atol=1e-6)
    assert [o.header['trace_type'] for o in out] == [15, 17, 16]
    # the inverse rotation puts it back
    back = run(hrot(a=[30.0], angle='deg', inv=True), out)
    npt.assert_allclose(np.asarray(back[1]), np.cos(az) * wavelet, atol=1e-5)
    assert hrot().parallelism == 'serial'


def test_hrot_angle_from_the_geometry_and_the_key():
    t = np.arange(100) * DT
    w = ricker(t - 0.2)
    # a receiver to the east of the source: azimuth 90 degrees (the program gets 0 for it): N = 0, E = w is radial
    st = [trace(w, tx_loc=[0.0, 0.0, 0.0], rx_loc=[500.0, 0.0, 0.0]) for _ in range(3)]
    st[0] = st[0].replace(np.zeros(100))
    st[1] = st[1].replace(np.zeros(100))
    out = run(hrot(), st)
    npt.assert_allclose(np.asarray(out[1]), w, atol=1e-6)
    npt.assert_allclose(np.asarray(out[2]), 0.0, atol=1e-6)
    # angles by a key, interpolated
    tr_a = station(np.zeros(100), w, 0 * w, ensemble_trace_number=5)
    out = run(hrot(a=[0.0, 90.0], x=[0.0, 10.0], angle='deg'), tr_a)
    az = np.radians(45.0)
    npt.assert_allclose(np.asarray(out[1]), np.cos(az) * w, atol=1e-5)
    with pytest.raises(ValueError):
        hrot(angle='mil')
    with pytest.raises(ValueError):
        hrot(a=[1.0, 2.0], x=[3.0, 1.0, 2.0])


def test_stations_by_key_and_by_component_type():
    t = np.arange(50) * DT
    z, n, e = np.zeros(50), ricker(t - 0.1), np.zeros(50)
    # by a key (a gather of three), with the traces out of order that components puts in order
    st = [trace(e, trace_type=14, ensemble_number=7), trace(z, trace_type=12, ensemble_number=7),
          trace(n, trace_type=13, ensemble_number=7)]
    out = run(hrot(a=[0.0], key='ensemble_number', components=(12, 13, 14)), st)
    assert [o.header['trace_type'] for o in out] == [15, 17, 16]
    npt.assert_allclose(np.asarray(out[1]), n, atol=1e-6)
    with pytest.raises(ValueError, match="needs 3"):
        run(hrot(a=[0.0], key='ensemble_number'), st[:2])
    with pytest.raises(ValueError, match="trace_type"):
        run(hrot(a=[0.0], components=(12, 13, 99)), st)
    with pytest.warns(UserWarning, match="do not make a station"):
        run(hrot(a=[0.0]), st + st[:1])


# ----------------------------------------------------------------------------------------------------------- polar
def linear_station(az_deg=40.0, inc_deg=30.0, n=300, noise=0.0):
    """A rectilinear particle motion along a direction given by its incidence and azimuth"""
    t = np.arange(n) * DT
    w = ricker(t - 0.6, 20.0) + 0.5 * ricker(t - 0.2, 20.0)
    inc, az = np.radians(inc_deg), np.radians(az_deg)
    comps = [np.cos(inc) * w, np.sin(inc) * np.cos(az) * w, np.sin(inc) * np.sin(az) * w]
    return [c + noise * RNG.normal(size=n) for c in comps]


def test_polar_of_a_rectilinear_motion():
    z, n, e = linear_station(az_deg=40.0, inc_deg=30.0)
    out = polar(station(z, n, e), wl=0.1, rl=1, theta=2, phi=3, angle='deg', tau=1, pln=1, l1=1, amp=1, dir=1)
    assert set(out) >= {'rl', 'theta', 'phi', 'tau', 'pln', 'l1', 'er', 'ir', 'qr', 'dir'}
    it = int(0.6 / DT)
    rl = np.asarray(out['rl'][0])
    assert rl[it] > 0.99 and np.asarray(out['tau'][0])[it] > 0.99
    assert np.asarray(out['theta'][0])[it] == pytest.approx(30.0, abs=0.5)
    assert np.asarray(out['phi'][0])[it] == pytest.approx(40.0, abs=0.5)
    # the direction of polarization has the direction cosines (up to sign)
    d = np.array([np.abs(np.asarray(t))[it] for t in out['dir'][:3]])
    npt.assert_allclose(d, [np.cos(np.radians(30)), np.sin(np.radians(30)) * np.cos(np.radians(40)),
                            np.sin(np.radians(30)) * np.sin(np.radians(40))], atol=0.02)
    # the instantaneous resultant is the vector length
    assert np.asarray(out['ir'][0])[it] == pytest.approx(np.sqrt(z[it] ** 2 + n[it] ** 2 + e[it] ** 2), rel=1e-4)
    assert len(out['dir']) == 3 and out['rl'][0].n_sample == 300
    # the ends are zero
    assert np.asarray(out['rl'][0])[:10].tolist() == [0.0] * 10


def test_polar_phi_option_one_and_units():
    z, n, e = linear_station(az_deg=40.0, inc_deg=30.0)
    out = polar(station(z, n, e), phi=1, theta=1, angle='rad', rl=0, dir=0)
    it = int(0.6 / DT)
    # (the azimuth of option 1 is in -pi/2 .. pi/2: atan(E / N) of the principal axis; the program tests the wrong element)
    assert np.asarray(out['phi'][0])[it] == pytest.approx(np.radians(40.0), abs=0.02)
    assert set(out) == {'theta', 'phi'}
    two = polar(station(z, n, e) + station(z, e, n), rl=1, dir=0)
    assert len(two['rl']) == 2
    with pytest.raises(ValueError):
        polar(station(z, n, e), win='cosine')
    with pytest.raises(ValueError, match="too long"):
        polar(station(z, n, e), wl=10.0)
    with pytest.raises(ValueError, match="positive"):
        polar(station(z, n, e), wl=0.0)


def test_polar_windows_and_all():
    z, n, e = linear_station()
    for win in ('boxcar', 'bartlett', 'hanning', 'welsh'):
        out = polar(station(z, n, e), win=win, wl=0.12)
        assert np.isfinite(np.asarray(out['rl'][0])).all()
    every = polar(station(z, n, e), all=1)
    assert {'rl', 'theta', 'phi', 'tau', 'e21', 'e31', 'e32', 'pln', 'f1', 'l1', 'er', 'ir', 'qr', 'dir'} <= set(every)


# ---------------------------------------------------------------------------------------------------------- pofilt
def test_pofilt_keeps_the_polarized_signal_and_attenuates_other_directions():
    z, n, e = linear_station(az_deg=40.0, inc_deg=30.0, noise=0.0)
    signal = station(z, n, e)
    rng = np.random.default_rng(3)
    noise = [0.1 * rng.normal(size=300) for _ in range(3)]
    noisy = [trace(np.asarray(s) + x) for s, x in zip(signal, noise)]
    attributes = polar(noisy, wl=0.1, rl=1, dir=1)
    out = run(pofilt(attributes['dir'], attributes['rl'], sl=0.04), noisy)
    assert len(out) == 3
    filtered = np.array([np.asarray(t) for t in out])
    clean = np.array([np.asarray(t) for t in signal])
    raw = np.array([np.asarray(t) for t in noisy])
    interior = slice(20, 280)
    assert np.abs(filtered[:, interior] - clean[:, interior]).sum() < 1.5 * np.abs(raw[:, interior] - clean[:, interior]).sum()
    # no smoothing: each component is the product with the directivity and the weights
    plain = run(pofilt(attributes['dir'], attributes['rl'], smooth=False), noisy)
    expected = np.asarray(noisy[0]) * np.abs(np.asarray(attributes['dir'][0])) * np.asarray(attributes['rl'][0])
    npt.assert_allclose(np.asarray(plain[0]), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="fewer"):
        run(pofilt(attributes['dir'][:3], attributes['rl']), noisy + noisy)
    with pytest.raises(ValueError):
        pofilt(attributes['dir'], attributes['rl'], sl=0.0)


# ---------------------------------------------------------------------------------------------------------- eipofi
def test_eipofi_passes_a_planar_motion_and_reduces_noise():
    t = np.arange(400) * DT
    w = ricker(t - 0.8, 15.0)
    # an elliptical motion in a plane: two components with a quarter period of shift (the Hilbert pair of w)
    h = np.imag(np.fft.ifft(np.fft.fft(w) * (-1j) * np.sign(np.fft.fftfreq(400))))
    clean = [np.zeros(400), w + 0.5 * h, 0.6 * h - 0.4 * w]
    rng = np.random.default_rng(5)
    noisy = [trace(c + 0.05 * rng.normal(size=400)) for c in clean]
    report = {}
    out = run(eipofi(wl=0.1, report=report), noisy)
    assert len(out) == 3 and len(report['rl1']) == 1 and report['rl1'][0].shape == (400,)
    filtered = np.array([np.asarray(o) for o in out])
    noisy_a = np.array([np.asarray(o) for o in noisy])
    n_whole = 400 // 25 * 25
    err_after = np.abs(filtered[:, :n_whole] - np.array(clean)[:, :n_whole]).sum()
    err_before = np.abs(noisy_a[:, :n_whole] - np.array(clean)[:, :n_whole]).sum()
    assert err_after < err_before
    # the weights are in 0..1 with the cubic interpolation
    assert 0.0 <= report['pln'][0].min() and report['pln'][0].max() <= 1.0
    lin = run(eipofi(wl=0.1, interp='linear'), noisy)
    assert len(lin) == 3
    with pytest.raises(ValueError):
        run(eipofi(wl=0.004), noisy)
    with pytest.raises(ValueError):
        eipofi(interp='quintic')


# ------------------------------------------------------------------------------------------------- shear waves
def shear_station(alpha_deg=25.0, n=300, delay=0, noise=0.0, seed=1):
    """The four components of a shear wave that is split: a fast wave along the angle alpha, and a slow one that is delayed
    and weaker (S1 and S2 in the natural coordinates, rotated into the in-line / cross-line ones)"""
    t = np.arange(n) * DT
    fast = ricker(t - 0.5, 25.0)
    slow = 0.6 * ricker(t - 0.5 - delay * DT, 25.0)
    a = np.radians(alpha_deg)
    c, s = np.cos(a), np.sin(a)
    # the response matrix in the natural coordinates is diag(fast, slow), and R^T D R in the survey ones
    s11 = c * c * fast + s * s * slow
    s22 = s * s * fast + c * c * slow
    s12 = s21 = c * s * (fast - slow)
    rng = np.random.default_rng(seed)
    return [trace(x + noise * rng.normal(size=n)) for x in (s11, s12, s21, s22)]


def test_ltt_finds_the_polarization_angle():
    for mode in (2, 3):
        out = run(ltt(mode=mode, wl=7, ntraces=1), shear_station(alpha_deg=25.0))
        assert len(out) == 1
        alpha = np.asarray(out[0])
        it = int(0.5 / DT)
        assert alpha[it] == pytest.approx(25.0, abs=1.0)
    # the asymmetry indexes
    both = run(ltt(mode=1, wl=7, ntraces=1), shear_station(alpha_deg=25.0))
    assert len(both) == 2
    gamma = np.asarray(both[1])
    assert np.isfinite(gamma).all() and gamma.min() >= 0.0 and gamma.max() <= 1.0 + 1e-6
    # several stations are mixed
    many = run(ltt(mode=3, wl=5, ntraces=3), shear_station(25.0) + shear_station(25.0) + shear_station(25.0))
    assert len(many) == 3
    assert np.asarray(many[2])[int(0.5 / DT)] == pytest.approx(25.0, abs=1.0)
    with pytest.raises(ValueError):
        ltt(mode=4)
    with pytest.raises(ValueError):
        ltt(wl=0)


def test_alford_rotates_the_data_into_the_natural_coordinates():
    st = shear_station(alpha_deg=35.0, delay=2)
    report = {}
    out = run(alford(angle_inc=5.0, ntraces=1, report=report), st)
    assert len(out) == 4
    s11, s12, s21, s22 = (np.asarray(o) for o in out)
    original = [np.asarray(t) for t in st]
    # the energy of the off-diagonal traces drops, and the fast wave is on S11
    off_before = np.sum(original[1] ** 2 + original[2] ** 2)
    off_after = np.sum(s12 ** 2 + s21 ** 2)
    assert off_after < 0.05 * off_before
    assert np.sum(s11 ** 2) > np.sum(s22 ** 2)
    assert report['azimuth'][0] == pytest.approx(35.0, abs=3.0)
    assert report['quality'][0] > report['input_quality'][0]
    assert report['lag'][0] == pytest.approx(2 * DT, abs=2e-3)
    assert 0.0 < report['xcorr'][0] <= 1.0 + 1e-5
    # it works the same with the window of analysis, and the unrotated part above it
    windowed = run(alford(angle_inc=5.0, ntraces=1, tmin=0.35, tmax=0.7, taper=3), st)
    w11 = np.asarray(windowed[0])
    npt.assert_allclose(w11[:60], original[0][:60], atol=1e-6)  # (above the overlap)
    assert np.sum(np.asarray(windowed[1]) ** 2 + np.asarray(windowed[2]) ** 2) < 0.2 * off_before


def test_alford_with_too_long_a_lag_does_not_rotate_and_errors():
    st = shear_station(alpha_deg=35.0, delay=30)  # (a 120 ms lag)
    report = {}
    out = run(alford(angle_inc=5.0, ntraces=1, maxlag=40.0, report=report), st)
    assert report['azimuth'] == [0.0] and report['quality'] == [0.0] and report['lag'] == [0.0]
    for o, t in zip(out, st):
        npt.assert_array_equal(np.asarray(o), np.asarray(t))
    with pytest.raises(ValueError):
        run(alford(tmin=0.9, tmax=0.1), st)
    with pytest.raises(ValueError):
        alford(ntraces=0)
    with pytest.raises(ValueError, match="same number"):
        run(alford(), st[:3] + [trace(np.zeros(100))])


def test_eipofi_with_windows_of_zeros_is_finite_and_repeatable():
    # (svd_sort of the library read one value past the singular values: garbage, and different from run to run)
    rng = np.random.default_rng(9)
    comps = []
    for _ in range(3):
        x = np.zeros(200)
        x[100:] = 1e-3 * rng.normal(size=100)  # (the first windows are zeros, one is nearly)
        comps.append(x)
    runs = [np.array([np.asarray(t) for t in run(eipofi(wl=0.1), [trace(c) for c in comps])]) for _ in range(4)]
    assert np.isfinite(runs[0]).all()
    for r in runs[1:]:
        npt.assert_array_equal(r, runs[0])
