"""
Finite-difference modelling of the acoustic wave equation: SUFDMOD1 and SUFDMOD2.
"""
import math

import numpy as np
import numpy.testing as npt
import pytest

from seispy.synthetics import Snapshots, fdmod1, fdmod1_snapshots, fdmod2, fdmod2_pml, ea2df, remac2d, remel2dan, fctanismod

# ----------------------------------------------------------------------------------------------------------- fdmod1
V1 = np.full(401, 2000.0, dtype=np.float32)  # 2 km in 5 m samples
FD1 = dict(sz=500.0, dz=5.0, tmax=0.7, styp=2, freq=25.0)


def one(**kwargs):
    (trace,) = list(fdmod1(V1, **{**FD1, **kwargs}))
    return trace


def peak_time(trace):
    return np.abs(np.asarray(trace)).argmax() * trace.header['d_sample'] + trace.header['sample_start']


def test_fdmod1_the_wave_travels_at_the_velocity():
    near, far = one(rz=1000.0), one(rz=1500.0)
    assert peak_time(far) - peak_time(near) == pytest.approx(500.0 / 2000.0, abs=0.008)  # (a little numerical dispersion)
    assert one(rz=1000.0).header['rx_loc'][2] == -1000.0 and one(rz=1000.0).header['tx_loc'][2] == -500.0


def test_fdmod1_the_time_step_is_stable_and_the_trace_is_shifted_by_the_source_delay():
    trace = one(rz=1000.0)
    dt = 5.0 / 1.414 / 2000.0 / 2
    assert trace.header['d_sample'] == pytest.approx(dt, rel=1e-5)
    assert abs(trace.n_sample - (int(1 + 0.7 / dt) + 1)) <= 1
    assert trace.header['sample_start'] < 0.0


def test_fdmod1_an_absorbing_bottom_leaves_less_energy_than_a_free_one():
    absorbing = np.asarray(one(rz=1000.0, abs=(0, 1), tmax=1.4))
    reflecting = np.asarray(one(rz=1000.0, abs=(0, 0), tmax=1.4))
    late = slice(len(absorbing) * 3 // 4, None)
    assert np.abs(absorbing[late]).max() < 0.2 * np.abs(reflecting[late]).max()


def test_fdmod1_density_contrast_reflects():
    rho = np.full(401, 2500.0, dtype=np.float32)
    rho[260:] = 4000.0  # an interface at 1.3 km
    plain = np.asarray(one(rz=500.0 + 5.0, tmax=1.4))
    contrast = np.asarray(one(rz=500.0 + 5.0, density=rho, tmax=1.4))
    assert np.abs(contrast - plain).max() > 0.05 * np.abs(plain).max()


def test_fdmod1_snapshots():
    snaps = fdmod1_snapshots(V1, zd=2, td=3, **FD1)
    assert isinstance(snaps, Snapshots)
    frames = list(snaps)
    assert len(frames) == snaps.n_frames == len(snaps) and all(f.shape == (200,) and f.dtype == np.float32 for f in frames)
    assert snaps.spacing == (10.0,) and snaps.dt == pytest.approx(3 * 5.0 / 1.414 / 2000.0 / 2, rel=1e-5)
    assert snaps.times[1] == pytest.approx(snaps.dt)
    # the seismogram undersampled with td=3 is the pressure at the receiver, so the snapshots at the receiver sample agree
    trace = one(rz=1000.0, td=3, zd=2)
    npt.assert_allclose([f[100] for f in frames], np.asarray(trace), atol=1e-6)


def test_fdmod1_particle_velocity_differs_from_pressure_and_checks_its_input():
    assert not np.allclose(np.asarray(one(rz=1000.0, press=False)), np.asarray(one(rz=1000.0)))
    with pytest.raises(ValueError, match="1-D"):
        fdmod1(np.zeros((3, 3)), **FD1)
    with pytest.raises(ValueError, match="outside"):
        fdmod1(V1, **{**FD1, 'sz': 5000.0})
    with pytest.raises(ValueError, match="styp"):
        fdmod1(V1, **{**FD1, 'styp': 4})
    with pytest.raises(ValueError, match="two values"):
        fdmod1(V1, abs=(1,), **FD1)
    with pytest.raises(ValueError, match="shape"):
        fdmod1(V1, density=np.ones(5), **FD1)


# ----------------------------------------------------------------------------------------------------------- fdmod2
V2 = np.full((101, 201), 2000.0, dtype=np.float32)  # (nz, nx), 1 km deep and 2 km wide in 10 m samples
FD2 = dict(xs=1000.0, zs=250.0, tmax=0.3, dx=10.0, dz=10.0)


def peak_times(traces, dt, delay):
    return np.array([np.abs(np.asarray(t)).argmax() for t in traces]) * dt - delay


def test_fdmod2_the_wave_travels_at_the_velocity():
    sim = fdmod2(V2, hsz=250.0, **{**FD2, 'tmax': 0.5})
    traces = list(sim.horizontal_line)
    assert len(traces) == 201 and traces[0].n_sample == sim.nt
    times = peak_times(traces, sim.dt, sim.tdelay)
    # the arrival at 300 m and at 600 m from the source on the horizontal line
    assert times[160] - times[130] == pytest.approx(300.0 / 2000.0, abs=0.012)
    npt.assert_allclose(times[130], times[70], atol=2 * sim.dt)  # (symmetric about the source)


def test_fdmod2_headers_and_the_delay_of_the_source():
    sim = fdmod2(V2, hsz=250.0, **FD2)
    t = list(sim.horizontal_line)[130]
    assert t.header['sample_start'] == pytest.approx(-1.0 / sim.fpeak)
    assert t.header['d_sample'] == pytest.approx(10.0 / (2.0 * 2000.0))
    assert t.header['tx_loc'][0] == pytest.approx(1000.0) and t.header['tx_loc'][2] == pytest.approx(-250.0)
    assert t.header['rx_loc'][0] == pytest.approx(1300.0) and t.header['rx_loc'][2] == pytest.approx(-250.0)
    v = list(fdmod2(V2, vsx=1000.0, **FD2).vertical_line)
    assert len(v) == 101 and v[5].header['rx_loc'][2] == pytest.approx(-50.0)


def test_fdmod2_snapshots_are_frames_made_as_they_are_asked_for():
    sim = fdmod2(V2, **FD2)
    snaps = sim.snapshots(mt=20)
    assert snaps.n_frames == len(range(0, sim.nt, 20)) and snaps.dt == pytest.approx(20 * sim.dt)
    assert snaps.spacing == (10.0, 10.0) and snaps.origin == (0.0, 0.0)
    frames = iter(snaps)
    first = next(frames)
    assert first.shape == (101, 201) and first.dtype == np.float32
    rest = list(frames)
    assert len(rest) + 1 == snaps.n_frames
    # a circular wavefront from the source: at the last frame the energy is symmetric about the source column
    last = rest[-1]
    npt.assert_allclose(last, last[:, ::-1], atol=1e-3 * np.abs(last).max())
    assert np.abs(last).max() > 0.0
    with pytest.raises(RuntimeError, match="run already"):
        sim.snapshots()


def test_fdmod2_plane_wave_arrives_everywhere_at_the_same_time():
    shallow = fdmod2(V2, pw=True, hsz=500.0, **{**FD2, 'tmax': 0.4})
    deep = fdmod2(V2, pw=True, hsz=750.0, **{**FD2, 'tmax': 0.4})
    t1 = peak_times(list(shallow.horizontal_line), shallow.dt, shallow.tdelay)[40:160]
    t2 = peak_times(list(deep.horizontal_line), deep.dt, deep.tdelay)[40:160]
    assert t1.max() - t1.min() < 2 * shallow.dt  # (a flat wavefront)
    assert t2.mean() - t1.mean() == pytest.approx(250.0 / 2000.0, abs=0.01)


def test_fdmod2_an_extended_source_and_source_points():
    sim = fdmod2(V2, **{**FD2, 'xs': [800.0, 1200.0], 'zs': [250.0, 250.0]}, source_points=True, hsz=500.0)
    ss = list(sim.source_points)
    assert len(ss) == 2 and ss[0].header['tx_loc'][0] == pytest.approx(800.0)
    assert np.abs(np.asarray(ss[0])).max() > 0.0
    point = fdmod2(V2, source_points=True, **FD2)
    assert len(list(point.source_points)) == 1


def test_fdmod2_density_absorbing_and_the_free_surface():
    rho = np.full((101, 201), 1000.0, dtype=np.float32)
    rho[60:, :] = 3000.0
    plain = samples_of(fdmod2(V2, hsz=250.0, **{**FD2, 'tmax': 0.5}).horizontal_line)
    layered = samples_of(fdmod2(V2, density=rho, hsz=250.0, **{**FD2, 'tmax': 0.5}).horizontal_line)
    assert np.abs(plain - layered).max() > 0.05 * np.abs(plain).max()
    # (a constant density is 1)
    constant = samples_of(fdmod2(V2, density=np.full_like(V2, 7.0), hsz=250.0, **{**FD2, 'tmax': 0.5}).horizontal_line)
    npt.assert_array_equal(constant, plain)
    # the free surface keeps the energy that the absorbing one lets out of the model
    absorbing = samples_of(fdmod2(V2, hsz=250.0, **{**FD2, 'tmax': 1.0}).horizontal_line)
    free = samples_of(fdmod2(V2, hsz=250.0, abs=(0, 1, 1, 1), **{**FD2, 'tmax': 1.0}).horizontal_line)
    assert np.abs(free[:, -50:]).max() > 2 * np.abs(absorbing[:, -50:]).max()


def test_fdmod2_checks_its_input():
    with pytest.raises(ValueError, match="2-D"):
        fdmod2(np.zeros(5), **FD2)
    with pytest.raises(ValueError, match="outside"):
        fdmod2(V2, **{**FD2, 'xs': 5000.0})
    with pytest.raises(ValueError, match="number of xs"):
        fdmod2(V2, **{**FD2, 'xs': [1.0, 2.0], 'zs': 3.0})
    with pytest.raises(ValueError, match="shape"):
        fdmod2(V2, density=np.ones((3, 3)), **FD2)
    with pytest.raises(ValueError, match="four values"):
        fdmod2(V2, abs=(1, 1), **FD2)
    with pytest.raises(ValueError, match="not asked"):
        fdmod2(V2, **FD2).horizontal_line
    with pytest.raises(ValueError, match="hsz"):
        fdmod2(V2, hsz=5000.0, **FD2)


# -------------------------------------------------------------------------------------------------------- fdmod2_pml
def test_fdmod2_pml_without_a_layer_is_fdmod2():
    kwargs = {**FD2, 'tmax': 0.4}
    a = samples_of(fdmod2(V2, hsz=250.0, **kwargs).horizontal_line)
    b = samples_of(fdmod2_pml(V2, hsz=250.0, **kwargs).horizontal_line)
    npt.assert_allclose(b, a, atol=1e-4 * np.abs(a).max())


def test_fdmod2_pml_absorbs_what_a_free_boundary_reflects():
    kwargs = {**FD2, 'tmax': 1.2, 'xs': 1000.0, 'zs': 500.0}
    layered = samples_of(fdmod2_pml(V2, hsz=500.0, pml_thick=10, abs=(1, 1, 1, 1), **kwargs).horizontal_line)
    rigid = samples_of(fdmod2(V2, hsz=500.0, abs=(0, 0, 0, 0), **kwargs).horizontal_line)
    late = slice(layered.shape[1] * 3 // 4, None)
    assert np.isfinite(layered).all()
    assert np.abs(layered[:, late]).max() < 0.1 * np.abs(rigid[:, late]).max()
    # (the direct wave is the same)
    early = slice(0, 80)
    npt.assert_allclose(layered[:, early], rigid[:, early], atol=0.05 * np.abs(rigid[:, early]).max())


def test_fdmod2_pml_with_density_an_extended_source_and_arguments():
    rho = np.full((101, 201), 1000.0, dtype=np.float32)
    rho[60:, :] = 2000.0
    sim = fdmod2_pml(V2, density=rho, hsz=250.0, pml_thick=6, **{**FD2, 'xs': [800.0, 1200.0], 'zs': [250.0, 250.0], 'tmax': 0.4})
    data = samples_of(sim.horizontal_line)
    assert np.isfinite(data).all() and np.abs(data).max() > 0.0
    frames = list(fdmod2_pml(V2, pml_thick=6, **{**FD2, 'tmax': 0.2}).snapshots(10))
    assert frames[0].shape == (101, 201)
    with pytest.raises(ValueError, match="pml_thick"):
        fdmod2_pml(V2, pml_thick=-1, **FD2)
    with pytest.raises(TypeError):
        fdmod2_pml(V2, pw=True, **FD2)


# ------------------------------------------------------------------------------------------------------------ ea2df
EA_NZ, EA_NX = 60, 100
EA_C11 = np.full((EA_NZ, EA_NX), 4.0e9, dtype=np.float32)  # vp = 2000 m/s for rho = 1000
EA_C55 = np.full((EA_NZ, EA_NX), 1.0e9, dtype=np.float32)  # vs = 1000 m/s
EA_RHO = np.full((EA_NZ, EA_NX), 1000.0, dtype=np.float32)
EA = dict(dx=10.0, fx=0.0, dt=0.001, lt=0.35, sx=500.0, sz=300.0, favg=30.0, ts=0.05, hsz=300.0, vsx=500.0, bc=(10, 10, 10, 10))


def ea_lines(sim):
    traces = list(sim.horizontal_line)
    n = len(traces) // 2
    return samples_of(traces[:n]), samples_of(traces[n:])


def test_ea2df_the_p_wave_travels_at_the_velocity():
    sim = ea2df(EA_C11, EA_C55, EA_RHO, **EA)
    traces = list(sim.horizontal_line)
    assert len(traces) == 2 * EA_NX and traces[0].n_sample == sim.nt == 351
    u, w = ea_lines(sim)
    peak = np.abs(u).argmax(axis=1) * 0.001
    # on the line through the source the horizontal motion is a P wave: 100 m more takes 0.05 s
    assert peak[80] - peak[70] == pytest.approx(0.05, abs=0.006)
    t = traces[85]
    assert t.header['rx_loc'][0] == pytest.approx(85 * 10.0) and t.header['rx_loc'][2] == pytest.approx(-300.0)
    assert t.header['tx_loc'][0] == pytest.approx(500.0) and t.header['ensemble_number'] == 1
    assert traces[EA_NX + 3].header['ensemble_number'] == 2


def test_ea2df_vertical_line_and_the_symmetry_of_the_source():
    sim = ea2df(EA_C11, EA_C55, EA_RHO, **EA)
    traces = list(sim.vertical_line)
    assert len(traces) == 2 * EA_NZ and traces[5].header['rx_loc'][2] == pytest.approx(-50.0)
    u, _ = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, **EA))
    left, right = np.abs(u[:50]).max(), np.abs(u[50:]).max()  # (a P source radiates alike to both sides)
    assert left == pytest.approx(right, rel=0.1)


def test_ea2df_an_anisotropic_medium_that_is_isotropic_matches_the_isotropic_one():
    iso = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, **EA))
    zero = np.zeros_like(EA_C11)
    ani = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, c13=EA_C11 - 2 * EA_C55, c33=EA_C11, c15=zero, c35=zero, **EA))
    npt.assert_allclose(ani[0], iso[0], atol=1e-3 * np.abs(iso[0]).max())
    npt.assert_allclose(ani[1], iso[1], atol=1e-3 * np.abs(iso[0]).max())
    # (a real anisotropy changes it)
    vti = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, c13=EA_C11 - 2 * EA_C55, c33=1.5 * EA_C11, c15=zero, c35=zero, **EA))
    assert np.abs(vti[0] - iso[0]).max() > 0.01 * np.abs(iso[0]).max()
    with pytest.raises(ValueError, match="together"):
        ea2df(EA_C11, EA_C55, EA_RHO, c13=zero, **EA)


def test_ea2df_attenuation_the_plane_wave_and_the_free_surface():
    base = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, **EA))[0]
    lossy = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, q=np.full_like(EA_C11, 20.0), **EA))[0]
    assert np.abs(lossy[95]).max() < 0.85 * np.abs(base[95]).max() and np.abs(lossy[95]).max() > 0.0  # (far from the source)
    # a plane wave arrives at the same time all over a line below the source
    pw = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, stype='pw', **{**EA, 'sz': 100.0, 'hsz': 300.0, 'lt': 0.3}))[1]
    peaks = np.abs(pw[30:70]).argmax(axis=1)
    assert peaks.max() - peaks.min() <= 4
    free = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'bc': (2, 10, 10, 10), 'sz': 100.0, 'hsz': 100.0}))[0]
    rigid = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'bc': (10, 10, 10, 10), 'sz': 100.0, 'hsz': 100.0}))[0]
    assert np.isfinite(free).all() and np.abs(free - rigid).max() > 0.05 * np.abs(rigid).max()


def test_ea2df_the_padding_keeps_the_whole_model():
    # a contrast in the last rows of the model is in the model (the program's padding dropped them)
    c11 = EA_C11.copy()
    c11[-6:, :] = 9.0e9
    plain = ea_lines(ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'lt': 0.3, 'bc': (10, 10, 3, 10)}))[0]
    stiff = ea_lines(ea2df(c11, EA_C55, EA_RHO, **{**EA, 'lt': 0.3, 'bc': (10, 10, 3, 10)}))[0]
    assert np.abs(stiff - plain).max() > 0.01 * np.abs(plain).max()


def test_ea2df_snapshots():
    sim = ea2df(EA_C11, EA_C55, EA_RHO, **EA)
    snaps = sim.snapshots([0.1, 0.2, 0.3])
    frames = list(snaps)
    assert len(frames) == snaps.n_frames == 3 and frames[0].shape == (2, EA_NZ, EA_NX)
    npt.assert_allclose(snaps.times, [0.1, 0.2, 0.3])
    assert np.abs(frames[2]).max() > 0.0
    stress = list(ea2df(EA_C11, EA_C55, EA_RHO, **EA).snapshots([0.2], stress=True))
    assert stress[0].shape == (3, EA_NZ, EA_NX)
    with pytest.raises(RuntimeError, match="run already"):
        sim.snapshots([0.1])


def test_ea2df_checks_its_input():
    with pytest.raises(ValueError, match="sx"):
        ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'sx': 5000.0})
    with pytest.raises(ValueError, match="sz"):
        ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'sz': -50.0})
    with pytest.raises(ValueError, match="Unknown"):
        ea2df(EA_C11, EA_C55, EA_RHO, stype='s', **EA)
    with pytest.raises(ValueError, match="shape"):
        ea2df(EA_C11, EA_C55[:3], EA_RHO, **EA)
    with pytest.raises(ValueError, match="positive"):
        ea2df(EA_C11, EA_C55, np.zeros_like(EA_RHO), **EA)
    with pytest.raises(ValueError, match="hsz"):
        ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'hsz': 9000.0})
    with pytest.raises(ValueError, match="four values"):
        ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'bc': (1, 2)})


# --------------------------------------------------------------------------------------------------------- remac2d
RM_V = np.full((99, 105), 2000.0, dtype=np.float32)  # (nz, nx): sizes that the Fourier transforms of SU can do, 10 m grid
RM = dict(nt=200, dt=0.002, dx=10.0, dz=10.0, isx=[52], isz=[45], irz=[45], irx=[52], fmax=40.0)


def test_remac2d_the_wave_travels_at_the_velocity_and_is_symmetric():
    sim = remac2d(RM_V, **RM)
    xs = list(sim.x_sections)
    assert len(xs) == 105 and xs[0].n_sample == 200
    peak = [np.abs(np.asarray(xs[i])).argmax() * 0.002 for i in (70, 100)]
    assert peak[1] - peak[0] == pytest.approx(300.0 / 2000.0, abs=0.006)
    data = samples_of(xs)
    npt.assert_allclose(data[52 + 20], data[52 - 20], atol=1e-3 * np.abs(data).max())
    t = xs[60]
    assert t.header['rx_loc'][0] == pytest.approx(600.0) and t.header['rx_loc'][2] == pytest.approx(-450.0)
    assert t.header['tx_loc'][0] == pytest.approx(520.0) and t.header['d_sample'] == pytest.approx(0.002)


def test_remac2d_vertical_sections_snapshots_and_repeatability():
    sim = remac2d(RM_V, dtsnap=0.1, **RM)
    zs = list(sim.z_sections)
    assert len(zs) == 99 and zs[3].header['rx_loc'][2] == pytest.approx(-30.0) and zs[3].header['rx_loc'][0] == pytest.approx(520.0)
    snaps = sim.snapshots
    frames = list(snaps)
    assert snaps.n_frames == 4 and len(frames) == 4 and frames[0].shape == (99, 105)
    npt.assert_allclose(snaps.times, [0.1, 0.2, 0.3, 0.4])
    assert np.abs(frames[3]).max() > 0.0
    again = samples_of(remac2d(RM_V, dtsnap=0.1, **RM).x_sections)
    npt.assert_allclose(again, samples_of(sim.x_sections), atol=1e-5 * np.abs(again).max())


def test_remac2d_the_wave_equations_and_the_other_options():
    constant = samples_of(remac2d(RM_V, **RM).x_sections)
    variable = samples_of(remac2d(RM_V, density=np.full_like(RM_V, 1500.0), **RM).x_sections)
    assert np.corrcoef(constant[75], variable[75])[0, 1] > 0.99  # (a constant density in the variable density equation)
    non_reflecting = samples_of(remac2d(RM_V, opflag=2, **RM).x_sections)
    assert np.isfinite(non_reflecting).all() and np.abs(non_reflecting).max() > 0
    free = samples_of(remac2d(RM_V, fsflag=True, **{**RM, 'isz': [25], 'irz': [25]}).x_sections)
    assert np.isfinite(free).all()
    # a spike source has the impulse response, a wavelet given as an array is used as it is
    spike = samples_of(remac2d(RM_V, wavelet='spike', **{**RM, 'fmax': None}).x_sections)
    assert np.isfinite(spike).all() and np.abs(spike).max() > 0
    wavelet = np.exp(-((np.arange(60) - 20) * 0.002 * 40.0) ** 2).astype(np.float32)
    user = samples_of(remac2d(RM_V, wavelet=wavelet, dt_wavelet=0.002, **RM).x_sections)
    assert np.isfinite(user).all() and np.abs(user).max() > 0
    doubled = samples_of(remac2d(RM_V, amps=[2.0], **RM).x_sections)
    npt.assert_allclose(doubled, 2.0 * constant, atol=1e-4 * np.abs(constant).max())


def test_remac2d_checks_its_input():
    with pytest.raises(ValueError, match="Fourier"):
        remac2d(np.full((99, 101), 2000.0), **RM).run()
    with pytest.raises(ValueError, match="Fourier"):
        remac2d(np.full((99, 104), 2000.0), density=np.full((99, 104), 1000.0), **RM).run()
    with pytest.raises(ValueError, match="source box"):
        remac2d(RM_V, **{**RM, 'isx': [3]}).run()
    with pytest.raises(ValueError, match="fmax"):
        remac2d(RM_V, **{**RM, 'fmax': None})
    with pytest.raises(ValueError, match="too high"):
        remac2d(RM_V, **{**RM, 'fmax': 150.0}).run()
    with pytest.raises(ValueError, match="outside"):
        remac2d(RM_V, **{**RM, 'irz': [500]}).run()
    with pytest.raises(ValueError, match="density"):
        remac2d(RM_V, opflag=0, **RM)
    with pytest.raises(ValueError, match="dtsnap"):
        remac2d(RM_V, dtsnap=-1.0, **RM)


# ------------------------------------------------------------------------------------------------------- remel2dan
EL_RHO = np.full((99, 105), 1000.0, dtype=np.float32)
EL_VP = np.full((99, 105), 2000.0, dtype=np.float32)
EL_VS = np.full((99, 105), 1000.0, dtype=np.float32)
EL = dict(nt=200, dt=0.002, dx=10.0, dz=10.0, isx=[52], isz=[45], fmax=40.0, vp=EL_VP, vs=EL_VS)


def el_peaks(traces, n):
    return np.array([np.abs(np.asarray(t)).argmax() for t in list(traces)[:n]]) * 0.002


def test_remel2dan_p_and_s_waves_travel_at_their_velocities():
    p_wave = remel2dan(EL_RHO, styp='fx', x_lines=[(45, 'ux')], **EL)
    peaks = el_peaks(p_wave.x_sections, 105)
    assert peaks[90] - peaks[70] == pytest.approx(200.0 / 2000.0, abs=0.008)
    s_wave = remel2dan(EL_RHO, styp='fz', x_lines=[(45, 'uz')], **{**EL, 'nt': 300})
    peaks = el_peaks(s_wave.x_sections, 105)
    assert peaks[90] - peaks[70] == pytest.approx(200.0 / 1000.0, abs=0.012)


def test_remel2dan_sections_snapshots_and_headers():
    sim = remel2dan(EL_RHO, styp='p', x_lines=[(45, 'p'), (50, 'ux')], z_lines=[(52, 'p'), (60, 'uz')], snapshots_of=['p', 'ux'],
                    dtsnap=0.1, **EL)
    xs = list(sim.x_sections)
    assert len(xs) == 2 * 105 and xs[0].n_sample == 200
    assert xs[105 + 7].header['ensemble_number'] == 2 and xs[7].header['rx_loc'][0] == pytest.approx(70.0)
    assert xs[7].header['rx_loc'][2] == pytest.approx(-450.0) and xs[7].header['tx_loc'][0] == pytest.approx(520.0)
    zs = list(sim.z_sections)
    assert len(zs) == 2 * 99 and zs[3].header['rx_loc'][0] == pytest.approx(520.0)
    # the pressure along the vertical line is made (SU never made it)
    assert np.abs(np.asarray(zs[45])).max() > 0.0
    snaps = sim.snapshots
    assert set(snaps) == {'p', 'ux'} and snaps['p'].n_frames == 4
    frames = list(snaps['p'])
    assert frames[0].shape == (99, 105) and np.abs(frames[3]).max() > 0.0
    npt.assert_allclose(snaps['ux'].times, [0.1, 0.2, 0.3, 0.4])


def test_remel2dan_an_anisotropic_medium_that_is_isotropic_is_the_isotropic_one():
    iso = samples_of(remel2dan(EL_RHO, styp='fx', x_lines=[(45, 'ux')], **EL).x_sections)
    c33 = EL_RHO * EL_VP ** 2
    c55 = EL_RHO * EL_VS ** 2
    zero = np.zeros_like(c33)
    el = {k: v for k, v in EL.items() if k not in ('vp', 'vs')}
    ani = samples_of(remel2dan(EL_RHO, styp='fx', x_lines=[(45, 'ux')], c11=c33, c13=c33 - 2 * c55, c15=zero, c33=c33, c35=zero,
                               c55=c55, vmax=2000.0, vmin=1000.0, **el).x_sections)
    npt.assert_allclose(ani, iso, atol=1e-3 * np.abs(iso).max())


def test_remel2dan_two_sources_are_the_sum_of_the_two():
    both = samples_of(remel2dan(EL_RHO, styp=['p', 'p'], x_lines=[(45, 'p')], **{**EL, 'isx': [40, 66], 'isz': [45, 45]}).x_sections)
    one = samples_of(remel2dan(EL_RHO, styp='p', x_lines=[(45, 'p')], **{**EL, 'isx': [40], 'isz': [45]}).x_sections)
    two = samples_of(remel2dan(EL_RHO, styp='p', x_lines=[(45, 'p')], **{**EL, 'isx': [66], 'isz': [45]}).x_sections)
    npt.assert_allclose(both, one + two, atol=1e-4 * np.abs(both).max())


def test_remel2dan_checks_its_input():
    with pytest.raises(ValueError, match="needs vp and vs"):
        remel2dan(EL_RHO, **{k: v for k, v in EL.items() if k not in ('vp', 'vs')})
    with pytest.raises(ValueError, match="anisotropic"):
        remel2dan(EL_RHO, c11=EL_VP, **{k: v for k, v in EL.items() if k not in ('vp', 'vs')})
    with pytest.raises(ValueError, match="Fourier"):
        remel2dan(np.full((99, 104), 1000.0), vp=np.full((99, 104), 2000.0), vs=np.full((99, 104), 1000.0),
                  **{k: v for k, v in EL.items() if k not in ('vp', 'vs')}).run()
    with pytest.raises(ValueError, match="source box"):
        remel2dan(EL_RHO, **{**EL, 'isx': [3]}).run()
    with pytest.raises(ValueError, match="receiver type"):
        remel2dan(EL_RHO, x_lines=[(45, 'q')], **EL)
    with pytest.raises(ValueError, match="source type"):
        remel2dan(EL_RHO, styp='q', **EL)
    with pytest.raises(ValueError, match="outside"):
        remel2dan(EL_RHO, x_lines=[(500, 'p')], **EL).run()
    with pytest.raises(ValueError, match="too high"):
        remel2dan(EL_RHO, **{**EL, 'fmax': 150.0}).run()
    with pytest.raises(ValueError, match="snapshots_of"):
        remel2dan(EL_RHO, dtsnap=0.1, **EL)


# ----------------------------------------------------------------------------------------------------- fctanismod
FC = dict(aa=4.0, cc=4.0, ff=2.0, ll=1.0, nn=1.0, rho=2.0, nx=120, nz=100, dx=0.02, dz=0.02, nt=160, dt=0.004, fpeak=20.0,
          force=(True, False, False), isurf=1)


def test_fctanismod_the_p_wave_travels_at_the_velocity_and_the_stability_is_reported():
    sim = fctanismod(**FC)
    assert sim.vmax == pytest.approx(math.sqrt(2.0), rel=1e-5) and sim.stability == pytest.approx(0.2, rel=1e-4)
    traces = list(sim.reflection('x'))
    assert len(traces) == 120 and traces[0].n_sample == 160
    peak = np.array([np.abs(np.asarray(t)).argmax() for t in traces]) * 0.004
    # the horizontal P velocity is sqrt(aa / rho) = 1.414 km/s: 0.2 km takes 0.141 s
    assert peak[90] - peak[80] == pytest.approx(0.2 / 2.0 ** 0.5, abs=0.012)
    t = traces[70]
    assert t.header['rx_loc'][0] == pytest.approx(1.4) and t.header['rx_loc'][2] == pytest.approx(-1.0)
    assert t.header['tx_loc'][0] == pytest.approx(1.2) and t.header['tx_loc'][2] == pytest.approx(-1.0)


def test_fctanismod_vsp_snapshots_and_the_final_snapshot():
    sim = fctanismod(**{**FC, 'force': (False, False, True)})
    snaps = sim.snapshots(mt=40)
    frames = list(snaps)
    assert snaps.n_frames == len(frames) == 4 and frames[0].shape == (3, 100, 120)
    assert snaps.dt == pytest.approx(0.16)
    vsp = list(sim.vsp('z'))
    assert len(vsp) == 100 and vsp[10].header['rx_loc'][0] == pytest.approx(1.2) and vsp[10].header['rx_loc'][2] == pytest.approx(-0.2)
    assert np.abs(sim.final_snapshot).max() > 0.0
    with pytest.raises(RuntimeError, match="run already"):
        sim.snapshots()


def test_fctanismod_the_correction_the_surface_and_the_profiles():
    plain = samples_of(fctanismod(dofct=False, **FC).reflection('x'))
    corrected = samples_of(fctanismod(dofct=True, **FC).reflection('x'))
    assert np.isfinite(corrected).all() and np.abs(corrected - plain).max() > 1e-4 * np.abs(plain).max()
    free = samples_of(fctanismod(**{**FC, 'isurf': 2, 'sz': 5}).reflection('x'))
    absorbing = samples_of(fctanismod(**{**FC, 'isurf': 1, 'sz': 5}).reflection('x'))
    assert np.abs(free - absorbing).max() > 0.05 * np.abs(absorbing).max()
    # a constant, an array and a linear profile (with no gradient) are the same medium
    arrays = {k: np.full((100, 120), FC[k], np.float32) for k in ('aa', 'cc', 'ff', 'll', 'nn', 'rho')}
    from_arrays = samples_of(fctanismod(**{**FC, **arrays}).reflection('x'))
    linear = samples_of(fctanismod(**{**FC, 'aa': (4.0, 0.0, 0.0)}).reflection('x'))
    npt.assert_array_equal(from_arrays, corrected)
    npt.assert_array_equal(linear, from_arrays)
    # the profile has its value at (0, 0)
    gradient = {**FC, 'aa': (4.0, 0.0, 1.0), 'dofct': False}
    assert not np.array_equal(samples_of(fctanismod(**gradient).reflection('x')), plain)


def test_fctanismod_another_source_the_moving_boundary_and_the_wavelets():
    base = samples_of(fctanismod(**FC).reflection('x'))
    xz = np.zeros((100, 120), dtype=np.float32)
    xz[50, 60] = 1.0
    npt.assert_array_equal(samples_of(fctanismod(xzsource=xz, **FC).reflection('x')), base)
    moving = samples_of(fctanismod(movebc=True, impulse=True, **FC).reflection('x'))
    assert np.isfinite(moving).all() and np.abs(moving).max() > 0
    for wavelet in (2, 3, 4):
        assert np.isfinite(samples_of(fctanismod(wavelet=wavelet, **FC).vsp('x'))).all()
    assert np.isfinite(samples_of(fctanismod(source=2, **FC).reflection('x'))).all()


def test_fctanismod_checks_its_input():
    with pytest.raises(ValueError, match="component"):
        fctanismod(**FC).reflection('q')
    with pytest.raises(ValueError, match="outside"):
        fctanismod(**{**FC, 'sx': 500})
    with pytest.raises(ValueError, match="wavelet"):
        fctanismod(**{**FC, 'wavelet': 9})
    with pytest.raises(ValueError, match="match"):
        fctanismod(**{**FC, 'aa': np.ones((3, 3))})
    with pytest.raises(ValueError, match="density"):
        fctanismod(**{**FC, 'rho': 0.0})
    with pytest.raises(ValueError, match="three flags"):
        fctanismod(**{**FC, 'force': (1, 0)})
    with pytest.raises(ValueError, match="xzsource"):
        fctanismod(xzsource=np.zeros((3, 3)), **FC)


def samples_of(traces):
    return np.array([np.asarray(t) for t in traces])


def test_the_finite_difference_models_can_be_run_by_several_threads_at_once():
    import concurrent.futures

    work = {
        'fdmod1': lambda: np.array([np.asarray(t) for t in fdmod1(V1, rz=1000.0, **FD1)]),
        'fdmod2': lambda: samples_of(fdmod2(V2, hsz=250.0, **{**FD2, 'tmax': 0.3}).horizontal_line),
        'fdmod2_pml': lambda: samples_of(fdmod2_pml(V2, hsz=250.0, pml_thick=6, **{**FD2, 'tmax': 0.3}).horizontal_line),
        'ea2df': lambda: samples_of(ea2df(EA_C11, EA_C55, EA_RHO, **{**EA, 'lt': 0.15}).horizontal_line),
        'remac2d': lambda: samples_of(remac2d(RM_V, **RM).x_sections),
        'remel2dan': lambda: samples_of(remel2dan(EL_RHO, styp='fx', x_lines=[(45, 'ux')], **EL).x_sections),
        'fctanismod': lambda: samples_of(fctanismod(**{**FC, 'nt': 60}).reflection('x')),
        'fdmod2 extended': lambda: samples_of(fdmod2(V2, hsz=250.0, **{**FD2, 'xs': [800.0, 1200.0], 'zs': [250.0, 250.0],
                                                                      'tmax': 0.3}).horizontal_line),
    }
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for name, make in work.items():
            expected = make()
            for got in pool.map(lambda _: make(), range(8)):
                # (the same to rounding: a time stepping amplifies the last bit of a single precision value)
                npt.assert_allclose(got, expected, atol=1e-4 * np.abs(expected).max(), err_msg=name)
