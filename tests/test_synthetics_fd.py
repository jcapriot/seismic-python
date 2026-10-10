"""
Finite-difference modelling of the acoustic wave equation: SUFDMOD1 and SUFDMOD2.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.synthetics import Snapshots, fdmod1, fdmod1_snapshots, fdmod2

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


def samples_of(traces):
    return np.array([np.asarray(t) for t in traces])


def test_the_finite_difference_models_can_be_run_by_several_threads_at_once():
    import concurrent.futures

    work = {
        'fdmod1': lambda: np.array([np.asarray(t) for t in fdmod1(V1, rz=1000.0, **FD1)]),
        'fdmod2': lambda: samples_of(fdmod2(V2, hsz=250.0, **{**FD2, 'tmax': 0.3}).horizontal_line),
        'fdmod2 extended': lambda: samples_of(fdmod2(V2, hsz=250.0, **{**FD2, 'xs': [800.0, 1200.0], 'zs': [250.0, 250.0],
                                                                      'tmax': 0.3}).horizontal_line),
    }
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for name, make in work.items():
            expected = make()
            for got in pool.map(lambda _: make(), range(8)):
                # (the same to rounding: a time stepping amplifies the last bit of a single precision value)
                npt.assert_allclose(got, expected, atol=1e-4 * np.abs(expected).max(), err_msg=name)
