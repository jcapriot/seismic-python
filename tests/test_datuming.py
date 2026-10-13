"""Datuming (SUDATUMFD)."""
import numpy as np
import pytest

from seispy import datuming
from seispy.container import Trace, from_iterable

DT = 0.002
NT = 256


def gather(nx, nsx=1, spike_at=60, dx=10.0, dt=DT, nt=NT):
    traces = []
    for isx in range(nsx):
        for ix in range(nx):
            data = np.zeros(nt, dtype=np.float32)
            data[spike_at] = 1.0
            traces.append(Trace(data, d_sample=dt).replace(rx_loc=[ix * dx, 0.0, 0.0], ensemble_number=isx + 1))
    return traces


def run(traces, **kwargs):
    return list(from_iterable(list(traces)) | datuming.datumfd(**kwargs))


def test_no_steps_is_the_identity():
    traces = gather(12)
    vel = np.full((12, 4), 2000.0, np.float32)
    out = run(traces, vel1=vel, vel2=vel, nx=12, nsx=1, nz=0, dz=10.0, dx=10.0)
    for a, b in zip(traces, out):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), atol=1e-5)


def test_thin_lens_moves_the_event_up_by_the_two_way_time():
    # (the diffraction term is switched off with a tiny velocity: only the time shift of the thin lens is left)
    nx, nz, dz, v = 24, 5, 10.0, 2000.0
    traces = gather(nx)
    vel1 = np.full((nx, nz), v, np.float32)
    vel2 = np.full((nx, nz), 1e-3, np.float32)
    out = run(traces, vel1=vel1, vel2=vel2, nx=nx, nsx=1, nz=nz, dz=dz, dx=10.0)
    shift = nz * dz / (0.5 * v) / DT  # samples
    middle = np.asarray(out[nx // 2])
    assert abs(int(np.argmax(middle)) - (60 - shift)) <= 1
    assert middle.max() > 0.9


def test_blocks_of_gathers_and_headers():
    nx, nsx = 10, 2
    traces = gather(nx, nsx) + gather(nx, nsx)
    vel = np.full((nx, 6), 2500.0, np.float32)
    out = run(traces, vel1=vel, vel2=vel, nx=nx, nsx=nsx, nz=3, dz=8.0, dx=10.0)
    assert len(out) == len(traces)
    for a, b in zip(traces, out):
        assert b.header['ensemble_number'] == a.header['ensemble_number']
        np.testing.assert_array_equal(b.header['rx_loc'], a.header['rx_loc'])
    # the same block gives the same result
    for a, b in zip(out[:nx * nsx], out[nx * nsx:]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_gathers_are_independent():
    nx = 10
    vel = np.full((nx, 6), 2500.0, np.float32)
    both = gather(nx, 2)
    both[nx + 3] = both[nx + 3].replace(np.random.default_rng(0).normal(size=NT).astype(np.float32))
    out2 = run(both, vel1=vel, vel2=vel, nx=nx, nsx=2, nz=3, dz=8.0, dx=10.0)
    out1 = run(both[:nx], vel1=vel, vel2=vel, nx=nx, nsx=1, nz=3, dz=8.0, dx=10.0)
    for a, b in zip(out1, out2[:nx]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_receiver_outside_the_model_warns_and_works():
    nx = 8
    vel = np.full((4, 3), 2000.0, np.float32)  # 4 columns of 10 m: the receivers go to 70 m
    with pytest.warns(UserWarning, match="outside"):
        out = run(gather(nx), vel1=vel, vel2=vel, nx=nx, nsx=1, nz=2, dz=5.0, dx=10.0)
    assert all(np.isfinite(np.asarray(t)).all() for t in out)


def test_errors():
    vel = np.full((8, 3), 2000.0, np.float32)
    with pytest.raises(ValueError):
        datuming.datumfd(vel, vel[:, :2], nx=8, nsx=1, nz=2, dz=5.0)
    with pytest.raises(ValueError):
        datuming.datumfd(vel, vel, nx=8, nsx=1, nz=4, dz=5.0)  # more steps than depth samples
    with pytest.raises(ValueError):
        datuming.datumfd(vel, vel, nx=2, nsx=1, nz=2, dz=5.0)
    with pytest.raises(ValueError):
        datuming.datumfd(vel, vel, nx=8, nsx=1, nz=2, dz=0.0)
    with pytest.raises(ValueError):  # not a whole number of gathers
        run(gather(7), vel1=vel, vel2=vel, nx=8, nsx=1, nz=2, dz=5.0, dx=10.0)
