"""
SUDIPFILT: slope filter in the f-k domain, checked with plane waves on the grid of the transforms.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.filters import dipfilt

DT = 0.004
NT, NX, DX = 200, 32, 10.0  # df = 1.25 Hz, dK = 1 / 320 per m
F0 = 25.0  # (bin 20)
K0 = 5.0 / (NX * DX)
SLOPE = K0 / F0  # (the slope delta_t / delta_x of cos(2 pi (F t - K x)))


def wave(k):
    t = np.arange(NT) * DT
    x = np.arange(NX) * DX
    return np.cos(2 * np.pi * (F0 * t[None, :] - k * x[:, None])).astype(np.float32)


def run(stage, data, dt=DT):
    traces = [Trace(row, d_sample=dt) for row in data]
    return np.array([np.asarray(t) for t in from_iterable(traces) | stage])


def test_default_does_nothing():
    data = np.random.default_rng(1).normal(size=(NX, NT)).astype(np.float32)
    npt.assert_allclose(run(dipfilt(dx=DX), data), data, atol=1e-4)
    npt.assert_allclose(run(dipfilt(slopes=[0.0, 1.0], amps=[1.0, 1.0], dx=DX), data), data, atol=1e-4)


def test_it_rejects_one_direction_and_keeps_the_other():
    up, down = wave(K0), wave(-K0)
    both = up + down
    stage = dipfilt(slopes=[SLOPE / 3, 2 * SLOPE / 3], amps=[1.0, 0.0], dx=DX)  # keeps negative slopes
    out = run(stage, both)
    npt.assert_allclose(out, down, atol=2e-3)
    stage = dipfilt(slopes=[-2 * SLOPE / 3, -SLOPE / 3], amps=[0.0, 1.0], dx=DX)  # keeps positive slopes
    npt.assert_allclose(run(stage, both), up, atol=2e-3)


def test_amplitudes_scale_the_slopes_that_they_are_given_for():
    stage = dipfilt(slopes=[0.0], amps=[0.25], dx=DX)  # one amplitude, for every slope
    npt.assert_allclose(run(stage, wave(K0)), 0.25 * wave(K0), atol=1e-3)


def test_the_bias_does_not_change_which_slopes_are_filtered():
    both = wave(K0) + wave(-K0)
    kwargs = dict(slopes=[SLOPE / 3, 2 * SLOPE / 3], amps=[1.0, 0.0], dx=DX)
    for bias in (0.0, SLOPE, -SLOPE):
        out = run(dipfilt(bias=bias, **kwargs), both)
        # (a bias that moves the events by a fraction of a sample leaks a little, it is the filter of the bias grid)
        npt.assert_allclose(out, wave(-K0), atol=5e-2)


def test_horizontal_events_have_slope_zero():
    flat = wave(0.0)
    keep_flat = dipfilt(slopes=[-SLOPE, -SLOPE / 2, SLOPE / 2, SLOPE], amps=[0.0, 1.0, 1.0, 0.0], dx=DX)
    npt.assert_allclose(run(keep_flat, flat), flat, atol=2e-3)
    npt.assert_allclose(run(keep_flat, wave(K0)), 0.0, atol=2e-3)


def test_dt_is_taken_from_the_traces_or_given():
    data = wave(K0)
    stage = dipfilt(slopes=[SLOPE / 3, 2 * SLOPE / 3], amps=[1.0, 0.0], dx=DX)
    with_header = run(stage, data)
    without = run(dipfilt(slopes=[SLOPE / 3, 2 * SLOPE / 3], amps=[1.0, 0.0], dx=DX, dt=DT), data, dt=0.0)
    npt.assert_allclose(with_header, without, atol=1e-6)
    with pytest.raises(ValueError, match="dt"):
        run(stage, data, dt=0.0)


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        dipfilt(slopes=[0.0, 1.0], amps=[1.0])
    with pytest.raises(ValueError):
        dipfilt(slopes=[1.0, 0.0], amps=[1.0, 1.0])
    with pytest.raises(ValueError, match="dx"):
        run(dipfilt(), wave(K0))
    with pytest.raises(TypeError):
        list(from_iterable([Trace((np.ones(NT) + 1j).astype(np.complex64), d_sample=DT)]) | dipfilt(dx=DX))
    assert dipfilt(dx=DX).parallelism == 'serial'
    assert list(from_iterable([]) | dipfilt(dx=DX)) == []
