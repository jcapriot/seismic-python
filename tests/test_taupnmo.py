"""
SUTAUPNMO: NMO of traces in the tau-p domain, tau^2 (1 - p^2 v^2), by the library version of the program.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.stretching import taupnmo

DT = 0.004
NT = 200


def pulse(sample, nt=NT, amp=1.0):
    x = np.zeros(nt, dtype=np.float32)
    x[sample] = amp
    return x


def trace(x, **header):
    return Trace(x, d_sample=DT).replace(**header)


def run(stage, traces):
    return [np.asarray(t) for t in from_iterable(traces) | stage]


def test_zero_ray_parameter_changes_nothing_but_the_last_sample():
    x = np.random.default_rng(1).normal(size=NT).astype(np.float32)
    (out,) = run(taupnmo(vnmo=2000.0, p=lambda t: 0.0, lmute=0), [trace(x)])
    # (the program mutes from the first sample whose stretch is more than smute: with none that is the last sample)
    npt.assert_allclose(out[:-1], x[:-1], atol=1e-5)
    assert out[-1] == 0.0


def test_a_pulse_moves_to_tau_over_the_square_root():
    v = 2000.0
    p = 0.6 / v  # p v = 0.6, so t = tn 0.8
    (out,) = run(taupnmo(vnmo=v, p=lambda t: p, lmute=0, sscale=False), [trace(pulse(40))])
    npt.assert_allclose(out[50], 1.0, atol=1e-3)
    # (the samples next to it read the pulse between samples, which the 8 point sinc interpolation spreads)
    assert np.argmax(np.abs(out)) == 50
    assert np.abs(np.delete(out, range(40, 61))).max() < 0.15


def test_stretch_scaling_divides_by_the_stretch():
    v = 2000.0
    p = 0.6 / v
    (out,) = run(taupnmo(vnmo=v, p=lambda t: p, lmute=0, sscale=True), [trace(pulse(40, amp=2.0))])
    # a(tn) is the inverse of the stretch, 0.8 here: the program scales the muted samples (nothing), this the others
    npt.assert_allclose(out[50], 2.0 * 0.8, atol=2e-3)


def test_samples_with_too_much_stretch_are_muted():
    v = 2000.0
    # stretch 1 / 0.6 = 1.667 is more than the default smute 1.5: all of it is muted
    p = np.sqrt(1 - 0.36) / v
    (out,) = run(taupnmo(vnmo=v, p=lambda t: p), [trace(np.ones(NT, dtype=np.float32))])
    assert not out.any()
    (out,) = run(taupnmo(vnmo=v, p=lambda t: p, smute=2.0, lmute=0, sscale=False), [trace(np.ones(NT, dtype=np.float32))])
    npt.assert_allclose(out[20:-20], 1.0, atol=5e-3)  # (not the ends, where the sinc interpolation runs out of samples)
    assert out[-1] == 0.0


def test_evanescent_samples_are_muted():
    v = 2000.0
    (out,) = run(taupnmo(vnmo=v, p=lambda t: 1.2 / v), [trace(np.ones(NT, dtype=np.float32))])
    assert not out.any()


def test_the_ramp_goes_down_to_the_mute():
    # velocity that increases with tau: the stretch passes smute at some time, and the lmute samples before it are ramped
    x = np.ones(NT, dtype=np.float32)
    stage = taupnmo(tnmo=[0.0, 0.8], vnmo=[1000.0, 3000.0], p=lambda t: 0.5 / 1000.0, sscale=False, lmute=10)
    (out,) = run(stage, [trace(x)])
    nonzero = np.flatnonzero(out)
    itmute = nonzero[-1] + 1
    assert 10 < itmute < NT - 1
    npt.assert_allclose(out[itmute - 11], 1.0, atol=1e-2)
    ramp = out[itmute - 10:itmute]
    assert np.all(np.diff(ramp) < 0) and ramp[-1] == pytest.approx(0.1, abs=1e-2)
    assert not out[itmute:].any()


def test_a_mute_that_starts_before_the_ramp_is_long_enough_does_not_fail():
    # (the program writes before the start of the trace here)
    x = np.ones(NT, dtype=np.float32)
    stage = taupnmo(vnmo=2000.0, p=lambda t: 0.99 / 2000.0, smute=1.01, lmute=500)
    (out,) = run(stage, [trace(x)])
    assert out.shape == (NT,) and np.isfinite(out).all()


def test_ray_parameter_from_a_header_name_and_from_a_spacing():
    v = 2000.0
    traces = [trace(pulse(40), trace_id=i + 1) for i in range(3)]
    by_spacing = run(taupnmo(vnmo=v, p0=0.0, dp=0.3 / v, lmute=0, sscale=False), traces)
    for i, out in enumerate(by_spacing):
        factor = np.sqrt(1 - (0.3 * i) ** 2)
        assert np.argmax(out) == pytest.approx(40 / factor, abs=0.6)
    by_header = run(taupnmo(vnmo=v, p='ensemble_number', lmute=0, sscale=False),
                    [trace(pulse(40), ensemble_number=1)])
    # (a ray parameter of 1 is a lot more than 1 / v: everything is evanescent)
    assert not by_header[0].any()


def test_the_cdp_functions_are_interpolated_in_v_squared():
    mid = np.sqrt((1000.0 ** 2 + 2000.0 ** 2) / 2)
    p = 0.3 / 1000.0
    kwargs = dict(p=lambda t: p, lmute=0, sscale=False)
    stage = taupnmo(tnmo=[[0.0], [0.0]], vnmo=[[1000.0], [2000.0]], cdp=[100, 200], **kwargs)
    (got,) = run(stage, [trace(pulse(30), ensemble_number=150)])
    (expected,) = run(taupnmo(vnmo=mid, **kwargs), [trace(pulse(30))])
    npt.assert_allclose(got, expected, atol=1e-4)
    # outside the cdps, the nearest function is used
    (low,) = run(stage, [trace(pulse(30), ensemble_number=50)])
    (ref,) = run(taupnmo(vnmo=1000.0, **kwargs), [trace(pulse(30))])
    npt.assert_allclose(low, ref, atol=1e-6)


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        taupnmo(vnmo=2000.0)  # no ray parameter
    with pytest.raises(ValueError):
        taupnmo(vnmo=2000.0, p='offset', dp=1.0)
    with pytest.raises(ValueError):
        taupnmo(vnmo=2000.0, dp=1.0, smute=0.0)
    with pytest.raises(ValueError):
        taupnmo(tnmo=[0.0, 0.0], vnmo=[1000.0, 2000.0], dp=1.0)
    s = taupnmo(vnmo=2000.0, dp=1.0)
    assert s.parallelism == 'trace'
