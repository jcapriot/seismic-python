"""
SUTAUP: forward and inverse slant stacks (tau-p transforms), by the routines of the SU library.
"""
import numpy as np
import pytest

from seispy.container import Trace, from_iterable
from seispy.transforms import taup

NT, NX, DT, DX = 200, 24, 0.004, 10.0
P0, T0 = 0.002, 40  # the slope of the event (s/m), and its intercept (samples)
KWARGS = dict(dx=DX, pmin=0.0, pmax=0.004, np=21)  # (dp = .0002, so P0 is the 10th slope)


def event():
    data = np.zeros((NX, NT), dtype=np.float32)
    for ix in range(NX):
        data[ix, T0 + int(round(P0 * ix * DX / DT))] = 1.0
    return data


def run(stage, data, dt=DT):
    return np.array([np.asarray(t) for t in from_iterable([Trace(r, d_sample=dt) for r in data]) | stage])


@pytest.mark.parametrize('option', [1, 2])
def test_forward_transform_puts_a_linear_event_at_its_slope_and_intercept(option):
    out = run(taup(option, **KWARGS), event())
    assert out.shape == (21, NT)
    ip, itau = np.unravel_index(np.argmax(np.abs(out)), out.shape)
    assert (ip, itau) == (10, T0)
    # the other slopes get less of it
    assert np.abs(out[ip, itau]) > 2 * np.abs(np.delete(out, ip, axis=0)).max()


@pytest.mark.parametrize('forward, inverse', [(2, 4), (1, 3)])
def test_inverse_transform_puts_the_event_back(forward, inverse):
    data = event()
    tau_p = run(taup(forward, **KWARGS), data)
    back = run(taup(inverse, nx=NX, **KWARGS), tau_p)
    assert back.shape == data.shape
    # (the amplitudes are only relative, but the event is where it was)
    npt_ok = np.abs(np.argmax(back, axis=1) - np.argmax(data, axis=1)) <= 1
    assert npt_ok.all()
    assert np.corrcoef(back.ravel(), data.ravel())[0, 1] > 0.3


def test_the_inverse_can_make_more_traces_than_there_are_slopes():
    # (the program writes beyond what it allocated here)
    tau_p = run(taup(2, **KWARGS), event())
    back = run(taup(4, nx=60, **KWARGS), tau_p)
    assert back.shape == (60, NT)


def test_a_different_number_of_slopes():
    out = run(taup(2, dx=DX, pmin=0.0, pmax=0.004, np=9), event())
    assert out.shape == (9, NT)
    assert np.unravel_index(np.argmax(np.abs(out)), out.shape) == (4, T0)  # dp = .0005: P0 is the 4th slope


def test_nx_uses_the_first_traces_of_a_forward_transform():
    data = event()
    few = run(taup(2, **{**KWARGS, 'nx': 12}), data)
    ref = run(taup(2, **KWARGS), data[:12])
    # (the same transform of the first 12 traces; the default is all of them)
    np.testing.assert_allclose(few, run(taup(2, **{**KWARGS, 'nx': 12}), data[:12]), atol=1e-6)
    assert ref.shape == few.shape


def test_headers_follow_the_traces_and_the_dt_can_be_given():
    out = list(from_iterable([Trace(r, d_sample=DT).replace(ensemble_number=7) for r in event()]) | taup(2, **KWARGS))
    assert len(out) == 21 and all(t.header['ensemble_number'] == 7 for t in out)
    assert [t.header['trace_id'] for t in out] == list(range(1, 22))
    assert out[0].d_sample == pytest.approx(DT)
    given = run(taup(2, dt=DT, **KWARGS), event(), dt=0.0)
    np.testing.assert_allclose(given, run(taup(2, **KWARGS), event()), atol=1e-6)


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        taup(5)
    with pytest.raises(ValueError):
        taup(2, np=1)
    with pytest.raises(ValueError, match="sample interval"):
        run(taup(2, **KWARGS), event(), dt=0.0)
    with pytest.raises(ValueError):
        run(taup(2, **{**KWARGS, 'nx': NX + 5}), event())  # more traces than there are
    with pytest.raises(ValueError):
        run(taup(4, **KWARGS), event()[:10])  # fewer than np traces to invert
    with pytest.raises(TypeError):
        list(from_iterable([Trace((np.ones(NT) + 1j).astype(np.complex64), d_sample=DT)]) | taup(2, **KWARGS))
    assert taup().parallelism == 'serial'
    assert list(from_iterable([]) | taup()) == []
