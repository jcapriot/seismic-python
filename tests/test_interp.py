"""
SUINTERP, SUINTERPFOWLER and SUOCEXT.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.dmo import ocext
from seispy.interp import interp, interpfowler

DT = 0.004


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def ricker(t, f=12.0):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def dipping(n_x=40, nt=200, dip=2.0, t0=0.3, dx=25.0):
    """A linear event of a dip of `dip` samples per trace, as the traces of a line"""
    t = np.arange(nt) * DT
    return [Trace(ricker(t - t0 - dip * DT * ix).astype(np.float32), d_sample=DT).replace(
        tx_loc=[dx * ix, 0.0, 0.0], rx_loc=[dx * ix, 0.0, 0.0], ensemble_number=1) for ix in range(n_x)]


# ------------------------------------------------------------------------------------------------------- interp
@pytest.mark.parametrize('xopt', [0, 1])
def test_interp_makes_the_traces_between_along_the_dip(xopt):
    dense = dipping(n_x=47)
    coarse = dense[::2]  # (24 traces, with a dip of 4 samples per trace)
    out = run(interp(ninterp=1, xopt=xopt), coarse)
    assert len(out) == 1 + (len(coarse) - 1) * 2
    # the original traces are passed through
    for k, trace in enumerate(coarse):
        npt.assert_allclose(np.asarray(out[2 * k]), np.asarray(trace), atol=1e-6)
    # the new ones are the traces that were dropped
    middle = np.array([np.asarray(out[2 * k + 1]) for k in range(1, len(coarse) - 2)])
    truth = np.array([np.asarray(dense[2 * k + 1]) for k in range(1, len(coarse) - 2)])
    assert np.corrcoef(middle.ravel(), truth.ravel())[0, 1] > 0.9
    # (and are more like them than the average of the neighbours is)
    neighbours = np.array([0.5 * (np.asarray(coarse[k]) + np.asarray(coarse[k + 1])) for k in range(1, len(coarse) - 2)])
    assert np.abs(middle - truth).sum() < 0.6 * np.abs(neighbours - truth).sum()


def test_interp_headers_options_and_outputs():
    coarse = dipping(n_x=20)[::2]
    out = run(interp(ninterp=2, xopt=1), coarse)
    assert len(out) == 1 + (len(coarse) - 1) * 3
    # the positions are between those of the neighbours
    xs = [t.header['tx_loc'][0] for t in out[:4]]
    npt.assert_allclose(xs, [0.0, 50.0 / 3, 100.0 / 3, 50.0])
    # the low-pass model, and the dips: one trace for each input trace
    low = run(interp(iopt=1, xopt=1), coarse)
    assert len(low) == len(coarse) and not np.allclose(np.asarray(low[3]), np.asarray(coarse[3]))
    dips = run(interp(iopt=2, xopt=1), coarse)
    assert len(dips) == len(coarse)
    values = np.asarray(dips[5])
    centre = int(0.3 / DT) + 4 * 5
    on_event = np.abs(values[centre - 2:centre + 3])
    assert np.median(on_event) == pytest.approx(4.0, rel=0.4)
    # linear time interpolation and the vertical derivative run, and ninterp=0 passes the traces
    assert len(run(interp(linear=True, deriv=True, xopt=1), coarse)) == 2 * len(coarse) - 1
    same = run(interp(ninterp=0), coarse)
    npt.assert_array_equal(np.array([np.asarray(t) for t in same]), np.array([np.asarray(t) for t in coarse]))
    # one trace is passed
    assert len(run(interp(), coarse[:1])) == 1


def test_interp_errors():
    with pytest.raises(ValueError):
        interp(ninterp=-1)
    with pytest.raises(ValueError):
        interp(iopt=3)
    with pytest.raises(ValueError, match="sample interval"):
        run(interp(), [Trace(np.ones(30, dtype=np.float32), d_sample=0.0)] * 4)
    with pytest.raises(ValueError, match="same number"):
        run(interp(), [Trace(np.ones(30, dtype=np.float32), d_sample=DT),
                       Trace(np.ones(20, dtype=np.float32), d_sample=DT)])
    assert interp().parallelism == 'serial'


# ---------------------------------------------------------------------------------------------------- interpfowler
def panels(n_cdp, nv, neta, nt, vmin, dv, etamin, deta):
    traces = []
    for icmp in range(n_cdp):
        for ie in range(neta):
            for iv in range(nv):
                value = vmin + iv * dv + 1000.0 * (etamin + ie * deta)
                traces.append(Trace(np.full(nt, value, dtype=np.float32), d_sample=DT).replace(
                    tx_loc=[10.0 * icmp, 0, 0], rx_loc=[10.0 * icmp, 0, 0]))
    return traces


def test_interpfowler_picks_the_panel_values_along_the_functions():
    nv, neta, n_cdp, nt = 5, 3, 10, 30
    traces = panels(n_cdp, nv, neta, nt, 1500.0, 250.0, 0.1, 0.1)
    # panels that hold their own velocity plus 1000 eta: the output is v(t) + 1000 eta(t)
    out = run(interpfowler(cdps=[0, 9], times=[[0.0, 0.1], [0.0, 0.1]], velocities=[[1600.0, 1800.0], [2400.0, 2200.0]],
                           etas=[[0.12, 0.12], [0.18, 0.18]], vmin=1500.0, vmax=2500.0, nv=nv, etamin=0.1, etamax=0.3,
                           neta=neta, ncdps=n_cdp),
              traces)
    assert len(out) == n_cdp and out[0].n_sample == nt
    t = np.arange(nt) * DT
    v0 = np.interp(t, [0.0, 0.1], [1600.0, 1800.0])
    v9 = np.interp(t, [0.0, 0.1], [2400.0, 2200.0])
    for icmp in (0, 3, 9):
        w = icmp / 9.0
        expected = (1 - w) * (v0 + 120.0) + w * (v9 + 180.0)
        npt.assert_allclose(np.asarray(out[icmp]), expected, rtol=2e-3)
    assert [o.header['ensemble_number'] for o in out] == list(range(n_cdp))
    assert out[0].d_sample == DT and out[0].header['offset'] == 0.0


def test_interpfowler_cvs_panels_and_errors():
    nvstack, n_cdp, nt = 11, 2, 20
    # panels of constant 1 / v^2, from 0 in steps of 1 / (vminstack^2 (nvstack - 5))
    vminstack = 1500.0
    dv = 1.0 / (vminstack ** 2 * (nvstack - 5))
    traces = []
    for _ in range(n_cdp):
        for k in range(nvstack):
            traces.append(Trace(np.full(nt, k * dv, dtype=np.float32), d_sample=DT))
    out = run(interpfowler(cdps=[0], times=[[0.0]], velocities=[[1500.0]], nvstack=nvstack, vminstack=vminstack,
                           ncdps=n_cdp, nv=1),
              traces)
    # (v = 1500: 1 / v^2 = 6 dv, which is on the panel with the index 6)
    npt.assert_allclose(np.asarray(out[0]), 6 * dv, rtol=1e-3)
    with pytest.raises(ValueError, match="fewer"):
        run(interpfowler(cdps=[0], times=[[0.0]], velocities=[[1500.0]], etas=[[0.1]], nv=2, neta=2, ncdps=3),
            traces[:5])
    with pytest.raises(ValueError, match="etas"):
        interpfowler(cdps=[0], times=[[0.0]], velocities=[[1500.0]])
    with pytest.raises(ValueError):
        interpfowler(cdps=[0, 1], times=[[0.0]], velocities=[[1500.0]], etas=[[0.1]])


# ------------------------------------------------------------------------------------------------------- ocext
def offset_gather(offset=1000.0, n_x=40, nt=200, dip=0.0, t0=0.5):
    t = np.arange(nt) * DT
    traces = []
    for ix in range(n_x):
        x = ricker(t - t0 - dip * DT * ix, 20.0)
        traces.append(Trace(x.astype(np.float32), d_sample=DT).replace(offset=float(offset), ensemble_number=ix + 1))
    return traces


def test_ocext_to_the_same_offset_leaves_the_data_and_sets_the_offset():
    traces = offset_gather(dip=3.0)
    out = run(ocext(cdpmin=1, cdpmax=40, dxcdp=25.0, noffmix=1, offextr=1000.0), traces)
    assert len(out) == 40
    a = np.array([np.asarray(t) for t in traces])
    b = np.array([np.asarray(t) for t in out])
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.98
    assert all(t.header['offset'] == 1000.0 for t in out)


def test_ocext_moves_a_dipping_event_and_keeps_a_flat_one():
    flat = offset_gather(dip=0.0)
    out = run(ocext(cdpmin=1, cdpmax=40, dxcdp=25.0, noffmix=1, offextr=200.0, sdmo=1.0), flat)
    a = np.array([np.asarray(t) for t in flat])
    b = np.array([np.asarray(t) for t in out])
    # a flat event is not moved by moveout corrections (the middle of the line, away from the ends)
    assert abs(int(np.argmax(b[20])) - int(np.argmax(a[20]))) <= 1
    assert all(t.header['offset'] == 200.0 for t in out)
    # a dipping one is changed by it
    dip = offset_gather(dip=3.0)
    moved = np.array([np.asarray(t) for t in run(ocext(cdpmin=1, cdpmax=40, dxcdp=25.0, noffmix=1, offextr=0.0), dip)])
    original = np.array([np.asarray(t) for t in dip])
    assert np.abs(moved - original).max() > 0.1 * np.abs(original).max()
    with pytest.raises(TypeError):
        ocext(cdpmin=1, cdpmax=2, dxcdp=25.0, noffmix=1)
