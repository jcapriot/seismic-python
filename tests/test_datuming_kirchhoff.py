"""Kirchhoff datuming (SUDATUMK2DR, SUDATUMK2DS, SUKDMDCR, SUKDMDCS)."""
import numpy as np
import pytest

from seispy.container import Trace, from_iterable
from seispy.datuming import datumk2dr, datumk2ds, kdmdcr, kdmdcs

V = 2000.0
DT = 0.004
NT = 300
F = 20.0
# the tables: x 0..1000 by 25, z 0..400 by 25, the sources at 0..1000 by 25
FXT, DXT, NXT = 0.0, 25.0, 41
FZT, DZT, NZT = 0.0, 25.0, 17
FS, DS, NS = 0.0, 25.0, 41


def ricker(t):
    a = (np.pi * F * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def tables(depth_of_source=0.0):
    xs = FS + DS * np.arange(NS)
    x = FXT + DXT * np.arange(NXT)
    z = FZT + DZT * np.arange(NZT)
    return (np.sqrt((x[None, None, :] - xs[:, None, None]) ** 2 + (z[None, :, None] - depth_of_source) ** 2) / V).astype(
        np.float32)


def envelope_peak(trace):
    x = np.asarray(trace, dtype=float)
    spec = np.fft.fft(x)
    h = np.zeros(len(x))
    h[0] = 1
    h[1:len(x) // 2] = 2
    h[len(x) // 2] = 1
    env = np.abs(np.fft.ifft(spec * h))
    return int(np.argmax(env)) * DT, env


def diffractor_gather(sx, gxs, xd=250.0, zd=400.0, at=0.0):
    """Shot gather over a point diffractor, for receivers on the plane z=at"""
    out = []
    t = np.arange(NT) * DT
    r_s = np.hypot(sx - xd, zd)
    for gx in gxs:
        tt = (r_s + np.hypot(gx - xd, zd - at)) / V
        out.append(Trace(ricker(t - tt).astype(np.float32), d_sample=DT).replace(
            tx_loc=[sx, 0.0, 0.0], rx_loc=[gx, 0.0, 0.0], ensemble_number=1))
    return out


def run(stage, traces):
    return list(from_iterable(list(traces)) | stage)


COMMON = dict(fxt=FXT, dxt=DXT, fzt=FZT, dzt=DZT, fs=FS, ds=DS)


def test_datumk2dr_moves_the_receivers_to_the_datum():
    zdat = 100.0
    gxs = np.arange(0.0, 525.0, 25.0)
    shot = diffractor_gather(0.0, gxs)
    out = run(datumk2dr(tables(), fxi=0.0, dxi=25.0, nxi=41, sgn=1, surf=[[(0, 0), (99999, 0)], [(0, zdat), (99999, zdat)]],
                        nxso=1, fxso=0.0, dxso=25.0, nxgo=21, fxgo=0.0, dxgo=25.0, fzo=zdat - 50.0, nzo=11, dzo=10.0,
                        aperx=500.0, **COMMON), shot)
    assert len(out) == len(shot)
    # the header of the input trace goes with the trace
    assert out[3].header['rx_loc'][0] == 75.0
    r_s = np.hypot(0.0 - 250.0, 400.0)
    for ix in (6, 8, 10, 12):
        x = 25.0 * ix
        expected = (r_s + np.hypot(x - 250.0, 400.0 - zdat)) / V
        peak, env = envelope_peak(out[ix])
        assert abs(peak - expected) <= 2 * DT
        assert env.max() > 5 * np.median(env)


def test_datumk2dr_flat_surfaces_changes_nothing_but_the_filter():
    # (the recording surface and the datum coincide: the time shift is zero, the event stays where it is)
    gxs = np.arange(0.0, 525.0, 25.0)
    shot = diffractor_gather(0.0, gxs)
    out = run(datumk2dr(tables(), fxi=0.0, dxi=25.0, nxi=41, sgn=1, surf=[[(0, 10), (99999, 10)], [(0, 10), (99999, 10)]],
                        nxso=1, fxso=0.0, dxso=25.0, nxgo=21, fxgo=0.0, dxgo=25.0, fzo=10.0, nzo=5, dzo=10.0,
                        aperx=500.0, **COMMON), shot)
    r_s = np.hypot(250.0, 400.0)
    expected = (r_s + np.hypot(250.0 - 250.0, 400.0)) / V
    peak, _ = envelope_peak(out[10])
    assert abs(peak - expected) <= 3 * DT


def test_datumk2ds_moves_the_sources():
    # a receiver gather (one receiver, the sources along the line), the sources go down to the datum
    zdat = 100.0
    sxs = np.arange(0.0, 525.0, 25.0)
    gx = 600.0
    t = np.arange(NT) * DT
    traces = []
    for sx in sxs:
        tt = (np.hypot(sx - 250.0, 400.0) + np.hypot(gx - 250.0, 400.0)) / V
        traces.append(Trace(ricker(t - tt).astype(np.float32), d_sample=DT).replace(
            tx_loc=[float(sx), 0.0, 0.0], rx_loc=[gx, 0.0, 0.0], ensemble_number=1))
    out = run(datumk2ds(tables(), fxi=0.0, dxi=25.0, nxi=41, sgn=1,
                        surf=[[(0, 0), (99999, 0)], [(0, zdat), (99999, zdat)]], nxso=21, fxso=0.0, dxso=25.0, nxgo=1,
                        fxgo=gx, dxgo=25.0, exgo=gx, fzo=zdat - 50.0, nzo=11, dzo=10.0, aperx=500.0, **COMMON), traces)
    assert len(out) == len(traces)
    for ix in (6, 8, 10):
        x = 25.0 * ix
        expected = (np.hypot(x - 250.0, 400.0 - zdat) + np.hypot(gx - 250.0, 400.0)) / V
        peak, env = envelope_peak(out[ix])
        assert abs(peak - expected) <= 2 * DT


def test_kdmdcr_datums_the_receivers():
    zdat = 100.0
    gxs = np.arange(0.0, 525.0, 25.0)
    shot = diffractor_gather(0.0, gxs)
    out = run(kdmdcr(tables(), nxso=1, dxso=25.0, nxgo=21, dxgo=25.0, zrec=0.0, zdat=zdat, aperx=500.0, **COMMON), shot)
    assert len(out) == len(shot)
    r_s = np.hypot(250.0, 400.0)
    # (the receivers of the first output are those from the first receiver after the shot)
    for ix in (8, 10, 12):
        x = 25.0 * ix
        expected = (r_s + np.hypot(x - 250.0, 400.0 - zdat)) / V
        peak, env = envelope_peak(out[ix])
        assert abs(peak - expected) <= 3 * DT


def test_kdmdcs_runs_for_receivers_on_the_surface_and_on_the_datum():
    sxs = np.arange(0.0, 300.0, 25.0)
    t = np.arange(NT) * DT
    traces = [Trace(ricker(t - 0.3 - 0.0004 * ix).astype(np.float32), d_sample=DT).replace(
        tx_loc=[float(sx), 0.0, 0.0], rx_loc=[600.0, 0.0, 0.0]) for ix, sx in enumerate(sxs)]
    kwargs = dict(nxso=12, dxso=25.0, nxgo=30, dxgo=25.0, fxgo=0.0, zrec=0.0, zdat=100.0, aperx=300.0, **COMMON)
    for dc in (0, 1):
        out = run(kdmdcs(tables(), dc=dc, **kwargs), traces)
        assert len(out) == len(traces)
        assert all(np.isfinite(np.asarray(o)).all() for o in out)


def test_errors():
    shot = diffractor_gather(0.0, np.arange(0.0, 125.0, 25.0))
    common = dict(fxi=0.0, dxi=25.0, nxi=41, sgn=1, nxso=1, fxso=0.0, dxso=25.0, nxgo=5, fxgo=0.0, dxgo=25.0, **COMMON)
    with pytest.raises(ValueError):  # the tables are out of the range of the output
        run(datumk2dr(tables(), **{**common, 'fxgo': -100.0}), shot)
    with pytest.raises(ValueError):  # a shot that is not of the output gathers
        run(datumk2dr(tables(), **{**common, 'nxso': 1, 'fxso': 500.0}), shot)
    with pytest.raises(ValueError):  # the surface does not cover the locations
        run(datumk2dr(tables(), **{**common, 'surf': [[(0, 0), (100, 0)], [(0, 50), (100, 50)]]}), shot)
    with pytest.raises(ValueError):
        run(datumk2dr(tables(), **{**common, 'sgn': 2}), shot)
    with pytest.raises(ValueError):
        run(kdmdcr(tables(), nxso=1, dxso=25.0, nxgo=5, dxgo=25.0, zdat=50.0, **COMMON), shot)  # no zrec
    with pytest.raises(ValueError):
        run(kdmdcr(tables(), nxso=2, dxso=50.0, nxgo=5, dxgo=25.0, zrec=0.0, zdat=50.0, **COMMON), shot)
