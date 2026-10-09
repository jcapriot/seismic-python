"""
SUVELAN and SURELAN: stacking velocity and residual moveout semblance of gathers, by the library versions of the programs.
"""
import math

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.parallel import pmap
from seispy.velocity import relan, velan

NT, DT = 200, 0.004
RNG = np.random.default_rng(3)


def ricker(t, f=25.0):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def gather(data, offsets, cdp=1, dt=DT, **header):
    return [Trace(np.asarray(row, dtype=np.float32), d_sample=dt).replace(offset=float(off), ensemble_number=cdp, **header)
            for row, off in zip(data, offsets)]


def run(stage, traces):
    return list(from_iterable(traces) | stage)


def panel(out):
    return np.array([np.asarray(t) for t in out])


# --------------------------------------------------------------------------------------------- the loops of the programs
def reference_velan(data, offsets, dt, ft, nv, dv, fv, anis1, anis2, smute, ratio, nsmooth, pwr):
    """suvelan.c, with numpy (and the whole smoothing window)"""
    nt = data.shape[1]
    num, den, nnz = (np.zeros((nv, nt)) for _ in range(3))
    for x, off in zip(data.astype(np.float64), offsets):
        offan = off ** 4 * anis1 / (1 + off * off * anis2)
        for iv in range(nv):
            v = fv + iv * dv
            offovs = off * off / (v * v) + offan
            tnmute = math.sqrt(offovs / (smute * smute - 1.0))
            itmute = int((tnmute - ft) / dt) if tnmute > ft else 0
            it = np.arange(itmute, nt)
            tn = ft + it * dt
            ti = (np.sqrt(tn * tn + offovs) - ft) / dt
            iti = ti.astype(int)
            ok = iti < nt - 1
            it, ti, iti = it[ok], ti[ok], iti[ok]
            frac = ti - iti
            temp = (1 - frac) * x[iti] + frac * x[iti + 1]
            nz = temp != 0
            np.add.at(num[iv], it[nz], temp[nz])
            np.add.at(den[iv], it[nz], temp[nz] ** 2)
            np.add.at(nnz[iv], it[nz], 1.0)
    ntout = 1 + (nt - 1) // ratio
    sem = np.zeros((nv, ntout))
    for iv in range(nv):
        for itout in range(ntout):
            it = itout * ratio
            lo, hi = max(it - nsmooth // 2, 0), min(it + nsmooth // 2, nt - 1)
            n = np.sum(num[iv, lo:hi + 1] ** 2)
            d = np.sum(nnz[iv, lo:hi + 1] * den[iv, lo:hi + 1])
            sem[iv, itout] = n / d if d else 0.0
    return sem ** pwr if pwr != 1.0 else sem


def test_velan_matches_the_loops_of_the_program():
    offsets = np.arange(0.0, 800.0, 100.0)
    data = RNG.normal(size=(8, NT))
    kwargs = dict(nv=12, dv=150.0, fv=1500.0, anis1=0.0, anis2=0.0, smute=1.5, dtratio=3, nsmooth=7, pwr=1.0)
    got = panel(run(velan(**kwargs), gather(data, offsets)))
    expected = reference_velan(data.astype(np.float32), offsets, DT, 0.0, 12, 150.0, 1500.0, 0.0, 0.0, 1.5, 3, 7, 1.0)
    assert got.shape == expected.shape == (12, 1 + (NT - 1) // 3)
    npt.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_velan_quartic_term_and_power():
    offsets = np.arange(0.0, 800.0, 100.0)
    data = RNG.normal(size=(8, NT))
    kwargs = dict(nv=6, dv=300.0, fv=1500.0, anis1=1e-13, anis2=1e-7, smute=1.3, dtratio=2, nsmooth=5, pwr=2.0)
    got = panel(run(velan(**kwargs), gather(data, offsets)))
    expected = reference_velan(data.astype(np.float32), offsets, DT, 0.0, 6, 300.0, 1500.0, 1e-13, 1e-7, 1.3, 2, 5, 2.0)
    npt.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


# --------------------------------------------------------------------------------------------------------------- velan
def hyperbolic_gather(velocity=2500.0, t0=0.6, nt=400):
    t = np.arange(nt) * DT
    offsets = np.arange(0.0, 1500.0, 50.0)
    data = [ricker(t - math.sqrt(t0 ** 2 + (off / velocity) ** 2)) for off in offsets]
    return data, offsets


def test_velan_semblance_is_high_at_the_velocity_of_the_event():
    data, offsets = hyperbolic_gather()
    sem = panel(run(velan(nv=40, dv=50.0, fv=1500.0, dtratio=1, nsmooth=5), gather(data, offsets)))
    at_t0 = sem[:, int(round(0.6 / DT))]
    velocities = 1500.0 + 50.0 * np.arange(40)
    assert at_t0[velocities == 2500.0][0] > 0.95
    assert at_t0[velocities == 2200.0][0] < 0.3 and at_t0[velocities == 2800.0][0] < 0.3
    # (near t0 the best velocity is close to the true one)
    window = sem[:, int(0.56 / DT):int(0.64 / DT) + 1]
    assert abs(velocities[np.unravel_index(np.argmax(window), window.shape)[0]] - 2500.0) <= 100.0


def test_velan_makes_a_trace_for_every_velocity_of_every_gather():
    data, offsets = hyperbolic_gather(nt=100)
    traces = gather(data, offsets, cdp=7) + gather(data, offsets, cdp=8)
    out = run(velan(nv=5, dtratio=4), traces)
    assert len(out) == 10
    assert [t.header['ensemble_number'] for t in out] == [7] * 5 + [8] * 5
    assert [t.header['ensemble_trace_number'] for t in out] == [1, 2, 3, 4, 5] * 2
    assert all(t.header['offset'] == 0.0 for t in out)
    assert all(t.d_sample == pytest.approx(4 * DT) and t.n_sample == 1 + 99 // 4 for t in out)


def test_velan_a_gather_is_the_same_alone_or_with_others():
    data, offsets = hyperbolic_gather(nt=100)
    alone = panel(run(velan(nv=5), gather(data, offsets, cdp=1)))
    together = panel(run(velan(nv=5), gather(data, offsets, cdp=1) + gather(data * 2, offsets, cdp=2)))
    npt.assert_array_equal(together[:5], alone)
    # (the semblance does not depend on the scale of the data)
    npt.assert_allclose(together[5:], alone, rtol=1e-4)


def test_velan_can_be_split_across_workers_by_gather():
    data, offsets = hyperbolic_gather(nt=100)
    traces = sum((gather(data, offsets, cdp=c) for c in range(1, 7)), [])
    sequential = panel(run(velan(nv=8), traces))
    parallel = panel(from_iterable(traces) | pmap(velan(nv=8), workers=3, chunk=14, key='ensemble_number'))
    npt.assert_array_equal(parallel, sequential)


def test_velan_checks_its_parameters_and_input():
    for bad in (dict(nv=0), dict(smute=1.0), dict(dtratio=0), dict(pwr=0.0), dict(pwr=-1.0)):
        with pytest.raises(ValueError):
            velan(**bad)
    data, offsets = hyperbolic_gather(nt=50)
    with pytest.raises(ValueError, match="anis2"):
        run(velan(anis2=-1.0), gather(data, [1000.0] * len(data)))
    with pytest.warns(UserWarning, match="moveout"):
        run(velan(anis1=-1e-6), gather(data, offsets))
    with pytest.raises(ValueError, match="same number of samples"):
        run(velan(), gather([np.ones(50), np.ones(60)], [0.0, 100.0]))
    with pytest.raises(TypeError):
        run(velan(), [Trace((np.ones(50) + 1j).astype(np.complex64), d_sample=DT)])
    with pytest.raises(ValueError, match="sample interval"):
        run(velan(), gather(data, offsets, dt=0.0))
    assert velan().parallelism == 'ensemble'
    assert run(velan(), []) == []


# --------------------------------------------------------------------------------------------------------------- relan
DZ, NZ = 10.0, 200


def reference_relan(data, offsets, dz, fz, nr, dr, fr, smute, ratio, nsmooth):
    """surelan.c, with numpy"""
    nz = data.shape[1]
    num, den, nnz = (np.zeros((nr, nz)) for _ in range(3))
    for x, h in zip(data.astype(np.float64), offsets):
        for ir in range(nr):
            r = fr + ir * dr
            roffs2 = r * h * h
            znmute = roffs2 / (smute * smute - 1.0)
            znmute = math.sqrt(znmute) if znmute > 0 else math.sqrt(-znmute)
            izmute = max(int((znmute - fz) / dz), 0)
            iz = np.arange(izmute, nz)
            zn = fz + iz * dz
            temp = zn * zn + roffs2
            zi = np.where(temp > fz * fz, (np.sqrt(np.maximum(temp, 0)) - fz) / dz, 0.0)
            izi = zi.astype(int)
            ok = izi < nz - 1
            iz, zi, izi = iz[ok], zi[ok], izi[ok]
            frac = zi - izi
            val = (1 - frac) * x[izi] + frac * x[izi + 1]
            nz_ = val != 0
            np.add.at(num[ir], iz[nz_], val[nz_])
            np.add.at(den[ir], iz[nz_], val[nz_] ** 2)
            np.add.at(nnz[ir], iz[nz_], 1.0)
    nzout = 1 + (nz - 1) // ratio
    sem = np.zeros((nr, nzout))
    for ir in range(nr):
        for izout in range(nzout):
            i = izout * ratio
            lo, hi = max(i - nsmooth // 2, 0), min(i + nsmooth // 2, nz - 1)
            n = np.sum(num[ir, lo:hi + 1] ** 2)
            d = np.sum(nnz[ir, lo:hi + 1] * den[ir, lo:hi + 1])
            sem[ir, izout] = n / d if d else 0.0
    return sem


def test_relan_matches_the_loops_of_the_program():
    # (offsets and r that do not put the mute exactly on a sample, where float32 and float64 round differently)
    offsets = np.arange(0.0, 800.0, 100.0) + 13.0
    data = RNG.normal(size=(8, NZ))
    traces = gather(data, offsets, dt=DZ)
    got = panel(run(relan(nr=9, dr=0.047, fr=-0.213, dzratio=4, nsmooth=5), traces))
    expected = reference_relan(data.astype(np.float32), offsets, DZ, 0.0, 9, 0.047, -0.213, 1.5, 4, 5)
    assert got.shape == expected.shape == (9, 1 + (NZ - 1) // 4)
    npt.assert_allclose(got, expected, rtol=2e-3, atol=2e-3)


def test_relan_finds_the_residual_moveout():
    # an event whose depth is z(h)^2 = z0^2 + r h^2 on migrated gathers: the semblance is high for that r
    r0, z0 = 0.05, 600.0
    offsets = np.arange(0.0, 400.0, 20.0)
    z = np.arange(NZ) * DZ
    # (a wavelet a few samples wide: ricker takes a width in 1/z here)
    data = [ricker(z - math.sqrt(z0 ** 2 + r0 * h * h), f=0.02) for h in offsets]
    out = run(relan(nr=21, dr=0.01, fr=-0.05, dzratio=1, nsmooth=5), gather(data, offsets, dt=DZ))
    sem = panel(out)
    at_z0 = sem[:, int(round(z0 / DZ))]
    rs = -0.05 + 0.01 * np.arange(21)
    assert rs[np.argmax(at_z0)] == pytest.approx(r0, abs=0.011)
    assert out[0].d_sample == pytest.approx(DZ) and out[0].header['ensemble_trace_number'] == 1


def test_relan_checks_its_parameters():
    for bad in (dict(nr=0), dict(smute=0.5), dict(dzratio=0)):
        with pytest.raises(ValueError):
            relan(**bad)
    assert relan().parallelism == 'ensemble'
    assert run(relan(), []) == []
