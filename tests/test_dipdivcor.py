"""
SUDIPDIVCOR: dip-dependent divergence correction, by the library version of the program (the table of corrections, and the dip
filter of a wavenumber) with numpy's FFT in x.
"""
import math

import numpy as np
import numpy.testing as npt
import pytest

from seispy.amplitudes import _dipdivcor as _c
from seispy.amplitudes import dipdivcor
from seispy.container import Trace, from_iterable

DT = 0.004
NT = 120


def traces(data, dt=DT):
    return [Trace(row.astype(np.float32), d_sample=dt) for row in data]


def run(stage, data, dt=DT):
    return np.array([np.asarray(t) for t in from_iterable(traces(data, dt)) | stage])


def table(n_slopes=20, v=2000.0, nt=NT, trans=False, norm=True):
    return _c.divcor_table(n_slopes, DT, np.full(nt, v, dtype=np.float32), trans, norm)


# ---------------------------------------------------------------------------------------------------------- the table
def test_zero_slope_correction_grows_with_time_for_a_constant_velocity():
    divcor, _ = table()
    assert divcor.shape == (20, NT)
    # sigma = v^2 t and q = v t: the correction is sqrt(2) v t, normalized by the first sample
    ratio = divcor[0, 1:] / divcor[0, 1]
    npt.assert_allclose(ratio, np.arange(1, NT), rtol=1e-3)
    assert divcor[0, 0] == 0.0


def test_the_table_is_normalized_by_the_zero_slope_correction():
    divcor, _ = table(norm=True)
    assert divcor[0, 1] == pytest.approx(1.0)
    raw, _ = table(norm=False)
    npt.assert_allclose(divcor, raw / raw[0, 1], rtol=1e-5)


def test_the_correction_depends_on_the_slope_only_if_the_velocity_changes():
    divcor, _ = table()
    npt.assert_allclose(divcor, np.tile(divcor[:1], (20, 1)), rtol=1e-4)  # (a constant velocity: rays are straight)
    v = np.linspace(1500.0, 3500.0, NT).astype(np.float32)
    varying, _ = _c.divcor_table(20, DT, v, False, True)
    assert not np.allclose(varying, np.tile(varying[:1], (20, 1)), rtol=1e-2)


def test_transmission_changes_the_table_only_when_the_velocity_changes():
    constant, _ = table(trans=True)
    plain, _ = table(trans=False)
    npt.assert_allclose(constant, plain, rtol=1e-5)
    v = np.linspace(1500.0, 3000.0, NT).astype(np.float32)
    with_trans, _ = _c.divcor_table(10, DT, v, True, False)
    without, _ = _c.divcor_table(10, DT, v, False, False)
    assert not np.allclose(with_trans, without)


def test_the_first_velocity_comes_back_scaled():
    _, v0 = table(v=2000.0)
    assert v0 == pytest.approx(2000.0 * 0.0005)


# ------------------------------------------------------------------------------------------------------- the dip filter
def reference_dip_filter(p, k, dpx, dt, n_slopes, nw, divcor):
    """dipfilt of sudipdivcor.c, with numpy's FFT"""
    nt = p.shape[0]
    dw = 2.0 * np.pi / (nw * dt)
    wny = np.pi / dt
    pmin = k / wny
    qq = np.zeros(nw, dtype=np.complex128)
    for ip in range(n_slopes - 1, -1, -1):
        ph, pm, pl = dpx * (ip + 1), dpx * ip, dpx * (ip - 1)
        iwl = int(k / ph / dw + 1)
        iwh = int(k / pl / dw) if pl > 0 else 0
        if pl < 1.01 * pmin:
            iwh = (nw - 1) // 2 if nw % 2 else nw // 2 - 1
        if pm < 1.01 * pmin:
            iwh = nw // 2 if nw % 2 else nw // 2 - 1
        if iwh >= iwl:
            kq = np.zeros(nw, dtype=np.complex128)
            kq[:nt] = p * divcor[ip]
            kq = np.fft.ifft(kq) * nw  # (pfacc(1, ...): positive exponent)
            qq[iwl:iwh + 1] += kq[iwl:iwh + 1] * 0.5
            lo, hi = nw - iwh, nw - iwl
            qq[lo:hi + 1] += kq[lo:hi + 1] * 0.5
        if pm < 1.01 * pmin:
            break
    return (np.fft.fft(qq) / nw)[:nt]


def test_the_dip_filter_matches_a_numpy_version_of_the_program():
    n_slopes = 15
    divcor, v0 = table(n_slopes)
    rng = np.random.default_rng(5)
    nw, _ = _c.fft_sizes(NT, 8)
    dpx = np.float32(1.0 / (n_slopes - 1) / v0)
    dk = np.float32(0.7)
    ptk = (rng.normal(size=(6, NT)) + 1j * rng.normal(size=(6, NT))).astype(np.complex64)
    expected = ptk.copy()
    k = dk
    for ik in range(1, 6):
        expected[ik] = reference_dip_filter(ptk[ik].astype(np.complex128), float(k), float(dpx), DT, n_slopes, nw, divcor)
        k = np.float32(k + dk)
    got = ptk.copy()
    _c.dip_filter(got, dk, dpx, np.float32(DT), n_slopes, nw, 6, divcor)
    npt.assert_array_equal(got[0], ptk[0])  # (wavenumber 0 is left to the caller)
    npt.assert_allclose(got[1:], expected[1:], rtol=2e-3, atol=2e-3 * np.abs(expected).max())


def test_the_dip_filter_checks_its_arguments():
    divcor, v0 = table(10)
    ptk = np.zeros((4, NT), dtype=np.complex64)
    with pytest.raises(ValueError):
        _c.dip_filter(ptk, 1.0, 0.1, DT, 10, NT - 1, 4, divcor)  # nw < nt
    with pytest.raises(ValueError):
        _c.dip_filter(ptk, 1.0, 0.1, DT, 10, NT, 5, divcor)  # nkmax too high
    with pytest.raises(ValueError):
        _c.dip_filter(ptk, 1.0, 0.1, DT, 9, NT, 4, divcor)  # table of another size


# ------------------------------------------------------------------------------------------------------------ the stage
def test_conventional_correction_multiplies_by_the_zero_slope_table():
    data = np.random.default_rng(2).normal(size=(5, NT)).astype(np.float32)
    got = run(dipdivcor(25.0, vmig=2000.0, np=20, conv=True), data)
    divcor, _ = table(20)
    npt.assert_allclose(got, data * divcor[0], rtol=1e-5)
    assert not got[:, 0].any()


def test_flat_events_get_the_conventional_correction():
    # traces that are all the same are wavenumber 0 only (when they fill the fft in x, 32 traces do)
    assert _c.fft_sizes(NT, 32)[1] == 32
    row = np.random.default_rng(3).normal(size=NT).astype(np.float32)
    data = np.tile(row, (32, 1))
    got = run(dipdivcor(25.0, vmig=2000.0, np=20), data)
    divcor, _ = table(20)
    npt.assert_allclose(got, data * divcor[0], rtol=1e-4, atol=1e-4)


def test_dipping_events_are_corrected_differently_from_flat_ones():
    nx = 32
    rng = np.random.default_rng(4)
    data = np.zeros((nx, NT), dtype=np.float32)
    for ix in range(nx):  # a dipping event: a wavelet that moves 2 samples per trace
        data[ix, 20 + 2 * ix:24 + 2 * ix] = np.array([1, -2, 2, -1], dtype=np.float32) if 24 + 2 * ix <= NT else 0
    data += rng.normal(scale=0.01, size=data.shape).astype(np.float32)
    dip = run(dipdivcor(25.0, vmig=2000.0, np=30), data)
    flat = run(dipdivcor(25.0, vmig=2000.0, np=30, conv=True), data)
    assert dip.shape == data.shape and np.isfinite(dip).all()
    assert not np.allclose(dip, flat, atol=1e-3)


def test_normalization_is_a_scale_factor_without_dips():
    row = np.random.default_rng(6).normal(size=NT).astype(np.float32)
    data = np.tile(row, (32, 1))
    on = run(dipdivcor(25.0, np=10, conv=True), data)
    off = run(dipdivcor(25.0, np=10, conv=True, norm=False), data)
    divcor_raw, _ = table(10, v=1500.0, norm=False)
    npt.assert_allclose(off, data * divcor_raw[0], rtol=1e-5)
    assert not np.allclose(on, off)


def test_velocity_can_be_a_function_of_time_or_given_per_sample():
    data = np.random.default_rng(7).normal(size=(4, NT)).astype(np.float32)
    times = np.arange(NT) * DT
    per_sample = np.interp(times, [0.0, 0.4], [1500.0, 3000.0])
    a = run(dipdivcor(25.0, np=10, conv=True, tmig=[0.0, 0.4], vmig=[1500.0, 3000.0]), data)
    b = run(dipdivcor(25.0, np=10, conv=True, vt=per_sample), data)
    npt.assert_allclose(a, b, rtol=1e-4, atol=1e-6)


def test_parameters_are_checked():
    with pytest.raises(ValueError):
        dipdivcor(0.0)
    with pytest.raises(ValueError):
        dipdivcor(25.0, np=1)
    with pytest.raises(ValueError):
        dipdivcor(25.0, tmig=[0.0, 0.0], vmig=[1500.0, 2000.0])
    with pytest.raises(ValueError):
        dipdivcor(25.0, tmig=[0.0, 1.0], vmig=[1500.0])
    with pytest.raises(ValueError):
        run(dipdivcor(25.0, vt=np.ones(7)), np.ones((3, NT)))
    with pytest.raises(TypeError):
        list(from_iterable([Trace((np.ones(NT) + 1j).astype(np.complex64), d_sample=DT)]) | dipdivcor(25.0))
    assert dipdivcor(25.0).parallelism == 'serial'
    assert list(from_iterable([]) | dipdivcor(25.0)) == []
