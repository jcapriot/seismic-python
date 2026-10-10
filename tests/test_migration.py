"""
Migration: SUSTOLT, SUMIGFD, SUMIGFFD, SUMIGPS and SUMIGPSPI.
"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.migration import migfd, migffd, migps, migpspi, stolt

NX, NT, DT, DX, V = 101, 250, 0.004, 10.0, 2000.0
X0, Z0 = 50, 500.0  # a point diffractor at the trace 50, 500 m deep


def diffraction(offset=0.0, cdp_start=1):
    """The zero-offset response of a point diffractor in a constant velocity, as spikes (two-way times t = 2 r / v)"""
    traces = []
    for ix in range(NX):
        data = np.zeros(NT, dtype=np.float32)
        t = 2.0 * np.hypot((ix - X0) * DX, Z0) / V
        it = int(round(t / DT))
        if it < NT:
            data[it] = 1.0
        x = ix * DX
        traces.append(Trace(data, d_sample=DT).replace(tx_loc=[x - offset / 2, 0.0, 0.0], rx_loc=[x + offset / 2, 0.0, 0.0],
                                                       ensemble_number=cdp_start + ix))
    return traces


def focus(traces):
    data = np.array([np.asarray(t) for t in traces])
    ix, it = np.unravel_index(np.abs(data).argmax(), data.shape)
    return ix, it


def test_stolt_focuses_a_diffraction():
    out = list(from_iterable(diffraction()) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX, vmig=V))
    assert len(out) == NX and out[0].n_sample == NT
    ix, it = focus(out)
    assert abs(ix - X0) <= 1 and abs(it - int(round(2 * Z0 / V / DT))) <= 3
    assert [t.header['ensemble_number'] for t in out[:3]] == [1, 2, 3]
    assert out[0].header['d_sample'] == pytest.approx(DT)


def test_stolt_mixes_offsets_and_drops_traces_outside_of_the_cdps():
    # two offsets of the same data, mixed: the sum of the migrated gathers on every trace of the mix (and the traces are
    # there for each offset)
    gathers = diffraction(offset=0.0) + diffraction(offset=100.0)
    single = list(from_iterable(diffraction()) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX, vmig=V))
    mixed = list(from_iterable(gathers) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX, vmig=V, noffmix=2))
    assert len(mixed) == 2 * NX
    npt.assert_allclose(np.asarray(mixed[10]), np.asarray(mixed[NX + 10]))
    unmixed = list(from_iterable(gathers) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX, vmig=V, noffmix=1))
    assert len(unmixed) == 2 * NX
    # (a cdp outside of the range is dropped)
    limited = list(from_iterable(diffraction()) | stolt(cdpmin=11, cdpmax=90, dxcdp=DX, vmig=V))
    assert len(limited) == 80 and limited[0].header['ensemble_number'] == 11
    assert np.abs(np.asarray(single[50])).max() > 0


def test_stolt_with_a_velocity_function_and_the_checks():
    out = list(from_iterable(diffraction()) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX, tmig=[0.0, 0.5], vmig=[1900.0, 2100.0],
                                                    lstaper=5, lbtaper=10, smig=0.8))
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="cdpmin"):
        stolt(cdpmin=5, cdpmax=1, dxcdp=DX)
    with pytest.raises(ValueError, match="monotonically"):
        stolt(cdpmin=1, cdpmax=3, dxcdp=DX, tmig=[0.5, 0.2], vmig=[1.0, 2.0])
    with pytest.raises(ValueError, match="equal"):
        stolt(cdpmin=1, cdpmax=3, dxcdp=DX, tmig=[0.0, 0.2], vmig=[1.0])
    shifted = [t.replace(np.asarray(t), sample_start=0.1) for t in diffraction()]
    with pytest.raises(ValueError, match="non-zero"):
        list(from_iterable(shifted) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX))


NZ, DZ = 100, 10.0
VEL = np.full((NZ, NX), V, dtype=np.float32)


@pytest.mark.parametrize('migrate, kwargs', [(migfd, dict(dip=90)), (migfd, dict(dip=65)), (migffd, {}), (migpspi, {})])
def test_depth_migrations_focus_a_diffraction(migrate, kwargs):
    out = list(from_iterable(diffraction()) | migrate(vel=VEL, nz=NZ, dz=DZ, dx=DX, **kwargs))
    assert len(out) == NX and out[0].n_sample == NZ
    assert out[0].header['d_sample'] == pytest.approx(DZ) and out[0].header['sampling_unit'] in (2, 'm', 'meters')
    ix, iz = focus(out)
    assert abs(ix - X0) <= 2 and abs(iz - int(Z0 / DZ)) <= 4


def test_migps_focuses_a_diffraction_and_uses_the_options():
    out = list(from_iterable(diffraction()) | migps(vmig=V, dx=DX))
    ix, it = focus(out)
    assert abs(ix - X0) <= 1 and abs(it - int(round(2 * Z0 / V / DT))) <= 4
    both = list(from_iterable(diffraction()) | migps(tmig=[0.0, 0.5], vmig=[2000.0, 2000.0], dx=DX, ntflag=3, ltaper=4, nxpad=20,
                                                      np_slopes=100, ffil=[0.0, 5.0, 60.0, 90.0]))
    assert np.isfinite(np.array([np.asarray(t) for t in both])).all()
    given = list(from_iterable(diffraction()) | migps(vt=np.full(NT, V), dx=DX))
    npt.assert_allclose(np.asarray(given[50]), np.asarray(out[50]))
    # the midpoint spacing from the positions of the traces
    from_positions = list(from_iterable(diffraction()) | migps(vmig=V))
    npt.assert_allclose(np.asarray(from_positions[50]), np.asarray(out[50]))


def test_migration_checks_its_input():
    with pytest.raises(ValueError, match="shape"):
        list(from_iterable(diffraction()) | migfd(vel=np.ones((3, 3)), nz=NZ, dz=DZ))
    with pytest.raises(ValueError, match="nz"):
        migfd(vel=VEL, nz=0, dz=DZ)
    with pytest.raises(ValueError, match="ntflag"):
        migps(ntflag=7)
    with pytest.raises(ValueError, match="4 values"):
        migps(ffil=[1.0, 2.0])
    assert list(from_iterable([]) | migpspi(vel=VEL, nz=NZ, dz=DZ)) == []


# --------------------------------------------------------------------------------------------------------- kdmig2d, ktmig2d
from seispy.migration import kdmig2d, ktmig2d  # noqa: E402

KD_FXT, KD_DXT, KD_NXT = 0.0, 20.0, 51
KD_FZT, KD_DZT, KD_NZT = 0.0, 20.0, 41
KD_FS, KD_DS, KD_NS = 0.0, 100.0, 11
_xt = KD_FXT + KD_DXT * np.arange(KD_NXT)
_zt = KD_FZT + KD_DZT * np.arange(KD_NZT)
_src = KD_FS + KD_DS * np.arange(KD_NS)
KD_TTAB = (np.hypot(_xt[None, None, :] - _src[:, None, None], _zt[None, :, None]) / V).astype(np.float32)  # (ns, nzt, nxt)
KD_KW = dict(ttab=KD_TTAB, fxt=KD_FXT, dxt=KD_DXT, fzt=KD_FZT, dzt=KD_DZT, fs=KD_FS, ds=KD_DS, dxm=10.0, v0=V, angmax=70.0)


def zero_offset_diffractor(x0=500.0, z0=400.0, nt=400, dt=0.004):
    traces = []
    for xm in np.arange(100.0, 901.0, 10.0):
        data = np.zeros(nt, dtype=np.float32)
        t = 2.0 * np.hypot(xm - x0, z0) / V
        it = int(round(t / dt))
        if it < nt - 1:
            data[it] = 1.0
        traces.append(Trace(data, d_sample=dt).replace(tx_loc=[xm, 0.0, 0.0], rx_loc=[xm, 0.0, 0.0]))
    return traces


def test_kdmig2d_focuses_a_diffraction():
    out = list(from_iterable(zero_offset_diffractor()) | kdmig2d(**KD_KW))
    assert len(out) == 101 and out[0].n_sample == 201  # (2 times finer in x, 5 times in z than the tables)
    ix, iz = focus(out)
    assert abs(ix - 50) <= 2 and abs(iz - 100) <= 6
    assert out[0].header['d_sample'] == pytest.approx(4.0) and out[10].header['ensemble_number'] == 11
    assert out[10].header['tx_loc'][0] == pytest.approx(100.0)


def test_kdmig2d_offsets_and_the_velocity_analysis_output():
    traces = []
    for off in (0.0, 100.0):
        for xm in np.arange(200.0, 801.0, 20.0):
            data = np.zeros(400, dtype=np.float32)
            r = np.hypot(xm - off / 2 - 500.0, 400.0) + np.hypot(xm + off / 2 - 500.0, 400.0)
            data[int(round(r / V / 0.004))] = 1.0
            traces.append(Trace(data, d_sample=0.004).replace(tx_loc=[xm - off / 2, 0.0, 0.0], rx_loc=[xm + off / 2, 0.0, 0.0]))
    out = list(from_iterable(traces) | kdmig2d(noff=2, off0=0.0, doff=100.0, **KD_KW))
    assert len(out) == 2 * 101
    assert [t.header['ensemble_trace_number'] for t in out[:4]] == [1, 2, 1, 2]
    ix, iz = focus(out[0::2])
    assert abs(ix - 50) <= 2 and abs(iz - 100) <= 6
    ix, iz = focus(out[1::2])
    assert abs(ix - 50) <= 2 and abs(iz - 100) <= 6
    # the velocity analysis tables
    tv = np.zeros_like(KD_TTAB) + 0.001
    cs = np.ones_like(KD_TTAB)
    both = list(from_iterable(traces) | kdmig2d(noff=2, doff=100.0, tv=tv, cs=cs, output='both', **KD_KW))
    assert len(both) == 2 * len(out)
    npt.assert_allclose(np.asarray(both[0]), np.asarray(out[0]))
    assert np.abs(np.asarray(both[1])).max() > 0
    limited = list(from_iterable(traces) | kdmig2d(noff=1, off0=0.0, doff=50.0, limoff=True, **KD_KW))
    assert len(limited) == 101


def test_kdmig2d_checks_its_input():
    with pytest.raises(ValueError, match="3-D"):
        kdmig2d(**{**KD_KW, 'ttab': KD_TTAB[0]})
    with pytest.raises(ValueError, match="out of the traveltime table"):
        kdmig2d(**{**KD_KW, 'fxo': -100.0})
    with pytest.raises(ValueError, match="velocity analysis"):
        kdmig2d(output='velan', **KD_KW)
    with pytest.raises(ValueError, match="dxm"):
        list(from_iterable(zero_offset_diffractor()) | kdmig2d(**{k: v for k, v in KD_KW.items() if k != 'dxm'}))
    with pytest.raises(ValueError, match="output"):
        kdmig2d(output='x', **KD_KW)


def test_ktmig2d_focuses_a_diffraction_and_uses_the_cdps():
    traces = diffraction()  # (the ensemble_number is the cdp)
    out = list(from_iterable(traces) | ktmig2d(dx=DX, vel=np.full(NT, V), angmax=50.0))
    assert len(out) == NX
    ix, it = focus(out)
    assert abs(ix - X0) <= 1 and abs(it - int(round(2 * Z0 / V / DT))) <= 4
    table = np.full((NX + 5, NT), V)
    again = list(from_iterable(traces) | ktmig2d(dx=DX, vel=table, angmax=50.0, firstcdp=1))
    npt.assert_allclose(np.asarray(again[50]), np.asarray(out[50]))
    with pytest.raises(ValueError, match="cdps"):
        list(from_iterable(traces) | ktmig2d(dx=DX, vel=table, firstcdp=1000))
    with pytest.raises(ValueError, match="80"):
        ktmig2d(dx=DX, vel=np.full(NT, V), angmax=85.0)
    with pytest.raises(ValueError, match="array"):
        list(from_iterable(traces) | ktmig2d(dx=DX, vel=np.zeros((3, 5))))


def test_the_migrations_can_be_run_by_several_threads_at_once():
    import concurrent.futures

    def samples(traces):
        return np.array([np.asarray(t) for t in traces])

    work = {
        'stolt': lambda: samples(from_iterable(diffraction()) | stolt(cdpmin=1, cdpmax=NX, dxcdp=DX, tmig=[0.0, 0.5], vmig=[1900.0, 2100.0])),
        'migfd': lambda: samples(from_iterable(diffraction()) | migfd(vel=VEL, nz=NZ, dz=DZ, dx=DX)),
        'migps': lambda: samples(from_iterable(diffraction()) | migps(vmig=V, dx=DX)),
        'kdmig2d': lambda: samples(from_iterable(zero_offset_diffractor()) | kdmig2d(**KD_KW)),
        'ktmig2d': lambda: samples(from_iterable(diffraction()) | ktmig2d(dx=DX, vel=np.full(NT, V))),
    }
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for name, make in work.items():
            expected = make()
            for got in pool.map(lambda _: make(), range(6)):
                npt.assert_allclose(got, expected, atol=1e-4 * np.abs(expected).max(), err_msg=name)
