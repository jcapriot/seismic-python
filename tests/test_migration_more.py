"""SUMIGSPLIT, SUGAZMIG, SUMIGTK and SUMIGPSTI."""
import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import from_iterable
from seispy.migration import gazmig, migpsti, migps, migsplit, migtk

from test_migration import DT, DX, NT, NX, NZ, DZ, V, VEL, X0, Z0, diffraction, focus


def test_migsplit_focuses_a_diffraction():
    out = list(from_iterable(diffraction()) | migsplit(vel=VEL, nz=NZ, dz=DZ, dx=DX))
    assert len(out) == NX and out[0].n_sample == NZ
    assert out[0].header['d_sample'] == pytest.approx(DZ)
    ix, iz = focus(out)
    assert abs(ix - X0) <= 2 and abs(iz - int(Z0 / DZ)) <= 4


def test_migsplit_with_a_velocity_that_changes_laterally_and_checks():
    vel = VEL.copy()
    vel[:, NX // 2:] *= 1.02
    out = list(from_iterable(diffraction()) | migsplit(vel=vel, nz=NZ, dz=DZ, dx=DX))
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="shape"):
        list(from_iterable(diffraction()) | migsplit(vel=np.ones((3, 3)), nz=NZ, dz=DZ))
    with pytest.raises(ValueError):
        list(from_iterable(diffraction()) | migsplit(vel=VEL * 0.0, nz=NZ, dz=DZ, dx=DX))


def test_gazmig_focuses_a_diffraction_and_the_options():
    out = list(from_iterable(diffraction()) | gazmig(vmig=V, dx=DX))
    ix, it = focus(out)
    assert abs(ix - X0) <= 1 and abs(it - int(round(2 * Z0 / V / DT))) <= 4
    # a velocity for each sample gives the same migration
    given = list(from_iterable(diffraction()) | gazmig(vt=np.full(NT, V), dx=DX))
    npt.assert_allclose(np.asarray(given[50]), np.asarray(out[50]))
    # a shorter output, from a later first time
    shorter = list(from_iterable(diffraction()) | gazmig(vmig=V, dx=DX, ntau=150, ftau=0.0, dtau=DT))
    assert shorter[0].n_sample == 150
    # attenuation changes the amplitudes (and stays finite)
    attenuated = list(from_iterable(diffraction()) | gazmig(vmig=V, dx=DX, Q=50.0))
    a, b = np.abs(np.asarray(attenuated[50])).max(), np.abs(np.asarray(out[50])).max()
    assert np.isfinite(a) and a != b
    with pytest.raises(ValueError, match="tmig"):
        list(from_iterable(diffraction()) | gazmig(tmig=[0.5, 0.2], vmig=[1.0, 2.0]))
    with pytest.raises(ValueError, match="equal"):
        list(from_iterable(diffraction()) | gazmig(tmig=[0.0, 0.2], vmig=[1.0]))


def test_migtk_focuses_a_diffraction_and_the_options():
    out = list(from_iterable(diffraction()) | migtk(dxcdp=DX, vmig=V))
    assert len(out) == NX and out[0].n_sample == NT
    ix, it = focus(out)
    assert abs(ix - X0) <= 2 and abs(it - int(round(2 * Z0 / V / DT))) <= 8
    padded = list(from_iterable(diffraction()) | migtk(dxcdp=DX, vmig=V, nxpad=20, ltaper=5, fmax=60.0))
    assert np.isfinite(np.array([np.asarray(t) for t in padded])).all()
    with pytest.raises(ValueError):
        migtk(dxcdp=0.0)


def test_migpsti_is_migps_for_an_isotropic_medium():
    iso = list(from_iterable(diffraction()) | migps(vmig=V, dx=DX))
    ti = list(from_iterable(diffraction()) | migpsti(vnmig=V, vmig=V, etamig=0.0, dx=DX))
    ix, it = focus(ti)
    assert abs(ix - X0) <= 1 and abs(it - int(round(2 * Z0 / V / DT))) <= 4
    a = np.array([np.asarray(t) for t in iso])
    b = np.array([np.asarray(t) for t in ti])
    # the same image (the tables are the same for eta=0 and v=vn, up to the rounding of the elastic coefficients)
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.9


def test_migpsti_with_anisotropy_and_options():
    out = list(from_iterable(diffraction()) | migpsti(vnmig=2200.0, vmig=2000.0, etamig=0.1, dx=DX, ntflag=3, ltaper=4,
                                                      nxpad=10, np_slopes=100, ffil=[0.0, 5.0, 60.0, 90.0]))
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="ntflag"):
        migpsti(ntflag=0) and None
        list(from_iterable(diffraction()) | migpsti(ntflag=0))
    with pytest.raises(ValueError, match="4 values"):
        list(from_iterable(diffraction()) | migpsti(ffil=[1.0]))


# ------------------------------------------------------------------------------------------------- prestack shots
from seispy.container import Trace  # noqa: E402
from seispy.migration import migprefd, migpreffd, migprepspi, migpresp  # noqa: E402

PNX, PNZ, PDX, PDZ, PNT, PDT, PV = 60, 50, 10.0, 10.0, 300, 0.004, 2000.0
PX0, PZ0 = 30, 250.0


def shot_gathers(shots=(10, 20, 30, 40, 50), x0=PX0, z0=PZ0):
    """Common-shot spike data of a point diffractor in a constant velocity"""
    out = []
    for isx in shots:
        sx = isx * PDX
        for igx in range(PNX):
            gx = igx * PDX
            t = (np.hypot(sx - x0 * PDX, z0) + np.hypot(gx - x0 * PDX, z0)) / PV
            data = np.zeros(PNT, dtype=np.float32)
            it = int(round(t / PDT))
            if it < PNT:
                data[it] = 1.0
            out.append(Trace(data, d_sample=PDT).replace(tx_loc=[sx, 0.0, 0.0], rx_loc=[gx, 0.0, 0.0]))
    return out


PVEL = np.full((PNZ, PNX), PV, dtype=np.float32)


@pytest.mark.parametrize('migrate, kwargs', [(migprefd, dict(dip=45)), (migpreffd, {}), (migpresp, {}),
                                             (migprepspi, {}), (migprepspi, dict(nflag=1))])
def test_prestack_shot_migrations_focus_a_diffractor(migrate, kwargs):
    out = list(from_iterable(shot_gathers()) | migrate(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, **kwargs))
    assert len(out) == PNX and out[0].n_sample == PNZ
    assert out[0].header['d_sample'] == pytest.approx(PDZ)
    image = np.array([np.asarray(t) for t in out])
    assert np.isfinite(image).all()
    # (the image has the focus at the diffractor, among the strongest values: the sum over the shots is largest there)
    ix, iz = np.unravel_index(np.abs(image[:, 5:]).argmax(), image[:, 5:].shape)
    assert abs(ix - PX0) <= 3 and abs((iz + 5) - int(PZ0 / PDZ)) <= 4


def test_prestack_dips_options_and_checks():
    gathers = shot_gathers(shots=(20, 40))
    for dip in (45, 65, 79, 80, 87, 89, 90):
        out = list(from_iterable(gathers) | migprefd(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, dip=dip, nxshot=1))
        assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    one = list(from_iterable(gathers) | migprefd(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, nxshot=1))
    two = list(from_iterable(gathers) | migprefd(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX))
    assert not np.allclose(np.asarray(one[30]), np.asarray(two[30]))
    narrow = list(from_iterable(gathers) | migpresp(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, lpad=10, rpad=10))
    assert np.isfinite(np.array([np.asarray(t) for t in narrow])).all()
    with pytest.raises(ValueError, match="dip"):
        list(from_iterable(gathers) | migprefd(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, dip=50))
    with pytest.raises(ValueError, match="shape"):
        list(from_iterable(gathers) | migpresp(nxo=PNX, nz=PNZ, dz=PDZ, vel=np.ones((3, 3)), dx=PDX))
    with pytest.raises(ValueError, match="frequencies"):
        list(from_iterable(gathers) | migpresp(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, f2=60.0))
    with pytest.raises(ValueError, match="nflag"):
        list(from_iterable(gathers) | migprepspi(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX, nflag=3))
    backwards = list(reversed(shot_gathers(shots=(20, 40))))
    with pytest.raises(ValueError, match="monotonically"):
        list(from_iterable(backwards) | migpresp(nxo=PNX, nz=PNZ, dz=PDZ, vel=PVEL, dx=PDX))


# ------------------------------------------------------------------------------------------------- migtopo2d
from seispy.migration import migtopo2d  # noqa: E402

TV, TDT, TNT = 2000.0, 0.004, 400
FS0, DS0, NS0 = 0.0, 25.0, 41
FXT0, DXT0, NXT0 = 0.0, 25.0, 41
FZT0, DZT0, NZT0 = 0.0, 25.0, 25


def topo_tables(source_z=None):
    xs = FS0 + DS0 * np.arange(NS0)
    x = FXT0 + DXT0 * np.arange(NXT0)
    z = FZT0 + DZT0 * np.arange(NZT0)
    sz = np.zeros(NS0) if source_z is None else np.asarray(source_z)
    return (np.sqrt((x[None, None, :] - xs[:, None, None]) ** 2 + (z[None, :, None] - sz[:, None, None]) ** 2) / TV).astype(
        np.float32)


def zero_offset_diffractor(x0=500.0, z0=300.0):
    out = []
    t = np.arange(TNT) * TDT
    for ix in range(41):
        x = 25.0 * ix
        tt = 2.0 * np.hypot(x - x0, z0) / TV
        data = np.exp(-((t - tt) * 60.0) ** 2).astype(np.float32)
        out.append(Trace(data, d_sample=TDT).replace(tx_loc=[x, 0.0, 0.0], rx_loc=[x, 0.0, 0.0], ensemble_number=ix + 1))
    return out


def test_migtopo2d_focuses_a_diffraction_from_a_flat_surface():
    out = list(from_iterable(zero_offset_diffractor()) | migtopo2d(
        ttab=topo_tables(), fxt=FXT0, dxt=DXT0, fzt=FZT0, dzt=DZT0, fs=FS0, ds=DS0, fxi=0.0, dxi=25.0, nxi=41, dxm=25.0,
        nxo=41, fxo=0.0, dxo=25.0, nzo=49, fzo=0.0, dzo=12.5, aperx=500.0))
    assert len(out) == 41 and out[0].n_sample == 49
    image = np.array([np.asarray(t) for t in out])
    ix, iz = np.unravel_index(np.abs(image[:, 5:]).argmax(), image[:, 5:].shape)
    assert abs(ix - 20) <= 2 and abs((iz + 5) * 12.5 - 300.0) <= 40.0


def test_migtopo2d_with_a_dipping_surface_and_checks():
    surf = [(0.0, 0.0), (1000.0, 0.0)]
    out = list(from_iterable(zero_offset_diffractor()) | migtopo2d(
        ttab=topo_tables(), fxt=FXT0, dxt=DXT0, fzt=FZT0, dzt=DZT0, fs=FS0, ds=DS0, fxi=0.0, dxi=25.0, nxi=41, dxm=25.0, nxo=21,
        fxo=250.0, dxo=25.0, nzo=25, fzo=0.0, dzo=25.0, aperx=300.0, surf=surf, noff=2, off0=0.0, doff=100.0, ls=False))
    assert len(out) == 21 * 2
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="range"):
        list(from_iterable(zero_offset_diffractor()) | migtopo2d(
            ttab=topo_tables(), fxt=FXT0, dxt=DXT0, fzt=FZT0, dzt=DZT0, fs=FS0, ds=DS0, fxi=0.0, dxi=25.0, nxi=41, dxm=25.0,
            nxo=200))
    with pytest.raises(ValueError, match="dxm"):
        list(from_iterable(zero_offset_diffractor()) | migtopo2d(
            ttab=topo_tables(), fxt=FXT0, dxt=DXT0, fzt=FZT0, dzt=DZT0, fs=FS0, ds=DS0, fxi=0.0, dxi=25.0, nxi=41))
    with pytest.raises(ValueError, match="surf"):
        list(from_iterable(zero_offset_diffractor()) | migtopo2d(
            ttab=topo_tables(), fxt=FXT0, dxt=DXT0, fzt=FZT0, dzt=DZT0, fs=FS0, ds=DS0, fxi=0.0, dxi=25.0, nxi=41, dxm=25.0,
            surf=[(0.0, 0.0)]))


# ------------------------------------------------------------------------------------------------- invvxzco
from seispy.migration import invvxzco  # noqa: E402


def test_invvxzco_focuses_a_diffraction():
    out = list(from_iterable(zero_offset_diffractor()) | invvxzco(
        vel=np.full((41, 41), 2000.0, np.float32), dx=25.0, dz=25.0, dxm=25.0, nxd=5, nxb=20, nxo=41, dxo=25.0, nzo=41,
        dzo=25.0, fxo=0.0, fzo=0.0))
    assert len(out) == 41 and out[0].n_sample == 41
    image = np.array([np.asarray(t) for t in out])
    assert np.isfinite(image).all()
    ix, iz = np.unravel_index(np.abs(image[:, 3:]).argmax(), image[:, 3:].shape)
    assert abs(ix - 20) <= 2 and abs((iz + 3) - 12) <= 3
    # no skipping gives nearly the same image
    exact = list(from_iterable(zero_offset_diffractor()) | invvxzco(
        vel=np.full((41, 41), 2000.0, np.float32), dx=25.0, dz=25.0, dxm=25.0, nxd=1, nxb=20, nxo=41, dxo=25.0, nzo=41,
        dzo=25.0))
    other = np.array([np.asarray(t) for t in exact])
    assert np.corrcoef(image.ravel(), other.ravel())[0, 1] > 0.9


def test_invvxzco_options_and_checks():
    kwargs = dict(vel=np.full((41, 41), 2000.0, np.float32), dx=25.0, dz=25.0, dxm=25.0, nxb=20, nxo=11, dxo=25.0, nzo=11,
                  dzo=25.0, fxo=200.0)
    out = list(from_iterable(zero_offset_diffractor()) | invvxzco(vp=np.full((41, 41), 1e-5, np.float32), ls=True, nxc=3,
                                                                 nzc=3, **kwargs))
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="velocity"):
        list(from_iterable(zero_offset_diffractor()) | invvxzco(vel=np.ones(5)))
    with pytest.raises(ValueError, match="shape"):
        list(from_iterable(zero_offset_diffractor()) | invvxzco(vp=np.ones((2, 2)), **kwargs))
    with pytest.raises(ValueError, match="outside"):
        list(from_iterable(zero_offset_diffractor()) | invvxzco(**{**kwargs, 'fxo': -50.0}))
    with pytest.raises(ValueError, match="band"):
        list(from_iterable(zero_offset_diffractor()[:21]) | invvxzco(offset=200.0, fxm=100.0, **{**kwargs, 'nxb': 3}))


def test_invvxzco_reports_a_raypath_of_too_large_a_curvature():
    rng = np.random.default_rng(3)
    vel = (1500.0 + 1500.0 * rng.random((41, 41))).astype(np.float32)  # (rough: rays turn)
    with pytest.raises(ValueError, match="curvature"):
        list(from_iterable(zero_offset_diffractor()) | invvxzco(vel=vel, dx=25.0, dz=25.0, dxm=25.0, nxb=20, nxo=11, dxo=25.0,
                                                               nzo=11, dzo=25.0))


# ------------------------------------------------------------------------------------------------- miggbzo
from seispy.migration import miggbzo  # noqa: E402


def test_miggbzo_focuses_a_diffraction():
    out = list(from_iterable(zero_offset_diffractor()) | miggbzo(vel=np.full((41, 41), 2000.0, np.float32), dz=25.0, dx=25.0))
    assert len(out) == 41 and out[0].n_sample == 41
    assert out[0].header['d_sample'] == pytest.approx(25.0)
    image = np.array([np.asarray(t) for t in out])
    assert np.isfinite(image).all()
    ix, iz = np.unravel_index(np.abs(image[:, 3:]).argmax(), image[:, 3:].shape)
    assert abs(ix - 20) <= 3 and abs((iz + 3) - 12) <= 4


def test_miggbzo_options_and_checks():
    kwargs = dict(vel=np.full((41, 41), 2000.0, np.float32), dz=25.0, dx=25.0)
    out = list(from_iterable(zero_offset_diffractor()) | miggbzo(fmin=5.0, fmax=40.0, amax=45.0, amin=-30.0, bwh=300.0, **kwargs))
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="nx"):
        list(from_iterable(zero_offset_diffractor()) | miggbzo(vel=np.ones((3, 41)), dz=25.0, dx=25.0))
    with pytest.raises(ValueError, match="too small"):
        list(from_iterable(zero_offset_diffractor()) | miggbzo(bwh=1.0, **kwargs))
    with pytest.raises(ValueError, match="not allowed"):
        list(from_iterable(zero_offset_diffractor()) | miggbzo(amax=95.0, **kwargs))


# ----------------------------------------------------------------------------------------------- miggbzoan
from seispy.migration import miggbzoan  # noqa: E402


def test_miggbzoan_is_close_to_miggbzo_for_an_isotropic_medium():
    v = 2000.0
    iso = list(from_iterable(zero_offset_diffractor()) | miggbzoan(
        a3333=np.full((41, 41), v * v, np.float32), dz=25.0, dx=25.0))
    assert len(iso) == 41
    image = np.array([np.asarray(t) for t in iso])
    assert np.isfinite(image).all()
    ix, iz = np.unravel_index(np.abs(image[:, 3:]).argmax(), image[:, 3:].shape)
    assert abs(ix - 20) <= 3 and abs((iz + 3) - 12) <= 4
    reference = np.array([np.asarray(t) for t in from_iterable(zero_offset_diffractor()) | miggbzo(
        vel=np.full((41, 41), v, np.float32), dz=25.0, dx=25.0)])
    assert np.corrcoef(image.ravel(), reference.ravel())[0, 1] > 0.9


def test_miggbzoan_with_thomsens_parameters_and_checks():
    a3333 = np.full((41, 41), 4.0e6, np.float32)
    out = list(from_iterable(zero_offset_diffractor()) | miggbzoan(
        a3333=a3333, delta=np.full((41, 41), 0.05, np.float32), epsilon=np.full((41, 41), 0.1, np.float32), dz=25.0, dx=25.0))
    assert np.isfinite(np.array([np.asarray(t) for t in out])).all()
    with pytest.raises(ValueError, match="shape"):
        list(from_iterable(zero_offset_diffractor()) | miggbzoan(a3333=a3333, a1313=np.ones((2, 2))))
    with pytest.raises(ValueError, match="nx"):
        list(from_iterable(zero_offset_diffractor()) | miggbzoan(a3333=np.ones((3, 41)) * 4e6, dz=25.0, dx=25.0))
