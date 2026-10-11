"""Dip moveout: dmofk, dmotx, dmofkcw, dmovz, dmotivz and tihaledmo (seispy/dmo)."""
import numpy as np
import numpy.testing as npt
import pytest

from seispy import dmo
from seispy.container import Trace, from_iterable

NX, NT, DT, DX = 61, 500, 0.004, 25.0
TN = 0.8  # the time of the spike, after NMO


def gather(offset=800.0, ix0=30, tn=TN, early=True, cdp0=1, ntr=NX, nt=NT, scale=1.0):
    """The traces of a common-offset gather that is a spike at cdp ix0 (and, to keep the mute of the programs from cutting
    the response, which is earlier than the spike, a very small sample early on the first trace)"""
    out = []
    for i in range(ntr):
        d = np.zeros(nt, np.float32)
        if i == ix0:
            d[int(round(tn / DT))] = scale
        if early and i == 0:
            d[1] = 1e-9
        xm = i * DX
        out.append(Trace(d, d_sample=DT).replace(tx_loc=[xm - offset / 2, 0, 0], rx_loc=[xm + offset / 2, 0, 0],
                                                  ensemble_number=cdp0 + i))
    return out


def samples(stream):
    return np.array([np.asarray(t) for t in stream])


def peak_times(data, traces_away=(0, 6, 10)):
    return [int(np.argmax(np.abs(data[30 + n]))) * DT for n in traces_away]


def ellipse(offset, traces_away=(0, 6, 10), tn=TN):
    """The DMO impulse response of a spike at tn, in constant velocity: t = tn sqrt(1 - (x / h)^2)"""
    h = offset / 2.0
    return [tn * np.sqrt(max(1.0 - (n * DX / h) ** 2, 0.0)) for n in traces_away]


COMMON = dict(cdpmin=1, cdpmax=NX, dxcdp=DX, noffmix=1)


def check_ellipse(out, offset=800.0, tol=0.03, traces_away=(0, 6, 10)):
    got = peak_times(samples(out), traces_away)
    npt.assert_allclose(got, ellipse(offset, traces_away), atol=tol)


# --------------------------------------------------------------------------------------------- impulse responses
@pytest.mark.parametrize('stage, away', [
    (dmo.dmofk(**COMMON), (0, 6, 10)),
    (dmo.dmotx(**COMMON), (0, 6, 10)),
    (dmo.dmofkcw(gamma=1.0, **COMMON), (0, 6, 10)),
    # (the ray tables of these are for a v(z), and the operator is weak and noisy on the steep parts of the ellipse)
    (dmo.dmovz(**COMMON), (0, 6)),
    (dmo.dmotivz(**COMMON), (0, 6)),
], ids=['dmofk', 'dmotx', 'dmofkcw', 'dmovz', 'dmotivz'])
def test_impulse_response_is_the_ellipse_of_dmo(stage, away):
    out = list(from_iterable(gather()) | stage)
    assert len(out) == NX
    check_ellipse(out, traces_away=away)
    data = samples(out)
    assert np.isfinite(data).all()
    # the spike's energy has moved to other traces
    assert (data[:30] ** 2).sum() > 0 and (data[31:] ** 2).sum() > 0


@pytest.mark.parametrize('option', [1, 2, 3, 4, 6])
def test_tihaledmo_impulse_response_is_the_ellipse(option):
    out = list(from_iterable(gather()) | dmo.tihaledmo(nxmax=NX, option=option, dx=DX, v=1500.0, h=400.0))
    assert len(out) == NX
    check_ellipse(out)


def test_tihaledmo_tsvankin_with_the_nmo_velocity_of_an_isotropic_medium_is_the_ellipse():
    # in a medium without anisotropy the NMO velocity of a dip is v / cos(dip) = v / sqrt(1 - (p v)^2), p = k / (2 w)
    v = 1500.0
    dp = 0.99 / v / 199
    p = np.arange(200) * dp
    vnmo = v / np.sqrt(1.0 - (p * v) ** 2)
    out = list(from_iterable(gather()) | dmo.tihaledmo(nxmax=NX, option=5, dx=DX, h=400.0, vnmo=vnmo, dp=dp))
    check_ellipse(out, traces_away=(0, 6))


def test_the_weak_anisotropy_operator_with_no_anisotropy_is_the_isotropic_one():
    iso = samples(from_iterable(gather()) | dmo.tihaledmo(nxmax=NX, option=1, dx=DX, v=1500.0, h=400.0))
    weak = samples(from_iterable(gather()) | dmo.tihaledmo(nxmax=NX, option=6, dx=DX, v=1500.0, h=400.0, e=0.0, d=0.0))
    npt.assert_allclose(weak, iso, atol=1e-4)
    aniso = samples(from_iterable(gather()) | dmo.tihaledmo(nxmax=NX, option=6, dx=DX, v=1500.0, h=400.0, e=0.2, d=0.05))
    assert np.abs(aniso - iso).max() > 1e-3


def test_ti_dmo_with_no_anisotropy_is_the_isotropic_vz():
    vz = samples(from_iterable(gather()) | dmo.dmovz(v=np.full(NT, 1500.0), **COMMON))
    ti = samples(from_iterable(gather()) | dmo.dmotivz(vn=np.full(NT, 1500.0), eta=np.zeros(NT), **COMMON))
    npt.assert_allclose(ti, vz, atol=1e-3 * np.abs(vz).max())


# -------------------------------------------------------------------------------------------------------- mixes
def test_mixes_are_the_sums_of_the_dmo_of_their_offsets():
    first = gather(800.0)
    second = gather(1200.0, ix0=25)
    both = list(from_iterable(first + second) | dmo.dmofk(cdpmin=1, cdpmax=NX, dxcdp=DX, noffmix=2))
    alone = [samples(from_iterable(g) | dmo.dmofk(cdpmin=1, cdpmax=NX, dxcdp=DX, noffmix=1)) for g in (first, second)]
    assert len(both) == 2 * NX
    # the traces come out in the order that they came in, each with the sum of the offsets of the mix
    npt.assert_allclose(samples(both[:NX]), alone[0] + alone[1], atol=1e-5)
    npt.assert_allclose(samples(both[NX:]), alone[0] + alone[1], atol=1e-5)
    # one offset per mix: each is the DMO of its own offset
    separate = list(from_iterable(first + second) | dmo.dmofk(cdpmin=1, cdpmax=NX, dxcdp=DX, noffmix=1))
    npt.assert_allclose(samples(separate), np.concatenate(alone), atol=1e-5)
    # the headers are those of the input traces
    assert [t.header['ensemble_number'] for t in both[:NX]] == list(range(1, NX + 1))


def test_traces_out_of_the_range_of_cdps_are_dropped_and_a_partial_last_mix_is_done():
    traces = gather(800.0) + gather(900.0, ix0=20) + gather(1000.0, ix0=40)
    out = list(from_iterable(traces) | dmo.dmofk(cdpmin=11, cdpmax=50, dxcdp=DX, noffmix=2))
    assert len(out) == 3 * 40
    assert all(11 <= t.header['ensemble_number'] <= 50 for t in out)
    # (the last mix has one offset only)
    one = list(from_iterable(gather(1000.0, ix0=40)) | dmo.dmofk(cdpmin=11, cdpmax=50, dxcdp=DX, noffmix=1))
    npt.assert_allclose(samples(out[80:]), samples(one), atol=1e-5)


def test_dmotx_mixes_and_the_range_too():
    traces = gather(800.0) + gather(1000.0, ix0=20)
    out = list(from_iterable(traces) | dmo.dmotx(cdpmin=6, cdpmax=55, dxcdp=DX, noffmix=2))
    assert len(out) == 2 * 50
    a = samples(from_iterable(gather(800.0)) | dmo.dmotx(cdpmin=6, cdpmax=55, dxcdp=DX, noffmix=1))
    b = samples(from_iterable(gather(1000.0, ix0=20)) | dmo.dmotx(cdpmin=6, cdpmax=55, dxcdp=DX, noffmix=1))
    # (the shaping filter is linear: the mix is the sum)
    npt.assert_allclose(samples(out[:50]), a + b, atol=1e-5)


def test_a_trace_out_of_the_range_with_another_offset_does_not_end_the_mix():
    # (SU counted a new offset only for traces of the range of cdps, and then every later trace made a new mix)
    traces = gather(800.0)
    stray = gather(1500.0, ix0=1000)[:1]  # (a trace out of the range, at another offset, in the middle of the gather)
    stray[0] = stray[0].replace(ensemble_number=500)
    mixed = traces[:30] + stray + traces[30:]
    out = samples(from_iterable(mixed) | dmo.dmofk(cdpmin=1, cdpmax=NX, dxcdp=DX, noffmix=2))
    # this makes two gathers of 30 and 31 traces: each is DMO'd on its own and they are mixed
    assert out.shape[0] == NX
    assert np.isfinite(out).all()


# --------------------------------------------------------------------------------------------------------- the mute
def test_dmofk_does_not_make_what_is_earlier_than_the_first_sample_of_a_gather():
    out = samples(from_iterable(gather(early=False)) | dmo.dmofk(**COMMON))
    # (the first non-zero sample is the spike, and the response of DMO is earlier than it)
    first_sample = int(round(TN / DT)) - 4
    assert np.abs(out[:, :first_sample]).max() == 0.0


def test_dmovz_mutes_by_the_stretch_and_says_when_nothing_survives():
    # (a short offset is stretched little: all of the data of the gather is there; a long one is stretched everywhere)
    out = samples(from_iterable(gather(offset=200.0, tn=0.8)) | dmo.dmovz(smute=1.5, **COMMON))
    assert out.shape == (NX, NT) and np.abs(out).max() > 0
    with pytest.raises(ValueError, match="mute"):
        list(from_iterable(gather(offset=8000.0, tn=0.2)) | dmo.dmovz(**COMMON))


# ------------------------------------------------------------------------------------------- the checks of input
def test_zero_offset_gathers_are_not_changed_by_the_vz_programs():
    # (the program divided by the offset: the table of rays had no rays, and it read and wrote what it should not)
    zero = gather(offset=0.0, tn=0.4)
    for stage in (dmo.dmovz(**COMMON), dmo.dmotivz(**COMMON)):
        out = samples(from_iterable(zero) | stage)
        npt.assert_array_equal(out, samples(zero))
    # a negative offset (a split spread) is the same DMO as the positive one
    plus = samples(from_iterable(gather(offset=800.0)) | dmo.dmotx(**COMMON))
    minus = samples(from_iterable(gather(offset=-800.0)) | dmo.dmotx(**COMMON))
    npt.assert_allclose(minus, plus, atol=1e-5)


def test_dmo_checks_its_input():
    with pytest.raises(ValueError, match="cdpmin"):
        dmo.dmofk(cdpmin=5, cdpmax=1, dxcdp=DX, noffmix=1)
    with pytest.raises(ValueError, match="noffmix"):
        dmo.dmofk(cdpmin=1, cdpmax=5, dxcdp=DX, noffmix=0)
    with pytest.raises(ValueError, match="dxcdp"):
        dmo.dmotx(cdpmin=1, cdpmax=5, dxcdp=0.0, noffmix=1)
    with pytest.raises(ValueError, match="monotonically"):
        list(from_iterable(gather()) | dmo.dmofk(tdmo=[0.0, 1.0, 0.5], vdmo=[1500.0, 2000.0, 2500.0], **COMMON))
    with pytest.raises(ValueError, match="equal"):
        list(from_iterable(gather()) | dmo.dmofk(tdmo=[0.0, 1.0], vdmo=[1500.0], **COMMON))
    with pytest.raises(ValueError, match="gamma"):
        dmo.dmofkcw(gamma=0.0, **COMMON)
    with pytest.raises(ValueError, match="nonzero ft"):
        list(from_iterable([t.replace(sample_start=0.1) for t in gather()]) | dmo.dmovz(**COMMON))
    with pytest.raises(ValueError, match="velocities"):
        list(from_iterable(gather()) | dmo.dmovz(v=np.full(10, 1500.0), **COMMON))
    with pytest.raises(ValueError, match="same number of samples"):
        list(from_iterable(gather()[:3] + gather(nt=100, tn=0.2)[:3]) | dmo.dmofk(**COMMON))
    with pytest.raises(ValueError, match="option 5"):
        dmo.tihaledmo(nxmax=10, option=5)
    with pytest.raises(ValueError, match="positive value for v"):
        dmo.tihaledmo(nxmax=10, option=3, v=0.0)
    with pytest.raises(ValueError, match="option"):
        dmo.tihaledmo(nxmax=10, option=7)
    with pytest.raises(ValueError, match="smute"):
        dmo.dmovz(smute=1.0, **COMMON)


def test_tihaledmo_keeps_at_most_nxmax_traces():
    out = list(from_iterable(gather()) | dmo.tihaledmo(nxmax=40, dx=DX, h=400.0))
    assert len(out) == 40
    assert list(from_iterable([]) | dmo.tihaledmo(nxmax=5)) == []


def test_dmo_stages_work_with_no_traces():
    for stage in (dmo.dmofk(**COMMON), dmo.dmotx(**COMMON), dmo.dmofkcw(**COMMON), dmo.dmovz(**COMMON), dmo.dmotivz(**COMMON)):
        assert list(from_iterable([]) | stage) == []
