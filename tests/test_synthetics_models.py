"""
Test patterns and modelling programs of su/main/synthetics_waveforms_testpatterns: SUNHMOSPIKE, SUADDEVENT, ...
"""
import math

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.synthetics import _born, _events, addevent, goupillaud, goupillaudpo, imp2d, imp3d, nhmospike, syncz, synlv, synlvcw, synlvfti, synvxz, synvxzcs, kdsyn2d


def samples(traces):
    return np.array([np.asarray(t) for t in traces])


def spikes_of(trace):
    return {int(i): float(v) for i, v in zip(np.flatnonzero(np.asarray(trace)), np.asarray(trace)[np.flatnonzero(np.asarray(trace))])}


# --------------------------------------------------------------------------------------------------------- nhmospike
# (dt = 0.5 is exact in single precision: the program's arithmetic is in single precision, and for a dt like 0.001 it puts an
#  event at 100 ms on sample 99, see test_the_single_precision_arithmetic_of_the_program)
DT = 0.5


def test_defaults_are_the_gather_of_the_program():
    traces = list(nhmospike())
    assert len(traces) == 20 and all(t.n_sample == 300 for t in traces)
    assert [t.header['offset'] for t in traces[:3]] == [100.0, 200.0, 300.0]
    assert traces[-1].header['offset'] == 2000.0
    assert all(t.header['ensemble_number'] == 1 for t in traces)
    assert [t.header['trace_id'] for t in traces] == list(range(1, 21))
    assert traces[0].d_sample == pytest.approx(0.001)


def test_parabolic_moveout_follows_the_square_of_the_offset():
    # an event of 100 ms with a moveout of 200 ms at the reference offset 2000: index = 100 + 200 (x / 2000)^2 samples of 1 ms
    # (dt is 0.5 s here, so use the times in units of it: 1000 dt = 500 ms per sample)
    events = [(500.0 * 20, 500.0 * 10, 1.0)]  # p = 20 samples at x = offref, t = 10 samples
    traces = list(nhmospike(nt=200, ntr=20, dt=DT, events=events, nspk=1))
    for k, trace in enumerate(traces):
        x = 100.0 * (k + 1)
        expected = int(10 + 20 * (x / 2000.0) ** 2)
        assert spikes_of(trace) == {expected: 1.0}, (k, expected)


def test_linear_and_absolute_moveouts():
    events = [(500.0 * 40, 500.0 * 5, 2.0)]
    for gopt in (3, 4):
        traces = list(nhmospike(nt=200, ntr=10, dt=DT, events=events, nspk=1, gopt=gopt, offref=1000.0, offinc=100.0))
        for k, trace in enumerate(traces):
            x = 100.0 * (k + 1)
            assert spikes_of(trace) == {int(5 + 40 * x / 1000.0): 2.0}


def test_pseudo_hyperbolic_moveout_uses_the_reference_depth():
    events = [(500.0 * 30, 500.0 * 8, 1.0)]
    traces = list(nhmospike(nt=300, ntr=10, dt=DT, events=events, nspk=1, gopt=2, offref=1000.0, depthref=500.0))
    reference = math.sqrt(500.0 ** 2 + 1000.0 ** 2)
    for k, trace in enumerate(traces):
        x = 100.0 * (k + 1)
        expected = int(8 + 30 * math.sqrt(500.0 ** 2 + x ** 2) / reference)
        assert spikes_of(trace) == {expected: 1.0}


def test_events_that_land_on_the_same_sample_add_and_ones_off_the_trace_are_left_out():
    events = [(0.0, 500.0 * 10, 1.0), (0.0, 500.0 * 10, 2.5), (0.0, 0.0, 9.0), (0.0, 500.0 * 100, 4.0)]
    (trace,) = list(nhmospike(nt=50, ntr=1, dt=DT, events=events, nspk=4))
    assert spikes_of(trace) == {10: 3.5}  # (sample 0 and sample 100 of 50 are left out)
    (trace,) = list(nhmospike(nt=50, ntr=1, dt=DT, events=events, nspk=2))
    assert spikes_of(trace) == {10: 3.5}
    (trace,) = list(nhmospike(nt=50, ntr=1, dt=DT, events=events, nspk=1))
    assert spikes_of(trace) == {10: 1.0}
    (trace,) = list(nhmospike(nt=50, ntr=1, dt=DT, events=events, nspk=0))
    assert spikes_of(trace) == {}


def test_the_single_precision_arithmetic_of_the_program():
    # dt = 0.001 is a little more than 0.001 in single precision, so 100 ms is 99.99999 samples, which the program truncates
    traces = list(nhmospike(nt=300, ntr=1, events=[(0.0, 100.0, 1.0)], nspk=1))
    assert list(spikes_of(traces[0])) == [99]


def test_the_documented_moveout_of_the_fourth_event():
    # the program documents 120 ms for p4 and uses 100: at the reference offset the fourth event is 120 ms from its intercept
    (trace,) = list(nhmospike(nt=1000, ntr=1, dt=0.001, offinc=2000.0, offref=2000.0, gopt=3, nspk=4))
    indices = sorted(spikes_of(trace))
    assert indices[-1] - indices[-2] > 0 and 319 in indices  # 200 + 120 = 320 ms, truncated to 319 by the arithmetic


def test_nhmospike_checks_its_parameters():
    for bad in (dict(gopt=5), dict(nt=0), dict(ntr=-1), dict(dt=0.0), dict(nspk=5), dict(events=[(1.0, 2.0)])):
        with pytest.raises(ValueError):
            nhmospike(**bad)
    with pytest.raises(ValueError, match="g\\(x\\) = 0"):
        nhmospike(gopt=1, offref=0.0)
    assert list(nhmospike(ntr=0)) == []


# ---------------------------------------------------------------------------------------------------------- addevent
DT_EVENT = 0.004
NT = 200


def gather(offsets, dt=DT_EVENT, data=None):
    return [Trace((np.zeros(NT) if data is None else data).astype(np.float32), d_sample=dt).replace(offset=float(h), trace_id=i + 1)
            for i, h in enumerate(offsets)]


def run(stage, traces):
    return list(from_iterable(traces) | stage)


def test_addevent_puts_a_hyperbola_in_the_gather():
    offsets = [0.0, 300.0, 600.0, 900.0]
    out = run(addevent(type='nmo', t0=0.2, vel=2000.0, amp=2.0), gather(offsets))
    for h, trace in zip(offsets, out):
        tx = math.sqrt(0.2 ** 2 + (h / 2000.0) ** 2)
        x = np.asarray(trace)
        assert abs(int(np.argmax(x)) - tx / DT_EVENT) <= 0.5
        # the peak is the band-limited spike: about the amplitude, less by the sinc at the distance to the sample (the 8 point
        # table sinc of the SU library is only an approximation of it), and the weights of the samples add up to the amplitude
        fraction = abs(tx / DT_EVENT - round(tx / DT_EVENT))
        assert x[int(np.argmax(x))] == pytest.approx(2.0 * np.sinc(fraction), rel=0.08)
        assert x.sum() == pytest.approx(2.0, rel=0.03)


def test_addevent_linear_event_on_a_sample_is_one_sample():
    out = run(addevent(type='lmo', t0=0.1, vel=1500.0, amp=1.0), gather([-300.0, 300.0]))
    for trace in out:
        x = np.asarray(trace)
        assert x[75] == pytest.approx(1.0, abs=1e-5)  # 0.1 + 300/1500 = 0.3 s = sample 75
        assert np.abs(np.delete(x, 75)).max() < 1e-5


def test_addevent_adds_to_what_is_there_and_scales_with_the_amplitude():
    base = np.random.default_rng(1).normal(size=NT)
    (plain,) = run(addevent(type='lmo', t0=0.1, vel=1500.0, amp=1.0), gather([0.0]))
    (more,) = run(addevent(type='lmo', t0=0.1, vel=1500.0, amp=3.0), gather([0.0]))
    (added,) = run(addevent(type='lmo', t0=0.1, vel=1500.0, amp=3.0), gather([0.0], data=base))
    npt.assert_allclose(np.asarray(more), 3.0 * np.asarray(plain), atol=1e-5)
    npt.assert_allclose(np.asarray(added), base.astype(np.float32) + np.asarray(more), atol=1e-5)


def test_addevent_uses_dt_for_traces_without_a_sample_interval():
    (with_dt,) = run(addevent(type='lmo', t0=0.1, vel=1500.0, dt=DT_EVENT), gather([0.0], dt=0.0))
    (header,) = run(addevent(type='lmo', t0=0.1, vel=1500.0), gather([0.0]))
    npt.assert_allclose(np.asarray(with_dt), np.asarray(header), atol=1e-6)
    with pytest.raises(ValueError, match="sample interval"):
        run(addevent(type='lmo'), gather([0.0], dt=0.0))


def test_addevent_checks_its_parameters():
    with pytest.raises(ValueError):
        addevent(type='hyperbolic')
    with pytest.raises(ValueError):
        addevent(vel=0.0)
    with pytest.raises(TypeError):
        run(addevent(), [Trace((np.ones(NT) + 1j).astype(np.complex64), d_sample=DT_EVENT)])
    assert addevent().parallelism == 'trace'


# ----------------------------------------------------------------------------------------------------- goupillaudpo
def reference_goupillaudpo(r, l, k, tmax, pV):
    """sugoupillaudpo.c, in python"""
    n = len(r) - 1
    r = [pV * v for v in r]
    skl = 1 if k > l else (-1 if k < l else 0)
    n1, n2 = min(k, l), max(k, l)
    x = [0.0] * (2 * tmax)
    transm1 = 1.0
    for i in range(n1, n2):
        transm1 *= 1 + skl * r[i]
    if pV == -1 or l == 1:
        x[n2 - n1] = transm1
    else:
        x[n2 - n1] = skl * transm1
    effsd = 1.0 if l == 1 else 1 + pV * r[l - 1]
    rmax = min(n, tmax - 1 + (n2 + n1 - 1) // 2)
    transm2 = 1.0
    for i in range(n2, rmax + 1):
        x[2 * (i + 1) - n2 - n1] = effsd * transm1 * r[i] * transm2
        transm2 *= 1 - r[i] * r[i]
    if l != 1:
        transm2 = 1.0
        i = n1 - 1
        while i >= 0 and n2 - n1 + 2 * (n1 - 1 - i) < 2 * tmax:
            x[n2 - n1 + 2 * (n1 - 1 - i)] += pV * transm1 * r[i] * transm2
            transm2 *= 1 - r[i] * r[i]
            i -= 1
    odd = (k - l) % 2 != 0
    return np.array(x[1::2] if odd else x[0::2][:tmax]), odd


def reflectivity(n, seed=0):
    return np.random.default_rng(seed).uniform(-0.6, 0.6, size=n + 1).astype(np.float32)


def run_gpo(r, **kwargs):
    (trace,) = list(from_iterable([Trace(np.asarray(r, dtype=np.float32), d_sample=0.004)]) | goupillaudpo(**kwargs))
    return trace


@pytest.mark.parametrize('l, k', [(1, 1), (1, 4), (4, 1), (3, 3), (3, 6), (6, 3), (2, 9), (9, 2), (7, 7)])
@pytest.mark.parametrize('pV', [1, -1])
def test_goupillaudpo_matches_the_algorithm_of_the_program(l, k, pV):
    n = 11
    r = reflectivity(n, seed=l * 10 + k)
    for tmax in (None, 6, 14):
        trace = run_gpo(r, l=l, k=k, pV=pV, tmax=tmax)
        length = trace.n_sample
        if tmax is None:
            n1, n2 = min(k, l), max(k, l)
            assert length == (n2 - n1 + 2 + 2 * max(n1 - 1, n + 1 - n2, n + 1 - n1)) // 2
        else:
            assert length == tmax
        expected, odd = reference_goupillaudpo(r.astype(np.float64), l, k, length, pV)
        npt.assert_allclose(np.asarray(trace), expected, atol=2e-6, rtol=1e-5)
        assert trace.header['sample_start'] == (pytest.approx(0.002) if odd else 0.0)
        assert trace.header['trace_type'] == 1


def test_goupillaudpo_surface_source_and_receiver_has_the_textbook_primaries():
    r = np.array([0.0, 0.3, -0.2, 0.5, 0.1], dtype=np.float32)
    x = np.asarray(run_gpo(r, tmax=5))
    transmission = np.cumprod(np.concatenate(([1.0], 1 - r[:-1].astype(np.float64) ** 2)))
    expected = r * transmission
    expected[0] = 1.0  # (the direct arrival, in place of the surface coefficient)
    npt.assert_allclose(x[1:], (r * transmission)[1:], atol=1e-6)
    assert x[0] == pytest.approx(1.0)


def test_goupillaudpo_buried_source_has_an_upgoing_minus_one_for_a_vector_field():
    r = reflectivity(8, seed=3)
    surface = np.asarray(run_gpo(r, l=3, k=1, pV=1, tmax=6))
    pressure = np.asarray(run_gpo(r, l=3, k=1, pV=-1, tmax=6))
    # the upgoing wave from the top of layer 3 crosses the interfaces 1 and 2 on its way to the surface (a coefficient of 1 - r each),
    # and arrives together with its reflection from the surface (r[0] times it): a receiver at the surface sees both
    transmission = (1 - r[1]) * (1 - r[2])
    assert surface[1] == pytest.approx(-transmission * (1 - r[0]), rel=1e-5)
    # (for pressure the coefficients have the other sign, and the upgoing spike has amplitude 1 and not -1)
    assert pressure[1] == pytest.approx((1 + r[0]) * (1 + r[1]) * (1 + r[2]), rel=1e-5)


def test_goupillaudpo_does_not_change_the_reflectivity_and_every_trace_has_one():
    traces = [Trace(reflectivity(6, seed=s), d_sample=0.004) for s in range(3)]
    before = [np.asarray(t).copy() for t in traces]
    out = list(from_iterable(traces) | goupillaudpo(l=2, k=4, pV=-1))
    assert len(out) == 3
    for t, b in zip(traces, before):
        npt.assert_array_equal(np.asarray(t), b)
    for t, o in zip(traces, out):
        expected, _ = reference_goupillaudpo(np.asarray(t).astype(np.float64), 2, 4, o.n_sample, -1)
        npt.assert_allclose(np.asarray(o), expected, atol=2e-6)


def test_goupillaudpo_checks_its_parameters_and_input():
    for bad in (dict(pV=0), dict(k=0), dict(l=0), dict(tmax=-1)):
        with pytest.raises(ValueError):
            goupillaudpo(**bad)
    r = reflectivity(4)
    with pytest.raises(ValueError, match=r"l<=n\+1"):
        run_gpo(r, l=7)
    with pytest.raises(ValueError, match="receiver layer"):
        run_gpo(r, k=7)  # (the program reads beyond the reflectivity here)
    with pytest.raises(ValueError, match="Invalid reflection coefficient"):
        run_gpo(np.array([0.0, 1.5, 0.1]))
    with pytest.raises(ValueError, match="too small"):
        run_gpo(r, l=1, k=5, tmax=2)
    with pytest.warns(UserWarning, match="without subsurface"):
        run_gpo(np.array([0.3]))
    assert goupillaudpo().parallelism == 'trace'


# ------------------------------------------------------------------------------------------------------ imp2d, imp3d
def reference_born(nt, dt, c, rs, rg, dim, direct=False, rd=0.0):
    """The response of suimp2d / suimp3d at one receiver, with numpy's FFT (of the length that the programs pad to)"""
    nfft = _born.BornResponse(nt, dt, c).nfft
    if dim == 2:
        k = 1.0 / (4.0 * math.sqrt(2.0 * c * dt * nfft) * c * dt * dt * nfft * nfft)
        spread = math.sqrt(rs * rg * (rs + rg))
    else:
        k = 1.0 / (4.0 * c * c * dt ** 3 * nfft ** 3)
        spread = rs * rg
    x = _events.spike_response((rs + rg) / c, k / spread, dt, 0.0, nt)
    if direct:
        x = x + _events.spike_response(rd / c, k / rd, dt, 0.0, nt)
    padded = np.zeros(nfft, dtype=np.float64)
    padded[:nt] = x
    i = np.arange(nfft // 2 + 1, dtype=np.float64)
    factor = i * np.sqrt(i) if dim == 2 else i * i
    return (np.fft.irfft(np.fft.rfft(padded) * factor, nfft) * nfft)[:nt]


def dist(*d):
    return math.sqrt(sum(v * v for v in d))


@pytest.mark.parametrize('dim', [2, 3])
def test_born_response_matches_an_independent_computation(dim):
    kwargs = dict(nshot=2, nrec=3, nt=200, dt=0.004, c=4000.0, x0=1300.0, z0=900.0, dsx=170.0, dgx=130.0, gxmin=-200.0)
    make = imp2d if dim == 2 else imp3d
    traces = list(make(**kwargs))
    assert len(traces) == 6
    for number, trace in enumerate(traces):
        s, g = divmod(number, 3)
        sx, gx = 170.0 * s, -200.0 + 130.0 * g
        rs, rg = dist(sx - 1300.0, 0.0 - 900.0), dist(gx - 1300.0, 0.0 - 900.0)
        if dim == 3:
            rs, rg = dist(sx - 1300.0, 0.0 - 0.0, 0.0 - 900.0), dist(gx - 1300.0, 0.0, 0.0 - 900.0)
        expected = reference_born(200, 0.004, 4000.0, rs, rg, dim)
        got = np.asarray(trace)
        npt.assert_allclose(got, expected, atol=2e-4 * np.abs(expected).max(), rtol=0)


@pytest.mark.parametrize('dim', [2, 3])
def test_born_response_arrives_at_the_traveltime_through_the_scatterer(dim):
    make = imp2d if dim == 2 else imp3d
    for gx in (0.0, 400.0, 900.0):
        (trace,) = list(make(nt=256, dt=0.004, c=5000.0, x0=1000.0, z0=1000.0, gxmin=gx, nrec=1))
        t = (dist(0.0 - 1000.0, 1000.0) + dist(gx - 1000.0, 1000.0)) / 5000.0
        peak = int(np.argmax(np.abs(np.asarray(trace))))
        assert abs(peak - t / 0.004) <= 3, (gx, peak, t / 0.004)  # (the filter makes a wavelet a few samples wide)


def test_born_amplitude_falls_with_the_geometrical_spreading():
    # Geometries with whole numbers of samples between the arrivals (scatterer at (900, 1200), shot at 0, c = 5000 and dt = 4 ms:
    # 20 m of path is 1 sample), so the wavelets are the same, shifted, and the peak amplitudes are in the ratio of the spreading.
    gxs = [900.0, 1400.0, 1800.0, 2500.0]  # receivers 1200, 1300, 1500 and 2000 m from the scatterer
    kwargs = dict(nt=400, dt=0.004, c=5000.0, x0=900.0, z0=1200.0, nrec=1)
    rs = 1500.0
    for dim, make in ((2, imp2d), (3, imp3d)):
        peaks, arrivals = [], []
        for gx, rg in zip(gxs, (1200.0, 1300.0, 1500.0, 2000.0)):
            (trace,) = list(make(gxmin=gx, **kwargs))
            x = np.asarray(trace)
            peaks.append(float(np.abs(x).max()))
            arrivals.append(int(np.argmax(np.abs(x))))
        spreads = [math.sqrt(rs * rg * (rs + rg)) if dim == 2 else rs * rg for rg in (1200.0, 1300.0, 1500.0, 2000.0)]
        for n in (1, 2, 3):
            assert peaks[n] / peaks[0] == pytest.approx(spreads[0] / spreads[n], rel=0.02), (dim, n)
        # (and the arrivals are where the traveltimes through the scatterer are: 2700, 2800, 3000 and 3500 m of path)
        assert [a - arrivals[0] for a in arrivals] == [0, 5, 15, 40]


def test_imp3d_direct_arrival():
    kwargs = dict(nt=300, dt=0.004, c=3000.0, x0=1000.0, z0=1500.0, gxmin=300.0, nrec=1)
    (without,) = list(imp3d(dir=0, **kwargs))
    (with_direct,) = list(imp3d(dir=1, **kwargs))
    expected = reference_born(300, 0.004, 3000.0, dist(1000.0, 1500.0), dist(700.0, 1500.0), 3, direct=True, rd=300.0)
    npt.assert_allclose(np.asarray(with_direct), expected, atol=2e-4 * np.abs(expected).max(), rtol=0)
    difference = np.asarray(with_direct) - np.asarray(without)
    # the direct wave is at rd / c = 0.1 s = sample 25
    assert abs(int(np.argmax(np.abs(difference))) - 25) <= 3
    # (a receiver at the shot makes the direct arrival infinite, as in the program)
    (at_shot,) = list(imp3d(dir=1, nt=64, gxmin=0.0, nrec=1))
    assert not np.isfinite(np.asarray(at_shot)).all()


def test_born_headers_follow_the_shots_and_receivers():
    traces = list(imp2d(nshot=2, nrec=3, sxmin=10.0, dsx=100.0, szmin=5.0, dsz=2.0, gxmin=0.0, dgx=50.0, gzmin=1.0, dgz=0.5))
    assert [t.header['trace_id'] for t in traces] == [1, 2, 3, 4, 5, 6]
    assert [t.header['ensemble_number'] for t in traces] == [1, 1, 1, 2, 2, 2]
    assert [t.header['ensemble_trace_number'] for t in traces] == [1, 2, 3, 1, 2, 3]
    assert traces[4].header['tx_loc'] == pytest.approx((110.0, 0.0, -7.0))
    assert traces[4].header['rx_loc'] == pytest.approx((50.0, 0.0, -1.5))
    assert all(t.d_sample == pytest.approx(0.004) and t.n_sample == 256 for t in traces)
    three = list(imp3d(nshot=1, nrec=2, symin=20.0, dsy=0.0, gymin=40.0, dgy=10.0, y0=0.0))
    assert three[1].header['tx_loc'] == pytest.approx((0.0, 20.0, 0.0)) and three[1].header['rx_loc'] == pytest.approx((100.0, 50.0, 0.0))


def test_born_sources_check_their_parameters():
    for make in (imp2d, imp3d):
        for bad in (dict(nt=0), dict(c=0.0), dict(dt=-1.0), dict(nshot=-1)):
            with pytest.raises(ValueError):
                make(**bad)
        assert list(make(nshot=0)) == []
    with pytest.raises(ValueError):
        imp3d(dir=2)
    with pytest.raises(ValueError, match="too big"):
        imp2d(nt=70000)


# ----------------------------------------------------------------------------------------------------------- syncz
def reference_syncz_spikes(zint, dip_deg, v, rho, x):
    """(time, amplitude) of the spike of each interface at the position x, from the algorithm of susyncz.c, in python"""
    ninf = len(zint)
    z = [0.0] + list(zint)
    dip = [0.0] + [math.radians(a) for a in dip_deg]
    theta = [[0.0] * (ninf + 1) for _ in range(ninf + 1)]
    for j in range(1, ninf + 1):
        theta[j][j - 1] = -dip[j]
        for i in range(j - 1, 0, -1):
            theta[j][i - 1] = math.asin(v[i - 1] * math.sin(theta[j][i] + dip[i]) / v[i]) - dip[i]
    spikes = []
    for j in range(1, ninf + 1):
        ex, t0, p1, p2, w = x, 0.0, 0.0, 0.0, 1.0
        for i in range(0, j):
            if i > 0:
                s = math.sin(theta[j][i - 1])
                temp = 1.0 - math.tan(dip[i]) * s
                ex = (1.0 - math.tan(dip[i - 1]) * s) / temp * ex + (z[i] - z[i - 1]) * s / temp
                c1 = math.cos(theta[j][i - 1] + dip[i])
                c2 = math.cos(theta[j][i] + dip[i])
                w = w * c1 * c1 / (c2 * c2)
            if dip[i + 1] == dip[i]:
                dist = (z[i + 1] - z[i]) * math.cos(dip[i]) / math.cos(theta[j][i] + dip[i])
            else:
                meet = (z[i + 1] - z[i]) / (math.tan(dip[i]) - math.tan(dip[i + 1]))
                k = math.sin(dip[i] - dip[i + 1]) / (math.cos(theta[j][i] + dip[i + 1]) * math.cos(dip[i]))
                dist = (meet - ex) * k
            t0 += dist / v[i]
            p1 += dist * v[i]
            p2 += dist * v[i] * (w if i > 0 else 1.0)
        # the transmission coefficients: the reflection at the interface, times 1 - r^2 of the interfaces above
        imp = [rho[i] * v[i] for i in range(ninf + 1)]
        trans = (imp[j] - imp[j - 1]) / (imp[j] + imp[j - 1])
        for i in range(1, j):
            t1 = imp[i] * math.cos(theta[j][i - 1] + dip[i])
            t2 = imp[i - 1] * math.cos(theta[j][i] + dip[i])
            r = (t1 - t2) / (t1 + t2)
            trans *= 1.0 - r * r
        spikes.append((2.0 * t0, v[0] * trans / (0.004 * 4.0 * math.pi * 2.0 * math.sqrt(p1 * p2))))
    return spikes


def syncz_traces(**kwargs):
    return list(syncz(**kwargs))


def test_syncz_one_flat_interface_is_the_textbook_reflection():
    (trace, _) = syncz_traces(ninf=1, zint=[300.0], dip=[0.0], v=[1500.0, 2500.0], ntr=2, nt=200)
    x = np.asarray(trace)
    peak = int(np.argmax(np.abs(x)))
    assert peak == 100  # 2 z / v = 0.4 s
    reflection = (2500.0 - 1500.0) / (2500.0 + 1500.0)
    assert x[peak] == pytest.approx(reflection / (8 * math.pi * 0.004 * 300.0), rel=1e-4)
    assert np.abs(np.delete(x, peak)).max() < 1e-6 * abs(x[peak])  # (a spike on a sample: the sinc is zero on the others)


def test_syncz_two_flat_layers_have_the_transmission_loss_and_the_spreading_of_the_path():
    kwargs = dict(ninf=2, zint=[150.0, 250.0], dip=[0.0, 0.0], v=[1500.0, 2000.0, 3000.0], rho=[1.0, 1.2, 1.5], ntr=1, nt=200)
    (trace,) = syncz_traces(**kwargs)
    x = np.asarray(trace)
    imp = [1500.0, 2400.0, 4500.0]
    r1, r2 = (imp[1] - imp[0]) / (imp[1] + imp[0]), (imp[2] - imp[1]) / (imp[2] + imp[1])
    path = 150.0 * 1500.0 + 100.0 * 2000.0  # sum of thickness * velocity
    expected_2 = 1500.0 * r2 * (1 - r1 * r1) / (0.004 * 8 * math.pi * path)
    assert x[50] == pytest.approx(r1 / (8 * math.pi * 0.004 * 150.0), rel=1e-4)  # 2 * 150 / 1500 = 0.2 s
    assert x[75] == pytest.approx(expected_2, rel=1e-4)  # + 2 * 100 / 2000 = 0.1 s


@pytest.mark.parametrize('dip_deg', [-30.0, -10.0, 7.5, 25.0])
def test_syncz_a_dipping_interface_is_seen_at_the_normal_distance(dip_deg):
    # the interface is z = z0 + x tan(dip), and the zero-offset ray is normal to it: the distance is (z0 + x tan(dip)) cos(dip)
    dip = math.radians(dip_deg)
    (first, *rest) = syncz_traces(ninf=1, zint=[600.0], dip=[dip_deg], v=[2000.0, 3000.0], ntr=12, dx=20.0, nt=300)
    for itr, trace in enumerate([first] + rest):
        x = 20.0 * itr
        distance = (600.0 + x * math.tan(dip)) * math.cos(dip)
        data = np.asarray(trace)
        peak = int(np.argmax(np.abs(data)))
        assert abs(peak - 2 * distance / 2000.0 / 0.004) <= 0.6
        # (the peak is the band-limited spike of the amplitude R / (8 pi dt distance), less by the sinc at the fraction)
        fraction = abs(2 * distance / 2000.0 / 0.004 - round(2 * distance / 2000.0 / 0.004))
        reflection = (3000.0 - 2000.0) / (3000.0 + 2000.0)
        assert data.sum() == pytest.approx(reflection / (8 * math.pi * 0.004 * distance), rel=0.03)


def test_syncz_matches_the_algorithm_of_the_program_for_a_dipping_model():
    models = [
        dict(ninf=4),  # the defaults
        dict(ninf=3, zint=[200.0, 320.0, 500.0], dip=[10.0, 14.0, 4.0], v=[1800.0, 2200.0, 2100.0, 3200.0], rho=[1.0, 1.3, 1.2, 1.6]),
        dict(ninf=2, zint=[150.0, 400.0], dip=[-12.0, -7.0], v=[1500.0, 1900.0, 2600.0], rho=[1.0, 1.1, 1.4]),
    ]
    for model in models:
        ninf = model['ninf']
        zint = model.get('zint', [100.0 * i for i in range(1, ninf + 1)])
        dips = model.get('dip', [5.0 * i for i in range(1, ninf + 1)])
        v = model.get('v', [1500.0 + 500.0 * i for i in range(ninf + 1)])
        rho = model.get('rho', [1.0] * (ninf + 1))
        traces = syncz_traces(ntr=6, dx=15.0, nt=256, dt=0.004, **model)
        for itr, trace in enumerate(traces):
            expected = np.zeros(256)
            for time, amp in reference_syncz_spikes(zint, dips, v, rho, 15.0 * itr):
                expected += _events.spike_response(time, amp, 0.004, 0.0, 256)
            got = np.asarray(trace)
            npt.assert_allclose(got, expected, atol=2e-3 * np.abs(expected).max(), rtol=0)


def test_syncz_headers_and_lines():
    traces = syncz_traces(nline=2, ntr=4, dx=25.0, tdelay=0.05, nt=64)
    assert len(traces) == 8
    assert [t.header['trace_id'] for t in traces] == [1, 2, 3, 4, 1, 2, 3, 4]
    assert [t.header['ensemble_number'] for t in traces] == [1] * 4 + [2] * 4
    assert traces[2].header['tx_loc'] == pytest.approx((50.0, 0.0, 0.0)) and traces[2].header['rx_loc'] == pytest.approx((50.0, 0.0, 0.0))
    assert traces[0].header['sample_start'] == pytest.approx(0.05)
    npt.assert_array_equal(np.asarray(traces[1]), np.asarray(traces[5]))  # (identical lines)
    # the delay moves the events earlier in the recording
    plain = syncz_traces(ntr=1, nt=64)
    delayed = syncz_traces(ntr=1, nt=64, tdelay=0.04)
    assert int(np.argmax(np.abs(np.asarray(plain[0])))) - int(np.argmax(np.abs(np.asarray(delayed[0])))) == 10


def test_syncz_checks_the_model():
    flat = dict(zint=[100.0, 200.0], v=[1500.0, 2000.0, 2500.0])
    cases = [
        (dict(ninf=2, dip=[95.0, 0.0], **flat), "from -90 to 90"),
        (dict(ninf=2, dip=[80.0, -20.0], **flat), "more than 90"),
        (dict(ninf=2, dip=[0.0, 0.0], zint=[200.0, 100.0], v=[1500.0, 2000.0, 2500.0]), "intercept"),
        (dict(ninf=2, dip=[0.0, 0.0], zint=[100.0, 200.0], v=[1500.0, 0.0, 2500.0]), "velocity of layer 1"),
        (dict(ninf=2, dip=[0.0, 0.0], zint=[100.0, 200.0], v=[1500.0, 2000.0, -1.0]), "velocity of layer 2"),  # (the deepest layer)
        (dict(ninf=2, dip=[0.0, 0.0], rho=[1.0, 0.0, 1.0], **flat), "density of layer 1"),
        (dict(ninf=2, dip=[0.0, -30.0], zint=[100.0, 150.0], v=[1500.0, 2000.0, 2500.0], ntr=32, dx=10.0), "meets"),
        # (a ray that goes up from interface 3 meets a layer that is 3 times as fast at 30 degrees: sin of the angle is 1.5)
        (dict(ninf=3, dip=[0.0, 30.0, 0.0], zint=[100.0, 200.0, 300.0], v=[1500.0, 3000.0, 1000.0, 2000.0]), "critical angle"),
    ]
    for kwargs, message in cases:
        with pytest.raises(ValueError, match=message):
            syncz(**kwargs)
    with pytest.raises(ValueError, match="zint and dip must have"):
        syncz(ninf=3, zint=[1.0, 2.0])
    with pytest.raises(ValueError, match="v and rho must have"):
        syncz(ninf=2, v=[1500.0, 2000.0])
    with pytest.raises(ValueError):
        syncz(dt=0.0)
    assert [np.asarray(t).any() for t in syncz(ninf=0, ntr=2)] == [False, False]  # (no interfaces, no signal)


# ---------------------------------------------------------------------------------------------------------- threads
def test_the_models_can_be_made_by_several_threads_at_once():
    import concurrent.futures

    def reflectivity_traces():
        return [Trace(reflectivity(40, seed=i), d_sample=0.004) for i in range(6)]

    work = {
        'imp2d': lambda: samples(imp2d(nshot=2, nrec=3, nt=200, dsx=40.0, dgx=70.0)),
        'imp3d': lambda: samples(imp3d(nshot=2, nrec=3, nt=200, dir=1, gxmin=50.0, dgx=70.0)),
        'syncz': lambda: samples(syncz(ntr=8, nt=200)),
        'goupillaudpo': lambda: samples(from_iterable(reflectivity_traces()) | goupillaudpo(l=3, k=9)),
        'synlvcw': lambda: samples(synlvcw(nt=101, nxm=4, nxo=3, dxo=0.2, gamma=0.7)),
        'synlvfti': lambda: samples(synlvfti(nt=101, nxm=3, nxo=3, dxo=0.3, fxo=0.1, delta=0.2, epsilon=0.2)),
        'synvxz': lambda: samples(synvxz(VXZ_V, **VXZ)),
        'synvxzcs': lambda: samples(synvxzcs(VXZ_V, **VXZCS)),
        'kdsyn2d': lambda: samples(kdsyn2d(KD_MIG, KD_TTAB, **KD)),
        'goupillaud': lambda: samples(from_iterable(reflectivity_traces()) | goupillaud(l=1, k=6)),
        'addevent': lambda: samples(from_iterable(gather([0.0, 300.0, 600.0])) | addevent(t0=0.2, vel=2000.0)),
    }
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for name, make in work.items():
            expected = make()
            for got in pool.map(lambda _: make(), range(16)):
                npt.assert_array_equal(got, expected, err_msg=name)


# ------------------------------------------------------------------------------------------------------- goupillaud
def simulate_layers(r, l, k, steps, pV):
    """The waves in the layers 1 .. n+1 of a Goupillaud medium, by brute force, stepped by the one-way time z: the field at the top of
    layer k at the times 0, z, 2z, ....  The interface i scatters with the coefficients r_i (r_i is pV times the input): reflection
    r_i of the downgoing, -r_i of the upgoing, transmission 1 + r_i down and 1 - r_i up; the surface reflects the upgoing with -r_0.
    The source is at the top of layer l: a downgoing spike 1, and an upgoing spike of -pV (in a surface source, only the first)."""
    n = len(r) - 1
    reff = [pV * v for v in r]
    kk = min(k, n + 1)
    down_before = np.zeros(n + 3)
    up_now = np.zeros(n + 3)
    field = []
    for t in range(steps):
        down = np.zeros(n + 3)
        up_next = np.zeros(n + 3)
        up_arriving = up_now.copy()
        source_down = np.zeros(n + 3)
        if t == 0:
            source_down[l] = 1.0
            if l > 1:
                up_arriving[l] += -pV
        down[1] = -reff[0] * up_arriving[1] + source_down[1]
        for i in range(1, n + 1):
            d_arriving = down_before[i]
            u_arriving = up_arriving[i + 1]
            down[i + 1] = (1 + reff[i]) * d_arriving - reff[i] * u_arriving + source_down[i + 1]
            up_next[i] = reff[i] * d_arriving + (1 - reff[i]) * u_arriving
        field.append(down[kk] + up_arriving[kk])
        down_before = down
        up_now = up_next
    field = np.array(field)
    if k > n + 1:  # the half-space: the wave of the last layer, delayed
        delay = k - (n + 1)
        delayed = np.zeros(steps)
        delayed[delay:] = simulate_layers(r, l, n + 1, steps, pV)[:steps - delay]
        return delayed
    return field


def run_goupillaud(r, **kwargs):
    (trace,) = list(from_iterable([Trace(np.asarray(r, dtype=np.float32), d_sample=0.004)]) | goupillaud(**kwargs))
    return trace


def brute_force(r, l, k, pV, length):
    x = simulate_layers(r, l, k, 2 * length + 2, pV)
    return x[1::2][:length] if (k - l) % 2 else x[0::2][:length]


@pytest.mark.parametrize('pV', [1, -1])
@pytest.mark.parametrize('k', [1, 2, 3, 5, 7, 8, 11])  # (7 layers + the half-space: 8 and 11 are in the half-space)
def test_goupillaud_surface_source_matches_a_simulation_of_the_layers(k, pV):
    r = np.random.default_rng(4).uniform(-0.5, 0.5, size=7).astype(np.float32)
    trace = run_goupillaud(r, l=1, k=k, pV=pV, tmax=16)
    npt.assert_allclose(np.asarray(trace), brute_force(r.astype(np.float64), 1, k, pV, 16), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('pV', [1, -1])
@pytest.mark.parametrize('l, k', [(2, 1), (4, 1), (4, 2), (4, 3), (7, 3), (7, 1), (7, 5)])
def test_goupillaud_receiver_above_a_buried_source_matches_a_simulation_of_the_layers(l, k, pV):
    r = np.random.default_rng(4).uniform(-0.5, 0.5, size=7).astype(np.float32)
    trace = run_goupillaud(r, l=l, k=k, pV=pV, tmax=16)
    npt.assert_allclose(np.asarray(trace), brute_force(r.astype(np.float64), l, k, pV, 16), atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('pV', [1, -1])
@pytest.mark.parametrize('l, k', [(1, 1), (1, 4), (3, 1), (5, 2)])
def test_goupillaud_is_the_primaries_plus_the_multiples(l, k, pV):
    # where the program is exact, the response is that of goupillaudpo (primaries) plus second order terms in the coefficients
    base = np.random.default_rng(9).uniform(-1, 1, size=8)
    gaps = []
    for scale in (0.02, 0.002):
        r = (scale * base).astype(np.float32)
        (full,) = list(from_iterable([Trace(r, d_sample=0.004)]) | goupillaud(l=l, k=k, pV=pV, tmax=12))
        (primaries,) = list(from_iterable([Trace(r, d_sample=0.004)]) | goupillaudpo(l=l, k=k, pV=pV, tmax=12))
        gaps.append(np.abs(np.asarray(full, dtype=float) - np.asarray(primaries, dtype=float)).max())
    assert gaps[0] / gaps[1] > 50  # (a second order difference falls by 100 when r falls by 10, a first order one by 10)


def test_goupillaud_spike_to_the_first_primary():
    (trace,) = list(from_iterable([Trace(np.array([0.0, 0.5, 0.0], dtype=np.float32), d_sample=0.004)]) | goupillaud(tmax=5))
    x = np.asarray(trace)
    assert x[0] == pytest.approx(1.0) and x[1] == pytest.approx(0.5)  # (the direct wave, and the reflection of the first interface)
    assert trace.header['sample_start'] == 0.0 and trace.header['trace_type'] == 1


def test_goupillaud_default_length_and_the_half_sample_shift():
    r = np.random.default_rng(1).uniform(-0.3, 0.3, size=6).astype(np.float32)  # n = 5
    assert run_goupillaud(r, l=1, k=1).n_sample == (2 * 5 + 2 - 0 - 0) // 2
    assert run_goupillaud(r, l=1, k=4).n_sample == (2 * 5 + 2 - 0 - 3) // 2
    assert run_goupillaud(r, l=1, k=9).n_sample == 9  # (a receiver below the layers: k)
    odd = run_goupillaud(r, l=1, k=2, tmax=6)
    even = run_goupillaud(r, l=1, k=3, tmax=6)
    assert odd.header['sample_start'] == pytest.approx(0.002) and even.header['sample_start'] == 0.0


@pytest.mark.parametrize('pV', [1, -1])
@pytest.mark.parametrize('l, k', [(2, 2), (3, 3), (2, 4), (3, 6), (3, 5), (7, 7), (7, 12), (4, 4), (2, 9), (5, 8)])
def test_goupillaud_receiver_at_or_below_a_buried_source_matches_a_simulation_of_the_layers(l, k, pV):
    r = np.random.default_rng(4).uniform(-0.5, 0.5, size=7).astype(np.float32)
    trace = run_goupillaud(r, l=l, k=k, pV=pV, tmax=16)
    ref = brute_force(r.astype(np.float64), l, k, pV, 16)
    if k == l:  # (just below and just above the source: the direct spikes 1 and -pV average)
        ref[0] -= 0.5 * (1 - pV)
    npt.assert_allclose(np.asarray(trace), ref, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize('pV', [1, -1])
def test_goupillaud_at_the_source_has_the_documented_start(pV):
    r = np.random.default_rng(6).uniform(-0.3, 0.3, size=6).astype(np.float32)
    x = np.asarray(run_goupillaud(r, l=3, k=3, pV=pV, tmax=6))
    # the direct spikes 1 and -pV average to (1 - pV) / 2 (displacement 0, pressure 1); the reflection of the upgoing one
    # from the interface above the source is there at once
    assert x[0] == pytest.approx((1 - pV) / 2 + r[2], abs=1e-6)


@pytest.mark.parametrize('pV', [1, -1])
@pytest.mark.parametrize('l, k', [(3, 3), (3, 6), (2, 7), (5, 5)])
def test_goupillaud_buried_source_is_consistent_with_the_primaries(l, k, pV):
    base = np.random.default_rng(9).uniform(-1, 1, size=8)
    gaps = []
    for scale in (0.02, 0.002):
        r = (scale * base).astype(np.float32)
        (full,) = list(from_iterable([Trace(r, d_sample=0.004)]) | goupillaud(l=l, k=k, pV=pV, tmax=12))
        (prim,) = list(from_iterable([Trace(r, d_sample=0.004)]) | goupillaudpo(l=l, k=k, pV=pV, tmax=12))
        a, b = np.asarray(full, dtype=float), np.asarray(prim, dtype=float)
        if k == l:
            a = a[1:]; b = b[1:]  # (the direct spike is on its own)
        gaps.append(np.abs(a - b).max())
    assert gaps[0] / gaps[1] > 50


def test_goupillaud_checks_its_parameters_and_input():
    r = np.random.default_rng(3).uniform(-0.4, 0.4, size=6).astype(np.float32)
    for bad in (dict(pV=0), dict(k=0), dict(l=0), dict(tmax=-1)):
        with pytest.raises(ValueError):
            goupillaud(**bad)
    with pytest.raises(ValueError, match="l<=n\\+1"):
        run_goupillaud(r, l=7)
    with pytest.raises(ValueError, match="n must be >=1"):
        run_goupillaud(np.array([0.3]))
    with pytest.raises(ValueError, match="Invalid reflection coefficient"):
        run_goupillaud(np.array([0.0, 1.5, 0.1]))
    assert goupillaud().parallelism == 'trace'
    # every trace is a reflectivity series that gets its seismogram, and the input is not changed
    traces = [Trace(np.random.default_rng(s).uniform(-0.4, 0.4, size=6).astype(np.float32), d_sample=0.004) for s in range(3)]
    before = [np.asarray(t).copy() for t in traces]
    out = list(from_iterable(traces) | goupillaud(l=1, k=3))
    assert len(out) == 3
    for t, b in zip(traces, before):
        npt.assert_array_equal(np.asarray(t), b)


# ------------------------------------------------------------------------------------------------------- synlvcw
FLAT = dict(nt=401, dt=0.004, nxm=1, nxo=1, dxm=0.1, v00=2.0, reflectors=[[1.0, (-5.0, 1.0), (5.0, 1.0)]], ls=True)


def test_synlvcw_with_gamma_one_is_synlv():
    kwargs = dict(nt=101, nxm=4, nxo=3, dxo=0.2)
    npt.assert_array_equal(samples(synlvcw(**kwargs)), samples(synlv(**kwargs)))
    npt.assert_array_equal(samples(synlvcw(gamma=1.0, **kwargs)), samples(synlv(**kwargs)))


@pytest.mark.parametrize('gamma', [0.5, 1.0, 2.0])
def test_synlvcw_zero_offset_time_is_down_at_v_and_up_at_gamma_v(gamma):
    # a flat reflector at 1 km in v = 2 km/s: down in 0.5 s, up in 0.5 / gamma s
    x = np.asarray(next(iter(synlvcw(gamma=gamma, **FLAT))))
    expected = 0.5 + 0.5 / gamma
    if expected >= 400 * 0.004:
        pytest.skip("beyond the record")
    assert abs(np.argmax(np.abs(x)) * 0.004 - expected) < 0.016


def test_synlvcw_amplitudes_change_with_gamma_and_sp():
    base = samples(synlvcw(**FLAT))
    assert not np.allclose(base, samples(synlvcw(gamma=0.8, **FLAT)))
    flat_amp = samples(synlvcw(sp=False, **FLAT))
    assert np.isfinite(flat_amp).all() and np.abs(flat_amp).max() > 0
    assert not np.allclose(flat_amp / np.abs(flat_amp).max(), base / np.abs(base).max())


def test_synlvcw_rejects_a_bad_gamma():
    with pytest.raises(ValueError, match="gamma"):
        synlvcw(gamma=0.0)


# ------------------------------------------------------------------------------------------------------- synlvfti
DIPPING = [[1.0, (-3.0, 0.8), (3.0, 1.0)]]


@pytest.mark.parametrize('extra', [{}, dict(dvdz=0.5), dict(dvdx=0.2)])
@pytest.mark.parametrize('nxo, fxo', [(4, 0.1), (1, 0.0)])
def test_synlvfti_of_an_isotropic_medium_is_synlv(extra, nxo, fxo):
    kwargs = dict(nt=401, dt=0.004, nxm=3, nxo=nxo, dxo=0.3, fxo=fxo, v00=2.0, reflectors=DIPPING, ob=True, **extra)
    npt.assert_array_equal(samples(synlvfti(**kwargs)), samples(synlv(**kwargs)))


def test_synlvfti_does_not_use_the_obliquity_factors_unless_asked():
    kwargs = dict(nt=401, dt=0.004, nxm=2, nxo=3, dxo=0.3, fxo=0.1, reflectors=DIPPING)
    npt.assert_array_equal(samples(synlvfti(**kwargs)), samples(synlv(ob=False, **kwargs)))
    assert not np.array_equal(samples(synlvfti(ob=True, **kwargs)), samples(synlvfti(**kwargs)))


@pytest.mark.parametrize('eps', [0.1, 0.2, -0.1])
def test_synlvfti_elliptical_medium_has_the_hyperbolic_moveout_of_the_nmo_velocity(eps):
    # delta = epsilon: a flat reflector at 1 km in a v = 2 km/s medium has t^2 = t0^2 + x^2 / (v^2 (1 + 2 delta))
    offsets = np.array([0.0, 0.4, 0.8, 1.2, 1.6])
    traces = samples(synlvfti(nt=401, dt=0.004, nxm=1, xo=offsets, v00=2.0, ls=True, delta=eps, epsilon=eps,
                              reflectors=[[1.0, (-8.0, 1.0), (8.0, 1.0)]]))
    got = np.abs(traces).argmax(axis=1)
    expected = np.round(np.sqrt(1.0 + (offsets / (2.0 * math.sqrt(1 + 2 * eps))) ** 2) / 0.004)
    npt.assert_allclose(got, expected, atol=1)


def test_synlvfti_a_f_l_and_thomsens_parameters_are_the_same_medium():
    kwargs = dict(nt=401, dt=0.004, nxm=1, xo=[0.0, 0.8], v00=2.0, ls=True, reflectors=[[1.0, (-8.0, 1.0), (8.0, 1.0)]])
    eps, delta, l = 0.2, 0.1, 0.3
    a = 1 + 2 * eps
    f = math.sqrt(2 * delta * (1 - l) + (1 - l) ** 2) - l
    npt.assert_allclose(samples(synlvfti(a=a, f=f, l=l, **kwargs)), samples(synlvfti(delta=delta, epsilon=eps, **kwargs)), atol=1e-4)


def test_synlvfti_the_symmetry_axis_matters():
    kwargs = dict(nt=401, dt=0.004, nxm=1, xo=[0.0, 0.8, 1.2], v00=2.0, ls=True, epsilon=0.2, delta=0.2,
                  reflectors=[[1.0, (-8.0, 1.0), (8.0, 1.0)]])
    assert not np.allclose(samples(synlvfti(angxs=30.0, **kwargs)), samples(synlvfti(**kwargs)), atol=1e-3)
    base = samples(synlvfti(**kwargs))  # (the rotation by half a turn is done in single precision and the ray search stops at epsx)
    npt.assert_allclose(samples(synlvfti(angxs=180.0, **kwargs)), base, atol=0.01 * np.abs(base).max())


def test_synlvfti_rejects_bad_iteration_counts():
    with pytest.raises(ValueError):
        synlvfti(ntries=0)
    with pytest.raises(ValueError):
        synlvfti(nitmax=0)


# -------------------------------------------------------------------------------------------------------- synvxz
VXZ_V = np.full((60, 80), 2000.0, dtype=np.float32)  # (nz, nx), on a 25 m grid: 2 km deep and wide
VXZ = dict(dx=25.0, dz=25.0, nt=401, dt=0.004, nxo=3, dxo=100.0, nxm=7, dxm=50.0, fxm=500.0, ls=True,
           reflectors=[[1.0, (0.0, 1000.0), (2000.0, 1000.0)]])


def test_synvxz_traveltimes_in_constant_velocity_are_the_hyperbola():
    traces = samples(synvxz(VXZ_V, **VXZ)).reshape(3, 7, -1)
    for io, xo in enumerate((0.0, 100.0, 200.0)):
        expected = math.sqrt(1.0 + (xo / 2000.0) ** 2) / 0.004  # (a flat reflector 1 km deep, v = 2 km/s)
        got = np.abs(traces[io]).argmax(axis=1)
        npt.assert_allclose(got, expected, atol=1.5)


def test_synvxz_order_and_headers():
    stream = synvxz(VXZ_V, **VXZ)
    assert stream.n_traces == 21
    traces = list(stream)
    assert len(traces) == 21
    # (each offset in turn, with its midpoints)
    assert [t.header['ensemble_number'] for t in traces[:8]] == [1, 2, 3, 4, 5, 6, 7, 1]
    assert [t.header['ensemble_trace_number'] for t in traces[::7]] == [1, 2, 3]
    t = traces[8]  # the second midpoint, the offset 100
    assert t.header['tx_loc'][0] == pytest.approx(550.0 - 50.0) and t.header['rx_loc'][0] == pytest.approx(550.0 + 50.0)
    assert t.header['d_sample'] == pytest.approx(0.004) and t.n_sample == 401


def test_synvxz_skipping_midpoints_interpolates_the_traveltimes():
    full = samples(synvxz(VXZ_V, **VXZ))
    skipped = samples(synvxz(VXZ_V, nxd=3, **VXZ))
    assert np.abs(full - skipped).max() < 0.01 * np.abs(full).max()


def test_synvxz_agrees_with_synlv_in_a_constant_velocity():
    # the same medium in km (synlv takes the midpoint outermost, and has other units of amplitude)
    vxz = samples(synvxz(VXZ_V, **VXZ))
    lv = samples(synlv(nt=401, dt=0.004, nxm=7, dxm=0.05, fxm=0.5, nxo=3, dxo=0.1, v00=2.0, ls=True, ob=True,
                       reflectors=[[1.0, (0.0, 1.0), (2.0, 1.0)]]))
    for io in range(3):
        for im in range(7):
            a, b = vxz[io * 7 + im], lv[im * 3 + io]
            assert np.corrcoef(a, b)[0, 1] > 0.99
            assert a.max() / b.max() == pytest.approx(0.5, rel=0.15)  # (the same, in other units)


def test_synvxz_deeper_reflector_comes_later_and_faster_medium_earlier():
    slow = np.asarray(next(iter(synvxz(VXZ_V, **{**VXZ, 'nxo': 1, 'nxm': 1, 'reflectors': [[1.0, (0.0, 600.0), (2000.0, 600.0)]]}))))
    fast_v = np.full((60, 80), 3000.0, dtype=np.float32)
    fast = np.asarray(next(iter(synvxz(fast_v, **{**VXZ, 'nxo': 1, 'nxm': 1, 'reflectors': [[1.0, (0.0, 600.0), (2000.0, 600.0)]]}))))
    assert np.abs(slow).argmax() * 0.004 == pytest.approx(0.6, abs=0.012)
    assert np.abs(fast).argmax() * 0.004 == pytest.approx(0.4, abs=0.012)


def test_synvxz_checks_its_input():
    with pytest.raises(ValueError, match="outside"):
        synvxz(VXZ_V, **{**VXZ, 'fxm': -500.0})
    with pytest.raises(ValueError, match="band"):
        synvxz(VXZ_V, **{**VXZ, 'nxb': 2})
    with pytest.raises(ValueError, match="2-D"):
        synvxz(np.zeros(5))
    with pytest.raises(ValueError):
        synvxz(VXZ_V, nxd=0)


# ------------------------------------------------------------------------------------------------------ synvxzcs
VXZCS = dict(dx=25.0, dz=25.0, nt=401, dt=0.004, nxg=9, dxg=100.0, fxg=0.0, nxs=2, dxs=100.0, fxs=600.0, nxb=30, nxd=3, ls=True,
             reflectors=[[1.0, (0.0, 1000.0), (2000.0, 1000.0)]])


def test_synvxzcs_traveltimes_in_constant_velocity_are_the_hyperbola():
    traces = samples(synvxzcs(VXZ_V, **VXZCS)).reshape(2, 9, -1)
    for ishot, xs in enumerate((600.0, 700.0)):
        offsets = np.arange(9) * 100.0 - xs  # (the spread rolls with the shot, so that the receivers are at 0 .. 800 + 100 ishot)
        offsets = np.arange(9) * 100.0 + 100.0 * ishot - xs
        expected = np.sqrt(1.0 + (offsets / 2000.0) ** 2) / 0.004
        npt.assert_allclose(np.abs(traces[ishot]).argmax(axis=1), expected, atol=1.5)


def test_synvxzcs_order_and_headers():
    stream = synvxzcs(VXZ_V, **VXZCS)
    assert stream.n_traces == 18
    traces = list(stream)
    assert len(traces) == 18
    assert [t.header['ensemble_number'] for t in traces[::9]] == [1, 2]
    assert [t.header['ensemble_trace_number'] for t in traces[:9]] == list(range(1, 10))
    assert traces[9].header['tx_loc'][0] == pytest.approx(700.0) and traces[9].header['rx_loc'][0] == pytest.approx(100.0)
    fixed = list(synvxzcs(VXZ_V, cable=False, **VXZCS))
    assert fixed[9].header['rx_loc'][0] == pytest.approx(0.0)  # (the receivers do not roll along)


def test_synvxzcs_skipping_receivers_interpolates_the_traveltimes():
    full = samples(synvxzcs(VXZ_V, **{**VXZCS, 'nxd': 1}))
    skipped = samples(synvxzcs(VXZ_V, **VXZCS))
    assert np.abs(full - skipped).max() < 0.02 * np.abs(full).max()


def test_synvxzcs_is_synvxz_with_the_same_shots_and_receivers():
    # in a constant velocity a shot and a receiver are the offset and the midpoint of a common-offset trace
    cs = samples(synvxzcs(VXZ_V, **{**VXZCS, 'nxs': 1, 'nxd': 1, 'nxb': 40}))
    xs = 600.0
    for ig, xg in enumerate(np.arange(9) * 100.0):
        xo, xm = xg - xs, 0.5 * (xg + xs)
        co = samples(synvxz(VXZ_V, dx=25.0, dz=25.0, nt=401, dt=0.004, xo=[xo], nxm=1, fxm=xm, ls=True,
                            reflectors=VXZCS['reflectors']))[0]
        assert np.corrcoef(cs[ig], co)[0, 1] > 0.99
        assert cs[ig].max() / co.max() == pytest.approx(1.0, rel=0.1)


def test_synvxzcs_zero_perturbation_and_corners_leave_a_constant_medium_alone():
    base = samples(synvxzcs(VXZ_V, **VXZCS))
    npt.assert_allclose(samples(synvxzcs(VXZ_V, pert=np.zeros_like(VXZ_V), **VXZCS)), base, atol=1e-5 * np.abs(base).max())
    npt.assert_allclose(samples(synvxzcs(VXZ_V, nxc=4, nzc=3, **VXZCS)), base, atol=1e-5 * np.abs(base).max())


def test_synvxzcs_perturbation_of_the_slowness_changes_the_traveltimes():
    pert = np.full_like(VXZ_V, 2.0e-5)
    changed = samples(synvxzcs(VXZ_V, pert=pert, **VXZCS))
    base = samples(synvxzcs(VXZ_V, **VXZCS))
    assert not np.allclose(changed, base, atol=1e-3 * np.abs(base).max())


def test_synvxzcs_checks_its_input():
    with pytest.raises(ValueError, match="shot"):
        synvxzcs(VXZ_V, **{**VXZCS, 'fxs': -100.0})
    with pytest.raises(ValueError, match="receiver"):
        synvxzcs(VXZ_V, **{**VXZCS, 'nxg': 40})
    with pytest.raises(ValueError, match="shape"):
        synvxzcs(VXZ_V, pert=np.zeros((3, 3)), **VXZCS)
    with pytest.raises(ValueError):
        synvxzcs(VXZ_V, **{**VXZCS, 'nxs': 0})


# ------------------------------------------------------------------------------------------------------- kdsyn2d
KD_V = 2000.0
_x = np.arange(101) * 25.0
_z = np.arange(81) * 25.0
_srcs = np.arange(26) * 100.0
KD_TTAB = (np.sqrt((_x[None, None, :] - _srcs[:, None, None]) ** 2 + _z[None, :, None] ** 2) / KD_V).astype(np.float32)  # (ns, nzt, nxt)
KD_MIG = np.zeros((81, 101), dtype=np.float32)
KD_MIG[40, :] = 1.0  # a flat reflector at z = 1000
KD = dict(dx=25.0, dz=25.0, dxt=25.0, dzt=25.0, fs=0.0, ds=100.0, nt=501, dt=0.004, nxo=3, dxo=200.0, fxo=0.0, nxs=3, dxs=300.0,
          fxs=600.0, v0=KD_V)


def test_kdsyn2d_reflector_arrives_at_the_specular_time():
    traces = list(kdsyn2d(KD_MIG, KD_TTAB, **KD))
    assert len(traces) == 9
    for tr in traces:
        xo = tr.header['rx_loc'][0] - tr.header['tx_loc'][0]
        expected = math.sqrt(xo ** 2 + 4 * 1000.0 ** 2) / KD_V
        assert np.abs(np.asarray(tr)).argmax() * 0.004 == pytest.approx(expected, abs=0.006)


def test_kdsyn2d_order_headers_and_the_shot_gathers_of_a_flat_section_are_alike():
    stream = kdsyn2d(KD_MIG, KD_TTAB, **KD)
    assert stream.n_traces == 9
    traces = list(stream)
    assert [t.header['ensemble_number'] for t in traces[::3]] == [1, 2, 3]
    assert [t.header['ensemble_trace_number'] for t in traces[:3]] == [1, 2, 3]
    assert [t.header['tx_loc'][0] for t in traces[::3]] == [600.0, 900.0, 1200.0]
    assert [t.header['rx_loc'][0] for t in traces[:3]] == [600.0, 800.0, 1000.0]
    x = samples(traces).reshape(3, 3, -1)
    # (a flat reflector in a constant velocity: the reflection is the same for every shot; the edges of the section are not)
    npt.assert_allclose(x[1][:, 230:290], x[2][:, 230:290], atol=0.02 * np.abs(x).max())


def test_kdsyn2d_is_linear_in_the_section_and_a_zero_section_gives_zero():
    base = samples(kdsyn2d(KD_MIG, KD_TTAB, **KD))
    npt.assert_allclose(samples(kdsyn2d(3.0 * KD_MIG, KD_TTAB, **KD)), 3.0 * base, rtol=1e-4, atol=1e-5 * np.abs(base).max())
    assert not np.any(samples(kdsyn2d(np.zeros_like(KD_MIG), KD_TTAB, **KD)))
    npt.assert_array_equal(samples(kdsyn2d(KD_MIG, KD_TTAB, **KD)), base)  # (the inputs are not changed, and it is repeatable)


def test_kdsyn2d_checks_its_input():
    with pytest.raises(ValueError, match="3-D"):
        kdsyn2d(KD_MIG, KD_TTAB[0], **KD)
    with pytest.raises(ValueError, match="two sources"):
        kdsyn2d(KD_MIG, KD_TTAB[:1], **KD)
    with pytest.raises(ValueError, match="outside"):
        kdsyn2d(KD_MIG, KD_TTAB, **{**KD, 'fxs': 2600.0})
    with pytest.raises(ValueError, match="traveltime table"):
        kdsyn2d(KD_MIG, KD_TTAB, **{**KD, 'fx': 100.0})
    with pytest.raises(ValueError, match="angmax"):
        kdsyn2d(KD_MIG, KD_TTAB, **{**KD, 'angmax': 0.0})
