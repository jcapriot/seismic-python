"""The stages that call the SU library give the same traces when they are split across threads (the functions have no state)"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy import attributes
from seispy.amplitudes import centsamp, dipdivcor
from seispy.container import from_iterable
from seispy.convolution import acorfrac, conv
from seispy.decon import cddecon, fxdecon, phidecon
from seispy.filters import frac, minphase, phase, tvband
from seispy.parallel import pmap
from seispy.stretching import dlmo, ilog, log, nmo_a, reduce, taupnmo, ttoz
from seispy.synthetics import synlv
from seispy.tapering import ramp, taper, txtaper
from seispy.interp import interp
from seispy.velocity import relanan, vel2df, velan_nsel, velan_uccs
from seispy.transforms import ccwt, cepstrum, clogfft, iclogfft, radon, taup, wfft
from seispy.windowing import mute

FILTER = [0.0, 0.4, 0.2, 0.5, 0.5]


def samples(traces):
    return np.array([np.asarray(t) for t in traces])


STAGES = {
    'taper': lambda: taper(100.0, 100.0, type=2),
    'ramp': lambda: ramp(tmin=0.4, tmax=3.0),
    'mute': lambda: mute([0.0, 1.0], [0.1, 0.5], ntaper=4),
    'amp': lambda: attributes.amp(),
    'freq': lambda: attributes.freq(),
    'q': lambda: attributes.q(),
    'conv': lambda: conv(np.array(FILTER, dtype=np.float32)),
    'centsamp': lambda: centsamp(),
    'frac': lambda: frac(power=0.5),
    'phase': lambda: phase(a=30.0),
    'minphase': lambda: minphase(),
    'tvband': lambda: tvband([0.5], [5.0, 10.0, 15.0, 20.0]),
    'acorfrac': lambda: acorfrac(a=0.5, b=0.5),
    'wfft': lambda: wfft(),
    'cepstrum': lambda: cepstrum(unwrap=0),
    'log': lambda: log(ntmin=5),
    'taupnmo': lambda: taupnmo(vnmo=2000.0, p=lambda trace: 0.1 / 2000.0),
    'clogfft': lambda: clogfft(unwrap=0) | iclogfft(),
    'nmo_a': lambda: nmo_a(vnmo=2000.0, anis1=1e-18, anis2=0.0),
    'dlmo': lambda: dlmo(vnmo=[400.0, 800.0], fnmo=[0.0, 40.0]),
    'cddecon': lambda: cddecon(np.array(FILTER, dtype=np.float32), pnoise=0.01),
    'phidecon': lambda: phidecon(),
    'ccwt': lambda: ccwt(noct=2, nv=3),
    'txtaper': lambda: txtaper(tbeg=20.0, tend=20.0, key='offset', min=100.0, max=800.0, dx=100.0),
}


@pytest.mark.parametrize('name', sorted(STAGES))
def test_threads_give_the_same_traces(name):
    stage = STAGES[name]()
    expected = samples(synlv(nt=101, nxm=40) | stage)
    got = samples(synlv(nt=101, nxm=40) | pmap(stage, workers=4, chunk=3))
    npt.assert_array_equal(got, expected)


PANEL_STAGES = {
    # (the stages of a panel are not split across workers, but several threads can each transform a panel)
    'taup_fk': lambda: taup(1, dx=10.0, pmin=0.0, pmax=0.004, np=11),
    'taup_tx': lambda: taup(2, dx=10.0, pmin=0.0, pmax=0.004, np=11),
    'dipdivcor': lambda: dipdivcor(25.0, np=10, vmig=2000.0),
    'fxdecon': lambda: fxdecon(ntrw=6, ntrf=2),
    'vel2df': lambda: vel2df(tn=0.2, nv=4, nvh=3),
    'interp': lambda: interp(ninterp=1, xopt=1),
    'relanan': lambda: relanan(x=[1.0, 10.0], z=[5.0, 5.0], nr1=3, nr2=3, dzratio=2),
    'velan_uccs': lambda: velan_uccs(nv=3),
    'velan_nsel': lambda: velan_nsel(nv=3),
    'radon': lambda: radon(choose=3, nwin=2, muts=lambda trace: 200.0),
}


@pytest.mark.parametrize('name', sorted(PANEL_STAGES))
def test_threads_can_each_transform_a_panel(name):
    import concurrent.futures

    stage = PANEL_STAGES[name]()
    panel = list(synlv(nt=101, nxm=12))
    expected = samples(from_iterable(panel) | stage)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda _: samples(from_iterable(panel) | stage), range(16)))
    for got in results:
        npt.assert_array_equal(got, expected)


def test_threads_can_each_analyse_three_component_stations():
    import concurrent.futures

    from seispy.multicomponent import alford, eipofi, polar

    panel = list(synlv(nt=120, nxm=12))
    stations = [panel[i:i + 3] for i in range(0, 12, 3)]
    flat = [t for s in stations for t in s]
    four = [panel[i:i + 4] for i in range(0, 12, 4)]
    flat4 = [t for s in four for t in s]

    def attributes(_):
        out = polar(flat, wl=0.1, theta=2, phi=3, tau=1, amp=1)
        return {k: np.array([np.asarray(t) for t in v]) for k, v in out.items()}

    def filtered(_):
        return samples(from_iterable(flat) | eipofi(wl=0.1))

    def rotated(_):
        return samples(from_iterable(flat4) | alford(ntraces=1, angle_inc=10.0))

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for function in (attributes, filtered, rotated):
            expected = function(0)
            for got in pool.map(function, range(8)):
                if isinstance(expected, dict):
                    assert got.keys() == expected.keys()
                    for k in expected:
                        npt.assert_allclose(got[k], expected[k], rtol=1e-4, atol=1e-6)
                else:
                    # (the float sums of the same code differ in the last bits on some machines: see the notes on the tests)
                    npt.assert_allclose(got, expected, rtol=1e-4, atol=1e-6 * max(1.0, float(np.abs(expected).max())))


def test_threads_can_each_estimate_residual_statics():
    import concurrent.futures

    from seispy.statics import resstat

    panel = list(synlv(nt=101, nxm=12))
    keyed = [t.replace(ensemble_number=i % 3 + 1, ensemble_trace_number=i // 3 + 1) for i, t in enumerate(panel)]
    expected = resstat(keyed, ntpick=5, niter=2, cmp=lambda t: t.header['ensemble_trace_number'] % 2)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        results = list(pool.map(
            lambda _: resstat(keyed, ntpick=5, niter=2, cmp=lambda t: t.header['ensemble_trace_number'] % 2), range(16)
        ))
    assert all(r == expected for r in results)


def test_threads_can_each_datum_gathers():
    import concurrent.futures

    from seispy.datuming import datumfd

    panel = list(synlv(nt=101, nxm=12))
    vel = np.full((12, 6), 2200.0, np.float32)
    keyed = [t.replace(rx_loc=[10.0 * i, 0.0, 0.0]) for i, t in enumerate(panel)]

    def datumed(_):
        return samples(from_iterable(keyed) | datumfd(vel, vel, nx=6, nsx=2, nz=4, dz=8.0, dx=10.0))

    expected = datumed(0)
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for got in pool.map(datumed, range(8)):
            npt.assert_allclose(got, expected, rtol=1e-4, atol=1e-6 * max(1.0, float(np.abs(expected).max())))
