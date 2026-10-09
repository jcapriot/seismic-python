"""The stages that call the SU library give the same traces when they are split across threads (the functions have no state)"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy import attributes
from seispy.amplitudes import centsamp, dipdivcor
from seispy.container import from_iterable
from seispy.convolution import acorfrac, conv
from seispy.filters import frac, minphase, phase, tvband
from seispy.parallel import pmap
from seispy.stretching import ilog, log, reduce, taupnmo, ttoz
from seispy.synthetics import synlv
from seispy.tapering import ramp, taper
from seispy.transforms import cepstrum, clogfft, iclogfft, taup, wfft
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
