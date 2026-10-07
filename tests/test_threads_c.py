"""The stages that call the SU library give the same traces when they are split across threads (the functions have no state)"""
import numpy as np
import numpy.testing as npt
import pytest

from seispy import attributes
from seispy.amplitudes import centsamp
from seispy.convolution import acorfrac, conv
from seispy.filters import frac, minphase, phase, tvband
from seispy.parallel import pmap
from seispy.stretching import ilog, log, reduce, ttoz
from seispy.synthetics import synlv
from seispy.tapering import ramp, taper
from seispy.transforms import cepstrum, clogfft, iclogfft, wfft
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
    'clogfft': lambda: clogfft(unwrap=0) | iclogfft(),
}


@pytest.mark.parametrize('name', sorted(STAGES))
def test_threads_give_the_same_traces(name):
    stage = STAGES[name]()
    expected = samples(synlv(nt=101, nxm=40) | stage)
    got = samples(synlv(nt=101, nxm=40) | pmap(stage, workers=4, chunk=3))
    npt.assert_array_equal(got, expected)
