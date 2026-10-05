import numpy as np
import numpy.testing as npt

import seispy.filters
from seispy.filters import bfilt
from seispy.container import TraceCollection


def _sine_collection(freqs, n_tr=3, nt=1024, dt=0.002):
    t = np.arange(nt) * dt
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return TraceCollection(np.tile(sig, (n_tr, 1)), d_sample=dt)


def _amp_at(trace, f, dt=0.002):
    spec = np.abs(np.fft.rfft(np.asarray(trace)))
    freqs = np.fft.rfftfreq(len(trace), dt)
    return spec[np.argmin(np.abs(freqs - f))]


def test_only_stages_are_public():
    assert seispy.filters.__all__ == ['bfilt', 'filter']
    assert not hasattr(seispy.filters, 'butterworth_bandpass')


def test_bandpass_attenuates_outside_band():
    coll = _sine_collection([2.0, 40.0, 200.0])
    out = list(coll | bfilt(f_pass_low=20, f_stop_low=10, f_pass_high=80, f_stop_high=120))
    assert len(out) == 3
    for tr in out:
        mid = _amp_at(tr, 40.0)
        assert _amp_at(tr, 2.0) < 0.05 * mid
        assert _amp_at(tr, 200.0) < 0.05 * mid


def test_bandpass_not_inplace_preserves_input():
    coll = _sine_collection([40.0]).to_memory()
    before = [np.array(t) for t in coll]
    list(coll | bfilt())
    for b, t in zip(before, coll):
        npt.assert_equal(b, np.asarray(t))


def test_bandpass_inplace_modifies_input():
    coll = _sine_collection([2.0]).to_memory()
    before = [np.array(t) for t in coll]
    list(coll | bfilt(inplace=True))
    assert any(not np.array_equal(b, np.asarray(t)) for b, t in zip(before, coll))


def test_bandpass_chain_matches_to_memory():
    coll = _sine_collection([5.0, 40.0])
    a = [np.asarray(t) for t in coll | bfilt()]
    b = [np.asarray(t) for t in (coll | bfilt()).to_memory()]
    for x, y in zip(a, b):
        npt.assert_equal(x, y)
