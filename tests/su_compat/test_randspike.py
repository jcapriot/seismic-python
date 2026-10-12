"""surandspike: the same random numbers, and the spikes of seispy are one sample earlier than SU's (on purpose)."""
import numpy as np
import pytest

from harness import parse_su, run_ours


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("n1", [100, 37])
def test_spikes_are_those_of_su_one_sample_earlier(su, mode, n1):
    cmd = f"surandspike n1={n1} n2=8 nspk=5 seed=3 mode={mode} amax=0.5"
    ref_h, ref = parse_su(su.run(cmd))
    ours_h, ours = parse_su(
        run_ours(None, (("synthetics.randspike", dict(n1=n1, n2=8, nspk=5, seed=3, mode=mode, amax=0.5)),))
    )
    assert len(ref) == len(ours) == 8
    for a, b in zip(ours, ref):
        assert b[0] == 0.0  # (SU never uses the first sample)
        np.testing.assert_array_equal(a[:-1], b[1:])
        assert a[-1] == 0.0  # (and seispy never the last one)
    # the trace numbers and the sampling are the same
    np.testing.assert_array_equal(ours_h["tracl"], ref_h["tracl"])
    np.testing.assert_array_equal(ours_h["dt"], ref_h["dt"])
    np.testing.assert_array_equal(ours_h["ns"], ref_h["ns"])


def test_su_spikes_stay_inside_the_trace(su):
    # (the docstring of randspike once said that SU writes one of them past the end)
    n1 = 50
    _, traces = parse_su(su.run(f"surandspike n1={n1} n2=2000 nspk=20 seed=7"))
    samples = np.concatenate([np.flatnonzero(x) for x in traces])
    assert samples.min() == 1 and samples.max() == n1 - 1
