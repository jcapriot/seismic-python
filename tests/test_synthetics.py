import numpy as np
import numpy.testing as npt
import pytest

from seispy.synthetics import plane, spike, synlv


def test_spike_default():
    traces = list(spike())
    assert len(traces) == 32
    data = np.array([np.asarray(t) for t in traces])
    assert data.shape == (32, 64)
    # Four unit spikes at the documented locations
    npt.assert_equal(
        np.argwhere(data == 1.0),
        [[8, 15], [8, 47], [24, 15], [24, 47]],
    )
    assert data.sum() == 4.0


def test_spike_custom_spikes():
    data = np.array([np.asarray(t) for t in spike(nt=8, ntr=4, spikes=[(2, 3)])])
    expected = np.zeros((4, 8), dtype=np.float32)
    expected[2, 3] = 1.0
    npt.assert_equal(data, expected)


def test_spike_headers():
    for i, tr in enumerate(spike(dt=0.002)):
        assert tr.d_sample == 0.002
        assert tr.header['trace_id'] == i + 1


def test_spike_is_single_pass():
    it = spike()
    assert len(list(it)) == 32
    assert len(list(it)) == 0


def test_plane_shape_and_energy():
    data = np.array([np.asarray(t) for t in plane()])
    assert data.shape == (32, 64)
    assert data.sum() > 0
    # linear interpolation conserves the unit amplitude of each hit
    assert np.all(data >= 0)


def test_plane_matches_to_memory():
    direct = np.array([np.asarray(t) for t in plane()])
    mem = np.array([np.asarray(t) for t in plane().to_memory()])
    npt.assert_equal(direct, mem)


def test_plane_bad_spec():
    with pytest.raises(TypeError):
        plane(planes=[(1, 2, 3)])


def test_synlv_runs():
    coll = synlv().to_memory()
    assert len(coll) > 0
    data = np.array([np.asarray(t) for t in coll])
    assert np.isfinite(data).all()
    assert np.abs(data).max() > 0
