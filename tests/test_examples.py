"""The examples in examples/cookbook.py (and the README) run, and do what they say"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parent.parent / "examples" / "cookbook.py"


@pytest.fixture(scope="module")
def cookbook():
    spec = importlib.util.spec_from_file_location("cookbook", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deconvolution_makes_a_spikier_autocorrelation(cookbook):
    reflectivity, seismogram, deconvolved = cookbook.deconvolve_noisy_spikes()
    before = cookbook.sidelobe_energy(seismogram)
    after = cookbook.sidelobe_energy(deconvolved)
    assert (after < 0.2 * before).all()
    # the reflectivity itself is nearly a spike in autocorrelation
    assert cookbook.sidelobe_energy(reflectivity).mean() < 0.5


def test_cmp_gathers(cookbook):
    gathers = cookbook.cmp_gathers_to_nmo()
    assert len(gathers) == 5 and all(g.shape == (6, 251) for g in gathers)
    assert all(np.isfinite(g).all() for g in gathers)


def test_wavelet_spectrum_peaks_near_its_frequency(cookbook):
    peak, spectrum = cookbook.spectrum_of_a_wavelet()
    assert peak == pytest.approx(30.0, abs=3.0)
    assert spectrum.dtype == np.float32  # (an amplitude spectrum is real)


def test_echo_is_found_in_the_cepstrum(cookbook):
    assert cookbook.find_an_echo_in_the_cepstrum() == pytest.approx(0.08)


def test_sweep_frequency_follows_the_panel(cookbook):
    f_start, f_end, gabor_shape = cookbook.time_frequency_of_a_sweep()
    assert f_start == pytest.approx(10.0 + 35.0 * 0.24, abs=3.0)
    assert f_end == pytest.approx(10.0 + 35.0 * 1.76, abs=3.0)
    assert gabor_shape[1] == 501


def test_your_own_stage_and_headers(cookbook):
    data = cookbook.your_own_stage()
    assert data.shape[0] == 101 and set(np.unique(data)) <= {-1.0, 0.0, 1.0}
    offset, moved_offset, rx, trace_id = cookbook.headers()
    assert offset == 250.0 and moved_offset == -500.0 and rx == [-400.0, 0.0, 12.0] and trace_id == 3
