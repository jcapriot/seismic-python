import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace, from_iterable
from seispy.transforms import cwt, gabor, st

DT = 0.004
N = 128


def traces(data, dt=DT, **per_trace):
    out = []
    for i, row in enumerate(np.atleast_2d(data)):
        t = Trace(np.asarray(row, dtype=np.float32), d_sample=dt)
        header = {k: (v[i] if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in per_trace.items()}
        out.append(t.replace(**header) if header else t)
    return out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


def tone(period, n=N):
    return np.cos(2 * np.pi * np.arange(n) / period)


# ------------------------------------------------------------------------------------------------------------- st
def test_st_of_a_tone_is_flat_in_time_at_its_frequency():
    k0 = 20
    panel = run(st(), traces(np.cos(2 * np.pi * k0 * np.arange(N) / N)))
    rows = arr(panel)
    # the first row is the first frequency after 0: row n is the frequency index n + 1
    assert rows.shape == (int(N / 2 + 1.5), N)
    on = rows[k0 - 1]
    npt.assert_allclose(on, 1.0, rtol=1e-6)  # (a unit tone has a unit amplitude)
    assert rows[k0 + 9].max() < 0.5 * on.min() and rows[k0 - 11].max() < 0.5 * on.min()


def test_st_headers_and_frequency_range():
    panel = run(st(fmin=20.0, fmax=60.0), traces(tone(16)[None, :], trace_id=7))
    d1 = 1.0 / (N * DT)
    first, last = int(20.0 / d1 + 1), int(60.0 / d1 + 1.5)
    assert len(panel) == last - first + 1
    assert {t.header['ensemble_number'] for t in panel} == {7}
    assert [t.header['ensemble_trace_number'] for t in panel] == list(range(1, len(panel) + 1))
    assert all(t.n_sample == N and t.d_sample == DT for t in panel)


def test_st_every_trace_is_a_panel():
    panel = run(st(fmin=10.0, fmax=20.0), traces(np.vstack([tone(16), tone(32)]), trace_id=[1, 2]))
    numbers = [t.header['ensemble_number'] for t in panel]
    assert numbers == sorted(numbers) and set(numbers) == {1, 2}


def test_st_needs_real_traces_and_a_sample_interval():
    with pytest.raises(TypeError):
        run(st(), [Trace((tone(16) + 0j).astype(np.complex64), d_sample=DT)])
    with pytest.raises(ValueError):
        run(st(), [Trace(tone(16).astype(np.float32), d_sample=0.0)])
    assert len(run(st(dt=DT), [Trace(tone(16).astype(np.float32), d_sample=0.0)])) > 0


# ---------------------------------------------------------------------------------------------------------- gabor
def test_gabor_filters_pick_out_the_frequency():
    x = np.cos(2 * np.pi * 40.0 * np.arange(512) * DT)
    panel = run(gabor(), traces(x))
    rows = arr(panel)
    assert rows.shape == (20, 512)  # (round(125 / 6.25) filters)
    # the center of filter k is 6.25 k Hz: the one at 37.5 Hz is the closest to the 40 Hz of the tone
    middle = rows[:, 100:400].mean(axis=1)
    assert np.argmax(middle) == 6
    alpha = 3.0 / 6.25 ** 2
    assert middle[6] == pytest.approx(np.exp(-4 * alpha * 2.5 ** 2), rel=0.1)
    # the envelope of a tone does not change in time
    assert rows[6, 100:400].std() < 0.05 * middle[6]


def test_gabor_parameters_and_headers():
    panel = run(gabor(fmin=10.0, fmax=50.0, band=10.0), traces(tone(8)[None, :], trace_id=3))
    assert len(panel) == 4 and all(t.header['ensemble_number'] == 3 for t in panel)
    assert [t.header['ensemble_trace_number'] for t in panel] == [1, 2, 3, 4]
    narrow = arr(run(gabor(fmin=10.0, fmax=50.0, band=10.0, alpha=5.0), traces(tone(8)[None, :])))
    wide = arr(run(gabor(fmin=10.0, fmax=50.0, band=10.0, alpha=0.5), traces(tone(8)[None, :])))
    assert not np.allclose(narrow, wide)
    with pytest.raises(ValueError):
        gabor(band=0.0)
    with pytest.raises(ValueError):
        gabor(beta=-1.0)
    assert run(gabor(fmin=10.0, fmax=11.0, band=10.0), traces(tone(16))) == []  # (no filters)


# ------------------------------------------------------------------------------------------------------------ cwt
def test_cwt_scales_follow_the_period():
    x = np.vstack([tone(20, 1500), tone(40, 1500)])  # (a trace that is longer than the wavelets)
    peaks = []
    for row in x:
        rows = arr(run(cwt(first=-0.5, last=1.5, expinc=0.01, nwavelet=512), traces(row)))
        assert rows.shape[0] == 201
        peaks.append(np.argmax(rows[:, 700:800].mean(axis=1)))
    # twice the period is twice the scale: log10(2) = .30 in the exponent, that is 30 scales
    assert peaks[1] - peaks[0] == pytest.approx(30, abs=3)


def test_cwt_number_of_scales_headers_and_checks():
    panel = run(cwt(first=0.0, last=1.0, expinc=0.1), traces(tone(20, 100)[None, :], trace_id=5))
    assert len(panel) == 11  # (0, .1, ... 1)
    assert [t.header['ensemble_trace_number'] for t in panel] == list(range(1, 12))
    assert all(t.n_sample == 100 and t.header['ensemble_number'] == 5 for t in panel)
    assert np.isfinite(arr(panel)).all() and (arr(panel) >= 0).all()
    for kwargs in (dict(nwavelet=1), dict(base=1.0), dict(expinc=0.0), dict(xmax=-30.0), dict(sigma=0.0),
                   dict(first=2.0, last=1.0)):
        with pytest.raises(ValueError):
            cwt(**kwargs)
