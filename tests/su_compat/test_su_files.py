"""SU files made by the real programs go through seispy's reader and writer unchanged, and SU reads ours."""
import io

import numpy as np
import pytest

from seispy.io import read_su, write_su

# every header word that seispy maps, with a scalar for the coordinates and for the elevations
WORDS = dict(
    fldr=(7, 1), tracl=(1, 1), tracr=(1, 1), ep=(5, 0), cdp=(100, 1), cdpt=(1, 1), scalco=(-100, 0),
    sx=(100050, 100), gx=(150025, 100), sy=(200000, 0), gy=(200000, 0), scalel=(-10, 0), selev=(125, 10),
    gelev=(100, 10), sstat=(5, 1), gstat=(-5, 0), tstat=(0, 1), delrt=(200, 0), muts=(40, 0), mute=(300, 0),
    trid=(1, 0), nhs=(6, 0), mark=(3, 0), ntr=(10, 0), counit=(1, 0), offset=(-50, 10),
)


def _with_words():
    cmds = ["suplane ntr=10 nt=64 dt=0.004"]
    for key, (a, b) in WORDS.items():
        cmds.append(f"sushw key={key} a={a} b={b}")
    return " | ".join(cmds)


PRODUCERS = {
    "suplane": "suplane npl=2 ntr=12 nt=64 dt=0.004 offset=100",
    "suspike": "suspike nt=100 ntr=5 ix1=2 it1=30 nspk=2 dt=0.002",
    "susynlv": "susynlv nt=128 dt=0.004 nxo=4 nxs=3 dxo=50 dxs=100 fxo=0 fxs=0 nvq=1 v00=2000 "
    "ob=0 fpeak=20",
    "sushw_all": _with_words(),
}


@pytest.mark.parametrize("name", list(PRODUCERS))
def test_round_trip_is_byte_identical(su, name):
    raw = su.run(PRODUCERS[name])
    assert len(raw) > 240
    out = io.BytesIO()
    n = write_su(read_su(io.BytesIO(raw)), out)
    assert n > 0
    assert out.getvalue() == raw


def test_su_reads_what_we_write(su):
    raw = su.run(PRODUCERS["susynlv"])
    traces = list(read_su(io.BytesIO(raw)))
    # a stage-like change that SU must see in the header words
    changed = [t.replace(sample_start=0.08, d_sample=0.002, ensemble_number=42) for t in traces]
    out = io.BytesIO()
    write_su(changed, out)
    text = su.run("sugethw key=ns,dt,delrt,cdp,offset", stdin=out.getvalue()).decode()
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == len(traces)
    for l in lines:
        vals = dict(item.split("=") for item in l.split())
        assert int(vals["dt"]) == 2000
        assert int(vals["delrt"]) == 80
        assert int(vals["cdp"]) == 42
        assert int(vals["ns"]) == 128


def test_data_values_are_the_same(su):
    raw = su.run(PRODUCERS["suplane"])
    traces = list(read_su(io.BytesIO(raw)))
    ref = np.frombuffer(raw, dtype=np.uint8).reshape(len(traces), -1)[:, 240:].copy().view(np.float32)
    for tr, row in zip(traces, ref):
        np.testing.assert_array_equal(np.asarray(tr), row)
