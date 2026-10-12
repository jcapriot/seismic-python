"""seispy stages in the middle of real SU pipelines."""
import io
import shlex
import sys

import numpy as np
import pytest

from seispy.io import read_su

PY = f"{shlex.quote(sys.executable)} -m seispy.su_pipe"


def traces(raw):
    return [np.asarray(t) for t in read_su(io.BytesIO(raw))]


def test_pass_through_is_identical(su):
    ref = su.run("suplane ntr=6 nt=100")
    out = su.run(f"suplane ntr=6 nt=100 | {PY}")
    assert out == ref


def test_stage_matches_su_program(su):
    ours = su.run(f"suplane ntr=6 nt=128 | {PY} amplitudes.gain tpow=2.0")
    ref = su.run("suplane ntr=6 nt=128 | sugain tpow=2.0")
    assert len(ours) == len(ref)
    for a, b in zip(traces(ours), traces(ref)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5 * np.abs(b).max())


def test_seispy_in_the_middle_of_su(su):
    mixed = su.run(f"suplane ntr=6 nt=128 | sugain tpow=1 | {PY} amplitudes.gain tpow=1 | sugain tpow=1")
    ref = su.run("suplane ntr=6 nt=128 | sugain tpow=1 | sugain tpow=1 | sugain tpow=1")
    for a, b in zip(traces(mixed), traces(ref)):
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5 * np.abs(b).max())


def test_su_reads_the_end_of_a_pipe_early(su):
    # `head` closes the pipe before we have written everything
    out = su.run(f"set +o pipefail; suplane ntr=400 nt=500 | {PY} | head -c 2000 | wc -c")
    assert int(out) == 2000
