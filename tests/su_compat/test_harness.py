"""The harness itself: it has to notice differences, and the registry has to be consistent. (No SU needed.)"""
import io

import numpy as np
import pytest

from cases import CASES
from coverage import readme_programs
from harness import compare, load_divergences, run_ours
from seispy.container import Trace
from seispy.io import write_su


def su_bytes(data, **extra):
    traces = [
        Trace(np.asarray(row, dtype=np.float32), d_sample=0.004).replace(
            trace_id=i + 1, extra=extra
        )
        for i, row in enumerate(data)
    ]
    out = io.BytesIO()
    write_su(traces, out)
    return out.getvalue()


def test_identical_files_match():
    raw = su_bytes(np.random.default_rng(0).standard_normal((3, 50)), fldr=3)
    assert compare(raw, raw) == []


def test_detects_a_wrong_gain():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 50))
    ref = su_bytes(data)
    ours = su_bytes(data * 1.001)  # a 0.1% error
    problems = compare(ours, ref, rtol=1e-5)
    assert len(problems) == 1 and problems[0].startswith("data:")
    assert compare(ours, ref, rtol=1e-2) == []


def test_detects_a_header_word_and_ignore_words():
    data = np.ones((2, 8))
    ref, ours = su_bytes(data, fldr=1), su_bytes(data, fldr=2)
    (problem,) = compare(ours, ref)
    assert problem.startswith("header fldr")
    assert compare(ours, ref, ignore_words=["fldr"]) == []


def test_detects_nan_and_shape_differences():
    ref = su_bytes([[1.0, 2.0, 3.0]])
    assert compare(su_bytes([[1.0, np.nan, 3.0]]), ref)
    assert "samples" in "".join(compare(su_bytes([[1.0, 2.0]]), ref))
    assert "number of traces" in compare(su_bytes([[1.0] * 3] * 2), ref)[0]


def test_case_names_are_unique_and_registry_refers_to_them():
    names = [c.name for c in CASES]
    assert len(names) == len(set(names))
    unknown = set(load_divergences()) - set(names)
    assert not unknown, f"divergences.toml has entries without a case: {sorted(unknown)}"


def test_cases_refer_to_ported_programs():
    from coverage import README

    if not README.exists():
        pytest.skip("README.md is not next to the tests (they are run from outside of the source tree)")
    programs = set(readme_programs())
    assert programs, "could not read the program table of the README"
    unknown = {c.program for c in CASES} - programs
    assert not unknown, f"cases for programs that README.md does not list as ported: {sorted(unknown)}"


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_stages_resolve(case):
    from seispy.su_pipe import resolve_stage

    for name, kwargs in case.stages:
        resolve_stage(name)  # (SystemExit if the stage does not exist)
