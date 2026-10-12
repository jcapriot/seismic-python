"""Every case: the seispy stages against the real SU program, on SU generated input."""
import pytest

from cases import CASES
from harness import compare, load_divergences, run_ours

DIVERGENCES = load_divergences()


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_matches_su(su, case):
    div = DIVERGENCES.get(case.name)
    if div is not None and div.mode == "skip":
        pytest.skip(div.reason)

    raw = su.run(case.source)
    ref = su.run(case.source + " | " + case.su)
    ours = run_ours(raw, case.stages)

    rtol, atol, ignore = case.rtol, case.atol, set(case.ignore_words)
    if div is not None and div.mode == "loose":
        rtol = div.rtol if div.rtol is not None else rtol
        atol = div.atol if div.atol is not None else atol
        ignore |= set(div.ignore_words)
    problems = compare(ours, ref, rtol=rtol, atol=atol, ignore_words=ignore)

    if div is not None and div.mode == "expect_diff":
        assert problems, (
            f"{case.name} matches SU now, so its entry in divergences.toml ({div.reason!r}) is out of date: remove it"
        )
        return
    assert not problems, f"{case.name} ({case.su}) differs from SU:\n  " + "\n  ".join(problems)
