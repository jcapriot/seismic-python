"""Options of the test suite."""
import pathlib
import re

import pytest

SU_COMPAT_DIR = pathlib.Path(__file__).parent / "su_compat"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "su_compat: compares seispy with real Seismic Unix programs. Opt in by naming the marker: "
        "`pytest -m su_compat` (only these), or `-m 'su_compat or not su_compat'` (everything)",
    )


def pytest_collection_modifyitems(config, items):
    # The tests that use the real programs are the ones that ask for the `su` fixture. They are skipped unless the
    # marker expression (-m) mentions su_compat; pytest itself then selects what the expression says.
    opted_in = re.search(r"\bsu_compat\b", config.option.markexpr or "") is not None
    skip = pytest.mark.skip(
        reason="comparisons with Seismic Unix are opt in: run pytest -m su_compat "
        "(and build SU with tools/su_reference/build.sh)"
    )
    for item in items:
        if "su" in getattr(item, "fixturenames", ()) and SU_COMPAT_DIR in pathlib.Path(str(item.path)).parents:
            item.add_marker(pytest.mark.su_compat)
            if not opted_in:
                item.add_marker(skip)
