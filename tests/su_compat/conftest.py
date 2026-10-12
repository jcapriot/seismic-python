"""Fixtures for comparing seispy against real Seismic Unix programs.

These tests are opt in: they only run when the marker is asked for, ``pytest -m su_compat`` (see
``tests/conftest.py``). Without it they are skipped, so that the plain test suite neither needs SU nor depends on a
build of it. When opted in, a missing SU build is a failure, not a skip.

SU is built by ``tools/su_reference/build.sh`` (upstream, in its own worktree), found in ``SU_REFERENCE_ROOT``
(default ``~/codes/SeisUnix-master``). CWPROOT/PATH are only ever set on the environment of the subprocesses
started here, never on the test process or the user's shell.
"""
import os
import pathlib
import subprocess

import pytest

DEFAULT_ROOT = pathlib.Path.home() / "codes" / "SeisUnix-master"




class SU:
    """Runs SU programs / shell pipelines of SU programs."""

    def __init__(self, root):
        self.root = pathlib.Path(root)
        self.env = {
            **os.environ,
            "CWPROOT": str(self.root),
            "PATH": f"{self.root / 'bin'}{os.pathsep}{os.environ['PATH']}",
        }

    def run(self, pipeline, stdin=None, **kwargs):
        """Run a shell command line (pipes allowed) and return its stdout bytes."""
        res = subprocess.run(
            ["bash", "-o", "pipefail", "-c", pipeline],
            input=stdin,
            capture_output=True,
            env=self.env,
            **kwargs,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"{pipeline!r} failed ({res.returncode}):\n{res.stderr.decode(errors='replace')}"
            )
        return res.stdout


@pytest.fixture(scope="session")
def su():
    root = pathlib.Path(os.environ.get("SU_REFERENCE_ROOT", DEFAULT_ROOT))
    if not (root / "bin" / "suplane").exists():
        # (only reached with -m su_compat: a broken SU build must not make the comparisons silently disappear)
        pytest.fail(f"-m su_compat was given, but there is no Seismic Unix build at {root} (run tools/su_reference/build.sh)")
    return SU(root)
