"""Fixtures for comparing seispy against real Seismic Unix programs.

SU is built by ``tools/su_reference/build.sh`` (upstream master, in its own worktree).
CWPROOT/PATH are only ever set on the environment of the subprocesses started here,
never on the test process or the user's shell. Everything is skipped if SU is absent.
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
        message = f"no Seismic Unix build at {root}"
        if os.environ.get("SU_REQUIRED"):
            # (CI sets this, so that a broken SU build can not make the comparisons silently skip)
            pytest.fail(message)
        pytest.skip(message + " (run tools/su_reference/build.sh)")
    return SU(root)
