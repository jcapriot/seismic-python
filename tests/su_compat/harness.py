"""Run a seispy stage and the SU program it ports on the same input, and compare what they write.

A ``Case`` names the SU program, its command line, and the stage (as ``module.function`` of seispy and
its keyword arguments, the way ``seispy.su_pipe`` takes them). The input is made by SU itself, so both sides
see the same bytes. The comparison is of the SU files that come out, so it covers the trace data (to a
tolerance that scales with the size of each reference trace) and every one of the 80 header words.

Where the port is known to differ from the program, ``divergences.toml`` says so (see ``Divergence``).
"""
import dataclasses
import io
import pathlib
import tomllib

import numpy as np

from seispy.io import read_su, write_su
from seispy.io.su import SU_HEADER_BYTES, _native_dtype
from seispy.su_pipe import resolve_stage

DIVERGENCES = pathlib.Path(__file__).with_name("divergences.toml")

# header words that are pure bookkeeping of the file, never compared
NEVER_COMPARED = frozenset({"shortpad"})


@dataclasses.dataclass(frozen=True)
class Case:
    name: str  # unique, used to look up divergences
    program: str  # the SU program that is ported ("sugain")
    source: str  # a shell pipeline of SU programs that makes the input traces
    su: str  # the SU command line (program and parameters)
    stages: tuple  # ((module.function, {kwargs}), ...) of seispy
    rtol: float = 1e-5  # of the largest |sample| of each reference trace
    atol: float = 0.0
    # header words that are known to be written differently (and why, in the divergence registry or the note)
    ignore_words: tuple = ()
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Divergence:
    """What ``divergences.toml`` records about a case.

    mode
        ``loose``: it matches, but only to a looser ``rtol``/``atol`` (and ``ignore_words``);
        ``expect_diff``: it is known to differ, the test checks that it *still* differs (so that the entry is
        deleted when the difference goes away);
        ``skip``: not compared (it can not be, e.g. random numbers).
    """

    mode: str
    reason: str
    rtol: float | None = None
    atol: float | None = None
    ignore_words: tuple = ()
    ref: str = ""  # where it is documented, e.g. "seispy/attributes/__init__.py"
    verified: str = ""  # how the reason was checked (the SU source, and the output of the program)


def load_divergences(path=DIVERGENCES):
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    out = {}
    for name, entry in raw.items():
        mode = entry.get("mode")
        if mode not in ("loose", "expect_diff", "skip"):
            raise ValueError(f"divergences.toml [{name}]: mode must be loose, expect_diff or skip, not {mode!r}")
        if not entry.get("reason"):
            raise ValueError(f"divergences.toml [{name}]: a divergence needs a reason")
        if not entry.get("verified"):
            raise ValueError(f"divergences.toml [{name}]: say how the reason was verified against SU")
        out[name] = Divergence(
            mode=mode,
            reason=entry["reason"],
            rtol=entry.get("rtol"),
            atol=entry.get("atol"),
            ignore_words=tuple(entry.get("ignore_words", ())),
            ref=entry.get("ref", ""),
            verified=entry.get("verified", ""),
        )
    return out


def run_ours(raw, stages):
    """The SU bytes that the seispy stages make from SU bytes"""
    traces = read_su(io.BytesIO(raw))
    for name, kwargs in stages:
        traces = traces | resolve_stage(name)(**kwargs)
    out = io.BytesIO()
    write_su(traces, out)
    return out.getvalue()


def parse_su(raw):
    """(header structured array, [data arrays]) of SU bytes (native byte order, any ns per trace)"""
    dtype = _native_dtype()
    headers, data = [], []
    pos, n = 0, len(raw)
    while pos < n:
        if pos + SU_HEADER_BYTES > n:
            raise ValueError(f"truncated trace header at byte {pos}")
        hdr = np.frombuffer(raw, dtype=dtype, count=1, offset=pos)[0]
        pos += SU_HEADER_BYTES
        ns = int(hdr["ns"])
        if pos + 4 * ns > n:
            raise ValueError(f"truncated trace data at byte {pos}")
        data.append(np.frombuffer(raw, dtype=np.float32, count=ns, offset=pos))
        pos += 4 * ns
        headers.append(hdr)
    return np.array(headers, dtype=dtype), data


def compare(ours_raw, ref_raw, rtol=1e-5, atol=0.0, ignore_words=(), max_report=6):
    """A list of human readable differences between two SU byte strings (empty if they match)"""
    ours_h, ours_d = parse_su(ours_raw)
    ref_h, ref_d = parse_su(ref_raw)
    problems = []
    if len(ours_d) != len(ref_d):
        return [f"number of traces: seispy {len(ours_d)}, SU {len(ref_d)}"]

    skip = set(ignore_words) | NEVER_COMPARED
    words = [w for w in ref_h.dtype.names if w not in skip]
    for w in words:
        a, b = ours_h[w], ref_h[w]
        bad = np.flatnonzero(np.any((a != b).reshape(len(a), -1), axis=1))
        if len(bad):
            i = int(bad[0])
            problems.append(
                f"header {w}: differs on {len(bad)}/{len(a)} traces, first at trace {i}: "
                f"seispy {np.atleast_1d(a[i]).tolist()}, SU {np.atleast_1d(b[i]).tolist()}"
            )

    worst = (0.0, None)
    n_bad = 0
    for i, (a, b) in enumerate(zip(ours_d, ref_d)):
        if a.shape != b.shape:
            problems.append(f"trace {i}: {a.shape[0]} samples from seispy, {b.shape[0]} from SU")
            continue
        if a.size == 0:
            continue
        scale = float(np.max(np.abs(b)))
        with np.errstate(invalid="ignore"):
            err = np.abs(a.astype(np.float64) - b)
        # (a NaN or an inf has to be in both)
        mismatch_nan = np.isnan(a) != np.isnan(b)
        tol = atol + rtol * scale
        bad = (err > tol) | mismatch_nan
        if np.any(bad):
            n_bad += 1
            j = int(np.argmax(np.where(bad, np.nan_to_num(err, nan=np.inf), -1.0)))
            rel = float(np.nan_to_num(err[j], nan=np.inf) / (scale if scale > 0 else 1.0))
            if rel >= worst[0]:
                worst = (rel, (i, j, float(a[j]), float(b[j]), scale))
    if n_bad:
        i, j, x, y, scale = worst[1]
        problems.append(
            f"data: {n_bad}/{len(ref_d)} traces differ by more than rtol={rtol:g}, atol={atol:g}; worst is trace {i} "
            f"sample {j}: seispy {x:.7g}, SU {y:.7g} (|trace|max {scale:.4g}, error/max {worst[0]:.3g})"
        )
    return problems[:max_report] + (["..."] if len(problems) > max_report else [])
