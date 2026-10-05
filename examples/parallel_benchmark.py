"""
Compare sequential, threaded-pipeline and chunked data-parallel runs of a bfilt pipeline.

    python examples/parallel_benchmark.py
"""
import os
import time

from seispy.filters import bfilt
from seispy.parallel import pmap, prefetch
from seispy.synthetics import spike

NT, NTR = 32_768, 4_000
stage = bfilt(f_pass_low=10.0, f_stop_low=5.0, n_poles_low=4) | bfilt(f_pass_high=60.0, f_stop_high=80.0)


def run(label, make):
    t0 = time.perf_counter()
    n = sum(1 for _ in make())
    dt = time.perf_counter() - t0
    print(f"{label:<30s}{dt:7.2f} s   ({n / dt:8.0f} traces/s)")
    return dt


if __name__ == "__main__":
    print(f"{NTR} traces x {NT} samples, {os.cpu_count()} cpus")
    base = run("sequential", lambda: spike(nt=NT, ntr=NTR) | stage)
    # one thread per filter, like `plane | bfilt | bfilt`
    low, high = stage.stages
    run("pipeline (thread per stage)", lambda: spike(nt=NT, ntr=NTR) | low | prefetch(buffer=4, batch=16) | high)
    for w in (1, 2, 4, 8):
        dt = run(f"pmap threads, {w} worker(s)", lambda: spike(nt=NT, ntr=NTR) | pmap(stage, workers=w, chunk=16))
        print(f"{'':30s}speedup x{base / dt:4.2f}")
