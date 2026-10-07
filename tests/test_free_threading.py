"""
Threads that use seispy at the same time, with nothing to serialize them: the same stage objects and the same traces in several
threads, and the SU library routines that used to keep state (the tables of the Hilbert transform, the sinc interpolation, and
the random numbers). They are for the free threaded build of python (where there is no GIL to do that), and are made to give
the same results as the same work done by one thread on any build.
"""
import importlib
import pkgutil
import sys
import sysconfig
import threading

import numpy as np
import numpy.testing as npt
import pytest

import seispy
from seispy import attributes
from seispy import operations as op
from seispy.amplitudes import centsamp, gain
from seispy.container import Trace, from_iterable
from seispy.convolution import conv
from seispy.filters import bfilt, frac, minphase
from seispy.noise import _rng
from seispy.stacking import stack
from seispy.stretching import log, nmo, resamp, ttoz
from seispy.synthetics import synlv
from seispy.tapering import taper
from seispy.transforms import cepstrum, hilb, st
from seispy.windowing import mute

N_THREADS = 8
ROUNDS = 6

FREE_THREADED = bool(sysconfig.get_config_var('Py_GIL_DISABLED'))


def in_threads(work, n=N_THREADS):
    """Run work(i) in n threads that all start at the same time, and return what they returned (an exception is raised)"""
    barrier = threading.Barrier(n)
    results = [None] * n
    errors = []

    def run(i):
        try:
            barrier.wait()
            results[i] = work(i)
        except BaseException as error:  # noqa: B902 (every failure of a thread has to get to the test)
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def samples(traces):
    return np.array([np.asarray(t) for t in traces])


RNG = np.random.default_rng(7)
DATA = RNG.normal(size=(24, 200)).astype(np.float32)


def traces():
    return [Trace(row, d_sample=0.004).replace(trace_id=i + 1, ensemble_number=i // 6 + 1) for i, row in enumerate(DATA)]


STAGES = {
    'bfilt': lambda: bfilt(f_pass_low=10.0, f_stop_low=5.0),
    'gain': lambda: gain(tpow=2.0, agc=True, wagc=0.2),
    'mute': lambda: mute([0.0, 1.0], [0.1, 0.4], ntaper=4),
    'taper': lambda: taper(80.0, 80.0, type=3),
    'nmo': lambda: nmo(vnmo=2000.0),
    'resamp': lambda: resamp(nt=150, dt=0.005),
    'log': lambda: log(ntmin=5),
    'ttoz': lambda: ttoz(v=2200.0),
    'hilb': lambda: hilb(),
    'amp': lambda: attributes.amp(),
    'freq': lambda: attributes.freq(),
    'conv': lambda: conv(np.array([0.5, 1.0, 0.5], dtype=np.float32)),
    'centsamp': lambda: centsamp(),
    'frac': lambda: frac(power=0.5),
    'minphase': lambda: minphase(),
    'cepstrum': lambda: cepstrum(unwrap=0),
    'st': lambda: st(fmin=5.0, fmax=40.0),
    'despike': lambda: op.despike(nw=5),
    'saf': lambda: op.saf(),
    'stack': lambda: stack(),
}


@pytest.mark.parametrize('name', sorted(STAGES))
def test_one_stage_in_many_threads(name):
    # the same stage object (its specification) is bound in every thread, and the same traces are processed by all of them
    stage = STAGES[name]()
    source = traces()
    expected = samples(from_iterable(source) | stage)

    def work(i):
        return [samples(from_iterable(source) | stage) for _ in range(ROUNDS)]

    for results in in_threads(work):
        for got in results:
            npt.assert_array_equal(got, expected)


def test_different_stages_at_the_same_time():
    source = traces()
    names = sorted(STAGES)
    expected = {name: samples(from_iterable(source) | STAGES[name]()) for name in names}

    def work(i):
        out = {}
        for r in range(ROUNDS):
            name = names[(i * 3 + r) % len(names)]
            out[name] = samples(from_iterable(source) | STAGES[name]())
        return out

    for results in in_threads(work):
        for name, got in results.items():
            npt.assert_array_equal(got, expected[name])


def test_synthetics_in_many_threads():
    # su_synlv uses the table of sinc coefficients of addsinc, which is made when the module is imported
    expected = samples(synlv(nt=101, nxm=6, nxo=4))

    def work(i):
        return [samples(synlv(nt=101, nxm=6, nxo=4)) for _ in range(3)]

    for results in in_threads(work):
        for got in results:
            npt.assert_array_equal(got, expected)


def test_random_numbers_do_not_share_state():
    seeds = list(range(N_THREADS))
    expected_uniform = [_rng.Uniform(seed).draw(2000) for seed in seeds]
    expected_normal = [_rng.Normal(seed).draw(2000) for seed in seeds]

    def work(i):
        out = []
        for _ in range(ROUNDS):
            generator, normal = _rng.Uniform(seeds[i]), _rng.Normal(seeds[i])
            out.append((generator.draw(2000), normal.draw(2000)))
        return out

    for i, results in enumerate(in_threads(work)):
        for uniform, normal in results:
            npt.assert_array_equal(uniform, expected_uniform[i])
            npt.assert_array_equal(normal, expected_normal[i])


def test_traces_that_are_shared_are_not_changed():
    # the traces are read by all of the threads (and given to stages that make new traces from them)
    source = traces()
    before = samples(source)
    headers = [t.header for t in source]

    def work(i):
        for _ in range(ROUNDS):
            for trace in source:
                np.asarray(trace)
                trace.header
                trace.replace(trace_id=i)
            list(from_iterable(source) | op.sqr() | op.neg())
        return None

    in_threads(work)
    npt.assert_array_equal(samples(source), before)
    assert [t.header for t in source] == headers


@pytest.mark.skipif(not FREE_THREADED, reason="the interpreter has a GIL")
def test_the_gil_is_not_turned_on_by_importing_the_modules():
    for module in pkgutil.walk_packages(seispy.__path__, 'seispy.'):
        try:
            importlib.import_module(module.name)
        except ModuleNotFoundError as error:
            if error.name != 'matplotlib':  # (seispy.plotting needs it, and it is not a dependency)
                raise
    assert not sys._is_gil_enabled(), "an extension module did not say that it can run without the GIL"
