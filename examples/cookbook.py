"""
Worked examples that use the stages together. Each function is one example, and returns something that it computed so
that the tests (``tests/test_examples.py``) can check that it works. The README has the same code.

    python examples/cookbook.py
"""
import numpy as np

from seispy import operations as op
from seispy.amplitudes import gain
from seispy.container import Trace, from_iterable
from seispy.convolution import conv
from seispy.decon import pef
from seispy.filters import bfilt, minphase
from seispy.noise import addnoise
from seispy.parallel import group_by, pmap
from seispy.stage import per_trace, stage
from seispy.stretching import nmo
from seispy.synthetics import randspike, synlv
from seispy.transforms import amp, cepstrum, fft, gabor, st
from seispy.waveforms import ricker1, vibro_linear
from seispy.windowing import mute


def samples(traces):
    """The samples of a stream of traces as a 2D array (every trace must be the same length)"""
    return np.array([np.asarray(trace) for trace in traces])


# ---------------------------------------------------------------------------------------------------------------------
def deconvolve_noisy_spikes():
    """Make a seismogram (random reflectivity, convolved with a minimum phase wavelet, plus noise) and deconvolve it.

    Spiking deconvolution assumes a minimum phase wavelet, so the zero phase Ricker wavelet is made so with `minphase`.
    The deconvolution makes the autocorrelation of the trace much more like that of the reflectivity, which is a spike.

    In SU:  suwaveform type=ricker1 fpeak=25 | suminphase > wavelet.su
            surandspike | suconv sufile=wavelet.su | suaddnoise sn=100 | supef maxlag=0.08
    """
    wavelet = next(iter(ricker1(fpeak=25.0, dt=0.002) | minphase()))  # (a source is an iterator, here of one trace)

    def spikes():  # (an iterator is used up by one pass, so make a new one each time. The seed makes them the same.)
        return randspike(n1=400, n2=8, dt=0.002, nspk=12, amax=1.0, seed=1)

    reflectivity = samples(spikes())
    seismogram = samples(spikes() | conv(wavelet) | addnoise(sn=100, seed=2))
    deconvolved = samples(spikes() | conv(wavelet) | addnoise(sn=100, seed=2) | pef(maxlag=0.08, pnoise=0.001))
    return reflectivity, seismogram, deconvolved


def sidelobe_energy(data, lags=60):
    """How far each trace is from having a spike for an autocorrelation: the energy of lags 1 to ``lags``, over that of 0"""
    energies = []
    for x in data:
        autocorrelation = np.correlate(x, x, 'full')[len(x) - 1:]
        energies.append(float(np.sum(autocorrelation[1:lags] ** 2) / autocorrelation[0] ** 2))
    return np.array(energies)


# ---------------------------------------------------------------------------------------------------------------------
def cmp_gathers_to_nmo():
    """Gain, mute, NMO correct and filter synthetic common midpoint gathers, one gather at a time in parallel.

    In SU:  susynlv ... | sugain tpow=2 | sumute ... | sunmo vnmo=2.0 | subfilt ...
    """
    flow = (
        gain(tpow=2.0)
        | mute(xmute=[0.0, 1.0], tmute=[0.0, 0.2])
        | nmo(vnmo=2.0)
        | bfilt(f_pass_low=2.0, f_stop_low=1.0)
    )
    source = synlv(nt=251, dt=0.004, nxm=5, nxo=6, dxo=0.1)  # (5 midpoints, with 6 offsets each)
    # the gathers are cut where ensemble_number changes, so each one can go to a worker
    traces = source | pmap(flow, workers=2, chunk=8, key='ensemble_number')
    return [samples(gather) for gather in group_by(traces, 'ensemble_number')]


# ---------------------------------------------------------------------------------------------------------------------
def spectrum_of_a_wavelet():
    """The amplitude spectrum of a Ricker wavelet: it peaks near the frequency that it was made with.

    In SU:  suwaveform type=ricker1 fpeak=30 | sufft | suamp
    """
    (spectrum,) = ricker1(fpeak=30.0, dt=0.002, ns=512) | fft() | amp()
    frequencies = np.arange(spectrum.n_sample) * spectrum.d_sample  # (a spectrum has its frequency step as its d_sample)
    return frequencies[np.argmax(np.asarray(spectrum))], spectrum


# ---------------------------------------------------------------------------------------------------------------------
def find_an_echo_in_the_cepstrum():
    """An echo is a peak in the cepstrum, at the delay of the echo (here 0.08 s).

    In SU:  suplane ... | sucepstrum
    """
    x = np.zeros(256, dtype=np.float32)
    x[0], x[40] = 1.0, 0.6  # a direct arrival and an echo, 40 samples (0.08 s) later
    (cepstrum_trace,) = from_iterable([Trace(x, d_sample=0.002)]) | cepstrum(unwrap=0)
    values = np.abs(np.asarray(cepstrum_trace)[1:128])
    return (np.argmax(values) + 1) * cepstrum_trace.d_sample


# ---------------------------------------------------------------------------------------------------------------------
def time_frequency_of_a_sweep():
    """A sweep from 10 to 80 Hz, as a time-frequency panel: the Stockwell transform and the Gabor multifilter analysis.

    Each trace becomes a gather of traces, one per frequency; ``ensemble_trace_number`` counts the frequencies.

    In SU:  suvibro f1=10 f2=80 tv=2 | sust     and     suvibro f1=10 f2=80 tv=2 | sugabor
    """
    sweep = vibro_linear(f1=10.0, f2=80.0, tv=2.0, dt=0.004, t1=0.0, t2=0.0)
    stockwell = samples(sweep | st(fmin=5.0, fmax=100.0))
    multifilter = samples(vibro_linear(f1=10.0, f2=80.0, tv=2.0, dt=0.004, t1=0.0, t2=0.0) | gabor(band=5.0))
    # the frequency of the loudest row at the start and at the end of the sweep
    d1 = 1.0 / (stockwell.shape[1] * 0.004)
    first_row = int(5.0 / d1 + 1)
    f_start = (first_row + np.argmax(stockwell[:, 60])) * d1
    f_end = (first_row + np.argmax(stockwell[:, -60])) * d1
    return f_start, f_end, multifilter.shape


# ---------------------------------------------------------------------------------------------------------------------
def your_own_stage():
    """Any function of one trace is a stage, with ``per_trace``. Here is a trace-by-trace dip in amplitude.

    ``stage`` makes it something that can be chained with ``|`` and given to ``pmap`` (``parallelism='trace'`` says that
    traces are independent).
    """

    def _scale_by_offset(upstream, *, per_km=0.1):
        def scaled(trace):
            return trace.replace(np.asarray(trace) * (1.0 + per_km * abs(trace.header['offset']) / 1000.0))

        return per_trace(upstream, scaled)

    scale_by_offset = stage(_scale_by_offset, parallelism='trace', name='scale_by_offset', validate=True)

    traces = synlv() | scale_by_offset(per_km=0.5) | op.sgn()
    return samples(traces)


# ---------------------------------------------------------------------------------------------------------------------
def headers():
    """What is in a trace header, and what is worked out from it.

    ``offset`` is not stored, it is the distance from ``tx_loc`` to ``rx_loc`` (negative if the receiver is the first one).
    """
    trace = Trace(
        np.zeros(100, dtype=np.float32), d_sample=0.004, tx_loc=[100.0, 0.0, 10.0], rx_loc=[350.0, 0.0, 12.0],
        coord_unit='length',
    )
    values = trace.header
    moved = trace.replace(offset=-500.0, trace_id=3)  # (changes the receiver: replace() makes a new trace)
    return values['offset'], moved.header['offset'], moved.header['rx_loc'], moved.header['trace_id']


if __name__ == "__main__":
    reflectivity, seismogram, deconvolved = deconvolve_noisy_spikes()
    print("autocorrelation sidelobes:", sidelobe_energy(seismogram).mean(), "->", sidelobe_energy(deconvolved).mean())
    print("cmp gathers:", [g.shape for g in cmp_gathers_to_nmo()])
    print("wavelet peak frequency:", spectrum_of_a_wavelet()[0])
    print("echo delay:", find_an_echo_in_the_cepstrum())
    print("sweep, loudest frequency at the start and end:", time_frequency_of_a_sweep()[:2])
    print("headers:", headers())
