# seispy
This package is currently a proof of concept of directly integrating seismic unix into python. For now
we have included SU source code with enough modifications to get this simple workflow to successfully
execute on Linux, MacOS, **and Windows**!

## Installing:
If you have a working C compiler, installing is as easy as:

```bash
pip install .
```

### Requirements:
At the moment, the external package requirements are quite light:

* `numpy>=1.26.0` To handle some numerical IO

and, for `seispy.plotting` and the examples, `matplotlib` (`pip install seismic-python[plot]`).

### In place builds:
In place builds, useful for developers, can be accomplished with:

```bash
pip install --no-build-isolation --editable .
```

which requires that you already have `numpy>=2.0`, in your build environment.


> **NOTE**: On Windows you might have to append `--config-settings=setup-args="--vsenv"` to the `pip install` command,
> if you have the mingw compilers on your system, and you require the visual studio compilers for ABI
> compatibility with your python installation.


## The idea:
Seismic unix is dependant on a bash styled unix system, piping outputs between different programs
that all consume and generate either 1 trace, or 1 gather of traces at a time, and dump their output
streams to a file.

E.G.
```bash
plane | bfilt > out.su
```


The process of working with limited amounts of data at a time, piping it through different programs
allowing each to work sequentially on each trace can easily be mimicked in Python using iterators!

Consider this simple iterator, which adds an echo to `range`:

```Python
>>> def create_data(n):
...     for i in range(n):
...         print(f"creating {i}")
...         yield i
...
```

Evocations of `create_data` return generators:
```Python
>>> type(create_data(3))
<class 'generator'>
```
that do not execute until asked for a value.

```Python
>>> items = [i for i in create_data(3)]
creating 0
creating 1
creating 2
```

But what we can also do is consume generators in other generators!
```Python
>>> def multiply(items, scale):
...     for item in items:
...         print(f"multiplying {item}")
...         yield item * scale
...
```

Chain operators together gives:
```Python
>>> mul_items = [item for item in multiply(create_data(3), 2.0)]
creating 0
multiplying 0
creating 1
multiplying 1
creating 2
multiplying 2
>>> mul_items
[0.0, 2.0, 4.0]
```
The way to think about iterators is that you ask it to return a value. This chain of iterators then
recursively asks for the next value from its input, and so on until one of them actually gives a value
back to work with. This behavoir mimics the exact operations done by SU programs.

## Interfacing with SU
Since we are basically replacing all IO operations of seismic unix with python iterators, we can just rely
on python to handle all of the IO operations when necessary, and we no longer rely on having a unix shell
like interface, meaning this code compiles and works on all systems. My intention is to make as much use
as possible of the proven C code already written in seismic unix. For this purpose, Cython was the clearest
candidate to facilitate that low-level communication between C and Python. Also, transferring the `make` files
in seismic unix to `meson` helped to ensure proper linkage within the project.

Currently, the base `cwp`, `par` and `su` libraries are built and linked into the python package.

A little bit more subtleties here is that I'm using cython iterators to do the above operations, with
the intention of releasing the GIL when inside calls to enable threading.


## Complex traces
A trace has samples of dtype `float32` (the default) or `complex64`, which is what `np.asarray(trace)` gives you (still a
zero-copy view). Make one with a complex array, or `Trace(data, d_sample, dtype='complex64')`. `fft` and `analytic`
make them, and `ifft`, `real`, `imag`, `amp`, `logamp` and `phase` take them apart. SU has these as traces of pairs of
floats with a trace id, here `n_sample` is the number of complex samples.

Each stage does what makes sense for complex numbers, rather than refusing them all:

* **Anything that is defined for complex numbers works**, and keeps the trace complex: scaling and adding (`gain`'s
  `tpow`, `epow`, `scale`, `norm` and `bias`, which is a real number added to the real part), `op.neg`, `op.sum`,
  `op.mean`, `op.sin`, `op.inv`, ..., `weight`, `zero`, `nan`, `taper`, `ramp`, `mute`, `wind`, `kill`, `shift`,
  `reduce`, `normalize` (by rms or max), and `conv`. `op.abs` and `op.db` give the modulus (a real trace), and so do the
  statistics `op.std` and `op.var`.
* **Filters with real coefficients are linear**, so they are applied to the real and to the imaginary part: `bfilt`,
  `filter`, `hilb`, `resamp` and `nmo`.
* **What needs an order, a sign, a median, or a real signal's spectrum is rejected** with a `TypeError`: `op.posonly`,
  `op.ssqrt`, `op.sgn`, `op.slog`, `op.spike`, `op.saf`, `op.despike`, `normalize('med')`, `gain`'s clipping, `agc`,
  `gpow` and balancing options, `frac`, `phase`, `zerophase`, `acor`, `xcor`, `ai2r`, `r2ai`, `seispy.attributes`,
  `fft` and `analytic`. SEG-Y has no complex traces.

A trace shares the memory of the array it is made from (as it always has), so copy the array if you are going to write to
the trace.

## The trace header
`trace.header` is a dict of what a trace needs that can not be worked out from the others:

| | |
|---|---|
| `n_sample`, `d_sample`, `sample_start` | the number of samples, the sample interval and the time of the first sample, in seconds (or meters for depth) |
| `sampling_unit`, `sampling_domain` | seconds or meters, and time (or space) or Fourier domain |
| `tx_loc`, `rx_loc` | source and receiver `(x, y, elevation)`, with `coord_unit` for x and y: `'length'` or `'degrees'` (SEG-Y's seconds of arc and degrees-minutes-seconds are converted on reading) |
| `ensemble_number`, `ensemble_trace_number` | the gather that the trace is in (the CDP, the shot, the panel it came from), and its number in it, from 1 (0 is not set) |
| `trace_id`, `iline`, `xline` | the number of the trace in its line, and the in-line and cross-line numbers of a 3D survey |
| `trace_type` | the SEG-Y trace identification code: 1 for seismic data, 2 for a dead trace, ... |
| `fold` | the number of traces that were stacked to make this one (0 if it is a single trace) |
| `source_static`, `receiver_static`, `total_static` | static shifts, in seconds |

The offset is `header['offset']`, but it is worked out from the source and the receiver (negative when the receiver comes
first), and so is the midpoint. `trace.replace(offset=500.0)` moves the receiver. `replace` makes a new trace (as the stages do,
unless they are given `inplace=True`).

## Examples
These are in `examples/cookbook.py`, which the tests run.

**Deconvolution.** A seismogram (random reflectivity convolved with a wavelet, plus noise), then Wiener deconvolution.
Sources are used up by one pass, so make a new one when it is needed again.

```python
from seispy.waveforms import ricker1
from seispy.filters import minphase
from seispy.synthetics import randspike
from seispy.convolution import conv
from seispy.noise import addnoise
from seispy.decon import pef

# the first trace of a source (spiking deconvolution assumes a minimum phase wavelet)
wavelet = next(iter(ricker1(fpeak=25.0, dt=0.002) | minphase()))

traces = randspike(n1=400, n2=8, dt=0.002, nspk=12, seed=1) | conv(wavelet) | addnoise(sn=100, seed=2) | pef(maxlag=0.08)
```

**A processing flow on gathers.** Each stage works on one trace; `pmap` splits the stream between gathers across threads, and
`group_by` cuts the result back into gathers.

```python
from seispy.synthetics import synlv
from seispy.amplitudes import gain
from seispy.windowing import mute
from seispy.stretching import nmo
from seispy.filters import bfilt
from seispy.parallel import group_by, pmap

flow = gain(tpow=2.0) | mute(xmute=[0.0, 1.0], tmute=[0.0, 0.2]) | nmo(vnmo=2.0) | bfilt(f_pass_low=2.0, f_stop_low=1.0)
gathers = group_by(
    synlv(nt=251, dt=0.004, nxm=5, nxo=6, dxo=0.1) | pmap(flow, workers=2, chunk=8, key='ensemble_number'),
    'ensemble_number',
)
```

**Stacking.** Stacking works on gathers: the run of traces that have the same value of `key` (default
`ensemble_number`, the CDP). The stacked trace has the header of the first trace, with the offset 0 and `fold` the number
of traces that were stacked. `sort` first if the traces are not in order, or use `stackup`, which stacks in any order.

```python
from seispy.stacking import stack

stacked = synlv(nt=251, dt=0.004, nxm=5, nxo=6, dxo=0.1) | nmo(vnmo=2.0) | pmap(stack(), workers=2, chunk=6, key='ensemble_number')
```

**Spectra, and the cepstrum.** Spectra are complex traces in the Fourier domain: their `d_sample` is the frequency step.

```python
import numpy as np
from seispy.transforms import fft, amp, cepstrum

(spectrum,) = ricker1(fpeak=30.0, dt=0.002, ns=512) | fft() | amp()
peak_frequency = np.argmax(np.asarray(spectrum)) * spectrum.d_sample      # 30.3 Hz

# an echo is a peak in the cepstrum, at its delay
from seispy.container import Trace, from_iterable
x = np.zeros(256, dtype=np.float32); x[0], x[40] = 1.0, 0.6
(c,) = from_iterable([Trace(x, d_sample=0.002)]) | cepstrum(unwrap=0)     # the peak is at 0.08 s
```

**Time-frequency panels.** Each trace becomes a gather of traces, one for each frequency (or scale), numbered by
`ensemble_trace_number`.

```python
from seispy.waveforms import vibro_linear
from seispy.transforms import st, gabor

sweep = lambda: vibro_linear(f1=10.0, f2=80.0, tv=2.0, dt=0.004, t1=0.0, t2=0.0)
stockwell = np.array([np.asarray(t) for t in sweep() | st(fmin=5.0, fmax=100.0)])    # (frequencies, time)
multifilter = np.array([np.asarray(t) for t in sweep() | gabor(band=5.0)])
```

**Your own stage.** Any function of one trace is a stage with `per_trace`, and `stage` makes it chainable, and usable with
`pmap` (`parallelism='trace'` says that traces are independent of each other).

```python
from seispy.stage import per_trace, stage

def _scale_by_offset(upstream, *, per_km=0.1):
    def scaled(trace):
        return trace.replace(np.asarray(trace) * (1.0 + per_km * abs(trace.header['offset']) / 1000.0))
    return per_trace(upstream, scaled)

scale_by_offset = stage(_scale_by_offset, parallelism='trace', name='scale_by_offset', validate=True)
traces = synlv() | scale_by_offset(per_km=0.5)
```

**SEG-Y.** `read_segy` gives an iterator of traces, with the coordinates scaled, the angles in degrees, and the times in
seconds. It can be the left hand side of a pipe.

```python
from seispy.io import read_segy

traces = read_segy("line.sgy") | bfilt(f_pass_low=40.0, f_stop_low=50.0) | gain(agc=True, wagc=0.5)
```

## Free threaded python
The extension modules say that they can run without the GIL, so they work with the free threaded builds of python (3.14t and
later, `python3.14t`), and importing them does not turn the GIL back on. The SU library routines that the stages call have no
state that threads share: their lookup tables and scratch arrays are arguments, the tables that SU made on first use (the
Hilbert transform, the sinc interpolation) are made when the modules are imported, and the random numbers of `addnoise`,
`jitter` and `randspike` come from a generator with its own state for each stage. So the parallelism below (`prefetch` and
`pmap`) needs no GIL to be free of it: the python parts of the stages run in parallel too, not only the C kernels.

What can be shared by threads is the stages (the specifications that are chained with `|`) and the traces, which are not changed
by the stages. What can not is an iterator in the middle of a pipeline: take its traces from one thread at a time (which is
what `prefetch` and `pmap` do). `tests/test_free_threading.py` has the threads that use the same stages and traces at once.

### Stable ABI (abi3) wheels
The modules are built against the limited C API of python 3.12, so one `cp312-abi3` wheel per platform works for every
GIL-enabled CPython from 3.12 on. The free threaded builds have no stable ABI (python 3.14t has none, and Cython can not yet
generate code for the one of 3.15t), so they get their own wheels (`cp314t`, `cp315t`) that are built without it. To build in
place with a free threaded python, turn the limited API off:

```
pip install --no-build-isolation --editable . --config-settings=setup-args="-Dpython.allow_limited_api=false"
```

## Temporary files
`sort` and `flip` read all of their traces before they write the first. They keep them in memory while they fit in a budget
(1 GiB), and past that they put them on disk: `sort` as chunks that it merges, so that data bigger than memory can be sorted,
and `flip` as a file that is memory mapped. When they do, they say so, loudly: a warning (from the logger `seispy.spool`, which
python prints to stderr unless logging is set up otherwise) names the stage and the temporary directory, and the directory is
removed when the stage is done, whether it finished, failed or was stopped.

```python
source | sort('ensemble_number', 'offset')                         # memory, or the default directory if it must
source | sort('ensemble_number', tmpdir='D:/scratch', memory=2**28)  # a directory, and a smaller budget
source | sort('offset', memory=0)                                    # everything on disk, as susort does
source | sort('offset', tmpdir=False)                                # never the disk
```

The directory is, from the first that is set: the `tmpdir` of the stage, `seispy.spool.tmpdir`, the environment variable
`SEISPY_TMPDIR`, `CWP_TMPDIR` (the one the SU programs use), and the temporary directory of the system. The budget likewise:
`memory`, `seispy.spool.memory`, `SEISPY_MEMORY` (bytes). The stages that transform a whole panel (`specfk`, `dipdivcor`,
`taup`, ...) need the panel in memory to do it, so they have no `tmpdir`.

## Pipes and parallelism
Processing steps can be written as input-less *stages* and chained with `|`, just like the shell:

```python
from seispy.synthetics import spike
from seispy.filters import bfilt

traces = spike() | bfilt(f_pass_low=10.0, f_stop_low=5.0) | bfilt(zerophase=False)
```

Any iterable of `Trace` objects (e.g. your own generator) can be the left hand side of a pipe.

In a unix pipe every program runs at the same time, and `xargs -P` runs one program on several pieces of data at once.
`seispy.parallel` provides both, using only the standard library:

```python
from seispy.parallel import prefetch, pmap

# pipeline parallelism: each side of a `prefetch()` runs in its own thread, with a bounded buffer between them
spike() | bfilt(...) | prefetch() | bfilt(...)

# data parallelism: several workers on different chunks of the stream, results come back in order
spike() | pmap(bfilt(...) | bfilt(...), workers=8, chunk=16)
```

Stages declare how finely they can be split: `parallelism='trace'` (cut anywhere), `'ensemble'` (cut only between
gathers, give `pmap` a `key=` header name), or `'serial'` (the default; `pmap` refuses it). Threads give a speedup
for stages that release the GIL (the SU filter kernels do), otherwise use `executor='process'`.
See `examples/parallel_benchmark.py`.


## Available programs
Stages are named after the SU program they port, without the `su` prefix, and live in packages named after SU's own
categories. Their parameters are the SU parameters, as keyword arguments.

| seispy | SU | |
|---|---|---|
| `seispy.synthetics.spike`, `plane`, `synlv` | `suspike`, `suplane`, `susynlv` | sources |
| `seispy.filters.bfilt` | `subfilt` | Butterworth filters |
| `seispy.filters.filter` | `sufilter` | zero-phase, tapered polygonal filter |
| `seispy.amplitudes.gain` | `sugain` | tpow, epow, gpow, agc, clipping, balancing, ... |
| `seispy.operations.*` | `suop` | one stage per operation: `abs()`, `sqr()`, `slog10()`, `diff()`, `mean(nw=11)`, ... |
| `seispy.amplitudes.zero`, `nan`, `normalize` | `suzero`, `sunan`, `sunormalize` | zero a time window, replace NaNs and Infs, normalize by rms/max/median |
| `seispy.amplitudes.weight`, `ai2r`, `r2ai` | `suweight`, `suai2r`, `sur2ai` | weight traces by a header value, impedance to reflectivity and back |
| `seispy.tapering.taper`, `ramp` | `sutaper`, `suramp` | taper the start and end of traces, and/or the edge traces of a panel |
| `seispy.windowing.mute`, `wind`, `kill` | `sumute`, `suwind`, `sukill` | mute above/below a curve (modes 0-4), window by header value and in time, zero traces |
| `seispy.stretching.shift`, `resamp`, `reduce`, `nmo` | `sushift`, `suresamp`, `sureduce`, `sunmo` | shift/window in time, sinc resampling, reduced time, NMO with velocity functions of time and CDP |
| `seispy.transforms.hilb`, `zerophase` | `suhilb`, `suzerophase` | Hilbert transform (SU's own FIR, whose sign is the opposite of the usual one), zero-phase equivalent |
| `seispy.transforms.analytic`, `fft`, `ifft`, `real`, `imag`, `amp`, `logamp`, `phase` | `suanalytic`, `sufft`, `suifft`, `suamp` | complex traces: make them, and take them apart |
| `seispy.filters.frac`, `phase` | `sufrac`, `suphase` | fractional derivative/integral plus a phase shift, linear phase manipulation (numpy FFT) |
| `seispy.filters.minphase`, `tvband` | `suminphase`, `sutvband` | minimum phase equivalent (Kolmogoroff), time-variant bandpass (numpy FFT) |
| `seispy.decon.pef`, `shape` | `supef`, `sushape` | Wiener predictive/spiking deconvolution, Wiener shaping filter (SU's Toeplitz solver) |
| `seispy.stretching.log`, `ilog`, `tsq`, `ttoz`, `ztot` | `sulog`, `suilog`, `sutsq`, `suttoz`, `suztot` | log and time-squared stretch, time-to-depth and depth-to-time resampling |
| `seispy.convolution.acor`, `conv`, `xcor` | `suacor`, `suconv`, `suxcor` | auto-correlation, convolution and cross-correlation with a filter |
| `seispy.attributes.amp`, `phase`, `freq`, `q`, ... | `suattributes` | instantaneous attributes, one stage per mode |
| `seispy.waveforms.akb`, `berlage`, `gauss`, `gaussd`, `ricker1`, `ricker2`, `spike`, `unit`, `dgauss` | `suwaveform`, `sudgwaveform` | wavelets (computed by the SU library), one function for each type: sources |
| `seispy.waveforms.vibro_linear`, `vibro_segments`, `vibro_octave`, `vibro_hertz`, `vibro_tpower` | `suvibro` | Vibroseis sweeps, with tapers: sources |
| `seispy.synthetics.null`, `randspike` | `sunull`, `surandspike` | traces of zeros, random spikes: sources |
| `seispy.noise.addnoise`, `addflatnoise`, `jitter` | `suaddnoise`, `sujitter` | Gaussian or uniform noise at a signal to noise ratio (optionally band limited), random time shifts |
| `seispy.amplitudes.divcor`, `pgc`, `centsamp`, `impedance` | `sudivcor`, `supgc`, `sucentsamp`, `suimpedance` | divergence correction, programmed gain control, centroid samples of lobes (SU's code), reflectivity to impedance |
| `seispy.convolution.acorfrac`, `refcon` | `suacorfrac`, `surefcon` | fractional autocorrelation/convolution, refraction convolution of forward and reverse shots |
| `seispy.transforms.clogfft`, `iclogfft`, `cepstrum`, `icepstrum`, `wfft` | `suclogfft`, `suiclogfft`, `sucepstrum`, `suicepstrum`, `suwfft` | complex log spectrum with phase unwrapping (SU's code), the cepstrum, spectrum flattening |
| `seispy.transforms.st`, `gabor`, `cwt` | `sust`, `sugabor`, `sucwt` | time-frequency panels: Stockwell transform, multifilter analysis, wavelet transform |
| `seispy.windowing.vlength` | `suvlength` | make traces the same length |
| `seispy.stacking.stack`, `divstack`, `pws`, `stackup` | `sustack`, `sudivstack`, `supws`, `sustackup` | stack the traces of each gather (a run of equal `key`): mean, diversity, phase-weighted; stacking to any key combination in any order |
| `seispy.windowing.sort`, `mixgathers` | `susort`, `sumixgathers` | sort by header values (on disk when it does not fit in memory), fill the gaps of a gather from another |
| `seispy.operations.mix`, `sum2`, `diff2`, `prod2`, `quo2`, `ptsum`, `ptdiff`, `ptprod`, `ptquo`, `zipper`, `zippol` | `sumix`, `suop2` | moving average over traces, arithmetic on two data sets (or a data set and a trace), complex traces from two real ones |
| `seispy.filters.median`, `medmix` | `sumedian` | median or mix about a moveout curve, to suppress events that have that moveout |
| `seispy.stretching.taupnmo` | `sutaupnmo` | NMO of tau-p traces, for a velocity function of tau and CDP, with the ray parameter from a header value or a function |
| `seispy.amplitudes.dipdivcor` | `sudipdivcor` | dip-dependent divergence correction in the wavenumber domain, for the traces of the stream as one panel |
| `seispy.transforms.specfx`, `specfk`, `speck1k2` | `suspecfx`, `suspecfk`, `suspeck1k2` | amplitude spectra: of each trace, f-k of a panel, and 2D (k1, k2) of a panel (numpy's FFT; the axis across the traces is described in the docstring) |
| `seispy.filters.dipfilt` | `sudipfilt` | dip (slope) filter in the f-k domain, with a bias slope that is made horizontal first (numpy's FFT) |
| `seispy.transforms.taup` | `sutaup` | forward and inverse slant stacks (tau-p transforms) of a panel, in the t-x and F-K domains (`option` 1 to 4) |
| `seispy.velocity.velan`, `relan` | `suvelan`, `surelan` | stacking velocity semblance of CDP gathers, residual moveout semblance of migrated gathers: one semblance trace per velocity (or r parameter) for each gather (`ensemble_trace_number` counts them) |
| `seispy.windowing.split`, `cleave`, `putgthr` | `susplit`, `sucleave`, `suputgthr` | write the traces that go through them to files, by the value of a header word, by ranges of it, or a file for each gather (in seispy's own format, `.spy`) |
| `seispy.windowing.getgthr`, `sorty` | `sugetgthr`, `susorty` | the traces of the files of a directory (a source), a small shot data set that shows the geometry in the data, to look at sorting |
| `seispy.synthetics.imp2d`, `imp3d` | `suimp2d`, `suimp3d` | Born-integral shot records for a line scatterer (2-D) and a point scatterer (3-D, with the direct arrival if `dir=1`): sources |
| `seispy.synthetics.syncz` | `susyncz` | zero-offset true-amplitude (2.5-D) data over dipping interfaces in constant-velocity layers: a source |
| `seispy.synthetics.goupillaudpo` | `sugoupillaudpo` | primaries-only impulse response of a lossless Goupillaud medium, one seismogram for each reflectivity series |
| `seispy.synthetics.goupillaud` | `sugoupillaud` | impulse response of a lossless Goupillaud medium with the multiples, one seismogram for each reflectivity series (exact for a surface source or a receiver above the source; warns otherwise) |
| `seispy.synthetics.nhmospike`, `addevent` | `sunhmospike`, `suaddevent` | a gather of spikes with parabolic, pseudo-hyperbolic or linear tau-p moveouts (source), and a linear or hyperbolic event added to traces |
| `seispy.tapering.gausstaper` | `sugausstaper` | multiply traces by a gaussian of a header value (the offset) |
| `seispy.operations.flip`, `vcat` | `suflip`, `suvcat` | turn a data set over (rotate, transpose, reverse), append a second data set to the ends of the traces with an overlap |
| `seispy.attributes.mean`, `max`, `quantile`, `histogram`, `cmp` | `sumean`, `sumax`, `suquantile`, `suhistogram`, `sucmp` | report on a data set (and return the results, rather than make traces): L-p means, maxima/minima/rms/threshold peaks, quantiles and ranks, histograms, comparison of two data sets |

```python
from seispy.synthetics import synlv
from seispy.filters import filter
from seispy.amplitudes import gain
from seispy import operations as op

traces = synlv() | gain(tpow=2.0) | filter(f=[10, 20, 60, 80]) | gain(agc=True, wagc=0.2) | op.sgn()
```

### What runs the SU C code
The work of each SU program is a library function in the SU sources (`su_gain`, `su_nmo`, `su_mute_above`, ...), with the
program's `main` left out of the build: the stages call those through Cython, without the GIL. The lookup tables and scratch
arrays that the programs keep in `static` variables, filled in by the first trace, are arguments in the library versions, which
is what lets the stages run in parallel (`tests/test_threads_c.py` checks that). The few places where the library versions
differ from the programs, because the program is plainly wrong, are listed at the top of each source file.

* **The program is the SU code:** `gain`, `bfilt`, `nmo`, `taupnmo`, `taup`, `velan`, `relan`, `imp2d`, `imp3d`, `syncz`, `goupillaudpo`, `goupillaud`, `addevent`, `resamp`, `hilb`, `analytic`, `synlv`, `centsamp`, `mute` (every mode),
  `taper`, `ramp`, the wavelets (`seispy.waveforms`) and the sweeps, `log`, `ilog`, `ttoz`, `ztot` and `tsq`, the attributes
  (`seispy.attributes`), `conv`, `acor`, `xcor` and `refcon` (the SU convolution and correlation), `pgc`, the stacks
  (`stack`, `divstack`, `pws`, `stackup`), and `median` and `medmix`. The random numbers of `addnoise`, `addflatnoise`, `jitter`
  and `randspike` are those of SU's generators, so for the same `seed` they make the numbers that the programs make.
* **numpy's FFT, with the SU code between the transforms:** `dipdivcor` (the transform in x), `specfx`, `specfk`, `speck1k2`, `dipfilt`, `frac`, `phase`, `minphase`, `tvband`, `wfft`, `acorfrac`,
  `clogfft`, `iclogfft`, `cepstrum`, `icepstrum` (phase unwrapping is SU's too), `st`, `gabor` and `cwt`. The SU programs
  pad every trace for their prime-factor FFT, which is not carried over. `filter` designs its filter with `polygonalFilter` of the
  SU sources, and filters with numpy's FFT.
* **Plain numpy, where the program is only arithmetic:** the operations of `suop` (all but `saf`, `freq` and `despike`),
  `zero`, `nan`, `normalize`, `weight`, `divcor`, `impedance`, `ai2r`, `r2ai`, `wind`, `kill`, `vlength`, `sort`,
  `mixgathers`, `suop2` (the binary operations), `shift`, `reduce`, `real`, `imag`, `amp`, `fft` and `ifft`,
  `gausstaper`, `flip`, `vcat`, the reports of `seispy.attributes` (`mean`, `max`, `quantile`, `histogram`, `cmp`),
  and the source `null`. (`mix` uses the SU weighted sum.)

The stages do not
support SU parameters that need header words that seispy does not have (`mark`, `tracl`, `muts`, ...) or
temporary files (`tmpdir`). Where SU takes a header word by name (`key=offset`), these take the name of a value in
`trace.header`, or a function of a trace.
