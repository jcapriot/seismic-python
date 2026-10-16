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
| `seispy.synthetics.spike`, `plane`, `synlv`, `synlvcw`, `synlvfti` | `suspike`, `suplane`, `susynlv`, `susynlvcw`, `susynlvfti` | sources (`synlvcw` adds the mode conversion `gamma`, `synlvfti` a transversely isotropic medium) |
| `seispy.filters.bfilt` | `subfilt` | Butterworth filters |
| `seispy.filters.filter` | `sufilter` | zero-phase, tapered polygonal filter |
| `seispy.amplitudes.gain` | `sugain` | tpow, epow, gpow, agc, clipping, balancing, ... |
| `seispy.operations.*` | `suop` | one stage per operation: `abs()`, `sqr()`, `slog10()`, `diff()`, `mean(nw=11)`, ... |
| `seispy.amplitudes.zero`, `nan`, `normalize` | `suzero`, `sunan`, `sunormalize` | zero a time window, replace NaNs and Infs, normalize by rms/max/median |
| `seispy.amplitudes.weight`, `ai2r`, `r2ai` | `suweight`, `suai2r`, `sur2ai` | weight traces by a header value, impedance to reflectivity and back |
| `seispy.tapering.taper`, `ramp` | `sutaper`, `suramp` | taper the start and end of traces, and/or the edge traces of a panel |
| `seispy.tapering.txtaper` | `sutxtaper` | taper in x and t: by the position of the traces in the panel or by a header value, and the ends of the traces |
| `seispy.windowing.mute`, `wind`, `kill` | `sumute`, `suwind`, `sukill` | mute above/below a curve (modes 0-4), window by header value and in time, zero traces |
| `seispy.stretching.shift`, `resamp`, `reduce`, `nmo` | `sushift`, `suresamp`, `sureduce`, `sunmo` | shift/window in time, sinc resampling, reduced time, NMO with velocity functions of time and CDP |
| `seispy.stretching.nmo_a`, `dlmo` | `sunmo_a`, `sudlmo` | NMO with a quartic (anisotropy) term in the traveltime, dynamic linear moveout of surface waves (the velocity is a function of frequency) |
| `seispy.transforms.hilb`, `zerophase` | `suhilb`, `suzerophase` | Hilbert transform (SU's own FIR, whose sign is the opposite of the usual one), zero-phase equivalent |
| `seispy.transforms.analytic`, `fft`, `ifft`, `real`, `imag`, `amp`, `logamp`, `phase` | `suanalytic`, `sufft`, `suifft`, `suamp` | complex traces: make them, and take them apart |
| `seispy.filters.frac`, `phase` | `sufrac`, `suphase` | fractional derivative/integral plus a phase shift, linear phase manipulation (SU's FFT) |
| `seispy.filters.minphase`, `tvband` | `suminphase`, `sutvband` | minimum phase equivalent (Kolmogoroff), time-variant bandpass (SU's FFT) |
| `seispy.decon.pef`, `shape` | `supef`, `sushape` | Wiener predictive/spiking deconvolution, Wiener shaping filter (SU's Toeplitz solver) |
| `seispy.decon.cddecon`, `phidecon`, `fxdecon` | `sucddecon`, `suphidecon`, `sufxdecon` | deconvolution by complex division with a given filter, phase inversion deconvolution (Lichman and Northwood), f-x prediction filtering for random noise (the SU library function, on a panel in memory) |
| `seispy.stretching.log`, `ilog`, `tsq`, `ttoz`, `ztot` | `sulog`, `suilog`, `sutsq`, `suttoz`, `suztot` | log and time-squared stretch, time-to-depth and depth-to-time resampling |
| `seispy.convolution.acor`, `conv`, `xcor` | `suacor`, `suconv`, `suxcor` | auto-correlation, convolution and cross-correlation with a filter |
| `seispy.attributes.amp`, `phase`, `freq`, `q`, ... | `suattributes` | instantaneous attributes, one stage per mode |
| `seispy.waveforms.akb`, `berlage`, `gauss`, `gaussd`, `ricker1`, `ricker2`, `spike`, `unit`, `dgauss` | `suwaveform`, `sudgwaveform` | wavelets (computed by the SU library), one function for each type: sources |
| `seispy.waveforms.vibro_linear`, `vibro_segments`, `vibro_octave`, `vibro_hertz`, `vibro_tpower` | `suvibro` | Vibroseis sweeps, with tapers: sources |
| `seispy.synthetics.null`, `randspike` | `sunull`, `surandspike` | traces of zeros, random spikes: sources |
| `seispy.noise.harlan` | `suharlan` | signal and noise separation by Harlan's invertible linear transformation (the tau-p transform, histograms of the data and of sign reversed data, and a reliability mask); the noise and the signal are the `report` and the stream (SU library function; the program's plots and static state are not here) |
| `seispy.noise.sembase`, `semgroup` | `susembase`, `susemgroup` | semblance based structure oriented smoothing (and, for `sembase`, interpolation to new locations): dips are scanned by semblance (SU library functions), and the dip corrected traces within a range are summed; the dips, powers and semblances are the `report` and can be given back as `indip` |
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
| `seispy.transforms.specfx`, `specfk`, `speck1k2` | `suspecfx`, `suspecfk`, `suspeck1k2` | amplitude spectra: of each trace, f-k of a panel, and 2D (k1, k2) of a panel (SU's FFT; the axis across the traces is described in the docstring) |
| `seispy.filters.dipfilt` | `sudipfilt` | dip (slope) filter in the f-k domain, with a bias slope that is made horizontal first (SU's FFT) |
| `seispy.transforms.taup` | `sutaup` | forward and inverse slant stacks (tau-p transforms) of a panel, in the t-x and F-K domains (`option` 1 to 4) |
| `seispy.velocity.velan`, `relan` | `suvelan`, `surelan` | stacking velocity semblance of CDP gathers, residual moveout semblance of migrated gathers: one semblance trace per velocity (or r parameter) for each gather (`ensemble_trace_number` counts them) |
| `seispy.windowing.split`, `cleave`, `putgthr` | `susplit`, `sucleave`, `suputgthr` | write the traces that go through them to files, by the value of a header word, by ranges of it, or a file for each gather (in seispy's own format, `.spy`) |
| `seispy.windowing.getgthr`, `sorty` | `sugetgthr`, `susorty` | the traces of the files of a directory (a source), a small shot data set that shows the geometry in the data, to look at sorting |
| `seispy.synthetics.synvxz`, `synvxzcs` | `susynvxz`, `susynvxzcs` | Kirchhoff-style common-offset sections and common-shot gathers in a V(x,z) medium (the velocity, and the slowness perturbation of `synvxzcs`, are arrays `(nz, nx)`, not files): sources |
| `seispy.synthetics.kdsyn2d` | `sukdsyn2d` | Kirchhoff demigration of a migrated section into common-shot gathers (the section and the traveltime tables are arrays, not files): a source |
| `seispy.synthetics.fdmod1`, `fdmod1_snapshots`, `fdmod2`, `fdmod2_pml` | `sufdmod1`, `sufdmod2`, `sufdmod2_pml` | finite-difference modelling of the acoustic wave equation: the model is an array, the seismograms are trace streams and the wavefield is `Snapshots` (an iterable of frames) |
| `seispy.synthetics.ea2df` | `suea2df` | (an)elastic anisotropic 2-D finite-difference forward modelling, fourth order in space: the model is arrays, the seismograms are trace streams, the wavefield is `Snapshots` at the times asked for |
| `seispy.filters.kfilter`, `k1k2filter` | `sukfilter`, `suk1k2filter` | sin^2 tapered polygonal filters of the (k1, k2) wavenumbers of a panel: radially symmetric, and the product of one in each of k1 and k2 (SU's FFT, with `polygonalFilter` of SU for the second; the gain of the first is 1 in its passband) |
| `seispy.filters.kfrac` | `sukfrac` | a fractional power of `i|k|` with a phase shift, in the (k1, k2) domain of a panel, optionally directional (SU's FFT) |
| `seispy.filters.smgauss2` | `susmgauss2` | Gaussian smoothing of a panel of velocities (as slowness, as its documentation says) |
| `seispy.filters.lfaf`, `ccfilt`, `fwatrim` | `sulfaf`, `succfilt`, `sufwatrim` | FX domain filters of gathers: low frequency trace mixing, removal of the frequencies that do not correlate in phase between traces, and an alpha trimmed mean of the traces of a gather |
| `seispy.filters.deghost` | `sudeghost` | deghosting (or ghosting) in (t, x) or (tau, p), with the delay of the ghost a function of the slowness |
| `seispy.filters.svdfilt` | `susvdfilt` | SVD (eigenimage) filter: the first principal planes of a window of traces, kept or subtracted |
| `seispy.welllogs.backus`, `backush`, `lprime` | `subackus`, `subackush`, `sulprime` | Thomsen parameters of the Backus average of vp, vs and rho logs (with the intrinsic anisotropy of shale from a gamma ray log), and the averaging length for a Backus number |
| `seispy.welllogs.wellrf`, `las2su` | `suwellrf`, `las2su` | the normal incidence reflectivity of time from depth, velocity and density logs (resampled with `ints8r`), and the curves of a LAS file as logs |
| `seispy.welllogs.gassmann`, `rock_*`, `fluid_*`, `brine_*`, `oil_*`, `gas_*` | `sugassmann` | Gassmann fluid substitution with the Batzle and Wang fluid properties: the functions of SU's library, applied to arrays |
| `seispy.dmo.dmofk`, `dmotx`, `dmofkcw` | `sudmofk`, `sudmotx`, `sudmofkcw` | dip moveout of NMO corrected common-offset gathers, mixed over `noffmix` offsets: Hale's F-K (log stretch) and T-X (Kirchhoff) methods, and the F-K method for converted waves; the gathers are cut where the offset changes, the cdp is `ensemble_number` |
| `seispy.dmo.dmovz`, `dmotivz` | `sudmovz`, `sudmotivz` | DMO for V(z) and transversely isotropic V(z) media (Artley; Alkhalifah): rays traced through arrays of interval velocities (and eta), the DMO of dip decomposition |
| `seispy.dmo.tihaledmo` | `sutihaledmo` | DMO of one common-offset gather in the f-k domain, with six operators (Hale, Bleistein two ways, Zhang, Tsvankin anisotropic and weakly anisotropic) |
| `seispy.filters.fwmix`, `groll` | `sufwmix`, `sugroll` | the weighted mix of a gather in the frequency domain (one trace per gather), and ground roll suppression by the Karhunen-Loeve transform (the singular vectors of the panel) |
| `seispy.operations.permute`, `recmo`, `simmat` | `supermute`, `surecmo`, `susimmat` | transpose a 3D cube of traces, compensate a moving streamer by mixing with the traces before, and the similarity matrix of two traces (cross-correlation, normalized, regression coefficient or angle) |
| `seispy.stacking.recip`, `cvs4fowler` | `surecip`, `sucvs4fowler` | sum the traces of the same absolute offset, and constant velocity stacks of CDP gathers for the Fowler codes |
| `seispy.multicomponent.hrot`, `polar`, `pofilt`, `eipofi` | `suhrot`, `supolar`, `supofilt`, `sueipofi` | three-component data: rotation to radial and transverse, polarization attributes (a dict of traces), polarization filters from them, and by the SVD of eigenimages. A station is three traces: consecutive, or grouped by a key, in the order of `trace_type` codes |
| `seispy.multicomponent.ltt`, `alford` | `sultt`, `sualford` | four-component shear wave data (S11, S12, S21, S22 as the four traces of a station): the polarization angle by the linear transform technique, and the Alford rotation into the natural coordinates |
| `seispy.datuming.datumk2dr`, `datumk2ds` | `sudatumk2dr`, `sudatumk2ds` | Kirchhoff datuming of the receivers of shot gathers and of the sources of receiver gathers between two spline surfaces, from traveltime tables as arrays (SU library functions; the program's out-of-range reads are errors or kept inside) |
| `seispy.datuming.kdmdcr`, `kdmdcs` | `sukdmdcr`, `sukdmdcs` | 2.5-D data mapping datuming of the receivers of shot gathers / the sources of receiver gathers from a recording surface to a datum given by depths, from traveltime tables (SU library functions) |
| `seispy.datuming.datumfd` | `sudatumfd` | zero-offset finite difference datuming of shot gathers by the one-way wave equation, with the velocity as `(mx, mz)` arrays (SU library function; the frequency step of the program is corrected) |
| `seispy.interp.interp`, `interpfowler` | `suinterp`, `suinterpfowler` | interpolation of traces along dips that are picked from the low frequencies (SU library function, on a gather in memory), an output image from constant velocity panels |
| `seispy.dmo.ocext` | `suocext` | extrapolation of common-offset gathers to a smaller offset by offset continuation (DMO of `dmofk`, to an offset that is given) |
| `seispy.velocity.vel2df`, `relanan` | `suvel2df`, `surelanan` | semblance over the NMO and horizontal velocities (eta) at one time, residual moveout semblance for anisotropic media (SU library functions, on gathers) |
| `seispy.velocity.velan_uccs`, `velan_nccs`, `velan_usel`, `velan_nsel` | `suvelan_uccs`, `suvelan_nccs`, `suvelan_usel`, `suvelan_nsel` | stacking velocity panels from the sum of the cross-correlations of the trace pairs (unnormalized, normalized, and selecting the pairs of large differential moveout) |
| `seispy.velocity.tivel` | `sutivel` | velocity tables (NMO, phase, derivatives, as functions of p or angle) of a transversely isotropic medium, as arrays |
| `seispy.statics.static`, `staticrrs` | `sustatic`, `sustaticrrs` | elevation statics from the elevations or from tables of statics (by geometry or key), residual and average refraction statics (SU's sinc interpolation; the statics are in seconds, in the header) |
| `seispy.statics.randstat`, `addstatics` | `surandstat`, `suaddstatics` | random static errors on each trace, random surface consistent statics (SU's random numbers) |
| `seispy.statics.resstat` | `suresstat` | surface consistent residual statics of sources and receivers by stack power maximization (the SU library function, on the traces in memory; it returns the statics, which `static(mode='keyed')` applies) |
| `seispy.windowing.pad`, `matrix` | `supad`, `sumatrix` | zero traces for the missing values of a key in each panel, a matrix of trace zones by source and receiver |
| `seispy.transforms.slowft`, `slowift` | `suslowft`, `suslowift` | Fourier transforms by the discrete sum, with no padding |
| `seispy.transforms.phasevel`, `ccwt` | `suphasevel`, `succwt` | phase velocity dispersion image of a shot record, complex continuous wavelet transform (Morlet or Fourier kernel) |
| `seispy.transforms.radon` | `suradon` | the generalized (parabolic, pseudo hyperbolic, linear) Radon transform of a gather: tau-p transform and its inverse, and removal of multiples (the SU library functions, on gathers in memory) |
| `seispy.windowing.mutecsv`, `windpoly`, `padmissing`, `braid`, `gprfb` | `sumutecsv`, `suwindpoly`, `supadmissing`, `subraid`, `sugprfb` | mutes that interpolate between CDPs, windows by a polygon or polyline, dead traces for missing key values, merging of ordered streams, and removal of the first break of GPR data |
| `seispy.picking.fbpickw`, `fnzero`, `pickamp` | `sufbpickw`, `sufnzero`, `supickamp` | first break picks, the time of the first and last non-zero sample, and amplitudes picked in a window (functions that return their results) |
| `seispy.velocity.intvel`, `stkvel` | `suintvel`, `sustkvel` | interval velocities from stacking velocities and the stacking velocities of a dipping layer model (Schneider's formulas) |
| `seispy.synthetics.remac2d`, `remel2dan` | `suremac2d`, `suremel2dan` | acoustic and elastic anisotropic 2-D Fourier-method modelling with the rapid expansion (REM) time integration, free of grid dispersion and of a limit on `dt`: arrays in, section streams and `Snapshots` out |
| `seispy.synthetics.fctanismod` | `sufctanismod` | elastic finite-difference modelling in media with a vertical symmetry axis, with the flux-corrected transport (FCT) against grid dispersion: arrays or profiles in, seismograms as trace streams and the wavefield as `Snapshots` |
| `seispy.migration.stolt` | `sustolt` | Stolt (f-k) migration of stacked data or common-offset gathers, with a v(t) stretch and offset mixing |
| `seispy.migration.migsplit`, `gazmig`, `migtk`, `migpsti` | `sumigsplit`, `sugazmig`, `sumigtk`, `sumigpsti` | split-step depth migration, Gazdag phase shift time migration with attenuation Q, the t-k domain method, and phase shift migration with turning rays for TI media (SU library functions; the TI table's slope range and an out-of-range read are fixed) |
| `seispy.migration.migprefd`, `migpreffd`, `migpresp`, `migprepspi` | `sumigprefd`, `sumigpreffd`, `sumigpresp`, `sumigprepspi` | 2-D prestack common-shot depth migration: finite-difference, Fourier finite-difference, split-step and phase shift plus interpolation (the shots are grouped by source position, one per gather; the program's reading of the traces at the shot boundaries and its Hamming window are fixed) |
| `seispy.migration.migtopo2d` | `sumigtopo2d` | Kirchhoff depth migration from a recording surface of variable topography (a spline), from traveltime tables as for `kdmig2d`; the surface and reference tables are those of the datuming programs (SU library function) |
| `seispy.migration.invvxzco` | `suinvvxzco` | 2.5-D Born inversion (migration with true amplitudes) of a common-offset gather for a smooth v(x, z) and an optional slowness perturbation, by a finite-difference eikonal solver (SU library function; a raypath of too large a curvature is an error where the library stopped the process) |
| `seispy.migration.miggbzo`, `miggbzoan` | `sumiggbzo`, `sumiggbzoan` | migration of zero-offset data by the accumulation of Gaussian beams, for v(x, z) and for an anisotropic (P-wave) medium of elastic coefficients or Thomsen's parameters (SU library functions) |
| `seispy.migration.migfd`, `migffd`, `migpspi` | `sumigfd`, `sumigffd`, `sumigpspi` | finite-difference (45 to 90 degrees), Fourier finite-difference and phase-shift-plus-interpolation depth migration of zero-offset sections, with the velocity as an `(nz, nx)` array |
| `seispy.migration.migps` | `sumigps` | phase-shift migration with turning rays for v(t) |
| `seispy.migration.kdmig2d`, `ktmig2d` | `sukdmig2d`, `suktmig2d` | Kirchhoff depth migration from traveltime tables (arrays), prestack or poststack, and prestack time migration with the double square root operator |
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

Programs that are not here yet: `sukdmig3d` (3-D Kirchhoff depth migration, from the tables of `sutetraray`, which is not ported),
`suinvzco3d` (3-D inversion for v(z), whose code mixes up its grids and its indices) and `sutifowler` (VTI velocity analysis by
Fowler's method, which keeps its data on disk with the `VND` virtual array library), and the programs of the csv Q-files and SPS
header words of `su/main/statics` (`sushiftcsv`,
`surescsv`, `sunszonecsv`, `sufarldcsv`, `sugetqcsv`), which are driven by files and by header words that traces here do
not have.

Programs that are left out on purpose: `succepstrum`, whose code transforms the one-sided spectrum with a complex FFT of
a length that is not valid for it and does not set the length of the trace, so that it does not make the complex cepstrum
that its documentation describes (`cepstrum` and `clogfft` do).

### What runs the SU C code
The work of each SU program is a library function in the SU sources (`su_gain`, `su_nmo`, `su_mute_above`, ...), with the
program's `main` left out of the build: the stages call those through Cython, without the GIL. The lookup tables and scratch
arrays that the programs keep in `static` variables, filled in by the first trace, are arguments in the library versions, which
is what lets the stages run in parallel (`tests/test_threads_c.py` checks that). The few places where the library versions
differ from the programs, because the program is plainly wrong, are listed at the top of each source file.

* **The program is the SU code:** `gain`, `bfilt`, `nmo`, `taupnmo`, `taup`, `velan`, `relan`, `imp2d`, `imp3d`, `syncz`, `goupillaudpo`, `goupillaud`, `addevent`, `resamp`, `hilb`, `analytic`, `synlv`, `synlvcw`, `synlvfti`, `synvxz`, `synvxzcs`, `kdsyn2d`, `fdmod1`, `fdmod2`, `fdmod2_pml`, `ea2df`, `remac2d`, `remel2dan`, `fctanismod`, `centsamp`, `mute` (every mode),
  `taper`, `ramp`, the wavelets (`seispy.waveforms`) and the sweeps, `log`, `ilog`, `ttoz`, `ztot` and `tsq`, the attributes
  (`seispy.attributes`), `conv`, `acor`, `xcor` and `refcon` (the SU convolution and correlation), `pgc`, the stacks
  (`stack`, `divstack`, `pws`, `stackup`), and `median` and `medmix`. The random numbers of `addnoise`, `addflatnoise`, `jitter`
  and `randspike` are those of SU's generators, so for the same `seed` they make the numbers that the programs make.
* **SU's FFT, with the SU code between the transforms:** `dipdivcor` (the transform in x), `specfx`, `specfk`, `speck1k2`, `dipfilt`, `frac`, `phase`, `minphase`, `tvband`, `wfft`, `acorfrac`, `cddecon`, `phidecon`, `phasevel`, `ccwt`, `dlmo`,
  `clogfft`, `iclogfft`, `cepstrum`, `icepstrum` (phase unwrapping is SU's too), `st`, `gabor`, `cwt`, `fft`, `ifft`, `zerophase`
  and the 2-D filters. Every frequency domain operation, in Python and in the C++, goes through one FFT, `seispy._fft`:
  SU's prime-factor FFT (`cwp/lib/pfafft`) in single precision, with its sign convention (`exp(+i w t)` to frequency), no normalization
  on the way to frequency, and its lengths: products of mutually prime factors from 2, 3, 4, 5, 7, 8, 9, 11, 13 and 16. The traces are
  padded for it as the programs pad them (`npfaro(nt, 2 nt)`, and so on), so the outputs have SU's lengths and frequency
  samplings (`seispy._fft.good_length` and `good_length_within` give them). `filter` designs its filter with `polygonalFilter` of the
  SU sources, and filters with the same FFT. Where a program's padding wraps a correlation around (`acorfrac`) the padding is made
  long enough not to.
* **Plain numpy, where the program is only arithmetic:** the operations of `suop` (all but `saf`, `freq` and `despike`),
  `zero`, `nan`, `normalize`, `weight`, `divcor`, `impedance`, `ai2r`, `r2ai`, `wind`, `kill`, `vlength`, `sort`,
  `mixgathers`, `suop2` (the binary operations), `shift`, `reduce`, `real`, `imag`, `amp`, `fft` and `ifft`,
  `gausstaper`, `flip`, `vcat`, the reports of `seispy.attributes` (`mean`, `max`, `quantile`, `histogram`, `cmp`),
  and the source `null`. (`mix` uses the SU weighted sum.)

The stages do not
support SU parameters that need header words that seispy does not have (`mark`, `tracl`, `muts`, ...) or
temporary files (`tmpdir`). Where SU takes a header word by name (`key=offset`), these take the name of a value in
`trace.header`, or a function of a trace.
