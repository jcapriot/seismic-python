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

* `numpy>=1.22.4` To handle some numerical IO
* `matplotlib`

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

```python
from seispy.synthetics import synlv
from seispy.filters import filter
from seispy.amplitudes import gain
from seispy import operations as op

traces = synlv() | gain(tpow=2.0) | filter(f=[10, 20, 60, 80]) | gain(agc=True, wagc=0.2) | op.sgn()
```

`gain`, `filter` and `bfilt` call the SU C code itself. Like `su_bfhighpass` and `su_synlv` before them, the work of
each program is a library function in the SU sources (`su_gain`, `su_filter`/`polygonalFilter`), with the program's
`main` left out of the build. The operations of `suop` are each a stage of their own: most are plain array arithmetic
on the (zero-copy) numpy view of the trace's samples, and the three that are more than that (`op.saf`, `op.freq`,
`op.despike`) are functions in the SU sources. The SU programs keep their lookup tables and scratch arrays in `static` variables filled in by
the first trace, so in the library versions those are arguments, which is what lets the stages run in parallel. The
few places where the library versions differ from the programs, because the program is plainly wrong, are listed at the
top of each source file (`suop.c` has the most, `sugain.c` has a typo in its first-sample agc gain). The stages do not
support SU parameters that need header words that seispy does not have (`mark`, `tracl`, `muts`, ...) or
temporary files (`tmpdir`). Where SU takes a header word by name (`key=offset`), these take the name of a value in
`trace.header`, or a function of a trace.
