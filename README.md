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
