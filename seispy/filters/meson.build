module_path = 'seispy/filters'

python_sources = [
    '__init__.py',
    '_fftfilter.py',
    '_moveout.py',
    '_phasefilters.py',
    '_spectral.py',
]

cython_sources = [
    '_bandpass.pyx',
    '_filter.pyx',
    '_kernels.pyx',
]

foreach cy_source : cython_sources
    py.extension_module(
        cy_source.replace('.pyx', ''),
        cy_source,
        cython_args: cython_args,
        c_args: cython_c_args,
        install: true,
        limited_api: limited_api,
        subdir: module_path,
        dependencies : [py_dep, su_dep],
    )
endforeach

py.install_sources(
  python_sources,
  subdir: module_path
)
