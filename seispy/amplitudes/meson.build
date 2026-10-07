module_path = 'seispy/amplitudes'

python_sources = [
    '__init__.py',
    '_pertrace.py',
    '_scan.py',
]

cython_sources = [
    '_gain.pyx',
    '_centsamp.pyx',
    '_pgc.pyx',
]

foreach cy_source : cython_sources
    py.extension_module(
        cy_source.replace('.pyx', ''),
        cy_source,
        cython_args: cython_args,
        c_args: cython_c_args,
        install: true,
        subdir: module_path,
        dependencies : [py_dep, su_dep],
    )
endforeach

py.install_sources(
  python_sources,
  subdir: module_path
)
