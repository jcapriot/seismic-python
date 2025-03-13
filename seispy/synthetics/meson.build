module_path = 'seispy/synthetics'

python_sources = [
    '__init__.py',
]

cython_sources = [
    '_plane.pyx',
    '_spike.pyx',
    '_synlv.pyx',
]

foreach cy_source : cython_sources
    py.extension_module(
        cy_source.strip('.pyx'),
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