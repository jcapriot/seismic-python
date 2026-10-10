module_path = 'seispy/synthetics'

python_sources = [
    '__init__.py',
    '_sources.py',
    '_patterns.py',
    '_layered.py',
    '_fd.py',
    '_scatterers.py',
]

cython_sources = [
    '_born.pyx',
    '_events.pyx',
    '_fdmod.pyx',
    '_goupillaud.pyx',
    '_kirchhoff.pyx',
    '_plane.pyx',
    '_syncz.pyx',
    '_spike.pyx',
    '_synlv.pyx',
]

foreach cy_source : cython_sources
    py.extension_module(
        cy_source.replace('.pyx', ''),
        cy_source,
        cython_args: cython_args,
        cpp_args: cython_c_args,
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