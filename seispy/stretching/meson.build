module_path = 'seispy/stretching'

python_sources = [
    '__init__.py',
]

foreach cy_source : ['_resample.pyx', '_nmo.pyx', '_taupnmo.pyx', '_stretch.pyx']
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
