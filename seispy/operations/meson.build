module_path = 'seispy/operations'

python_sources = [
    '__init__.py',
    '_numpy_ops.py',
    '_panels.py',
    '_dataset.py',
]

cython_sources = [
    '_ops.pyx',
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
