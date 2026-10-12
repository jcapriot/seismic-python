module_path = 'seispy/velocity'

python_sources = [
    '__init__.py',
    '_extra.py',
]

foreach name : ['_kernels', '_more']
    py.extension_module(
        name,
        name + '.pyx',
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
