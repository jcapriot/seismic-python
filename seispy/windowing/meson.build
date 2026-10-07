module_path = 'seispy/windowing'

python_sources = [
    '__init__.py',
]

py.extension_module(
    '_mutec',
    '_mutec.pyx',
    cython_args: cython_args,
    c_args: cython_c_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, su_dep],
)

py.install_sources(
  python_sources,
  subdir: module_path
)
