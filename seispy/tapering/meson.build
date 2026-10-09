module_path = 'seispy/tapering'

python_sources = [
    '__init__.py',
]

py.extension_module(
    '_taperc',
    '_taperc.pyx',
    cython_args: cython_args,
    cpp_args: cython_c_args,
    install: true,
    limited_api: limited_api,
    subdir: module_path,
    dependencies : [py_dep, su_dep],
)

py.install_sources(
  python_sources,
  subdir: module_path
)
