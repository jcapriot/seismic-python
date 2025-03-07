subdir('su')
subdir('include')
subdir('io')
subdir('filters')
subdir('synthetics')

python_sources = [
    '__init__.py',
    'plotting.py',
]

module_path = 'seispy'

py.install_sources(
  python_sources,
  subdir: module_path
)

py.extension_module(
    'container',
    'container.pyx',
    cython_args: cython_args,
    c_args: cython_c_args,
    install: true,
    subdir: module_path,
    dependencies : [py_dep, su_dep],
    include_directories: spy_include,
)
