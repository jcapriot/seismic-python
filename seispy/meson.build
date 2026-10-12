subdir('include')
subdir('io')
subdir('filters')
subdir('amplitudes')
subdir('operations')
subdir('tapering')
subdir('windowing')
subdir('stretching')
subdir('transforms')
subdir('convolution')
subdir('attributes')
subdir('decon')
subdir('stacking')
subdir('velocity')
subdir('migration')
subdir('noise')
subdir('waveforms')
subdir('synthetics')
subdir('welllogs')
subdir('dmo')
subdir('picking')
subdir('statics')
subdir('interp')

python_sources = [
    '__init__.py',
    'plotting.py',
    'stage.py',
    'parallel.py',
    'spool.py',
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
    cpp_args: cython_c_args,
    install: true,
    limited_api: limited_api,
    subdir: module_path,
    dependencies : [py_dep, su_dep],
    include_directories: spy_include,
)

py.extension_module(
    '_fft',
    '_fft.pyx',
    cython_args: cython_args,
    cpp_args: cython_c_args,
    install: true,
    limited_api: limited_api,
    subdir: module_path,
    dependencies : [py_dep, su_dep],
    include_directories: spy_include,
)
