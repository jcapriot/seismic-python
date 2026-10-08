module_path = 'seispy/transforms'

python_sources = [
    '__init__.py',
    '_cepstral.py',
    '_timefreq.py',
]

foreach name : ['_hilbert', '_unwrap', '_kernels']
    py.extension_module(
        name,
        name + '.pyx',
        cython_args: cython_args,
        c_args: cython_c_args,
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
