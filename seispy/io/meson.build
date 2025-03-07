module_path = 'seispy/io'

io_args = []
# check for file io options:
if cc.has_function('fseeko', prefix: '#define _GNU_SOURCE\n#include <stdio.h>')
    io_args += ['-DHAVE_FSEEKO']
endif

if cc.has_function('ftello', prefix: '#define _GNU_SOURCE\n#include <stdio.h>')
    io_args += ['-DHAVE_FTELLO']
endif

py.extension_module(
    '_io',
    '_io.pyx',
    cython_args: cython_args,
    c_args: io_args,
    install: true,
    subdir: module_path,
    dependencies: [py_dep],
    include_directories: spy_include,
)

py.extension_module(
    'byteswapping',
    'byteswapping.pyx',
    cython_args: cython_args,
    c_args: io_args,
    install: true,
    subdir: module_path,
    dependencies: [py_dep],
    include_directories: spy_include,
)

py.extension_module(
    'segy_standard',
    'segy_standard.pyx',
    cython_args: cython_args,
    c_args: io_args,
    install: true,
    subdir: module_path,
    dependencies: [py_dep, su_dep],
    include_directories: spy_include,
)

python_sources = ['__init__.py']

py.install_sources(
  python_sources,
  subdir: module_path
)