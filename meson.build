project(
    'seispy',
    'c', 'cpp', 'cython',
    version : '0.0.1a0',
    license: 'MIT',
    # (the SU sources are C++ too, so the extension modules are, and no declaration needs to be wrapped in extern "C")
    default_options: ['cython_language=cpp', 'cpp_std=c++17'],
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

m_dep = cc.find_library('m', required : false)
if m_dep.found()
  add_project_link_arguments('-lm', language : ['c', 'cpp'])
endif

_global_c_args = cc.get_supported_arguments(
  '-Wno-unused-but-set-variable',
  '-Wno-unused-function',
  '-Wno-conversion',
  '-Wno-misleading-indentation',
)
add_project_arguments(_global_c_args, language : 'c')
add_project_arguments(cpp.get_supported_arguments(_global_c_args), language : 'cpp')
# (the SU headers have string literals as `char *`, which GCC and clang allow in C++, and MSVC needs to be told to)
if cpp.get_id() == 'msvc'
  add_project_arguments('/Zc:strictStrings-', language : 'cpp')
else
  add_project_arguments(cpp.get_supported_arguments('-Wno-write-strings', '-Wno-writable-strings'), language : 'cpp')
endif

# NumPy include directory
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_22_API_VERSION']
np_dep = dependency('numpy')

# Deal with M_PI & friends; add `use_math_defines` to c_args or cpp_args
# Cython doesn't always get this right itself (see, e.g., gh-16800), so
# explicitly add the define as a compiler flag for Cython-generated code.
is_windows = host_machine.system() == 'windows'
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

c_undefined_ok = ['-Wno-maybe-uninitialized']

cython_c_args = [numpy_nodepr_api, use_math_defines]
# (the `linetrace` directive of the modules only does something if this is defined: -Dline_tracing=true)
if get_option('line_tracing')
  cython_c_args += ['-DCYTHON_TRACE_NOGIL=1']
endif

# Target the 3.12 limited API, so that one abi3 wheel covers every GIL-enabled CPython >= 3.12 (see [tool.meson-python] in
# pyproject.toml). Meson only uses this while `python.allow_limited_api` is true: `-Dpython.allow_limited_api=false` turns it
# off, which free-threaded builds need (3.14t has no stable ABI, and Cython can not yet target the one of 3.15t).
limited_api = '3.12'

cython_args = []
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

# (the warnings of the SU sources, which are upstream's, are not shown)
su_proj = subproject('su', default_options: ['warning_level=0'])
su_dep = su_proj.get_variable('su_dep')

subdir('seispy')