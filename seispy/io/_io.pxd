# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from libc.stdio cimport FILE


cdef extern from * nogil:
    """
    #ifndef _IO_HEADER_PXD
    #define _IO_HEADER_PXD

    #include <stdio.h>
    #include <stddef.h>
    #include <stdint.h>
    #include "Python.h"

    #if defined _MSC_VER && _MSC_VER >= 1900

        #include <stdlib.h>   // _set_thread_local_invalid_parameter_handler()
        /*
         * Macros to protect CRT calls against instant termination when passed an
         * invalid parameter (https://bugs.python.org/issue23524).
         */
        extern _invalid_parameter_handler _Py_silent_invalid_parameter_handler;
        #define _BEGIN_SUPPRESS_IPH { _invalid_parameter_handler _Py_old_handler = \
            _set_thread_local_invalid_parameter_handler(_Py_silent_invalid_parameter_handler);
        #define _END_SUPPRESS_IPH _set_thread_local_invalid_parameter_handler(_Py_old_handler); }

    #else

        #define _BEGIN_SUPPRESS_IPH
        #define _END_SUPPRESS_IPH

    #endif

    #ifdef _WIN32
    static FILE *spy_fdopen(int fd, const char *mode){
        FILE *f;
        _BEGIN_SUPPRESS_IPH
        f = _fdopen(fd, mode);
        _END_SUPPRESS_IPH
        return f;
    }
    #else
        #define spy_fdopen fdopen
    #endif

    /* 64 bit file position support, also on win-amd64. Issue gh-2256 */
    #if defined(_MSC_VER) && defined(_WIN64) && (_MSC_VER > 1400) || \
        defined(__MINGW32__) || defined(__MINGW64__)
        #include <io.h>
        #include <stdint.h>

        #define spy_off_t int64_t
        #define spy_fseek _fseeki64
        #define spy_ftell _ftelli64
        #define spy_lseek _lseeki64
    #else
        #ifdef HAVE_FSEEKO
            #define spy_fseek fseeko
        #else
            #define spy_fseek fseek
        #endif
        #ifdef HAVE_FTELLO
            #define spy_ftell ftello
        #else
            #define spy_ftell ftell
        #endif
        #include <sys/types.h>
        #ifndef _WIN32
            #include <unistd.h>
        #endif
        #define spy_lseek lseek
        #define spy_off_t off_t
    #endif

    #endif
    """
    ctypedef int spy_off_t

    FILE *spy_fdopen(int fd, const char * mode)
    spy_off_t spy_lseek(int fd, spy_off_t offset, int whence)
    int spy_fseek(FILE *stream, spy_off_t offset, int whence)
    spy_off_t spy_ftell(FILE *stream)



cdef (FILE *, spy_off_t) PyFile_Dup(object file, char* mode)
cdef int PyFile_DupClose(object file, FILE* handle, spy_off_t orig_pos)

cdef size_t write_struct_to_file(
    char *st,
    size_t *offsets,
    size_t *sizes,
    size_t *n_elements,
    size_t n_attr,
    FILE *fd
) noexcept nogil

cdef size_t read_struct_from_file(
    char *st,
    size_t *offsets,
    size_t *sizes,
    size_t *n_elements,
    size_t n_attr,
    FILE *fd
) noexcept nogil

cdef void copy_struct_to_char(
    char *st,
    size_t *offsets,
    size_t *sizes,
    size_t *n_elements,
    size_t n_attr,
    char *out
) noexcept nogil

cdef void copy_struct_from_char(
    char *st,
    size_t *offsets,
    size_t *sizes,
    size_t *n_elements,
    size_t n_attr,
    char *out
) noexcept nogil

cpdef size_t[:,::1] struct_dtype_info(object struct_dtype)