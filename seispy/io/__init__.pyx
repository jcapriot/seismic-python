# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from cpython.object cimport PyObject_AsFileDescriptor
from libc.stdio cimport FILE, fclose, SEEK_SET, fwrite, fread
from libc.limits cimport INT_MIN, INT_MAX
import os
import io

"""
This module is meant to handle getting a C FILE pointer to an open python file object. It's contents
were adapted from numpy's npy_3kcompat.h file. As suggested in that file, they recommend copying that
file due to there lack of backwards compatibility guarantees.

Permanent link to the version of the reference file is at:

https://github.com/numpy/numpy/blob/a1f2d582f84878b6c67bd641fa671a4cf868fe8c/numpy/_core/include/numpy/npy_3kcompat.h
"""

# The docstring below is raw C code necessary for Windows
cdef extern from *:
    """
    #ifdef _MSC_VER

    #include <stdlib.h>

    #if _MSC_VER >= 1900
    /* _io.h uses this function in the _Py_BEGIN/END_SUPPRESS_IPH
     * macros. It does not need to be defined when building using MSVC
     * earlier than 14.0 (_MSC_VER == 1900).
     */

    static void __cdecl _silent_invalid_parameter_handler(
        wchar_t const* expression,
        wchar_t const* function,
        wchar_t const* file,
        unsigned int line,
        uintptr_t pReserved) { }

    _invalid_parameter_handler _Py_silent_invalid_parameter_handler = _silent_invalid_parameter_handler;

    #endif

    #endif
    """


cdef (FILE *, spy_off_t) PyFile_Dup(object file, char* mode):
    cdef:
        int fd, fd2
        Py_ssize_t fd2_tmp
        spy_off_t pos, orig_pos
        FILE *handle

    file.flush()

    try:
        fd = PyObject_AsFileDescriptor(file)
    except OSError:
        raise IOError(f"Cannot write to a {type(file).__name__} object.")
    if fd == -1:
        return NULL, 0
    fd2_tmp = os.dup(fd)
    if fd2_tmp < INT_MIN or fd2_tmp > INT_MAX:
        raise IOError("Getting an 'int' from os.dup() failed")

    fd2 = <int> fd2_tmp
    handle = spy_fdopen(fd2, mode)
    orig_pos = spy_ftell(handle)
    if orig_pos == -1:
        if isinstance(file, io.RawIOBase):
            return handle, orig_pos
        else:
            fclose(handle)
            raise IOError("obtaining file position failed")

    # raw handle to the Python-side position
    try:
        pos = file.tell()
    except Exception:
        fclose(handle)
    if spy_fseek(handle, pos, SEEK_SET) == -1:
        fclose(handle)
        raise IOError("seeking file failed")
    return handle, orig_pos

cdef int PyFile_DupClose(object file, FILE* handle, spy_off_t orig_pos):
    cdef:
        int fd
        spy_off_t position = spy_ftell(handle)
    fclose(handle)

    # Restore original file handle position,
    fd = PyObject_AsFileDescriptor(file)
    if fd == -1:
        return -1
    if spy_lseek(fd, orig_pos, SEEK_SET) == -1:
        if isinstance(file, io.RawIOBase):
            return 0
        else:
            raise IOError("seeking file failed")
    if position == -1:
        raise IOError("obtaining file position failed")

    # Seek Python-side handle to the FILE* handle position
    file.seek(position)
    return 0

cdef size_t write_struct_to_file(
    char *st, size_t *offsets, size_t *sizes, size_t n_attrs, FILE *fd
) noexcept nogil:
    """
    If a struct has padding between its members, you should use this to
    write to a file, as it explicitly does not include any platform/compiler
    specific padding between members.
    
    Otherwise, you can just use fwrite(&st, sizeof(st), 1, fd)
    """
    cdef size_t i, n_bytes_written
    for i in range(n_attrs):
        n_bytes_written += fwrite(st, 1, sizes[i], fd)
        st += offsets[i]
    return n_bytes_written

cdef size_t read_struct_from_file(
    char *st, size_t *offsets, size_t *sizes, size_t n_attrs, FILE *fd
) noexcept nogil:
    """
    If a struct has padding, you should use this to write to a file,
    as it explicitly does not include any platform/compiler specific
    padding between members.
    
    Otherwise, you can just use fread(&st, sizeof(st), 1, fd)
    """
    cdef size_t i, n_bytes_read
    for i in range(n_attrs):
        n_bytes_read += fread(st, 1, sizes[i], fd)
        st += offsets[i]
    return n_bytes_read

cdef void copy_struct_to_char(
    char *st, size_t *offsets, size_t *sizes, size_t n_attrs, char *out
) noexcept nogil:
    """
    If a struct has padding, and you don't want to include that padding
    in the byte array, you should use this to copy it in, as it explicitly
    does not include any platform/compiler specific padding between members.

    Otherwise, you can just memcpy the two...
    """
    cdef size_t i, j
    for i in range(n_attrs):
        for j in range(offsets[i], offsets[i] + sizes[i]):
            out[0] = st[j]
            out += 1

cdef void copy_struct_from_char(
    char *st, size_t *offsets, size_t *sizes, size_t n_attrs, char *out
) noexcept nogil:
    """
    If a struct has padding, and that padding isn't included in the byte
    array, you should use this to copy it in, as it explicitly does not
    include any platform/compiler specific padding between members.

    Otherwise, you can just memcpy the two...
    """
    cdef size_t i, j
    for i in range(n_attrs):
        for j in range(offsets[i], offsets[i] + sizes[i]):
            st[j] = out[0]
            out += 1