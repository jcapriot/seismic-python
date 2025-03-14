# cython: embedsignature=True, language_level=3
# cython: linetrace=True
from cpython.object cimport PyObject_AsFileDescriptor
from libc.stdio cimport FILE, fclose, SEEK_SET, fwrite, fread
from libc.limits cimport INT_MIN, INT_MAX
from libc.stdlib cimport malloc
from libc.string cimport memcpy
cimport cython
import os
import io
from . cimport byteswapping as bswap

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


cdef (FILE *, spy_off_t) PyFile_Dup(object file, char * mode):
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

cdef int PyFile_DupClose(object file, FILE * handle, spy_off_t orig_pos):
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

@cython.boundscheck(False)
cdef size_t write_struct_to_file(
        void *in_struct,
        size_t[:, ::1] struct_info,
        bint is_packed,
        size_t expected_size,
        FILE *fd,
) noexcept nogil:
    cdef:
        size_t i, n_bytes_written
        char *st = <char *> in_struct
        char *attr
        size_t[::1] offsets = struct_info[0]
        size_t[::1] sizes = struct_info[1]
        size_t[::1] n_elements = struct_info[2]
        size_t n_attr = struct_info.shape[1]
    if is_packed:
        n_bytes_written = fwrite(st, 1, expected_size, fd)
    else:
        n_bytes_written = 0
        for i in range(n_attr):
            attr = st + offsets[i]
            n_bytes_written += fwrite(attr, 1, sizes[i] * n_elements[i], fd)
    return n_bytes_written

@cython.boundscheck(False)
cdef size_t read_struct_from_file(
        void *out_struct,
        size_t[:, ::1] struct_info,
        bint is_packed,
        size_t expected_size,
        FILE *fd,
        str file_endian_flag,
) noexcept nogil:
    cdef:
        size_t i, n_bytes_read
        char *st = <char *> out_struct
        char *attr
        size_t[::1] offsets = struct_info[0]
        size_t[::1] sizes = struct_info[1]
        size_t[::1] n_elements = struct_info[2]
        size_t n_attr = struct_info.shape[1]

    if is_packed:
        n_bytes_read = fread(st, 1, expected_size, fd)
    else:
        n_bytes_read = 0
        for i in range(n_attr):
            attr = st + offsets[i]
            n_bytes_read += fread(attr, 1, sizes[i] * n_elements[i], fd)

    if n_bytes_read == expected_size:
        if file_endian_flag == ">":
            bswap.swapXX_big_and_system(
                st, &offsets[0], &sizes[0], &n_elements[0], n_attr
            )
        elif file_endian_flag == "<":
            bswap.swapXX_little_and_system(
                st, &offsets[0], &sizes[0], &n_elements[0], n_attr
            )
        elif file_endian_flag == "<>":
            bswap.swapXX_pairwise_and_system(
                st, &offsets[0], &sizes[0], &n_elements[0], n_attr
            )

    return n_bytes_read

@cython.boundscheck(False)
cdef void copy_struct_to_char(
    void *in_struct,
    size_t[:, ::1] struct_info,
    bint is_packed,
    size_t expected_size,
    char *out_chrs
) noexcept nogil:
    cdef:
        size_t i, j
        char * st = <char *> in_struct
        size_t[::1] offsets = struct_info[0]
        size_t[::1] sizes = struct_info[1]
        size_t[::1] n_elements = struct_info[2]
        size_t n_attr = offsets.shape[1]

    if is_packed:
        memcpy(out_chrs, st, expected_size)
    else:
        for i in range(n_attr):
            for j in range(offsets[i], offsets[i] + n_elements[i] * sizes[i]):
                out_chrs[0] = st[j]
                out_chrs += 1

@cython.boundscheck(False)
cdef void copy_struct_from_char(
    void *out_struct,
    size_t[:, ::1] struct_info,
    bint is_packed,
    size_t expected_size,
    char *in_chrs
) noexcept nogil:
    cdef:
        size_t i, j
        char * st = <char *> out_struct
        size_t[::1] offsets = struct_info[0]
        size_t[::1] sizes = struct_info[1]
        size_t[::1] n_elements = struct_info[2]
        size_t n_attr = offsets.shape[1]

    if is_packed:
        memcpy(out_struct, st, expected_size)
    else:
        for i in range(n_attr):
            for j in range(offsets[i], offsets[i] + n_elements[i] * sizes[i]):
                st[j] = in_chrs[j]
                in_chrs += 1

cpdef size_t[:,::1] struct_dtype_info(object struct_dtype):
    """
    Accepts a numpy structured dtype, and unpacks the attributes into a
    3 by n_attrs information, such that:

    Returns
    -------
    offsets, sizes, n_elements : (n_attr,) memoryview of size_t

    Examples
    --------
    Take the following cython struct:
    ```
    cdef struct ex:
        int i1
        short i2
        float[3] f1
    ```

    The corresponding info about the dtype would be:
    >>> ex_dtype = np.dtype('i4,i2,3f4')
    >>> offsets, sizes, n_elements = struct_dtype_info(ex_dtype)
    >>> for s, o, n in zip(sizes, offsets, n_elements):
    ...    print(f'offset: {o}, itemsize: {s}, n_elements:{n}')
    ...
    offset: 0, itemsize: 4, n_elements:1
    offset: 4, itemsize: 2, n_elements:1
    offset: 6, itemsize: 4, n_elements:3

    Notes
    -----
    This does work for all structured dtypes, but it might not be useful for nested
    structures.
    """

    cdef:
        size_t n_attrs = len(struct_dtype)
        size_t[:,::1] info = <size_t[:3, :n_attrs]> malloc(3 * sizeof(size_t)*n_attrs)
        # info[0] offsets
        # info[1] sizes
        # info[2] n_elements

    for i, name in enumerate(struct_dtype.names):
        dtype, offset = struct_dtype.fields[name]
        n_elements = 1
        subdtype = dtype.subdtype
        if subdtype:
            subdtype = subdtype[0]
            n_elements = dtype.itemsize // subdtype.itemsize
            dtype = subdtype
        else:
            n_elements = 1
        info[0, i] = offset
        info[1, i] = dtype.itemsize
        info[2, i] = n_elements
    return info