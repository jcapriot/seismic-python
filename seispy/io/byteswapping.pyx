cimport cython
from Cython.Shadow import struct
from libc.stdlib cimport malloc

cpdef swappable swap_endian_and_system(swappable x, str endian='>'):
    cdef swappable _x

    if endian == '>':
        target = 0
    elif endian == '<':
        target = 1
    elif endian == '<>':
        target = 2
    else:
        raise ValueError("endian must be one of '>' (big), '<' (little), or '<>' (pairwise byteswapped)")

    with nogil:
        _x = x
        if swappable is uint16_t or swappable is int16_t:
            if target == 0:
                swap16_big_and_system(<uint16_t*> &_x, 1)
            elif target == 1:
                swap16_little_and_system(<uint16_t*> &_x, 1)
            else:
                swap16_pairwise_and_system(<uint16_t*> &_x, 1)
        elif swappable is uint32_t or swappable is int32_t or swappable is float32_t:
            if target == 0:
                swap32_big_and_system(<uint32_t*> &_x, 1)
            elif target == 1:
                swap32_little_and_system(<uint32_t*> &_x, 1)
            else:
                swap32_pairwise_and_system(<uint32_t*> &_x, 1)
        elif swappable is uint64_t or swappable is int64_t or swappable is float64_t:
            if target == 0:
                swap64_big_and_system(<uint64_t*> &_x, 1)
            elif target == 1:
                swap64_little_and_system(<uint64_t*> &_x, 1)
            else:
                swap64_pairwise_and_system(<uint64_t*> &_x, 1)
    return _x


@cython.boundscheck(False)
cpdef swappable[::1] swap_endian_and_system_array(swappable[::1] x, str endian='>', bint inplace=True):
    cdef size_t n = x.shape[0]
    cdef size_t target_size = sizeof(swappable)
    cdef int endian_target
    cdef swappable[::1] _x
    if inplace:
        _x = x
    else:
        _x = <swappable[:n]> malloc(sizeof(swappable) * n)
        _x[...] = x
    if n == 0:
        return _x

    if endian == '>':
        target = 0
    elif endian == '<':
        target = 1
    elif endian == '<>':
        target = 2
    else:
        raise ValueError("endian must be one of '>' (big), '<' (little), or '<>' (pairwise byteswapped)")

    with nogil:
        if swappable is uint16_t or swappable is int16_t:
            if target == 0:
                swap16_big_and_system(<uint16_t*> &_x[0], n)
            elif target == 1:
                swap16_little_and_system(<uint16_t*> &_x[0], n)
            else:
                swap16_pairwise_and_system(<uint16_t*> &_x[0], n)
        elif swappable is uint32_t or swappable is int32_t or swappable is float32_t:
            if target == 0:
                swap32_big_and_system(<uint32_t*> &_x[0], n)
            elif target == 1:
                swap32_little_and_system(<uint32_t*> &_x[0], n)
            else:
                swap32_pairwise_and_system(<uint32_t*> &_x[0], n)
        elif swappable is uint64_t or swappable is int64_t or swappable is float64_t:
            if target == 0:
                swap64_big_and_system(<uint64_t*> &_x[0], n)
            elif target == 1:
                swap64_little_and_system(<uint64_t*> &_x[0], n)
            else:
                swap64_pairwise_and_system(<uint64_t*> &_x[0], n)
    return _x

cdef void swap_struct_endian_and_system(void *x, size_t[:, ::1] struct_info, str endian=">"):
    """Swaps all structure item byte orders in place."""
    cdef int endian_target
    cdef char* _x = <char *> x
    cdef size_t[:] offsets = struct_info[0]
    cdef size_t[:] sizes = struct_info[1]
    cdef size_t[:] n_elements = struct_info[2]

    if endian == '>':
        target = 0
    elif endian == '<':
        target = 1
    elif endian == '<>':
        target = 2
    else:
        raise ValueError("endian must be one of '>' (big), '<' (little), or '<>' (pairwise byteswapped)")

    with nogil:
        if target == 0:
            swapXX_big_and_system(_x, &offsets[0], &sizes[0], &n_elements[0], offsets.shape[0])
        elif target == 1:
            swapXX_little_and_system(_x, &offsets[0], &sizes[0], &n_elements[0], offsets.shape[0])
        elif target == 2:
            swapXX_pairwise_and_system(_x, &offsets[0], &sizes[0], &n_elements[0], offsets.shape[0])