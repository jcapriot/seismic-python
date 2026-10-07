import sys

import numpy as np
import numpy.testing as npt
import pytest

from seispy.io.byteswapping import swap_endian_and_system, swap_endian_and_system_array


NP_DTYPES = [
    np.uint16,
    np.uint32,
    np.uint64,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]


@pytest.mark.parametrize('dtype', NP_DTYPES)
@pytest.mark.parametrize('arr', [False, True, "inplace"])
def test_swap_big_and_system(dtype, arr):

    big_ref_bytes = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.byte)
    big_ref = big_ref_bytes.view(dtype)[0]

    if sys.byteorder == 'little':
        sys_ref = np.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.byte).view(dtype)[-1]
    else:
        sys_ref = big_ref

    if arr:
        inplace = arr == 'inplace'
        big_ref = np.full(10, big_ref, dtype=dtype)
        big_swapped = np.asarray(swap_endian_and_system_array(big_ref, '>', inplace))
        if inplace:
            big_swapped = big_ref
    else:
        func_index = dtype.__name__ + "_t"
        big_swapped = swap_endian_and_system[func_index](big_ref, '>')

    npt.assert_equal(big_swapped, sys_ref)

@pytest.mark.parametrize('dtype', NP_DTYPES)
@pytest.mark.parametrize('arr', [False, True, "inplace"])
def test_swap_little_and_system(dtype, arr):

    little_ref_bytes = np.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.byte)
    little_ref = little_ref_bytes.view(dtype)[0]

    if sys.byteorder == 'big':
        sys_ref = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.byte).view(dtype)[-1]
    else:
        sys_ref = little_ref


    if arr:
        inplace = arr == 'inplace'
        little_ref = np.full(10, little_ref, dtype=dtype)
        little_swapped = np.asarray(swap_endian_and_system_array(little_ref, '<', inplace))
        if inplace:
            little_swapped = little_ref
    else:
        func_index = dtype.__name__ + "_t"
        little_swapped = swap_endian_and_system[func_index](little_ref, '<')

    npt.assert_equal(little_swapped, sys_ref)

@pytest.mark.parametrize('dtype', NP_DTYPES)
@pytest.mark.parametrize('arr', [False, True, "inplace"])
def test_swap_pairwise_and_system(dtype, arr):

    pairwise_ref_bytes = np.asarray([2, 1, 4, 3, 6, 5, 8, 7], dtype=np.byte)
    pairwise_ref = pairwise_ref_bytes.view(dtype)[0]

    if sys.byteorder == 'big':
        sys_ref = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.byte).view(dtype)[0]
    else:
        sys_ref = np.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.byte).view(dtype)[-1]

    if arr:
        inplace = arr == 'inplace'
        pairwise_ref = np.full(10, pairwise_ref, dtype=dtype)
        pairwise_swapped = np.asarray(swap_endian_and_system_array(pairwise_ref, '<>', inplace))
        if inplace:
            pairwise_swapped = pairwise_ref
    else:
        func_index = dtype.__name__ + "_t"
        pairwise_swapped = swap_endian_and_system[func_index](pairwise_ref, '<>')

    npt.assert_equal(pairwise_swapped, sys_ref)