from ..config cimport (
    uint16_t, uint32_t, uint64_t,
    int16_t, int32_t, int64_t,
    float32_t, float64_t
)

cdef extern from "bswap.h" nogil:

    uint16_t spy_bswap_u16(uint16_t x)
    uint32_t spy_bswap_u32(uint32_t x)
    uint64_t spy_bswap_u64(uint64_t x)

    uint16_t spy_prw_big_bswap_u16(uint32_t x)
    uint32_t spy_prw_big_bswap_u32(uint32_t x)
    uint64_t spy_prw_big_bswap_u64(uint32_t x)

    uint16_t spy_prw_lil_bswap_u16(uint32_t x)
    uint32_t spy_prw_lil_bswap_u32(uint32_t x)
    uint64_t spy_prw_lil_bswap_u64(uint32_t x)

    void swap16_big_and_system(uint16_t *x, size_t n)
    void swap32_big_and_system(uint32_t *x, size_t n)
    void swap64_big_and_system(uint64_t *x, size_t n)
    void swap16_little_and_system(uint16_t *x, size_t n)
    void swap32_little_and_system(uint32_t *x, size_t n)
    void swap64_little_and_system(uint64_t *x, size_t n)
    void swap16_pairwise_and_system(uint16_t *x, size_t n)
    void swap32_pairwise_and_system(uint32_t *x, size_t n)
    void swap64_pairwise_and_system(uint64_t *x, size_t n)

    void swapXX_big_and_system(
            char *str, size_t *offsets, size_t *sizes, size_t *n_elements, size_t n_attr
    )
    void swapXX_little_and_system(
            char *str, size_t *offsets, size_t *sizes, size_t *n_elements, size_t n_attr
    )
    void swapXX_pairwise_and_system(
            char *str, size_t *offsets, size_t *sizes, size_t *n_elements, size_t n_attr
    )

ctypedef fused swappable:
    uint16_t
    uint32_t
    uint64_t
    int16_t
    int32_t
    int64_t
    float32_t
    float64_t

cpdef swappable swap_endian_and_system(swappable x, str endian=?)
cpdef swappable[::1] swap_endian_and_system_array(swappable[::1] x, str endian=?, bint inplace=?)

cdef void swap_struct_endian_and_system(void *x, size_t[:, ::1] struct_info, str endian=?)