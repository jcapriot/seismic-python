#ifndef _BSWAP_H
#define _BSWAP_H

#include <stddef.h> // size_t, _byteswap_ushort on windows
#include <string.h> // memcpy

#include "spy_config.h"

#ifdef _MSC_VER
#define spy_bswap_u16(x) _byteswap_ushort(x) //relying on ushort being 16 bits on msvc compiler
#define spy_bswap_u32(x) _byteswap_ulong(x)  //relying on ulong being 32 bits on msvc compiler
#define spy_bswap_u64(x) _byteswap_uint64(x)

#else

#ifdef HAVE___BUILTIN_BSWAP16
#define spy_bswap_u16(x) __builtin_bswap16(x)
#else
static inline spy_uint16
spy_bswap_u16(spy_uint16 x)
{
    return ((x & 0x00ffu) << 8) | ((x 0xff00u) >> 8);
}
#endif

#ifdef HAVE___BUILTIN_BSWAP32
#define spy_bswap_u32(x) __builtin_bswap32(x)
#else
static inline spy_uint32
spy_bswap_u32(spy_uint32 x)
{
    return ((x & 0x000000ffu) << 24) |
           ((x & 0x0000ff00u) <<  8) |
           ((x & 0x00ff0000u) >>  8) |
           ((x & 0xff000000u) >> 24);
}
#endif

#ifdef HAVE___BUILTIN_BSWAP64
#define spy_bswap_u64(x) __builtin_bswap64(x)
#else
static inline spy_uint64
spy_bswap_u64(spy_uint64 x)
{
    return ((x & 0x00000000000000ffULL) << 56) |
           ((x & 0x000000000000ff00ULL) << 40) |
           ((x & 0x0000000000ff0000ULL) << 24) |
           ((x & 0x00000000ff000000ULL) <<  8) |
           ((x & 0x000000ff00000000ULL) >>  8) |
           ((x & 0x0000ff0000000000ULL) >> 24) |
           ((x & 0x00ff000000000000ULL) >> 40) |
           ((x & 0xff00000000000000ULL) >> 56);
}
#endif

#endif

#define spy_prw_big_bswap_u16 spy_bswap_u16

static inline spy_uint32
spy_prw_big_bswap_u32(spy_uint32 x){
    return ((x & 0x00ff00ffu) << 8) |
           ((x & 0xff00ff00u) >> 8);
}

static inline spy_uint64
spy_prw_big_bswap_u64(spy_uint64 x){
    return ((x & 0x00ff00ff00ff00ffULL) << 8) |
           ((x & 0xff00ff00ff00ff00ULL) >> 8);
}

#define spy_prw_lil_bswap_u16(x) (x)

static inline spy_uint32
spy_prw_lil_bswap_u32(spy_uint32 x){
    return ((x & 0x0000ffffu) << 16) |
           ((x & 0xffff0000u) >> 16);
}

static inline spy_uint64
spy_prw_lil_bswap_u64(spy_uint64 x){
    return ((x & 0x000000000000ffffULL) << 48) |
           ((x & 0x00000000ffff0000ULL) << 16) |
           ((x & 0x0000ffff00000000ULL) >> 16) |
           ((x & 0xffff000000000000ULL) >> 48);
}

// Unaligned versions:

static inline void
spy_bswap2_unaligned(char * x)
{
    char a = x[0];
    x[0] = x[1];
    x[1] = a;
}

static inline void
spy_bswap4_unaligned(char * x)
{
    char a = x[0];
    x[0] = x[3];
    x[3] = a;
    a = x[1];
    x[1] = x[2];
    x[2] = a;
}

static inline void
spy_bswap8_unaligned(char * x)
{
    char a = x[0]; x[0] = x[7]; x[7] = a;
    a = x[1]; x[1] = x[6]; x[6] = a;
    a = x[2]; x[2] = x[5]; x[5] = a;
    a = x[3]; x[3] = x[4]; x[4] = a;
}


#define spy_prw_big_bswap2_unaligned spy_bswap2_unaligned

static inline void
spy_prw_big_bswap4_unaligned(char * x)
{
    char a = x[0]; x[0] = x[1]; x[1] = a;
    a = x[2]; x[2] = x[3]; x[3] = a;
}

static inline void
spy_prw_big_bswap8_unaligned(char * x)
{
    char a = x[0]; x[0] = x[1]; x[1] = a;
    a = x[2]; x[2] = x[3]; x[3] = a;
    a = x[4]; x[4] = x[5]; x[5] = a;
    a = x[6]; x[6] = x[7]; x[7] = a;
}

static inline void
spy_prw_lil_bswap4_unaligned(char * x)
{
    // bytes 0 and 2 are swapped, bytes 1 and 3 are swapped
    char a = x[0]; x[0] = x[2]; x[2] = a;
    a = x[1]; x[1] = x[3]; x[3] = a;
}

static inline void
spy_prw_lil_bswap8_unaligned(char * x)
{
    char a = x[0]; x[0] = x[6]; x[6] = a;
    a = x[1]; x[1] = x[7]; x[7] = a;
    a = x[2]; x[2] = x[4]; x[4] = a;
    a = x[3]; x[3] = x[5]; x[5] = a;
}

static inline int
spy_is_aligned(const void *restrict p, const uintptr_t alignment)
{
    /*
     * Assumes alignment is a power of two, as required by the C standard.
     * Assumes cast from pointer to uintptr_t gives a sensible representation we
     * can use bitwise & on (not required by C standard, but used by glibc).
     * This test is faster than a direct modulo.
     * Note alignment value of 0 is allowed and returns False.
     */
    return ((uintptr_t)(p) & alignment) == 0;
}

// array swapping endian
static inline void swap16_big_and_system(spy_uint16 *x, size_t n){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    char *a;
    if (spy_is_aligned(x, 2)){
        for(i=0; i < n; ++i){
            x[i] = spy_bswap_u16(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 2){
            spy_bswap2_unaligned(a);
        }
    }
    #endif
}

static inline void swap32_big_and_system(spy_uint32 *x, size_t n){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    char *a;
    if(spy_is_aligned(x, 4)){
        for(i=0; i < n; ++i){
            x[i] = spy_bswap_u32(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 4){
            spy_bswap4_unaligned(a);
        }
    }
    #endif
}

static inline void swap64_big_and_system(spy_uint64 *x, size_t n){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    char *a;
    if(spy_is_aligned(x, 8)){
        for(i=0; i < n; ++i){
            x[i] = spy_bswap_u64(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 8){
            spy_bswap8_unaligned(a);
        }
    }
    #endif
}

static inline void swap16_little_and_system(spy_uint16 *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    char *a;
    if(spy_is_aligned(x, 2)){
        for(i=0; i < n; ++i){
            x[i] = spy_bswap_u16(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 2){
            spy_bswap2_unaligned(a);
        }
    }
    #endif
}

static inline void swap32_little_and_system(spy_uint32 *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    char *a;
    if(spy_is_aligned(x, 4)){
        for(i=0; i < n; ++i){
            x[i] = spy_bswap_u32(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 4){
            spy_bswap4_unaligned(a);
        }
    }
    #endif
}

static inline void swap64_little_and_system(spy_uint64 *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    char *a;
    if(spy_is_aligned(x, 8)){
        for(i=0; i < n; ++i){
            x[i] = spy_bswap_u64(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 8){
            spy_bswap8_unaligned(a);
        }
    }
    #endif
}

static inline void swap16_pairwise_and_system(spy_uint16 *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    char *a;
    if(spy_is_aligned(x, 4)){
        for(i=0; i < n; ++i){
            x[i] = spy_prw_big_bswap_u16(x[i]);
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 2){
            spy_prw_big_bswap2_unaligned(a);
        }
    }
    #endif
}

static inline void swap32_pairwise_and_system(spy_uint32 *x, size_t n){
    size_t i;
    char *a;
    if(spy_is_aligned(x, 4)){
        for(i=0; i < n; ++i){
            #ifdef IS_BIG_ENDIAN
            x[i] = spy_prw_big_bswap_u32(x[i]);
            #elif defined(IS_LITTLE_ENDIAN)
            x[i] = spy_prw_lil_bswap_u32(x[i]);
            #endif
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 4){
            #ifdef IS_BIG_ENDIAN
            spy_prw_big_bswap4_unaligned(a);
            #elif defined(IS_LITTLE_ENDIAN)
            spy_prw_lil_bswap4_unaligned(a);
            #endif
        }
    }
}

static inline void swap64_pairwise_and_system(spy_uint64 *x, size_t n){
    size_t i;
    char *a;
    if(spy_is_aligned(x, 8)){
        for(i=0; i < n; ++i){
            #ifdef IS_BIG_ENDIAN
            x[i] = spy_prw_big_bswap_u64(x[i]);
            #elif defined(IS_LITTLE_ENDIAN)
            x[i] = spy_prw_lil_bswap_u64(x[i]);
            #endif
        }
    }else{
        for(a=(char *) x, i=0; i < n; ++i, a += 8){
            #ifdef IS_BIG_ENDIAN
            spy_prw_big_bswap8_unaligned(a);
            #elif defined(IS_LITTLE_ENDIAN)
            spy_prw_lil_bswap8_unaligned(a);
            #endif
        }
    }
}

// variable byte swapping (usually for structs)
static inline void swapXX_big_and_system(
    char *str, size_t *offsets, size_t *sizes, size_t *n_elements, size_t n_attr
){
    #ifdef IS_LITTLE_ENDIAN
    size_t i, j;
    char *attr;
    for(i=0; i < n_attr; ++i, ++offsets, ++sizes, ++n_elements){
        attr = str + *offsets;
        if(*sizes == 2){
            spy_uint16 *t16 = (spy_uint16 *) attr;
            for(j=0; j<*n_elements; ++j, ++t16){
                *t16 = spy_bswap_u16(*t16);
            }
        }else if(*sizes == 4){
            spy_uint32 *t32 = (spy_uint32 *) attr;
            for(j=0; j<*n_elements; ++j, ++t32){
                *t32 = spy_bswap_u32(*t32);
            }
        }else if(*sizes == 8){
            spy_uint64 *t64 = (spy_uint64 *) attr;
            for(j=0; j<*n_elements; ++j, ++t64){
                *t64 = spy_bswap_u64(*t64);
            }
        }
    }
    #endif
}

static inline void swapXX_little_and_system(
    char *str, size_t *offsets, size_t *sizes, size_t *n_elements, size_t n_attr
){
    #ifdef IS_BIG_ENDIAN
    size_t i, j;
    char *attr;
    for(i=0; i < n_attr; ++i, ++offsets, ++sizes, ++n_elements){
        attr = str + *offsets;
        if(*sizes == 2){
            spy_uint16 *t16 = (spy_uint16 *) attr;
            for(j=0; j<*n_elements; ++j, ++t16){
                *t16 = spy_bswap_u16(*t16);
            }
        }else if(*sizes == 4){
            spy_uint32 *t32 = (spy_uint32 *) attr;
            for(j=0; j<*n_elements; ++j, ++t32){
                *t32 = spy_bswap_u32(*t32);
            }
        }else if(*sizes == 8){
            spy_uint64 *t64 = (spy_uint64 *) attr;
            for(j=0; j<*n_elements; ++j, ++t64){
                *t64 = spy_bswap_u64(*t64);
            }
        }
    }
    #endif
}

static inline void swapXX_pairwise_and_system(
    char *str, size_t *offsets, size_t *sizes, size_t *n_elements, size_t n_attr
){
    size_t i, j;
    char *attr;
    for(i=0; i < n_attr; ++i, ++offsets, ++sizes, ++n_elements){
        attr = str + *offsets;
        if(*sizes == 2){
            #ifdef IS_BIG_ENDIAN
            spy_uint16 *t16 = (spy_uint16 *) attr;
            for(j=0; j<*n_elements; ++j, ++t16){
                *t16 = spy_prw_big_bswap_u16(*t16);
            }
            #endif
        }else if(*sizes == 4){
            spy_uint32 *t32 = (spy_uint32 *) attr;
            for(j=0; j<*n_elements; ++j, ++t32){
                #ifdef IS_BIG_ENDIAN
                *t32 = spy_prw_big_bswap_u32(*t32);
                #elif defined(IS_LITTLE_ENDIAN)
                *t32 = spy_prw_lil_bswap_u32(*t32);
                #endif
            }
        }else if(*sizes == 8){
            spy_uint64 *t64 = (spy_uint64 *) attr;
            for(j=0; j<*n_elements; ++j, ++t64){
                #ifdef IS_BIG_ENDIAN
                *t64 = spy_prw_big_bswap_u64(*t64);
                #elif defined(IS_LITTLE_ENDIAN)
                *t64 = spy_prw_lil_bswap_u64(*t64);
                #endif
            }
        }
    }
}
#endif