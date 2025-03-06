#ifndef _BSWAP_H
#define _BSWAP_H

#include <stddef.h> // size_t, _byteswap_ushort on windows
#include <stdint.h> // uint16_t and the like
#include <string.h> // memcpy

#ifdef _MSC_VER
#define spy_bswap_u16(x) _byteswap_ushort(x) //relying on ushort being 16 bits on mscv compiler
#define spy_bswap_u32(x) _byteswap_ulong(x)  //relying on ulong being 32 bits on mscv compiler
#define spy_bswap_u64(x) _byteswap_uint64(x)

#elif defined(__APPLE__)

// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#define spy_bswap_u16(x) OSSwapInt16(x)
#define spy_bswap_u32(x) OSSwapInt32(x)
#define spy_bswap_u64(x) OSSwapInt64(x)

#else

#ifdef HAVE___BUILTIN_BSWAP16
#define spy_bswap_u16(x) __builtin_bswap16(x)
#else
static inline uint16_t
spy_bswap_u16(uint16_t x)
{
    return ((x & 0xffu) << 8) | (x >> 8);
}
#endif

#ifdef HAVE___BUILTIN_BSWAP32
#define spy_bswap_u32(x) __builtin_bswap32(x)
#else
static inline uint32_t
spy_bswap_u32(uint32_t x)
{
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8) |
   ((x & 0xff0000u) >> 8) | (x >> 24);
}
#endif

#ifdef HAVE___BUILTIN_BSWAP64
#define spy_bswap_u64(x) __builtin_bswap64(x)
#else
static inline uint64_t
spy_bswap_u64(uint64_t x)
{
    return ((x & 0xffULL) << 56) |
           ((x & 0xff00ULL) << 40) |
           ((x & 0xff0000ULL) << 24) |
           ((x & 0xff000000ULL) << 8) |
           ((x & 0xff00000000ULL) >> 8) |
           ((x & 0xff0000000000ULL) >> 24) |
           ((x & 0xff000000000000ULL) >> 40) |
           ( x >> 56);
}
#endif

#endif

#define spy_bswap_pair_big_u16 spy_bswap_u16

static inline uint32_t
spy_prw_big_bswap_u32(uint32_t x){
    return ((x & 0xffULL) << 8) |
           ((x & 0xff00ULL) >> 8) |
           ((x & 0xff0000ULL) << 8) |
           ((x & 0xff000000ULL) >> 8);
}

static inline uint64_t
spy_prw_big_bswap_u64(uint64_t x){
    return ((x & 0xffULL) << 8) |
           ((x & 0xff00ULL) >> 8) |
           ((x & 0xff0000ULL) << 8) |
           ((x & 0xff000000ULL) >> 8) |
           ((x & 0xff00000000ULL) << 8) |
           ((x & 0xff0000000000ULL) >> 8) |
           ((x & 0xff000000000000ULL) << 8) |
           ((x & 0xff00000000000000ULL) >> 8);
}

static inline uint32_t
spy_prw_lil_bswap_u32(uint32_t x){
    return ((x & 0xffULL) << 16) |
           ((x & 0xff00ULL) << 16) |
           ((x & 0xff0000ULL) >> 16) |
           ((x & 0xff000000ULL) >> 16);
}

static inline uint64_t
spy_prw_lil_bswap_u64(uint64_t x){
    return ((x & 0xffULL) << 48) |
           ((x & 0xff00ULL) << 48) |
           ((x & 0xff0000ULL) << 16) |
           ((x & 0xff000000ULL) << 16) |
           ((x & 0xff00000000ULL) >> 16) |
           ((x & 0xff0000000000ULL) >> 16) |
           ((x & 0xff000000000000ULL) >> 48) |
           ((x & 0xff00000000000000ULL) >> 48);
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
static inline void swap16_big_and_system(uint16_t *x, size_t n){
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

static inline void swap32_big_and_system(uint32_t *x, size_t n){
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

static inline void swap64_big_and_system(uint64_t *x, size_t n){
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

static inline void swap16_little_and_system(uint16_t *x, size_t n){
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

static inline void swap32_little_and_system(uint32_t *x, size_t n){
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

static inline void swap64_little_and_system(uint64_t *x, size_t n){
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

static inline void swap16_pairwise_and_system(uint16_t *x, size_t n){
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

static inline void swap32_pairwise_and_system(uint32_t *x, size_t n){
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

static inline void swap64_pairwise_and_system(uint64_t *x, size_t n){
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

// struct swapping
static inline void swap_struct_big_and_system(char *str, size_t *offsets, size_t *sizes, size_t n_attr){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    char *attr;
    for(i=0; i < n_attr; ++i, ++offsets, ++sizes){
        attr = str + *offsets;
        if(*sizes == 2){
            if(spy_is_aligned(attr, 2)){
                uint16_t *t16 = (uint16_t *) attr;
                *t16 = spy_bswap_u16(*t16);
            }else{
                spy_bswap2_unaligned(attr);
            }
        }else if(*sizes == 4){
            if(spy_is_aligned(attr, 4)){
                uint32_t *t32 = (uint32_t *) attr;
                *t32 = spy_bswap_u32(*t32);
            }else{
                spy_bswap4_unaligned(attr);
            }
        }else if(*sizes == 8){
            if(spy_is_aligned(attr, 8)){
                uint64_t *t64 = (uint64_t *) (str + *offsets);
                *t64 = spy_bswap_u64(*t64);
            }else{
                spy_bswap8_unaligned(attr);
            }
        }
    }
    #endif
}
static inline void swap_struct_little_and_system(char *str, size_t *offsets, size_t *sizes, size_t n_attr){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    char *attr;
    for(i=0; i < n_attr; ++i, ++offsets, ++sizes){
        attr = str + *offsets;
        if(*sizes == 2){
            if(spy_is_aligned(attr, 2)){
                uint16_t *t16 = (uint16_t *) attr;
                *t16 = spy_bswap_u16(*t16);
            }else{
                spy_bswap2_unaligned(attr);
            }
        }else if(*sizes == 4){
            if(spy_is_aligned(attr, 4)){
                uint32_t *t32 = (uint32_t *) attr;
                *t32 = spy_bswap_u32(*t32);
            }else{
                spy_bswap4_unaligned(attr);
            }
        }else if(*sizes == 8){
            if(spy_is_aligned(attr, 8)){
                uint64_t *t64 = (uint64_t *) (str + *offsets);
                *t64 = spy_bswap_u64(*t64);
            }else{
                spy_bswap8_unaligned(attr);
            }
        }
    }
    #endif
}

static inline void swap_struct_pairwise_and_system(char *str, size_t *offsets, size_t *sizes, size_t n_attr){
    size_t i;
    char *attr;
    for(i=0; i < n_attr; ++i, ++offsets, ++sizes){
        attr = str + *offsets;
        if(*sizes == 2){
            // 2 byte pairswapped is same a little endian?
            #ifdef IS_BIG_ENDIAN
            if(spy_is_aligned(attr, 2)){
                uint16_t *t16 = (uint16_t *) attr;
                *t16 = spy_bswap_u16(*t16);
            }else{
                spy_bswap2_unaligned(attr);
            }
            #endif
        }else if(*sizes == 4){
             if(spy_is_aligned(attr, 4)){
                uint32_t *t32 = (uint32_t *) attr;
                #ifdef IS_BIG_ENDIAN
                *t32 = spy_prw_big_bswap_u32(*t32);
                #elif defined(IS_LITTLE_ENDIAN)
                *t32 = spy_prw_lil_bswap_u32(*t32);
                #endif
            }else{
                #ifdef IS_BIG_ENDIAN
                spy_prw_big_bswap4_unaligned(attr);
                #elif defined(IS_LITTLE_ENDIAN)
                spy_prw_lil_bswap4_unaligned(attr);
                #endif
            }
        }else if(*sizes == 8){
             if(spy_is_aligned(attr, 8)){
                uint64_t *t64 = (uint64_t *) attr;
                t64 = (uint64_t *) attr;
                #ifdef IS_BIG_ENDIAN
                *t64 = spy_prw_big_bswap_u64(*t64);
                #elif defined(IS_LITTLE_ENDIAN)
                *t64 = spy_prw_lil_bswap_u64(*t64);
                #endif
            }else{
                #ifdef IS_BIG_ENDIAN
                spy_prw_big_bswap8_unaligned(attr);
                #elif defined(IS_LITTLE_ENDIAN)
                spy_prw_lil_bswap8_unaligned(attr);
                #endif
            }
        }
    }
}
#endif