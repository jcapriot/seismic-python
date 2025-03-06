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

#ifdef DHAVE___BUILTIN_BSWAP16
#define spy_bswap_u16(x) __builtin_bswap16(x)
#else
static inline uint16_t
spy_bswap_u16(uint16_t x)
{
    return ((x & 0xffu) << 8) | (x >> 8);
}
#endif

#ifdef DHAVE___BUILTIN_BSWAP32
#define spy_bswap_u32(x) __builtin_bswap32(x)
#else
static inline uint32_t
spy_bswap_u32(uint32_t x)
{
    return ((x & 0xffu) << 24) | ((x & 0xff00u) << 8) |
   ((x & 0xff0000u) >> 8) | (x >> 24);
}
#endif

#ifdef DHAVE___BUILTIN_BSWAP64
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

// array swapping endian
static inline void swap16_big_and_system(uint16_t *x, size_t n){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u16(x[i]);
    }
    #endif
}

static inline void swap32_big_and_system(uint32_t *x, size_t n){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u32(x[i]);
    }
    #endif
}

static inline void swap64_big_and_system(uint64_t *x, size_t n){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u64(x[i]);
    }
    #endif
}

static inline void swap16_little_and_system(uint16_t *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u16(x[i]);
    }
    #endif
}

static inline void swap32_little_and_system(uint32_t *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u32(x[i]);
    }
    #endif
}

static inline void swap64_little_and_system(uint64_t *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u64(x[i]);
    }
    #endif
}

static inline void swap16_pairwise_and_system(uint16_t *x, size_t n){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u16(x[i]);
    }
    #endif
}

static inline void swap32_pairwise_and_system(uint32_t *x, size_t n){
    size_t i;
    uint16_t *x2 = (uint16_t *) x;
    for(i=0; i<2*n; ++i){
        x2[i] = spy_bswap_u16(x2[i]);
    }
    #ifdef IS_LITTLE_ENDIAN
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u32(x[i]);
    }
    #endif
}

static inline void swap64_pairwise_and_system(uint64_t *x, size_t n){
    size_t i;
    uint16_t *x2 = (uint16_t *) x;
    for(i=0; i<4*n; ++i){
        x2[i] = spy_bswap_u16(x2[i]);
    }
    #ifdef IS_LITTLE_ENDIAN
    for(i=0; i < n; ++i){
        x[i] = spy_bswap_u64(x[i]);
    }
    #endif
}

// struct swapping
static inline void swap_struct_big_and_system(char *str, size_t *offsets, size_t *sizes, size_t n_attr){
    #ifdef IS_LITTLE_ENDIAN
    size_t i;
    uint16_t *t16;
    uint32_t *t32;
    uint64_t *t64;
    for(i=0; i < n_attr; ++i, str += *(++offsets), ++sizes){ // *sizes++ gets the value pointed to by size, then increments the sizes pointer
        if(*sizes == 2){
            t16 = (uint16_t *) str;
            *t16 = spy_bswap_u16(*t16);
        }else if(*sizes == 4){
            t32 = (uint32_t *) str;
            *t32 = spy_bswap_u32(*t32);
        }else if(*sizes == 8){
            t64 = (uint64_t *) str;
            *t64 = spy_bswap_u64(*t64);
        }
    }
    #endif
}
static inline void swap_struct_little_and_system(char *str, size_t *offsets, size_t *sizes, size_t n_attr){
    #ifdef IS_BIG_ENDIAN
    size_t i;
    uint16_t *t16;
    uint32_t *t32;
    uint64_t *t64;
    for(i=0; i < n_attr; ++i, str += *(++offsets), ++sizes){ // *sizes++ gets the value pointed to by size, then increments the sizes pointer
        if(*sizes == 2){
            t16 = (uint16_t *) str;
            *t16 = spy_bswap_u16(*t16);
        }else if(*sizes == 4){
            t32 = (uint32_t *) str;
            *t32 = spy_bswap_u32(*t32);
        }else if(*sizes == 8){
            t64 = (uint64_t *) str;
            *t64 = spy_bswap_u64(*t64);
        }
    }
    #endif
}
static inline void swap_struct_pairwise_and_system(char *str, size_t *offsets, size_t *sizes, size_t n_attr){
    #ifdef IS_BIG_ENDIAN
    size_t i, j;
    uint16_t *t16;
    uint32_t *t32;
    uint64_t *t64;
    for(i=0; i < n_attr; ++i, str += *(++offsets), ++sizes{ // *sizes++ gets the value pointed to by size, then increments the sizes pointer
        if(*sizes == 2){
            #ifdef IS_BIG_ENDIAN
            t16 = (uint16_t *) str;
            *t16 = spy_bswap_u16(*t16);
            #endif
        }else{
            // Swapping each byte pair gets us to big endian
            t16 = (*uint16_t) str
            for(j=0; j<*sizes; j+=2, ++t16){
                *t16 = spy_bswap_u16(*t16);
            }
            #ifdef IS_LITTLE_ENDIAN
            if(*sizes == 4){
                t32 = (uint32_t *) str;
                *t32 = spy_bswap_u32(*t32);
            }else if{*sizes == 8){
                t64 = (uint64_t *) str;
                *t64 = spy_bswap_u64(*t64);
            }
            #endif
        }
    }
    #endif
}
#endif