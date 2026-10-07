
cdef extern from "spy_config.h":
    int SPY_SYS_ENDIAN
    ctypedef float          spy_float32
    ctypedef double         spy_float64

    ctypedef char       spy_int8
    ctypedef short      spy_int16
    ctypedef int        spy_int32
    ctypedef long long  spy_int64

    ctypedef unsigned char          spy_uint8
    ctypedef unsigned short         spy_uint16
    ctypedef unsigned int           spy_uint32
    ctypedef unsigned long long     spy_uint64

ctypedef spy_float32    float32_t
ctypedef spy_float64    float64_t

ctypedef spy_int8       int8_t
ctypedef spy_int16      int16_t
ctypedef spy_int32      int32_t
ctypedef spy_int64      int64_t

ctypedef spy_uint8      uint8_t
ctypedef spy_uint16     uint16_t
ctypedef spy_uint32     uint32_t
ctypedef spy_uint64     uint64_t