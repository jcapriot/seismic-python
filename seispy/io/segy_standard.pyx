# cython: embedsignature=True, language_level=3
# cython: linetrace=True
import os
import sys
import warnings
from contextlib import nullcontext

from libc.math cimport floor, ceil, log10, fabs, ldexpf
from libc.stdint cimport (
    INT8_MAX, INT16_MAX, INT32_MAX, INT64_MAX,
    UINT8_MAX, UINT16_MAX, UINT32_MAX, UINT64_MAX,
    SIZE_MAX, int8_t, int32_t, uint32_t, uint8_t, uint16_t, uint64_t
)
from libc.stdio cimport fwrite, fread, FILE, SEEK_CUR, SEEK_SET, feof
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport cython
cimport cpython.buffer as pybuf

from . cimport byteswapping as bswap
from . cimport _io as spy_io

from .. cimport container as spyc
from ..config cimport SPY_SYS_ENDIAN

import numpy as np

cdef:
    # Expected Sizes
    size_t BIN_HDR_SIZE = sizeof(binary_header)
    size_t TRC_HDR_SIZE = sizeof(trace_header)
    size_t EXT_HDR_SIZE = sizeof(extended_trace_header)
    size_t SU_HDR_SIZE = sizeof(su_trace)

    bint BH_IS_PACKED = BIN_HDR_SIZE == BIN_HDR_BYTES
    bint STDH_IS_PACKED = TRC_HDR_SIZE == TRC_HDR_BYTES
    bint EXTH_IS_PACKED = EXT_HDR_SIZE == TRC_HDR_BYTES
    bint SUTH_IS_PACKED = SU_HDR_SIZE == TRC_HDR_BYTES

# define a numpy dtypes consistent with the header structs
cdef:
    binary_header tmp_bin_hdr
    size_t[:,::1] _bh_info
binary_header_dtype = np.asarray(<binary_header[:1]> &tmp_bin_hdr).dtype
_bh_info  = spy_io.struct_dtype_info(binary_header_dtype)

cdef:
    trace_header tmp_trc_hdr
    size_t[:,::1] _th_info
trace_header_dtype = np.asarray(<trace_header[:1]> &tmp_trc_hdr).dtype
_th_info  = spy_io.struct_dtype_info(trace_header_dtype)

cdef:
    extended_trace_header tmp_ext_hdr
    size_t[:,::1] _eh_info
extended_trace_header_dtype = np.asarray(<extended_trace_header[:1]> &tmp_ext_hdr).dtype
_eh_info  = spy_io.struct_dtype_info(extended_trace_header_dtype)


cdef:
    su_trace tmp_su_hdr
    size_t[:,::1] _su_info
su_trace_header_dtype = np.asarray(<su_trace[:1]> &tmp_su_hdr).dtype
_su_info  = spy_io.struct_dtype_info(su_trace_header_dtype)


cdef:
    dict data_format_sizes = {
        DataFormat.float32_ibm : 4,
        DataFormat.int32 : 4,
        DataFormat.int16 : 2,
        DataFormat.fixed32 : 4,
        # Rev 1
        DataFormat.float32_ieee : 4,
        DataFormat.float64_ieee : 8,
        DataFormat.int24 : 3,
        DataFormat.int08 : 1,
        # rev 2
        DataFormat.int64 : 8,
        DataFormat.uint32 : 4,
        DataFormat.uint16 : 2,
        DataFormat.uint64 : 8,
        DataFormat.uint24 : 3,
        DataFormat.uint08 : 1,
    }

class SEGYTruncatedError(IOError):
    """The file ended in the middle of a trace."""


cdef dict _DTYPE_STRING_MAP = {
    'f32': DataFormat.float32_ieee,
    'f64': DataFormat.float64_ieee,
    'ibmf32': DataFormat.float32_ibm,
    'fixed32': DataFormat.fixed32,
    'i08': DataFormat.int08,
    'i16': DataFormat.int16,
    'i24': DataFormat.int24,
    'i32': DataFormat.int32,
    'i64': DataFormat.int64,
    'u08': DataFormat.uint08,
    'ui16': DataFormat.uint16,
    'ui24': DataFormat.uint24,
    'ui32': DataFormat.uint32,
    'ui64': DataFormat.uint64,
}

cdef DataFormat str_to_data_format(str dtype_string):
    try:
        return _DTYPE_STRING_MAP[dtype_string]
    except KeyError:
        raise ValueError(f"Unrecognized dtype_string: {dtype_string}")

@cython.boundscheck(False)
cdef void ibm_to_float(uint8_t *inp, size_t n_items) noexcept nogil:
    cdef:
        size_t i
        float mantissa
        uint32_t IBM_SIGN = 0x80000000U
        uint32_t IBM_EXP = 0x7f000000U
        uint32_t IBM_FRAC = 0x00ffffffU
        uint32_t IBM_TOP = 0x00f00000U
        uint32_t TIES_TO_EVEN = 0xfffffffdU
        uint32_t BITCOUNT = 0x000055afU
        uint32_t INFINITY = 0x7f800000U
        uint32_t sign, exp, frac
        uint32_t top_digit, leading_zeros, mask, round_up
        # reinterpret it as a 4 byte integer:
        uint32_t *_in = <uint32_t *> &inp[0]

    with nogil:
        for i in range(n_items):
            sign = _in[i] & IBM_SIGN
            frac = _in[i] & IBM_FRAC
            if frac == 0:
                _in[i] = 0
            else:
                exp = (_in[i] & IBM_EXP) >> 22
                top_digit = frac & IBM_TOP
                while top_digit == 0:
                    frac <<= 4
                    exp -= 4
                    top_digit= frac & IBM_TOP
                leading_zeros = (BITCOUNT >> (top_digit >> 19U)) & 3U
                frac <<= leading_zeros

                exp = exp - 131 - leading_zeros

                if exp >= 0 and exp < 254:
                    _in[i] = sign + (exp << 23U) + frac
                elif exp >= 254:
                    _in[i] = sign + INFINITY
                elif exp >= -32:
                    mask = ~(TIES_TO_EVEN << (-1 - exp))
                    round_up = (frac & mask) > 0U
                    frac = ((frac >> (-1 - exp)) + round_up) >> 1
                    _in[i] = sign + frac
                else:
                    _in[i] = sign


@cython.boundscheck(False)
cdef void float_to_ibm(uint8_t *inp, size_t n_items) noexcept nogil:
    cdef:
        uint32_t it[4]
        uint32_t mt[4]

        uint32_t manthi, iexp, ix

        uint32_t * _u = <uint32_t *> &inp[0]

    with nogil:
        it[:] = [0x21200000U, 0x21400000U, 0x21800000U, 0x22100000U]
        mt[:] = [2, 4, 8, 1]

        for i in range(n_items):
            ix = (_u[i] & 0x01800000U) >> 23
            iexp = ((_u[i] & 0x7e000000U) >> 1) + it[ix]
            manthi = (mt[ix] * (_u[i] & 0x007fffffU)) >> 3
            manthi = (manthi + iexp) | (_u[i] & 0x80000000U)
            _u[i] = manthi if (_u[i] & 0x7fffffffU) else 0


@cython.boundscheck(False)
cpdef void segy_fixed_to_float(uint8_t *inp, size_t n_items) noexcept nogil:
    cdef:
        size_t i
        float f_value
        # reinterpret it as a 2 byte integers:
        # to determine absolute value and exponent
        i2 *  _in = <i2 *> &inp[0]

    with nogil:
        # this function is written assuming that the data format was in big
        # endian.
        if SPY_SYS_ENDIAN == 0:
            # On big endian, the 4 bytes are not swapped, thus it can be interpretted
            # using two int16 types (one for the value, the other for the power of
            # two exponent)
            for i in range(n_items):
                f_value = ldexpf(<float> _in[2 * i + 1], _in[2 * i])
                memcpy(&inp[4 * i], &f_value, 4)
        else:
            # On little endian the 4 bytes are all reversed, so the value and gain
            # also end up in different locations...
            for i in range(n_items):
                f_value = ldexpf(<float> _in[2 * i], _in[2 * i + 1])
                memcpy(&inp[4 * i], &f_value, 4)

ctypedef fused convertible:
    ui1
    i1
    ui2
    i2
    ui4
    i4
    f4
    ui8
    i8
    f8

@cython.boundscheck(False)
cdef void convert_to_float(convertible *inp, float *out, size_t n_items) noexcept nogil:
    """Converts any array type to an array of floats.
    
    Note
    ----
    The input and output *could* be the same memory location for an inplace transformation, of course
    the output array will still need enough allocated space to hold every item as a float. 

    If the input type is smaller than a float type, it loops from the end of the array backwards
    so there's no chance of overwriting the input bytes before they're needed. If the convertible
    size is bigger than a float, it loops from the start of the array forward.
    """
    cdef:
        size_t i

    with nogil:
        if sizeof(convertible) < sizeof(float):
            for i in range(n_items-1, -1, -1):
                out[i] = <float> inp[i]
        else:
            for i in range(n_items):
                out[i] = <float> inp[i]


@cython.boundscheck(False)
cdef void unpack3bytes_to_4(uint8_t *inp, size_t n_items, bint big_endian, bint signed) noexcept nogil:
    cdef:
        size_t i, i4, i3

    with nogil:
        if big_endian:
            # pad to 4 bytes by inserting a (signed) 0 before
            # (it is big endian after all)
            if not signed:
                for i in range(n_items - 1, -1, -1):
                    i4 = i * 4
                    i3 = i * 3
                    inp[i4 + 3] = inp[i3 + 2]
                    inp[i4 + 2] = inp[i3 + 1]
                    inp[i4 + 1] = inp[i3    ]
                    inp[i4    ] = 0  # input is unsigned, so just insert a 0 byte
            else:
                for i in range(n_items - 1, -1, -1):
                    i4 = i * 4
                    i3 = i * 3
                    inp[i4 + 3] = inp[i3 + 2]
                    inp[i4 + 2] = inp[i3 + 1]
                    inp[i4 + 1] = inp[i3    ]
                    # if it was a signed value and it was negative
                    # (meaning the most significant bit of the most significant byte was a 1)
                    if inp[i4 + 1] >> 7:
                        # insert 1111 (because it's a twos-compliment)
                        inp[i4] = 0xFF
                    else:
                        inp[i4] = 0

        else:
            if not signed:
                for i in range(n_items-1, -1, -1):
                    i4 = i * 4
                    i3 = i * 3
                    inp[i4 + 3] = 0
                    inp[i4 + 2] = inp[i3 + 2]
                    inp[i4 + 1] = inp[i3 + 1]
                    inp[i4    ] = inp[i3    ]
            else:
                for i in range(n_items-1, -1, -1):
                    i4 = i * 4
                    i3 = i * 3
                    # same logic as above.
                    if inp[i3 + 2] >> 7:
                        inp[i4 + 3] = 0xFF
                    else:
                        inp[i4 + 3] = 0
                    inp[i4 + 2] = inp[i3 + 2]
                    inp[i4 + 1] = inp[i3 + 1]
                    inp[i4    ] = inp[i3    ]

def trace_label_size():
    return TAP_LBL_BYTES

def text_header_size():
    return TXT_HDR_BYTES

def binary_header_size():
    return BIN_HDR_BYTES

def trace_header_size():
    return TRC_HDR_BYTES

def get_sizes_and_offsets(name):
    if name == 'binary_header':
        print(name)
        return _bh_info
    elif name == 'trace_header':
        print(name)
        return _th_info
    elif name =='extended_trace_header':
        print(name)
        return _eh_info
    else:
        print('su')
        return _su_info

def get_endian_keys():
    return {
        'big':SEGY_BIG_ENDIAN_FLAG,
        'little':SEGY_LITTLE_ENDIAN_FLAG,
        'pairwise':SEGY_PAIRWISE_ENDIAN_FLAG,
        'system':SEGY_SYSTEM_ENDIAN_FLAG,
    }

cdef double _to_degrees(double value, int coorunit) noexcept nogil:
    """A scaled coordinate, which is in the unit that the file says, in decimal degrees"""
    cdef double magnitude, deg, minutes, seconds, whole
    if coorunit == 2:  # seconds of arc
        return value / 3600.0
    if coorunit == 4:  # degrees, minutes and seconds, packed: DDDMMSS.ss
        magnitude = fabs(value)
        deg = floor(magnitude / 10000.0)
        minutes = floor((magnitude - deg * 10000.0) / 100.0)
        seconds = magnitude - deg * 10000.0 - minutes * 100.0
        whole = deg + minutes / 60.0 + seconds / 3600.0
        return -whole if value < 0.0 else whole
    return value


cdef i2 _time_scalar(double largest_ms) noexcept nogil:
    """The scalar (SEG-Y's tm_scal) that keeps the most digits of times, up to this large (ms), in 16 bits"""
    cdef double log_scale
    cdef i2 scalar
    if largest_ms == 0.0:
        return 0
    log_scale = log10(largest_ms / INT16_MAX)
    if log_scale > 0:
        log_scale = min(ceil(log_scale), 4)
        return <i2> (10 ** <int> log_scale)
    log_scale = min(floor(-log_scale), 4)
    scalar = <i2> (10 ** <int> log_scale)
    return -scalar


cdef i2 _scaled_time(double ms, i2 scalar) noexcept nogil:
    """A time (ms) as the 16 bit integer that goes with the scalar"""
    if scalar > 0:
        ms = ms / scalar
    elif scalar < 0:
        ms = ms * -scalar
    return <i2> floor(ms + 0.5)


cdef class SEGYTrace:
    cdef:
        trace_header hdr
        extended_trace_header ext_hdr
        size_t n_ext_hdr
        uint8_t[::1] data
        int _dtype
        size_t _itemsize

    @property
    def trace_header(self):
        return self.hdr

    @property
    def extended_trace_header(self):
        return self.ext_hdr

    @staticmethod
    cdef SEGYTrace from_spy_trace(spyc.Trace spy_tr):
        cdef:
            SEGYTrace segy = SEGYTrace.__new__(SEGYTrace)
            spyc.spy_trace_header *spy_hdr = spy_tr.hdr
            trace_header *hdr = &segy.hdr
            extended_trace_header *ext_hdr = &segy.ext_hdr
            double t0_mantissa
            int t0_exp
            i2 t_scale = 0
            char * trc_name = b'SEG00000'
            char * ext_name = b'SEG00001'
            size_t n_bytes = spy_hdr.n_sample * sizeof(f4)
        spyc.require_real(spy_tr)  # (SEG-Y has no complex traces)
        segy.data = np.empty(n_bytes, dtype=np.uint8)
        segy._dtype = DataFormat.float32_ieee
        segy._itemsize = sizeof(f4)
        with nogil:
            segy.n_ext_hdr = 1

            ext_hdr.nsamps = spy_hdr.n_sample
            if spy_hdr.n_sample <= UINT16_MAX:
                hdr.nsamps = spy_hdr.n_sample
            if n_bytes > 0:
                memcpy(&segy.data[0], &spy_tr.data[0], n_bytes)

            ext_hdr.dt = spy_hdr.d_sample * 1_000_000.0 # in micro seconds

            # The delay and the statics are in milliseconds, and share one scalar (which keeps as many significant
            # digits as can be kept for the largest of them).
            ms_t0 = spy_hdr.sample_start * 1_000.0
            ms_ss = spy_hdr.source_static * 1_000.0
            ms_rs = spy_hdr.receiver_static * 1_000.0
            ms_ts = spy_hdr.total_static * 1_000.0
            hdr.tm_scal = _time_scalar(max(max(fabs(ms_t0), fabs(ms_ss)), max(fabs(ms_rs), fabs(ms_ts))))
            hdr.delay = _scaled_time(ms_t0, hdr.tm_scal)
            hdr.shstat = _scaled_time(ms_ss, hdr.tm_scal)
            hdr.rcstat = _scaled_time(ms_rs, hdr.tm_scal)
            hdr.stapply = _scaled_time(ms_ts, hdr.tm_scal)

            ext_hdr.offset = spyc.hdr_offset(spy_hdr)

            ext_hdr.sht_x = spy_hdr.tx_loc[0]
            ext_hdr.sht_y = spy_hdr.tx_loc[1]
            ext_hdr.selev = spy_hdr.tx_loc[2]

            ext_hdr.rec_x = spy_hdr.rx_loc[0]
            ext_hdr.rec_y = spy_hdr.rx_loc[1]
            ext_hdr.relev = spy_hdr.rx_loc[2]

            # (the midpoint is not kept, it is the mean of the source and the receiver)
            ext_hdr.cdp_x = 0.5 * (spy_hdr.tx_loc[0] + spy_hdr.rx_loc[0])
            ext_hdr.cdp_y = 0.5 * (spy_hdr.tx_loc[1] + spy_hdr.rx_loc[1])

            ext_hdr.n_exttrchdr = 1

            hdr.trctype = spy_hdr.trace_type
            hdr.fold = spy_hdr.fold
            hdr.iline = spy_hdr.iline
            hdr.xline = spy_hdr.xline
            ext_hdr.linetrc = spy_hdr.trace_id
            if spy_hdr.coord_unit == spyc.CoordinateUnit.length:
                hdr.coorunit = 1
            elif spy_hdr.coord_unit == spyc.CoordinateUnit.degrees:
                hdr.coorunit = 3  # (decimal degrees)
            else:
                hdr.coorunit = 0
            ext_hdr.cdp = spy_hdr.ensemble_number
            hdr.cdptrc = spy_hdr.ensemble_trace_number  # (1 based, 0 is not set)
            hdr.header_name = trc_name
            ext_hdr.header_name = ext_name

        return segy

    cdef spyc.Trace to_spy_trace(self):
        cdef:
            spyc.Trace spy_tr = spyc.Trace.__new__(spyc.Trace)
            spyc.spy_trace_header *spy_hdr = spyc.new_hdr()

        spy_tr.hdr = spy_hdr

        spy_tr.hdr_owner = True

        cdef size_t nsamps = self.ext_hdr.nsamps

        spy_tr.data = spyc.alloc_data(nsamps)

        cdef:
            bint supported = True
            double xy_scale = 1.0
            double z_scale = 1.0
            double file_offset = 0.0
            double time_scale = 1.0

        with nogil:
            spy_hdr.n_sample = self.ext_hdr.nsamps

            if nsamps == 0:
                pass
            elif self._dtype == DataFormat.int08:
                convert_to_float(<i1 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.int16:
                convert_to_float(<i2 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.int32:
                convert_to_float(<i4 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.int64:
                convert_to_float(<i8 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.uint08:
                convert_to_float(<ui1 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.uint16:
                convert_to_float(<ui2 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.uint32:
                convert_to_float(<ui4 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.uint64:
                convert_to_float(<ui8 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.float32_ieee:
                convert_to_float(< f4 *> &self.data[0], &spy_tr.data[0], nsamps)
            elif self._dtype == DataFormat.float64_ieee:
                convert_to_float(< f8 *> &self.data[0], &spy_tr.data[0], nsamps)
            else:
                # ibm float4 conversion should've already been handled on read in.
                # 3 byte integers should have also been converted to 4 byte integers
                # as well, so this is a format we really do not know how to convert.
                supported = False

            # The delay and the statics are in ms, times the scalar (a positive scalar multiplies and a negative one
            # divides, and 0 is the same as 1).
            if self.hdr.tm_scal > 0:
                time_scale = self.hdr.tm_scal
            elif self.hdr.tm_scal < 0:
                time_scale = 1.0 / -self.hdr.tm_scal
            time_scale /= 1_000.0
            spy_hdr.sample_start = self.hdr.delay * time_scale
            spy_hdr.source_static = self.hdr.shstat * time_scale
            spy_hdr.receiver_static = self.hdr.rcstat * time_scale
            spy_hdr.total_static = self.hdr.stapply * time_scale

            if self.n_ext_hdr:
                spy_hdr.d_sample = self.ext_hdr.dt / 1_000_000.0
                file_offset = self.ext_hdr.offset

                spy_hdr.tx_loc[0] = self.ext_hdr.sht_x
                spy_hdr.tx_loc[1] = self.ext_hdr.sht_y
                spy_hdr.tx_loc[2] = self.ext_hdr.selev

                spy_hdr.rx_loc[0] = self.ext_hdr.rec_x
                spy_hdr.rx_loc[1] = self.ext_hdr.rec_y
                spy_hdr.rx_loc[2] = self.ext_hdr.relev

                spy_hdr.trace_id = self.ext_hdr.linetrc
                spy_hdr.ensemble_number = self.ext_hdr.cdp
            else:
                # positive scalars multiply and negative ones divide, 0 is the same as 1.
                if self.hdr.co_scal > 0:
                    xy_scale = self.hdr.co_scal
                elif self.hdr.co_scal < 0:
                    xy_scale = 1.0 / -self.hdr.co_scal
                if self.hdr.ed_scal > 0:
                    z_scale = self.hdr.ed_scal
                elif self.hdr.ed_scal < 0:
                    z_scale = 1.0 / -self.hdr.ed_scal

                spy_hdr.d_sample = self.hdr.dt / 1_000_000.0
                file_offset = self.hdr.offset

                spy_hdr.tx_loc[0] = self.hdr.sht_x * xy_scale
                spy_hdr.tx_loc[1] = self.hdr.sht_y * xy_scale
                spy_hdr.tx_loc[2] = self.hdr.selev * z_scale

                spy_hdr.rx_loc[0] = self.hdr.rec_x * xy_scale
                spy_hdr.rx_loc[1] = self.hdr.rec_y * xy_scale
                spy_hdr.rx_loc[2] = self.hdr.relev * z_scale

                spy_hdr.trace_id = self.hdr.linetrc
                spy_hdr.ensemble_number = self.hdr.cdp

            if spy_hdr.d_sample == 0:
                # (an extended header can leave this blank)
                spy_hdr.d_sample = self.hdr.dt / 1_000_000.0

            # Angles are always decimal degrees (the file's can be seconds of arc, or degrees, minutes and seconds).
            if self.hdr.coorunit == 1:
                spy_hdr.coord_unit = spyc.CoordinateUnit.length
            elif self.hdr.coorunit >= 2 and self.hdr.coorunit <= 4:
                spy_hdr.coord_unit = spyc.CoordinateUnit.degrees
                spy_hdr.tx_loc[0] = _to_degrees(spy_hdr.tx_loc[0], self.hdr.coorunit)
                spy_hdr.tx_loc[1] = _to_degrees(spy_hdr.tx_loc[1], self.hdr.coorunit)
                spy_hdr.rx_loc[0] = _to_degrees(spy_hdr.rx_loc[0], self.hdr.coorunit)
                spy_hdr.rx_loc[1] = _to_degrees(spy_hdr.rx_loc[1], self.hdr.coorunit)
            else:
                spy_hdr.coord_unit = spyc.CoordinateUnit.unknown

            # The offset is not kept in the header, it is the distance from the source to the receiver. A file with an
            # offset but no coordinates (common) puts the receiver that far from the source, along x (a length, so not
            # when the coordinates are angles).
            if spy_hdr.coord_unit != spyc.CoordinateUnit.degrees and file_offset != 0.0 and spy_hdr.rx_loc[0] == spy_hdr.tx_loc[0] and spy_hdr.rx_loc[1] == spy_hdr.tx_loc[1]:
                spy_hdr.rx_loc[0] = spy_hdr.tx_loc[0] + file_offset

            spy_hdr.trace_type = self.hdr.trctype
            spy_hdr.fold = self.hdr.fold
            spy_hdr.iline = self.hdr.iline
            spy_hdr.xline = self.hdr.xline
            spy_hdr.ensemble_trace_number = max(self.hdr.cdptrc, 0)  # (1 based, 0 is not set)

        if not supported:
            raise ValueError(f"Unable to convert SEG-Y data format {self._dtype} to floating point.")

        return spy_tr

    cdef to_file_descriptor(self, FILE *fd):
        cdef size_t n_write
        n_write = spy_io.write_struct_to_file(&self.hdr, _th_info, STDH_IS_PACKED, TRC_HDR_SIZE, fd)
        if n_write != TRC_HDR_SIZE:
            raise IOError("Error writing trace header to file.")

        n_write = spy_io.write_struct_to_file(&self.ext_hdr, _eh_info, EXTH_IS_PACKED, TRC_HDR_SIZE, fd)
        if n_write != TRC_HDR_SIZE:
            raise IOError("Error writing trace header to file.")

        # `data` is a plain buffer of bytes (already in the file's data format), so write it as bytes.
        cdef size_t n_bytes = self.data.shape[0]
        if n_bytes > 0:
            n_write = fwrite(&self.data[0], 1, n_bytes, fd)
            if n_write != n_bytes:
                raise IOError("Error writing trace data to file.")

    @staticmethod
    def from_seispy(spyc.Trace trace):
        """Convert a seispy Trace to a SEG-Y trace (float32 samples)."""
        return SEGYTrace.from_spy_trace(trace)

    def write(self, file):
        """Write this trace (trace header, extended trace header, then the data) to a binary file object."""
        cdef spy_io.DupResult res = spy_io.PyFile_Dup(file, "wb")
        try:
            self.to_file_descriptor(res.handle)
        finally:
            spy_io.PyFile_DupClose(file, res.handle, res.offset)
        file.flush()

    def __getitem__(self, item):
        return np.asarray(self)[item]

    @property
    def dtype(self):
        return np.asarray(self).dtype

    @staticmethod
    cdef SEGYTrace from_file_descriptor(FILE *fd, binary_header *bhdr):
        # file endian is 0, 1, or 2, for big, little, pairwise.
        cdef:
            SEGYTrace tr = SEGYTrace.__new__(SEGYTrace)
            size_t n_read
            int n_ext_headers
        cdef str endian_flag
        if bhdr.byte_order_id == SEGY_BIG_ENDIAN_FLAG:
            endian_flag = ">"
        elif bhdr.byte_order_id == SEGY_LITTLE_ENDIAN_FLAG:
            endian_flag = "<"
        else:
            endian_flag = "<>"
        n_read = spy_io.read_struct_from_file(&tr.hdr, _th_info, STDH_IS_PACKED, TRC_HDR_BYTES, fd, endian_flag)
        if n_read != TRC_HDR_BYTES:
            if n_read == 0 and feof(fd):
                # a clean end of the file, in between traces.
                raise EOFError("Reached the end of the file.")
            elif feof(fd):
                raise SEGYTruncatedError("The file ended in the middle of a trace header.")
            else:
                raise IOError("Error reading trace header from file.")

        if bhdr.is_fixed_traces:
            tr.hdr.nsamps = bhdr.n_sample_per_trace
            tr.hdr.dt = bhdr.d_sample
            tr.ext_hdr.nsamps = bhdr.next_sample_per_trace
            tr.ext_hdr.dt = bhdr.dext_sample
        # otherwise the trace header values are used, unless they are left blank.
        if tr.hdr.nsamps == 0:
            tr.hdr.nsamps = bhdr.n_sample_per_trace
        if tr.hdr.dt == 0:
            tr.hdr.dt = bhdr.d_sample
        cdef size_t nsamps = tr.hdr.nsamps
        if bhdr.nmax_ext_trc_hdr>0:
            n_read = spy_io.read_struct_from_file(&tr.ext_hdr, _eh_info, EXTH_IS_PACKED, TRC_HDR_BYTES, fd, endian_flag)
            if n_read != TRC_HDR_BYTES:
                if feof(fd):
                    raise SEGYTruncatedError("The file ended in the middle of an extended trace header.")
                else:
                    raise IOError("Error reading extended trace header from file.")

            n_ext_headers = tr.ext_hdr.n_exttrchdr
            if n_ext_headers == 0:
                n_ext_headers = bhdr.nmax_ext_trc_hdr
            tr.n_ext_hdr = n_ext_headers

            if tr.ext_hdr.nsamps > 0:
                nsamps = tr.ext_hdr.nsamps

            # seek through any extra headers (for now)
            # should likely just stash these bytes on the spyc.Trace object
            # for passing through to a segy output.
            spy_io.spy_fseek(fd, TRC_HDR_SIZE * (n_ext_headers - 1), SEEK_CUR)
        # always set nsamps on the extended header for future use.
        tr.ext_hdr.nsamps = nsamps

        tr._dtype = bhdr.data_format
        tr._itemsize = data_format_sizes[bhdr.data_format]

        cdef size_t n_data_bytes = tr._itemsize * nsamps

        if n_data_bytes > 0:
            if tr._itemsize == 3:
                tr.data = np.empty(nsamps * 4, dtype=np.uint8)
            else:
                tr.data = np.empty(n_data_bytes, dtype=np.uint8)
        else:
            tr.data = np.empty((0,), dtype=np.uint8)
        if nsamps > 0:
            #tr.data `= <float[:nsamps]> malloc(sizeof(float) * nsamps)
            n_read = fread(&tr.data[0], 1, n_data_bytes, fd)
            if n_read != n_data_bytes:
                if feof(fd):
                    raise SEGYTruncatedError("The file ended in the middle of a trace's data.")
                else:
                    raise IOError("Error reading trace data from file.")
            if tr._itemsize == 3:
                if endian_flag == "<>":
                    raise NotImplementedError("Reading pairwise byteswapped 3 byte values is not a defined behavoir.")
                unpack3bytes_to_4(&tr.data[0], nsamps, endian_flag == ">", bhdr.data_format==DataFormat.int24)
                # now it's the corresponding 4 byte size.
                tr._itemsize = 4
                tr._dtype = DataFormat.int32 if bhdr.data_format == DataFormat.int24 else DataFormat.uint32
            if tr._itemsize == 1:
                pass
            elif endian_flag == ">":
                if tr._itemsize == 2:
                    bswap.swap16_big_and_system(<uint16_t*> &tr.data[0], nsamps)
                elif tr._itemsize == 4:
                    bswap.swap32_big_and_system(<uint32_t*> &tr.data[0], nsamps)
                elif tr._itemsize == 8:
                    bswap.swap64_big_and_system(<uint64_t*> &tr.data[0], nsamps)
                else:
                    raise NotImplementedError
            elif endian_flag == "<":
                if tr._itemsize == 2:
                    bswap.swap16_little_and_system(<uint16_t*> &tr.data[0], nsamps)
                elif tr._itemsize == 4:
                    bswap.swap32_little_and_system(<uint32_t*> &tr.data[0], nsamps)
                elif tr._itemsize == 8:
                    bswap.swap64_little_and_system(<uint64_t*> &tr.data[0], nsamps)
                else:
                    raise NotImplementedError
            elif endian_flag == "<>":
                if tr._itemsize == 2:
                    bswap.swap16_pairwise_and_system(<uint16_t*> &tr.data[0], nsamps)
                elif tr._itemsize == 4:
                    bswap.swap32_pairwise_and_system(<uint32_t*> &tr.data[0], nsamps)
                elif tr._itemsize == 8:
                    bswap.swap64_pairwise_and_system(<uint64_t*> &tr.data[0], nsamps)
                else:
                    raise NotImplementedError

            if tr._dtype == DataFormat.float32_ibm:
                ibm_to_float(&tr.data[0], nsamps)
                tr._dtype = DataFormat.float32_ieee
            elif tr._dtype == DataFormat.fixed32:
                segy_fixed_to_float(&tr.data[0], nsamps)
                tr._dtype = DataFormat.float32_ieee
        return tr

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        buffer.obj = self
        buffer.buf = <void *> &self.data[0]
        buffer.len = self.ext_hdr.nsamps
        buffer.itemsize = self._itemsize
        buffer.ndim = 1

        buffer.shape = <Py_ssize_t *> malloc(sizeof(Py_ssize_t))
        buffer.shape[0] = self.ext_hdr.nsamps
        buffer.strides = <Py_ssize_t *> malloc(sizeof(Py_ssize_t))
        buffer.strides[0] = buffer.itemsize
        buffer.readonly = 0

        if flags & pybuf.PyBUF_FORMAT:
            if self._dtype == DataFormat.int08:
                buffer.format = 'b'
            elif self._dtype == DataFormat.uint08:
                buffer.format = 'B'
            elif self._dtype == DataFormat.int16:
                buffer.format = '=h'
            elif self._dtype == DataFormat.uint16:
                buffer.format = '=H'
            elif self._dtype == DataFormat.int32:
                buffer.format = '=i'
            elif self._dtype == DataFormat.uint32:
                buffer.format = '=I'
            elif self._dtype == DataFormat.int64:
                buffer.format = '=q'
            elif self._dtype == DataFormat.uint64:
                buffer.format = '=Q'
            elif self._dtype == DataFormat.float32_ieee:
                buffer.format = '=f'
            else: # self._dtype == DataFormat.float64_ieee
                buffer.format = '=d'
        else:
            buffer.format = NULL

        buffer.internal = NULL
        buffer.suboffsets = NULL #self.data.suboffsets

    def __releasebuffer__(self, Py_buffer *buffer):
        free(buffer.shape)
        free(buffer.strides)

_SEGY_SORT_TO_SPY_ENSEMBLE = {
    TraceSorting.unknown : spyc.EnsembleType.unknown,
    TraceSorting.none : spyc.EnsembleType.unsorted,
    TraceSorting.cdp : spyc.EnsembleType.common_midpoint,
    TraceSorting.midpoint : spyc.EnsembleType.common_midpoint,
    TraceSorting.source : spyc.EnsembleType.tx_gather,
    TraceSorting.receiver : spyc.EnsembleType.rx_gather,
    TraceSorting.offset : spyc.EnsembleType.common_offset,
}

_SEGY_SORT_SPY_SORT = {
    spyc.EnsembleType.unknown : TraceSorting.unknown,
    spyc.EnsembleType.unsorted : TraceSorting.none,
    spyc.EnsembleType.tx_gather : TraceSorting.source,
    spyc.EnsembleType.rx_gather : TraceSorting.receiver,
    spyc.EnsembleType.common_midpoint : TraceSorting.midpoint,
    spyc.EnsembleType.common_offset : TraceSorting.offset,
}

cdef class SEGYCollection:
    cdef:
        binary_header hdr
        object file
        object text_header
        list extra_text_headers

    def __cinit__(self):
        self.file = None
        self.text_header = None
        self.extra_text_headers = []

    def traces(self):
        """A one-pass iterator of seispy Traces, converted from this file's traces.

        Call it again to read the file again.
        """
        return _FileSEGYToTraceIterator(self)

    def __iter__(self):
        return _FileSEGYIterator(self)

    @property
    def file_endian(self):
        if self.hdr.byte_order_id == SEGY_BIG_ENDIAN_FLAG:
            return "big"
        elif self.hdr.byte_order_id == SEGY_LITTLE_ENDIAN_FLAG:
            return "little"
        elif self.hdr.byte_order_id == SEGY_PAIRWISE_ENDIAN_FLAG:
            return "pairwise"
        else:
            return "unknown"

    @staticmethod
    cdef SEGYCollection from_spy_collection(spyc.CollectionHeader spy_hdr, spyc.spy_trace_header *spy_tr_hdr=NULL):
        cdef:
            SEGYCollection segy = SEGYCollection.__new__(SEGYCollection)
            binary_header *hdr = &segy.hdr

        hdr.meters_or_feet = 1
        hdr.data_format = DataFormat.float32_ieee
        hdr.byte_order_id = SEGY_SYSTEM_ENDIAN_FLAG
        hdr.major_rev = 2
        hdr.minor_rev = 1
        hdr.nmax_ext_trc_hdr = 1
        hdr.sort_method = _SEGY_SORT_SPY_SORT[spy_hdr.ensemble_type]
        hdr.n_traces = spy_hdr.n_traces

        if spy_hdr.uniform_traces:
            hdr.is_fixed_traces = True
            if spy_tr_hdr is NULL:
                raise TypeError('A trace header is required if marked for uniform traces.')
            hdr.next_sample_per_trace = spy_tr_hdr.n_sample
            hdr.dext_sample = spy_tr_hdr.d_sample

    @staticmethod
    cdef SEGYCollection from_header_bytes(const unsigned char[::1] bys, endian=None, data_format=None, fixed_traces=None):
        cdef SEGYCollection coll = SEGYCollection.__new__(SEGYCollection)
        if bys.shape[0] < BIN_HDR_SIZE:
            raise ValueError(f"Incorrect number of bytes, expected at least {BIN_HDR_SIZE}, got {bys.shape[0]}.")

        spy_io.copy_struct_from_char(&coll.hdr, _bh_info, BH_IS_PACKED, BIN_HDR_SIZE, &bys[0])
        # first check if the endian flag is defined:
        cdef str endian_type
        if coll.hdr.byte_order_id == SEGY_BIG_ENDIAN_FLAG:
            endian_type = ">"
        elif coll.hdr.byte_order_id == SEGY_LITTLE_ENDIAN_FLAG:
            endian_type = "<"
        elif coll.hdr.byte_order_id == SEGY_PAIRWISE_ENDIAN_FLAG:
            endian_type = "<>"
        else:
            # then use endian if it was passed
            if endian is not None:
                if endian not in ["<", ">", "<>"]:
                    raise ValueError(
                        "Endian flag must be '>', '<', or '<>' for big, little or pairwise swapped, respectively"
                    )
                endian_type = endian
            else:
                # Try to guess
                if coll.hdr.data_format >= 1 and coll.hdr.data_format <= 16:
                    # If big, then good...
                    endian_type = {"little":"<", "big":">"}[sys.byteorder]
                    # if not big, then it is likely little
                    # (but could be pairwise swapped as this is only a 2byte identifier
                elif coll.hdr.data_format >= 256 and coll.hdr.data_format <= 4096:
                    # If on little system (or pairwise) this will say it was big order
                    # If on big system this will say file was little order (or pairwise).
                    # basically means it needs to be swapped.
                    endian_type = {"little":">", "big":"<"}[sys.byteorder]
                    # for now, don't worry about a pairwise file...
                    # as it is technically only allowed in rev 2, and therefore
                    # should be properly identified by the byte_order_id field.
                else:
                    # Default to big endian as described in rev < 2
                    endian_type = '>'
                    warnings.warn(
                        "Unable to guess endian from segy file, defaulted to big endian.",
                        UserWarning,
                        stacklevel=2,
                    )
        bswap.swap_struct_endian_and_system(&coll.hdr, _bh_info, endian=endian_type)
        # Then write back the endian identifier for later use on read in.
        if endian_type == '>':
            coll.hdr.byte_order_id = SEGY_BIG_ENDIAN_FLAG
        elif endian_type == "<":
            coll.hdr.byte_order_id = SEGY_LITTLE_ENDIAN_FLAG
        elif endian_type == "<>":
            coll.hdr.byte_order_id = SEGY_PAIRWISE_ENDIAN_FLAG

        # do some validation on the major minor revision numbers
        if coll.hdr.major_rev not in [0, 1, 2]:
            warnings.warn("Invalid major revision, setting to revision 0")
            coll.hdr.major_rev = 0
            coll.hdr.minor_rev = 0
        elif coll.hdr.major_rev == 2 and coll.hdr.minor_rev not in [0, 1]:
            warnings.warn("Invalid minor revision, setting to revision 0")
            coll.hdr.major_rev = 0
            coll.hdr.minor_rev = 0

        if coll.hdr.major_rev < 2:
            coll.hdr.nmax_ext_trc_hdr = 0
            if data_format is not None:
                coll.hdr.data_format = str_to_data_format(data_format)
            elif coll.hdr.data_format == 0:
                coll.hdr.data_format = DataFormat.float32_ibm

        if fixed_traces is not None:
            coll.hdr.is_fixed_traces = <bint> fixed_traces
        return coll

    @classmethod
    def from_file(cls, filename, endian=None, data_format=None, fixed_traces=None):
        cdef:
            SEGYTrace trace
            SEGYCollection coll

        if hasattr(filename, 'read'):
            ctx = nullcontext(filename)
        else:
            filename = os.fspath(filename)
            ctx = open(filename, "rb")

        with ctx as f:
            # an open file is taken to be a whole SEG-Y file, wherever it was left.
            f.seek(0)
            text_header_bytes = f.read(TXT_HDR_BYTES)
            coll = SEGYCollection.from_header_bytes(
                f.read(BIN_HDR_BYTES),
                endian=endian,
                data_format=data_format,
                fixed_traces=fixed_traces,
            )
            coll.text_header = _decode_text(text_header_bytes)
            # then try to read any extra text headers (and find the start of the traces)
            coll.extra_text_headers = get_extra_text_headers(coll, f)

        coll.hdr.next_txt_hdr = len(coll.extra_text_headers)

        coll.file = filename

        if coll.hdr.major_rev < 2:
            coll.hdr.n_traces = 0
        return coll

    @property
    def textual_header(self):
        return self.text_header

    @property
    def extra_textual_headers(self):
        return self.extra_text_headers

    @property
    def binary_header(self):
        return self.hdr


cdef list get_extra_text_headers(SEGYCollection coll, f):
    cdef:
        # I'll at least have a text header and a binary header
        spyc.spy_off_t trace_start
        size_t next_txt_hdr

    trace_start = TXT_HDR_BYTES + BIN_HDR_BYTES
    next_txt_hdr = coll.hdr.next_txt_hdr
    # If the collection header info contains the first trace offset use that.
    if coll.hdr.major_rev >= 2 and coll.hdr.first_trace_byte_offset > 0:
        trace_start = coll.hdr.first_trace_byte_offset
        next_txt_hdr = (trace_start - (TXT_HDR_BYTES + BIN_HDR_BYTES)) // TXT_HDR_BYTES
    elif coll.hdr.major_rev == 0:
        next_txt_hdr = 0

    cdef list text_headers = []
    cdef bint get_another = next_txt_hdr != 0
    cdef size_t i_txt_hdr = 0
    while get_another:
        txt_hdr_bytes = f.read(TXT_HDR_BYTES)
        text_headers.append(_decode_text(txt_hdr_bytes))

        i_txt_hdr += 1
        if next_txt_hdr < 0:
            get_another = not text_headers[-1].startswith('((SEG: EndText))')
        else:
            get_another = i_txt_hdr < next_txt_hdr
    return text_headers


cdef str _decode_text(bytes text):
    if b'C' not in [text[0], text[3040], text[3120]]:
        decoded = text.decode('EBCDIC-CP-BE')
        try:
            text = decoded.encode('ascii')
            return decoded
        except UnicodeEncodeError:
            pass
    try:
        return text.decode('ascii')
    except UnicodeDecodeError:
        return text.decode('EBCDIC-CP-BE')


def _warn_if_short(size_t n_read, size_t n_expected):
    if n_expected > 0 and n_read < n_expected:
        warnings.warn(
            f"The SEG-Y file ended after {n_read} traces, but its binary header says it holds {n_expected}.",
            UserWarning,
            stacklevel=3,
        )


def read_segy(filename, endian=None, data_format=None, fixed_traces=None):
    """Read a SEG-Y file as a stream of seispy Traces, ready to be piped into stages::

        read_segy("shots.segy") | bfilt(...)

    Parameters
    ----------
    filename : path or binary file object
    endian, data_format, fixed_traces
        Overrides for files that do not identify themselves properly, see `SEGYCollection.from_file`.

    Returns
    -------
    A trace iterator. It can only be used once, call `read_segy` again to re-read the file.
    """
    return SEGYCollection.from_file(
        filename, endian=endian, data_format=data_format, fixed_traces=fixed_traces
    ).traces()


cdef class _FileSEGYIterator:
    cdef:
        FILE *fd
        bint owner
        object file
        spy_io.spy_off_t orig_pos
        binary_header bhdr_local  # my own copy, so I never depend on the collection staying alive
        binary_header *bhdr
        str endian_flag
        size_t i

    def __cinit__(self):
        self.fd = NULL
        self.owner = False
        self.file = None
        self.i = 0

    def __dealloc__(self):
        # make sure I get closed up when I'm garbage collected
        self._close_file()

    cdef _close_file(self):
        # first close my duped file
        if self.fd is not NULL and self.file is not None:
            spy_io.PyFile_DupClose(self.file, self.fd, self.orig_pos)
            self.fd = NULL
        # If I own the original, close it
        if self.owner and self.file is not None:
            self.file.close()
        # and clear my reference to the original
        self.file = None

    def __init__(self, SEGYCollection coll):
        file = coll.file
        if not hasattr(file, "read"):
            # open the file
            file = open(os.fspath(file), "rb")
            self.owner = True
        else:
            self.owner = False
        self.file = file
        self.bhdr_local = coll.hdr
        self.bhdr = &self.bhdr_local
        if coll.hdr.byte_order_id == SEGY_BIG_ENDIAN_FLAG:
            self.endian_flag = ">"
        elif coll.hdr.byte_order_id == SEGY_LITTLE_ENDIAN_FLAG:
            self.endian_flag = "<"
        elif coll.hdr.byte_order_id == SEGY_PAIRWISE_ENDIAN_FLAG:
            self.endian_flag = "<>"
        else:
            raise ValueError("Unknown File Endian.")

        cdef spyc.spy_off_t trace_start
        cdef spy_io.DupResult res

        try:
            res = spy_io.PyFile_Dup(file, "rb")
            self.fd = res.handle
            self.orig_pos = res.offset
            if coll.hdr.major_rev >= 2 and coll.hdr.first_trace_byte_offset != 0:
                trace_start = coll.hdr.first_trace_byte_offset
            else:
                trace_start = (coll.hdr.next_txt_hdr + 1) * TXT_HDR_BYTES + BIN_HDR_BYTES
            # Go to the start of the traces. This is absolute so that it works for file objects
            # that were already read from, or are being read a second time.
            spy_io.spy_fseek(self.fd, trace_start, SEEK_SET)
        except Exception as err:
            self._close_file()
            raise err

    cdef SEGYTrace next_trace(self):
        if self.fd is NULL or self.i == SIZE_MAX or (self.bhdr.n_traces > 0 and self.i == self.bhdr.n_traces):
            raise StopIteration()
        cdef:
            SEGYTrace trace_in
        try:
            trace_in = SEGYTrace.from_file_descriptor(self.fd, self.bhdr)
        except EOFError:
            _warn_if_short(self.i, self.bhdr.n_traces)
            self.bhdr.n_traces = self.i
            self._close_file()
            raise StopIteration()
        except Exception as err:
            # if something else goes wrong reading in from the file descriptor
            # close myself and re-raise the error.
            self._close_file()
            raise
        if trace_in.ext_hdr.last_trc == LastID.in_file:
            # set i as SIZE_MAX to trigger ending on the next request.
            self.bhdr.n_traces = self.i + 1
            self.i = SIZE_MAX
        else:
            self.i += 1
        if self.i == SIZE_MAX or (self.bhdr.n_traces > 0 and self.i == self.bhdr.n_traces):
            # The next request will raise a StopIteration so close myself now.
            self._close_file()
        return trace_in

    def __next__(self):
        return self.next_trace()


## same code as above except that it also converts to a seispy trace.
cdef class _FileSEGYToTraceIterator(spyc.BaseTraceIterator):
    cdef:
        FILE *fd
        bint owner
        object file
        spy_io.spy_off_t orig_pos
        binary_header bhdr_local  # my own copy, so I never depend on the collection staying alive
        binary_header *bhdr
        str endian_flag

    def __cinit__(self):
        self.fd = NULL
        self.owner = False
        self.file = None
        self.hdr.n_traces = 0

    def __dealloc__(self):
        # make sure I get closed up when I'm garbage collected
        self._close_file()

    cdef _close_file(self):
        # first close my duped file
        if self.fd is not NULL and self.file is not None:
            spy_io.PyFile_DupClose(self.file, self.fd, self.orig_pos)
            self.fd = NULL
        # If I own the original, close it
        if self.owner and self.file is not None:
            self.file.close()
        # and clear my reference to the original
        self.file = None

    def __init__(self, SEGYCollection coll):
        file = coll.file
        if not hasattr(file, "read"):
            # open the file
            file = open(os.fspath(file), "rb")
            self.owner = True
        else:
            self.owner = False
        self.file = file
        self.bhdr_local = coll.hdr
        self.bhdr = &self.bhdr_local
        if coll.hdr.byte_order_id == SEGY_BIG_ENDIAN_FLAG:
            self.endian_flag = ">"
        elif coll.hdr.byte_order_id == SEGY_LITTLE_ENDIAN_FLAG:
            self.endian_flag = "<"
        elif coll.hdr.byte_order_id == SEGY_PAIRWISE_ENDIAN_FLAG:
            self.endian_flag = "<>"
        else:
            raise ValueError("Unknown File Endian.")

        self.hdr.n_traces = self.bhdr.n_traces
        self.hdr.uniform_traces = self.bhdr.is_fixed_traces
        self.hdr.ensemble_type = _SEGY_SORT_TO_SPY_ENSEMBLE.get(self.bhdr.sort_method, spyc.EnsembleType.unknown)

        cdef spyc.spy_off_t trace_start
        cdef spy_io.DupResult res

        try:
            res = spy_io.PyFile_Dup(file, "rb")
            self.fd = res.handle
            self.orig_pos = res.offset
            if coll.hdr.major_rev >= 2 and coll.hdr.first_trace_byte_offset != 0:
                trace_start = coll.hdr.first_trace_byte_offset
            else:
                trace_start = (coll.hdr.next_txt_hdr + 1) * TXT_HDR_BYTES + BIN_HDR_BYTES
            # Go to the start of the traces. This is absolute so that it works for file objects
            # that were already read from, or are being read a second time.
            spy_io.spy_fseek(self.fd, trace_start, SEEK_SET)
        except Exception as err:
            self._close_file()
            raise err

    cdef spyc.Trace next_trace(self):
        if self.fd is NULL or self.i == SIZE_MAX or (self.hdr.n_traces > 0 and self.i == self.hdr.n_traces):
            raise StopIteration()
        cdef:
            SEGYTrace trace_in
        try:
            trace_in = SEGYTrace.from_file_descriptor(self.fd, self.bhdr)
        except EOFError:
            _warn_if_short(self.i, self.hdr.n_traces)
            self.hdr.n_traces = self.i
            self._close_file()
            raise StopIteration()
        except Exception as err:
            # if something else goes wrong reading in from the file descriptor
            # close myself and re-raise the error.
            self._close_file()
            raise
        if trace_in.ext_hdr.last_trc == LastID.in_file:
            # set i as SIZE_MAX to trigger ending on the next request.
            self.hdr.n_traces = self.i + 1
            self.i = SIZE_MAX
        else:
            self.i += 1
        if self.i == SIZE_MAX or (self.hdr.n_traces > 0 and self.i == self.hdr.n_traces):
            # The next request will raise a StopIteration so close myself now.
            self._close_file()
        return trace_in.to_spy_trace()