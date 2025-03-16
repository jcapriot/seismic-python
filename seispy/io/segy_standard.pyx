# cython: embedsignature=True, language_level=3
# cython: linetrace=True
import os
import sys
import warnings
from contextlib import nullcontext

from libc.math cimport floor, ceil, log10, fabs
from libc.stdint cimport (
    INT8_MAX, INT16_MAX, INT32_MAX, INT64_MAX,
    UINT8_MAX, UINT16_MAX, UINT32_MAX, UINT64_MAX,
    SIZE_MAX, int8_t, int32_t, uint32_t, uint8_t, uint16_t, uint64_t
)
from libc.stdio cimport fwrite, fread, FILE, SEEK_CUR, feof
from libc.stdlib cimport malloc
cimport cython

from . cimport byteswapping as bswap
from . cimport _io as spy_io

from .. cimport container as spyc

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

@cython.boundscheck(False)
cpdef void ibm_to_float(uint8_t[::1] inp) noexcept nogil:
    cdef:
        size_t n_bytes = inp.shape[0]
        size_t n_items = n_bytes // 4
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


ctypedef fused convertible4:
    ui4
    i4
    f4

@cython.boundscheck(False)
cdef int convert4_to_float(convertible4 *inp, size_t n_items) noexcept nogil:
    cdef:
        size_t i
        float holder
        convertible4 *hld_ptr

    if convertible4 is not float:
        with nogil:
            for i in range(n_items):
                holder = <float> inp[i]
                hld_ptr = <convertible4 *> &holder
                inp[i] = hld_ptr[0]

ctypedef fused convertibleX:
    ui1
    i1
    ui2
    i2
    ui8
    i8
    f8

@cython.boundscheck(False)
cdef float *convert_to_float(convertibleX *inp, size_t n_items) noexcept nogil:
    cdef:
        size_t i
        float * out

    out = <float *> malloc(sizeof(float) * n_items)

    with nogil:
        for i in range(n_items):
            out[i] = <float> inp[i]

    return out


@cython.boundscheck(False)
cdef void unpack3bytes_to_4(uint8_t[::1] inp, str endian_flag, bint signed) noexcept nogil:
    cdef:
        size_t i, i4, i3
        size_t n_bytes = inp.shape[0]
        size_t n_items = n_bytes // 4

    with nogil:
        if endian_flag == ">":
            # pad to 4 bytes by inserting a (signed) 0 before
            # (it is big endian after all)
            for i in range(n_items - 1, -1, -1):
                i4 = i * 4
                i3 = i * 3
                inp[i4 + 3] = inp[i3 + 2]
                inp[i4 + 2] = inp[i3 + 1]
                inp[i4 + 1] = inp[i3    ]
                # if it was a signed value and it was negative
                # meaning the first bit of the MSB was a 1
                # insert 1111
                if signed and inp[i4 + 1] & 1 == 1:
                    inp[i4] = 0xFF
                else:
                    inp[i4    ] = 0
        elif endian_flag == "<":
            # pad to 4 bytes by appending a (signed) 0
            for i in range(n_items - 1, -1, -1):
                i4 = i * 4
                i3 = i * 3
                # same as above for properly accounting for sign
                if signed and inp[i3 + 2] & 1 == 1:
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

cdef class SEGYTrace:
    cdef:
        trace_header hdr
        extended_trace_header ext_hdr
        size_t n_ext_hdr
        float[::1] data

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
        with nogil:
            segy.data = spy_tr.data
            segy.n_ext_hdr = 1

            if spy_hdr.n_sample <= UINT16_MAX:
                hdr.nsamps = spy_hdr.n_sample
            else:
                ext_hdr.nsamps = spy_hdr.n_sample
            ext_hdr.dt = spy_hdr.d_sample * 1_000_000.0 # in micro seconds

            # in milliseconds
            ms_t0 = spy_hdr.sample_start * 1_000.0

            # shift to store as many significant digits as possible
            if ms_t0 != 0.0:
                log_scale = log10(fabs(ms_t0)/INT16_MAX)
                if log_scale > 0:
                    log_scale = min(ceil(log_scale), 4)
                    hdr.tm_scal = <i2> (10**log_scale)
                    hdr.delay = <i2> (ms_t0 / hdr.tm_scal)
                else:
                    log_scale = min(floor(-log_scale), 4)
                    hdr.tm_scal = <i2> (10**log_scale)
                    hdr.delay = <i2> (ms_t0 * hdr.tm_scal)
                    hdr.tm_scal *= -1
            else:
                hdr.delay = 0
                hdr.tm_scal = 0

            ext_hdr.offset = spy_hdr.offset

            ext_hdr.sht_x = spy_hdr.tx_loc[0]
            ext_hdr.sht_y = spy_hdr.tx_loc[1]
            ext_hdr.selev = spy_hdr.tx_loc[2]

            ext_hdr.rec_x = spy_hdr.rx_loc[0]
            ext_hdr.rec_y = spy_hdr.rx_loc[1]
            ext_hdr.relev = spy_hdr.rx_loc[2]

            ext_hdr.cdp_x = spy_hdr.mid_point[0]
            ext_hdr.cdp_y = spy_hdr.mid_point[1]

            ext_hdr.n_exttrchdr = 1

            # hdr.trctype = spy_hdr.line_id
            hdr.chan = spy_hdr.line_id
            ext_hdr.linetrc = spy_hdr.trace_id
            hdr.coorunit = 1
            ext_hdr.cdp = spy_hdr.ensemble_number
            hdr.cdptrc = spy_hdr.ensemble_trace_number + 1
            hdr.header_name = trc_name
            ext_hdr.header_name = ext_name

    cdef spyc.Trace to_spy_trace(self):
        cdef:
            spyc.Trace spy_tr = spyc.Trace.__new__(spyc.Trace)
            spyc.spy_trace_header *spy_hdr = spyc.new_hdr()

        spy_tr.hdr = spy_hdr

        spy_tr.hdr_owner = True

        spy_tr.data = self.data
        spy_hdr.n_sample = self.data.shape[0]

        with nogil:
            if self.hdr.tm_scal > 0:
                spy_hdr.sample_start = (<double> self.hdr.delay) * self.hdr.tm_scal
            elif self.hdr.tm_scal < 0:
                spy_hdr.sample_start = (<double> self.hdr.delay) / -self.hdr.tm_scal
            spy_hdr.sample_start /= 1_000.0

            if self.n_ext_hdr:
                spy_hdr.d_sample = self.ext_hdr.dt / 1_000_000.0
                spy_hdr.offset = self.ext_hdr.offset

                spy_hdr.tx_loc[0] = self.ext_hdr.sht_x
                spy_hdr.tx_loc[1] = self.ext_hdr.sht_y
                spy_hdr.tx_loc[2] = self.ext_hdr.selev

                spy_hdr.rx_loc[0] = self.ext_hdr.rec_x
                spy_hdr.rx_loc[1] = self.ext_hdr.rec_y
                spy_hdr.rx_loc[2] = self.ext_hdr.relev

                spy_hdr.mid_point[0] = self.ext_hdr.cdp_x
                spy_hdr.mid_point[1] = self.ext_hdr.cdp_y

                spy_hdr.trace_id = self.ext_hdr.linetrc
                spy_hdr.ensemble_number = self.ext_hdr.cdp
            else:
                spy_hdr.d_sample = self.hdr.dt / 1_000_000.0
                spy_hdr.offset = self.hdr.offset

                spy_hdr.tx_loc[0] = self.hdr.sht_x
                spy_hdr.tx_loc[1] = self.hdr.sht_y
                spy_hdr.tx_loc[2] = self.hdr.selev

                spy_hdr.rx_loc[0] = self.hdr.rec_x
                spy_hdr.rx_loc[1] = self.hdr.rec_y
                spy_hdr.rx_loc[2] = self.hdr.relev

                spy_hdr.mid_point[0] = self.hdr.cdp_x
                spy_hdr.mid_point[1] = self.hdr.cdp_y

                spy_hdr.trace_id = self.hdr.linetrc
                spy_hdr.ensemble_number = self.hdr.cdp

            spy_hdr.mid_point[2] = 0.5 * (spy_hdr.tx_loc[2] + spy_hdr.rx_loc[2])

            # hdr.trctype = spy_hdr.line_id
            spy_hdr.line_id = self.hdr.chan
            spy_hdr.ensemble_trace_number = self.hdr.cdptrc - 1

        return spy_tr

    cdef to_file_descriptor(self, FILE *fd):
        cdef size_t n_write
        n_write = spy_io.write_struct_to_file(&self.hdr, _th_info, STDH_IS_PACKED, TRC_HDR_SIZE, fd)
        if n_write != TRC_HDR_SIZE:
            raise IOError("Error writing trace header to file.")

        n_write = spy_io.write_struct_to_file(&self.ext_hdr, _eh_info, EXTH_IS_PACKED, TRC_HDR_SIZE, fd)
        if n_write != TRC_HDR_SIZE:
            raise IOError("Error writing trace header to file.")

        n_write = fwrite(&self.data[0], sizeof(float), self.data.shape[0], fd)
        if n_write != self.data.shape[0]:
            raise IOError("Error writing trace data to file.")

    @staticmethod
    cdef SEGYTrace from_file_descriptor(FILE *fd, str file_endian_flag, int max_ext_header, int data_type):
        # file endian is 0, 1, or 2, for big, little, pairwise.
        cdef:
            SEGYTrace tr = SEGYTrace.__new__(SEGYTrace)
            size_t n_read
            int n_ext_headers
        n_read = spy_io.read_struct_from_file(&tr.hdr, _th_info, STDH_IS_PACKED, TRC_HDR_BYTES, fd, file_endian_flag)
        if n_read != TRC_HDR_BYTES:
            if feof(fd):
                raise EOFError("Reached the end of the file while reading header.")
            else:
                raise EOFError("Error reading trace header from file.")

        cdef size_t nsamps = tr.hdr.nsamps
        if max_ext_header>0:
            n_read = spy_io.read_struct_from_file(&tr.hdr, _eh_info, EXTH_IS_PACKED, TRC_HDR_BYTES, fd, file_endian_flag)
            if n_read != TRC_HDR_BYTES:
                if feof(fd):
                    raise EOFError("Reached the end of the file while reading extended header.")
                else:
                    raise IOError("Error reading extended trace header from file.")

            n_ext_headers = tr.ext_hdr.n_exttrchdr
            if n_ext_headers == 0:
                n_ext_headers = max_ext_header

            if tr.ext_hdr.nsamps > 0:
                nsamps = tr.ext_hdr.nsamps

            # seek through any extra headers (for now)
            # should likely just stash these bytes on the spyc.Trace object
            # for passing through to a segy output.
            spy_io.spy_fseek(fd, TRC_HDR_SIZE * (n_ext_headers - 1), SEEK_CUR)

        cdef size_t data_item_size = data_format_sizes[data_type]

        cdef size_t n_data_bytes = data_item_size * nsamps

        cdef uint8_t[::1] data_bytes

        if data_item_size == 3:
            data_bytes = <uint8_t[:4]> malloc(nsamps * 4)
        else:
            data_bytes = <uint8_t[:n_data_bytes]> malloc(n_data_bytes)

        #tr.data = <float[:nsamps]> malloc(sizeof(float) * nsamps)
        n_read = fread(&data_bytes[0], 1, n_data_bytes, fd)
        if n_read != n_data_bytes:
            if feof(fd):
                raise EOFError("Reached the end of the file while reading data.")
            else:
                raise IOError("Error reading trace data from file.")
        if data_item_size == 3:
            if file_endian_flag == "<>":
                raise NotImplementedError("Reading pairwise byteswapped 3 byte values is not a defined behavoir.")
            unpack3bytes_to_4(data_bytes, file_endian_flag, data_type==DataFormat.int24)
            # now it's the corresponding 4 byte size.
            data_item_size = 4
            data_type = DataFormat.int32 if data_type == DataFormat.int24 else DataFormat.uint32
        if data_item_size == 1:
            pass
        elif file_endian_flag == ">":
            if data_item_size == 2:
                bswap.swap16_big_and_system(<uint16_t*> &data_bytes[0], nsamps)
            elif data_item_size == 4:
                bswap.swap32_big_and_system(<uint32_t*> &data_bytes[0], nsamps)
            elif data_item_size == 8:
                bswap.swap64_big_and_system(<uint64_t*> &data_bytes[0], nsamps)
            else:
                raise NotImplementedError
        elif file_endian_flag == "<":
            if data_item_size == 2:
                bswap.swap16_little_and_system(<uint16_t*> &data_bytes[0], nsamps)
            elif data_item_size == 4:
                bswap.swap32_little_and_system(<uint32_t*> &data_bytes[0], nsamps)
            elif data_item_size == 8:
                bswap.swap64_little_and_system(<uint64_t*> &data_bytes[0], nsamps)
            else:
                raise NotImplementedError
        elif file_endian_flag == "<>":
            if data_item_size == 2:
                bswap.swap16_pairwise_and_system(<uint16_t*> &data_bytes[0], nsamps)
            elif data_item_size == 4:
                bswap.swap32_pairwise_and_system(<uint32_t*> &data_bytes[0], nsamps)
            elif data_item_size == 8:
                bswap.swap64_pairwise_and_system(<uint64_t*> &data_bytes[0], nsamps)
            else:
                raise NotImplementedError

        if data_item_size == sizeof(float):
            # we already allocated the memory we need, so just transform it in place
            if data_type == DataFormat.float32_ibm:
                ibm_to_float(data_bytes)
            elif data_type == DataFormat.int32:
                convert4_to_float(<i4 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.uint32:
                convert4_to_float(<ui4 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.float32_ieee:
                pass # nothing needs to be done...
            tr.data = <float[:nsamps]> <float *> &data_bytes[0]
        else:
            # need to create a new array and copy it into it,
            # as the bytes array does not have the same space
            # as an array of floats...
            if data_type == DataFormat.int16:
                tr.data = <float[:nsamps]> convert_to_float(<i2 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.float64_ieee:
                tr.data = <float[:nsamps]> convert_to_float(<f8 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.int08:
                tr.data = <float[:nsamps]> convert_to_float(<i1 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.int64:
                tr.data = <float[:nsamps]> convert_to_float(<i8 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.uint16:
                tr.data = <float[:nsamps]> convert_to_float(<ui2 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.uint64:
                tr.data = <float[:nsamps]> convert_to_float(<ui8 *> &data_bytes[0], nsamps)
            elif data_type == DataFormat.uint08:
                tr.data = <float[:nsamps]> convert_to_float(<ui1 *> &data_bytes[0], nsamps)
        return tr

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
    cdef SEGYCollection from_header_bytes(const unsigned char[::1] bys, endian=None, data_format=None):
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
                if coll.hdr.data_format > 1 and coll.hdr.data_format <= 16:
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
                getattr(DataFormat, data_format)
            elif coll.hdr.data_format == 0:
                coll.hdr.data_format = DataFormat.float32_ieee

        return coll

    @classmethod
    def from_file(cls, filename, file_endian=None, data_format=None):
        cdef:
            SEGYTrace trace
            SEGYCollection coll

        if hasattr(filename, 'read'):
            ctx = nullcontext(filename)
        else:
            filename = os.fspath(filename)
            ctx = open(filename, "rb")

        with ctx as f:
            text_header_bytes = f.read(TXT_HDR_BYTES)
            coll = SEGYCollection.from_header_bytes(f.read(BIN_HDR_BYTES))
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

    return text.decode('ascii')


cdef class _FileSEGYIterator(spyc.BaseTraceIterator):
    cdef:
        FILE *fd
        bint owner
        object file
        spy_io.spy_off_t orig_pos
        binary_header bhdr
        str endian_flag

    def __cinit__(self):
        self.fd = NULL
        self.owner = False
        self.file = None
        self.hdr.n_traces = 0

    def __dealoc__(self):
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
        self.bhdr = coll.hdr
        if coll.hdr.byte_order_id == SEGY_BIG_ENDIAN_FLAG:
            self.endian_flag = ">"
        elif coll.hdr.byte_order_id == SEGY_LITTLE_ENDIAN_FLAG:
            self.endian_flag = "<"
        elif coll.hdr.byte_order_id == SEGY_PAIRWISE_ENDIAN_FLAG:
            self.endian_flag = "<>"
        else:
            raise ValueError("Unknown File Endian.")

        self.hdr.n_traces = self.bhdr.n_traces

        cdef spyc.spy_off_t trace_start

        try:
            self.fd, self.orig_pos = spy_io.PyFile_Dup(file, "rb")
            if self.owner:
                if coll.hdr.major_rev >= 2 and coll.hdr.first_trace_byte_offset != 0:
                    trace_start = coll.hdr.first_trace_byte_offset
                else:
                    trace_start = (coll.hdr.next_txt_hdr + 1) * TXT_HDR_BYTES + BIN_HDR_BYTES
                # Advance fd to the start of the traces since I opened it:
                spy_io.spy_fseek(self.fd, trace_start, SEEK_CUR)
        except Exception as err:
            self._close_file()
            raise err

    cdef spyc.Trace next_trace(self):
        if self.i == SIZE_MAX or (self.hdr.n_traces > 0 and self.i == self.hdr.n_traces):
            raise StopIteration()
        cdef:
            SEGYTrace trace_in
        print(self.i)
        try:
            trace_in = SEGYTrace.from_file_descriptor(self.fd, self.endian_flag, self.bhdr.nmax_ext_trc_hdr, self.bhdr.data_format)
        except EOFError as err:
            print("Reached End of File:", print(err))
            self.hdr.n_traces = self.i
            self._close_file()
            raise StopIteration()
        except Exception as err:
            # if something else goes wrong reading in from the file descriptor
            # close myself and re-raise the error.
            self._close_file()
            print("I errored reading from the file", err)
            raise err
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