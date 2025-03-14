# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from libc.math cimport floor, ceil, log10, fabs
from libc.stdint cimport (
    INT8_MAX, INT16_MAX, INT32_MAX, INT64_MAX,
    UINT8_MAX, UINT16_MAX, UINT32_MAX, UINT64_MAX,
)
from libc.stdio cimport fwrite, fread, FILE, SEEK_CUR
from libc.stdlib cimport malloc

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

cdef class SEGYTrace:
    cdef:
        trace_header hdr
        extended_trace_header ext_hdr
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

        spy_tr.header_owner = True

        spy_tr.data = self.data
        spy_hdr.n_sample = self.data.shape[0]

        with nogil:
            spy_hdr.d_sample = self.ext_hdr.dt / 1_000_000.0
            if self.hdr.tm_scal > 0:
                spy_hdr.sample_start = (<double> self.hdr.delay) * self.hdr.tm_scal
            elif self.hdr.tm_scal < 0:
                spy_hdr.sample_start = (<double> self.hdr.delay) / -self.hdr.tm_scal
            spy_hdr.sample_start /= 1_000.0


            spy_hdr.offset = self.ext_hdr.offset

            spy_hdr.tx_loc[0] = self.ext_hdr.sht_x
            spy_hdr.tx_loc[1] = self.ext_hdr.sht_y
            spy_hdr.tx_loc[2] = self.ext_hdr.selev

            spy_hdr.rx_loc[0] = self.ext_hdr.rec_x
            spy_hdr.rx_loc[1] = self.ext_hdr.rec_y
            spy_hdr.rx_loc[2] = self.ext_hdr.relev

            spy_hdr.mid_point[0] = self.ext_hdr.cdp_x
            spy_hdr.mid_point[1] = self.ext_hdr.cdp_y
            spy_hdr.mid_point[2] = 0.5 * (spy_hdr.tx_loc[2] + spy_hdr.rx_loc[2])

            # hdr.trctype = spy_hdr.line_id
            spy_hdr.line_id = self.hdr.chan
            spy_hdr.trace_id = self.ext_hdr.linetrc
            spy_hdr.ensemble_number = self.ext_hdr.cdp
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
    cdef SEGYTrace from_file_descriptor(FILE *fd, str file_endian_flag, int max_ext_header=0):
        # file endian is 0, 1, or 2, for big, little, pairwise.
        cdef:
            SEGYTrace tr = SEGYTrace.__new__(SEGYTrace)
            size_t n_read
            int n_ext_headers
        n_read = spy_io.read_struct_from_file(&tr.hdr, _th_info, STDH_IS_PACKED, TRC_HDR_SIZE, fd, file_endian_flag)
        if n_read != 1:
            raise IOError("Error reading trace header from file.")

        cdef size_t nsamps = tr.hdr.nsamps
        if max_ext_header>0:
            n_read = spy_io.read_struct_from_file(&tr.hdr, _eh_info, EXTH_IS_PACKED, TRC_HDR_SIZE, fd, file_endian_flag)
            if n_read != 1:
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

        tr.data = <float[:nsamps]> malloc(sizeof(float) * nsamps)
        n_read = fread(&tr.data[0], sizeof(float), nsamps, fd)
        if n_read != nsamps:
            raise IOError("Error reading trace data from file.")
        bswap.swap_endian_and_system_array(tr.data, file_endian_flag, inplace=True)
        return tr


_SEGY_SORT_SPY_SORT = {
    spyc.EnsembleType.unknown : TraceSorting.unknown,
    spyc.EnsembleType.unsorted : TraceSorting.none,
    spyc.EnsembleType.tx_gather : TraceSorting.source,
    spyc.EnsembleType.rx_gather : TraceSorting.receiver,
    spyc.EnsembleType.common_midpoint : TraceSorting.midpoint,
    spyc.EnsembleType.common_offset : TraceSorting.offset,
}

cdef class SEGYCollectionHeader:
    cdef:
        binary_header hdr

    @staticmethod
    cdef SEGYCollectionHeader from_spy_collection(spyc.CollectionHeader spy_hdr, spyc.spy_trace_header *spy_tr_hdr=NULL):
        cdef:
            SEGYCollectionHeader segy = SEGYCollectionHeader.__new__(SEGYCollectionHeader)
            binary_header *hdr = &segy.hdr

        hdr.meters_or_feet = 1
        hdr.data_format = DataFormat.float32_ieee
        hdr.byte_order_id = EndianKey.system
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
