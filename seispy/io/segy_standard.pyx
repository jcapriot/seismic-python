# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from libc.math cimport floor, ceil, log10, fabs
from libc.string cimport memset
from libc.limits cimport INT_MIN, INT_MAX, SHRT_MAX
from .. cimport container as spyc
from ._io cimport struct_dtype_info


import numpy as np

cdef:
    # Expected Sizes
    size_t TXT_HDR_SIZE = 3200
    size_t BIN_HDR_SIZE = 400
    size_t TRC_HDR_SIZE = 240
    size_t TAP_LBL_SIZE = 180

    i2 UNKNOWN = 0

    i4 BIG_ENDIAN = 0x01020304
    i4 LITTLE_ENDIAN = 0x04030201
    i4 PAIRWISE_BYTESWAP = 0x02010403

    # data_format codes
    i2 DAT_F32_IBM = 1
    i2 DAT_I32 = 2
    i2 DAT_I16 = 3
    i2 DAT_F32_FGN = 4
    i2 DAT_F32_I3E = 5
    i2 DAT_F64_I3E = 6
    i2 DAT_I24 = 7
    i2 DAT_I08 = 8
    i2 DAT_I64 = 9
    i2 DAT_U32 = 10
    i2 DAT_U16 = 11
    i2 DAT_U64 = 12
    i2 DAT_U24 = 15
    i2 DAT_U08 = 16

    # trace sorting codes
    i2 SORT_NONE = 1
    i2 SORT_CMN_DP_PT = 2
    i2 SORT_SINGLE = 3
    i2 SORT_HORIZ_STK = 4
    # rev 1
    i2 SORT_OTHER = -1
    i2 SORT_CMN_SRC_PT = 5
    i2 SORT_CMN_RX_PT = 6
    i2 SORT_CMN_OFF_PT = 7
    i2 SORT_CMN_MD_PT = 8
    i2 SORT_CMN_CNV_PT = 9

    # Sweep type code
    ui2 SWP_LINEAR = 1
    ui2 SWP_PARABOLIC = 2
    ui2 SWP_EXP = 3
    ui2 SWP_OTHER = 4
    ui2 TPR_LINEAR = 1
    ui2 TPR_COS = 2
    ui2 TPR_OTHER = 3

    i2 CORR_DATA_YES = 2
    i2 CORR_DATA_NO = 1

    i2 BIN_GAIN_YES = 1
    i2 BIN_GAIN_NO = 2

    i2 AMP_REC_NONE = 1
    i2 AMP_REC_SPH = 2
    i2 AMP_REC_AGC = 3
    i2 AMP_REC_OTHER = 4

    i2 MEASURE_METERS = 1
    i2 MEASURE_FEET = 2

    i2 POLARITY_UP_NEG = 1
    i2 POLARITY_UP_POS = 2

    i2 VIB_POL_338_023 = 1
    i2 VIB_POL_023_068 = 2
    i2 VIB_POL_067_113 = 3
    i2 VIB_POL_113_158 = 4
    i2 VIB_POL_158_203 = 5
    i2 VIB_POL_203_248 = 6
    i2 VIB_POL_248_293 = 7
    i2 VIB_POL_293_338 = 8

    #Time codes, rev 2
    i2 TIME_LOCAL = 1
    i2 TIME_GMT = 2
    i2 TIME_OTHER = 3
    i2 TIME_UTC = 4
    i2 TIME_GPS = 5

    # Survey Types, rev 2.1
    i2 SRV_LAND = 1
    i2 SRV_MARINE = 2
    i2 SRV_TRANS = 3
    i2 SRV_DWN_HOLE = 4
    i2 SRV_1D= 8
    i2 SRV_2D = 16
    i2 SRV_3D = 24
    i2 SRV_TIME_LAPSE = 32
    i2 SRV_PARALLEL_LINES = 128
    i2 SRV_CRS_SPREAD = 256
    i2 SRV_PATCHES = 684
    i2 SRV_TWD_STRMR = 1024
    i2 SRV_OBS = 1152
    i2 SRV_RAND = 1280

    i2 TRC_ID_OTHER = -1
    i2 TRC_ID_TIME_SEISMIC = 1
    i2 TRC_ID_DEAD = 2
    i2 TRC_ID_DUMMY = 3
    i2 TRC_ID_TIMEBREAK = 4
    i2 TRC_ID_UPHOLE = 5
    i2 TRC_ID_SWEEP = 6
    i2 TRC_ID_TIMING = 7
    i2 TRC_ID_WATERBRK = 8
    i2 TRC_ID_NEARGUNSIG = 9
    i2 TRC_ID_FARGUNSIG = 10
    i2 TRC_ID_PRESSURE = 11
    i2 TRC_ID_VERT = 12
    i2 TRC_ID_CROSS = 13
    i2 TRC_ID_INLINE = 14
    i2 TRC_ID_ROT_VERT = 15
    i2 TRC_ID_ROT_TRANS = 16
    i2 TRC_ID_ROT_RADIAL = 17
    i2 TRC_ID_VIBEMASS = 18
    i2 TRC_ID_VIBEBASE = 19
    i2 TRC_ID_VIBEGFORCE = 20
    i2 TRC_ID_VIBEREF = 21
    i2 TRC_ID_TV_PAIR = 22
    i2 TRC_ID_TD_PAIR = 23
    i2 TRC_ID_DV_PAIR = 24
    i2 TRC_ID_DEPTH_DOMAIN = 25
    i2 TRC_ID_GRAV_POT = 26
    i2 TRC_ID_EFIELD_VERT = 27
    i2 TRC_ID_EFIELD_CROSS = 28
    i2 TRC_ID_EFIELD_INLINE = 29
    i2 TRC_ID_ROT_EFIELD_VERT = 30
    i2 TRC_ID_ROT_EFIELD_TRANS = 31
    i2 TRC_ID_ROT_EFIELD_RAD = 32
    i2 TRC_ID_BFIELD_VERT = 33
    i2 TRC_ID_BFIELD_CROSS = 34
    i2 TRC_ID_BFIELD_INLINE = 35
    i2 TRC_ID_ROT_BFIELD_VERT = 36
    i2 TRC_ID_ROT_BFIELD_TRANS = 37
    i2 TRC_ID_ROT_BFIELD_RAD = 38
    i2 TRC_ID_PITCH = 39
    i2 TRC_ID_YAW = 40
    i2 TRC_ID_ROLL = 41



# define a numpy dtypes consistent with the header structs
cdef:
    binary_header tmp_bin_hdr
    size_t[:,::1] _bh_info
binary_header_dtype = np.asarray(<binary_header[:1]> &tmp_bin_hdr).dtype
_bh_info  = struct_dtype_info(binary_header_dtype)

cdef:
    trace_header tmp_trc_hdr
    size_t[:,::1] _th_info
trace_header_dtype = np.asarray(<trace_header[:1]> &tmp_trc_hdr).dtype
_th_info  = struct_dtype_info(trace_header_dtype)

cdef:
    extended_trace_header tmp_ext_hdr
    size_t[:,::1] _eh_info
extended_trace_header_dtype = np.asarray(<extended_trace_header[:1]> &tmp_ext_hdr).dtype
_eh_info  = struct_dtype_info(extended_trace_header_dtype)


cdef:
    su_trace tmp_su_hdr
    size_t[:,::1] _su_info
su_trace_header_dtype = np.asarray(<su_trace[:1]> &tmp_su_hdr).dtype
_su_info  = struct_dtype_info(su_trace_header_dtype)

cdef:
    unocal_trace tmp_unocal_hdr
    size_t[:,::1] _unc_info
unocal_trace_header_dtype = np.asarray(<unocal_trace[:1]> &tmp_unocal_hdr).dtype
_unc_info  = struct_dtype_info(unocal_trace_header_dtype)

def trace_label_size():
    return TAP_LBL_SIZE

def text_header_size():
    return TXT_HDR_SIZE

def binary_header_size():
    return BIN_HDR_SIZE

def trace_header_size():
    return TRC_HDR_SIZE

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
    cdef SEGYTrace from_spy_trace(spyc.Trace spy_tr, spyc.CollectionHeader coll_hdr):
        cdef:
            SEGYTrace segy = SEGYTrace.__new__(SEGYTrace)
            spyc.spy_trace_header *spy_hdr = &spy_tr.tr.hdr
            trace_header *hdr = &segy.hdr
            extended_trace_header *ext_hdr = &segy.ext_hdr
            double t0_mantissa
            int t0_exp
            i2 t_scale = 0
        with nogil:
            segy.data = spy_tr.trace_data

            ext_hdr.nsamps = spy_hdr.n_sample
            ext_hdr.dt = spy_hdr.d_sample * 1_000_000.0 # in micro seconds

            # in milliseconds
            ms_t0 = spy_hdr.sample_start * 1_000.0

            # shift to store as many significant digits as possible
            if ms_t0 != 0.0:
                log_scale = log10(fabs(ms_t0)/SHRT_MAX)
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
            ext_hdr.ffid = spy_hdr.trace_id
            hdr.coorunit = 1
            hdr.cdp = spy_hdr.ensemble_trace_number


cdef class TraceCollectionBinaryHeader:
    cdef:
        binary_header hdr


    def __cinit__(self):
        cdef binary_header *hdr = &self.hdr
        # initialize everything to 0
        memset(hdr, 0, BIN_HDR_SIZE)

    cpdef binary_header get_c_hdr(self):
        return self.hdr