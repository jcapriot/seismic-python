from ..config cimport (
    uint8_t as ui1,
    uint16_t as ui2,
    uint32_t as ui4,
    uint64_t as ui8,
    int16_t as i2,
    int32_t as i4,
    int64_t as i8,
    float32_t as f4,
    float64_t as f8,
)

from ._segy_enums cimport *

cdef extern from "spy_config.h":
    pass

cdef extern from *:
    """
    #define SEGY_UNKNOWN 0
    #define TXT_HDR_BYTES 3200
    #define BIN_HDR_BYTES 400
    #define TRC_HDR_BYTES 240
    #define TAP_LBL_BYTES 180

    #ifdef IS_BIG_ENDIAN

    #define SEGY_BIG_ENDIAN_FLAG 0x01020304
    #define SEGY_LITTLE_ENDIAN_FLAG 0x04030201
    #define SEGY_PAIRWISE_ENDIAN_FLAG 0x02010403

    #elif defined(IS_LITTLE_ENDIAN)

    #define SEGY_BIG_ENDIAN_FLAG 0x04030201
    #define SEGY_LITTLE_ENDIAN_FLAG 0x01020304
    #define SEGY_PAIRWISE_ENDIAN_FLAG 0x03040102

    #endif
    #define SEGY_SYSTEM_ENDIAN_FLAG 16909060
    """
    int SEGY_UNKNOWN
    int TXT_HDR_BYTES
    int BIN_HDR_BYTES
    int TRC_HDR_BYTES
    int TAP_LBL_BYTES

    int SEGY_BIG_ENDIAN_FLAG
    int SEGY_LITTLE_ENDIAN_FLAG
    int SEGY_PAIRWISE_ENDIAN_FLAG
    int SEGY_SYSTEM_ENDIAN_FLAG

# constants to be defined in segy_standard.pyx
cdef:
    # Expected Sizes
    size_t BIN_HDR_SIZE
    size_t TRC_HDR_SIZE
    size_t EXT_HDR_SIZE
    size_t SU_HDR_SIZE

cdef struct binary_header:
    i4 job_id
    i4 line_number
    i4 reel_number
    ui2 n_trace_per_ensemble
    ui2 n_aux_trace_per_ensemble
    ui2 d_sample # in \mu s, Hz, m, or ft.
    ui2 d_sample_field # in \mu s, Hz, m, or ft.
    ui2 n_sample_per_trace
    ui2 n_sample_per_trace_field
    i2 data_format
    ui2 n_fold
    i2 sort_method
    i2 vert_sum_code
    ui2 sweep_freq_start
    ui2 sweep_freq_end
    ui2 sweep_dur # in ms
    i2 sweep_type
    ui2 sweep_trace
    ui2 sweep_start_taper_dur # in ms
    ui2 sweep_end_taper_dur # in ms
    i2 taper_type
    i2 traces_are_correlated
    i2 bin_gain_is_recovered
    i2 amp_recovery_method
    i2 meters_or_feet
    i2 signal_polarity
    i2 vib_polarity_code
    # rev 2 (up to unassigned_1)
    ui4 next_trace_per_ensemble
    ui4 next_aux_trace_per_ensemble
    ui4 next_sample_per_trace
    f8 dext_sample # in \mu s, Hz, m, or ft.
    f8 dext_sample_field # in \mu s, Hz, m, or ft.
    ui4 next_sample_per_trace_field
    ui4 next_fold
    i4 byte_order_id
    char[200] unassigned_1
    # rev 1
    ui1 major_rev
    ui1 minor_rev
    ui2 is_fixed_traces
    i2 next_txt_hdr
    # rev 2 (up to unassigned_2)
    ui2 nmax_ext_trc_hdr # rev 2.1 (In rev1 this is a ui4, and there is no survey type
    i2 survey_type # rev 2.1
    i2 time_basis
    ui8 n_traces
    ui8 first_trace_byte_offset
    ui4 n_trailer_stanzas
    char[68] unassigned_2


cdef struct trace_header:
    # rev 0
    ui4 linetrc # 1
    ui4 reeltrc # 5
    i4 ffid # 9
    i4 chan # 13
    i4 epsnum # 17
    i4 cdp # 21
    i4 cdptrc # 25
    i2 trctype # 29
    ui2 vstack # 31
    ui2 fold # 33
    i2 rectype # 35
    i4 offset # 37
    i4 relev # 41
    i4 selev # 45
    i4 sdepth # 49
    i4 rdatum # 53
    i4 sdatum # 57
    ui4 wdepthso # 61
    ui4 wdepthrc # 65
    i2 ed_scal # 67
    i2 co_scal # 69
    i4 sht_x # 73
    i4 sht_y # 77
    i4 rec_x # 81
    i4 rec_y # 85
    i2 coorunit # 89
    ui2 wvel # 91
    ui2 subwvel # 93
    i2 shuphole # 95
    i2 fcuphole # 97
    i2 shstat # 99
    i2 rcstat # 101
    i2 stapply # 103
    i2 lagtimea # 105
    i2 lagtimeb # 107
    i2 delay # 109
    i2 mutestrt # 111
    i2 muteend # 113
    ui2 nsamps # 115
    ui2 dt # 117
    i2 gain # 119
    i2 ingconst # 121
    i2 initgain # 123
    i2 corrflag # 125
    ui2 sweepstrt # 127
    ui2 sweepend  # 129
    ui2 sweeplng # in ms # 131
    i2 seeptyp # 133
    ui2 sweepstp # in ms # 135
    ui2 sweepetp # in ms # 137
    i2 tapertyp # 139
    ui2 aliasfil # 141
    i2 aliasslop # 143
    ui2 notchfil # 145
    i2 notchslop # 147
    ui2 lowcut # 149
    ui2 highcut # 151
    i2 lowcslop # 153
    i2 hicslop # 155
    ui2 year # 157
    ui2 day # 159
    ui2 hour # 161
    ui2 minute # 163
    ui2 second # 165
    i2 timebase # 167
    i2 trweight # 169
    i2 rstaswp1 # 171
    i2 rstatrc1 # 173
    i2 rstatrcn # 175
    i2 gapsize # 177
    i2 overtrvl # 179
    # rev 1
    i4 cdp_x # 181
    i4 cdp_y # 185
    i4 iline # 189
    i4 xline # 193
    i4 sp # 197 ## spnum
    i2 sp_scal # 201
    i2 samp_unit # 203
    i4 trans_const_mantis # 205
    i2 trans_const_pow10 # 209
    i2 trans_unit # 211
    i2 dev_id # 213
    i2 tm_scal # to be applied to give true time in milliseconds of bytes 95-114 # 215
    i2 src_type # 217
    i2 src_dir1 # 219
    i2 src_dir2 # 221
    i2 src_dir3 # 223
    i4 smeasure_mantis # 225
    i2 smeasuret_scale # 229
    i2 src_units # 231
    char[8] header_name

cdef struct extended_trace_header:
    ui8 linetrc
    ui8 reeltrc
    i8 ffid
    i8 cdp
    f8 relev
    f8 rdepth
    f8 selev
    f8 sdepth
    f8 rdatum
    f8 sdatum
    f8 wdepthso
    f8 wdepthrc
    f8 sht_x
    f8 sht_y
    f8 rec_x
    f8 rec_y
    f8 offset
    ui4 nsamps
    i4 nanosec
    f8 dt
    i4 cable_num
    ui2 n_exttrchdr
    i2 last_trc
    f8 cdp_x
    f8 cdp_y
    char[56] reserved
    char[8] header_name

cdef struct su_trace:
    int tracl
    int tracr
    int fldr
    int tracf
    int ep
    int cdp
    int cdpt
    short trid
    short nvs
    short nhs
    short duse
    int offset
    int gelev
    int selev
    int sdepth
    int gdel
    int sdel
    int swdep
    int gwdep
    short scalel
    short scalco
    int  sx
    int  sy
    int  gx
    int  gy
    short counit
    short wevel
    short swevel
    short sut
    short gut
    short sstat
    short gstat
    short tstat
    short laga
    short lagb
    short delrt
    short muts
    short mute
    unsigned short ns
    unsigned short dt
    short gain
    short igc
    short igi
    short corr
    short sfs
    short sfe
    short slen
    short styp
    short stas
    short stae
    short tatyp
    short afilf
    short afils
    short nofilf
    short nofils
    short lcf
    short hcf
    short lcs
    short hcs
    short year
    short day
    short hour
    short minute
    short sec
    short timbas
    short trwf
    short grnors
    short grnofr
    short grnlof
    short gaps
    short otrav
    # SU Specifics
    float d1
    float f1
    float d2
    float f2
    float ungpow
    float unscale
    int ntr
    short mark
    short shortpad
    short unass[14]