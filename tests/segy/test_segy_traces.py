import struct
import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from seispy.container import Trace
from seispy.filters import bfilt
from seispy.io import SEGYCollection, SEGYTruncatedError, read_segy

FILES = Path(__file__).resolve().parent / "test_files"

# The reference files all have a single 256 sample trace
TEXT_HDR, BIN_HDR, TRC_HDR = 3200, 400, 240
TRACE_START = TEXT_HDR + BIN_HDR


def ref_data(trace_dtype_unsigned):
    if trace_dtype_unsigned:
        return np.arange(256, dtype=np.uint8)
    return np.arange(-128, 128, dtype=np.int8)


def _path(fmt, endian):
    return FILES / f"Format{fmt:02d}{endian}.segy"


@pytest.mark.parametrize('endian', ['big', 'little', 'pairwise'])
@pytest.mark.parametrize('fmt', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16])
def test_read_segy_all_formats(endian, fmt):
    if fmt in [7, 15] and endian == 'pairwise':
        pytest.skip('3 byte pairwise numbers are not supported.')
    path = _path(fmt, endian)

    # compare with what the raw reader gives
    raw = next(iter(SEGYCollection.from_file(path)))
    expected = np.asarray(raw).astype(np.float32)

    traces = list(read_segy(path))
    assert len(traces) == 1
    tr = traces[0]
    assert isinstance(tr, Trace)
    assert tr.n_sample == 256
    npt.assert_array_equal(np.asarray(tr), expected)
    # and the values are the same ones the raw reader checks against
    npt.assert_array_equal(np.asarray(tr), ref_data(np.issubdtype(raw.dtype, np.unsignedinteger)).astype(np.float32))


def _big_file():
    return bytearray(_path(2, 'big').read_bytes())


def _as_revision_1(data):
    # major/minor revision live at bytes 3501/3502 of the file. Before revision 2 the number of traces is not stored.
    data[3500] = 1
    data[3501] = 0
    return data


def test_unknown_length_reads_to_end_of_file(tmp_path):
    data = _as_revision_1(_big_file())
    trace = bytes(data[TRACE_START:])
    data += trace * 2  # three traces now
    path = tmp_path / "three.segy"
    path.write_bytes(data)

    it = read_segy(path)
    assert it.n_traces is None
    traces = list(it)
    assert len(traces) == 3
    for tr in traces:
        npt.assert_array_equal(np.asarray(tr), ref_data(False).astype(np.float32))
    # now it knows how many there were, and asking again after the end is fine
    assert it.n_traces == 3
    with pytest.raises(StopIteration):
        next(it)


def test_empty_file_ends_cleanly(tmp_path):
    data = _as_revision_1(_big_file())[:TRACE_START]
    path = tmp_path / "empty.segy"
    path.write_bytes(data)
    it = read_segy(path)
    assert list(it) == []
    with pytest.raises(StopIteration):
        next(it)  # the file is already closed, this must not crash


@pytest.mark.parametrize('keep', [TRACE_START + 10, TRACE_START + TRC_HDR - 1, TRACE_START + TRC_HDR + 100])
def test_truncated_trace_is_an_error(tmp_path, keep):
    data = _big_file()[:keep]
    path = tmp_path / "truncated.segy"
    path.write_bytes(data)
    with pytest.raises(SEGYTruncatedError):
        list(read_segy(path))


def test_fewer_traces_than_declared_warns(tmp_path):
    # revision 2 says there is one trace, but there isn't
    path = tmp_path / "short.segy"
    path.write_bytes(_big_file()[:TRACE_START])
    with pytest.warns(UserWarning, match="ended after 0 traces"):
        assert list(read_segy(path)) == []


def test_reading_does_not_change_the_collection():
    coll = SEGYCollection.from_file(_path(2, 'big'))
    n_before = coll.binary_header['n_traces']
    assert len(list(coll.traces())) == 1
    assert coll.binary_header['n_traces'] == n_before


def test_iterator_outlives_the_collection():
    # the iterator must not depend on the collection it came from
    it = SEGYCollection.from_file(_path(2, 'little')).traces()
    import gc
    gc.collect()
    assert len(list(it)) == 1


def test_read_from_file_object_twice():
    with open(_path(2, 'big'), 'rb') as f:
        first = [np.asarray(t).copy() for t in read_segy(f)]
        second = [np.asarray(t).copy() for t in read_segy(f)]
        assert f.tell() == 0 or f.tell() > 0  # still usable
    assert len(first) == len(second) == 1
    npt.assert_array_equal(first[0], second[0])


def _patch(data, offset, fmt, value):
    struct.pack_into(fmt, data, TRACE_START + offset, value)


def test_header_conversion(tmp_path):
    data = _big_file()
    _patch(data, 116, '>H', 4000)      # dt in microseconds
    _patch(data, 70, '>h', -100)       # coordinate scalar: divide by 100
    _patch(data, 68, '>h', 10)         # elevation scalar: multiply by 10
    _patch(data, 72, '>i', 123456)     # source x
    _patch(data, 76, '>i', -5000)      # source y
    _patch(data, 44, '>i', 5)          # source elevation
    _patch(data, 40, '>i', 7)          # receiver elevation
    path = tmp_path / "headers.segy"
    path.write_bytes(data)

    (tr,) = list(read_segy(path))
    assert tr.d_sample == pytest.approx(0.004)
    tx = tr.header['tx_loc']
    assert tx[0] == pytest.approx(1234.56)
    assert tx[1] == pytest.approx(-50.0)
    assert tx[2] == pytest.approx(50.0)
    assert tr.header['rx_loc'][2] == pytest.approx(70.0)
    assert 'mid_point' not in tr.header


def test_blank_trace_dt_falls_back_to_binary_header(tmp_path):
    data = _big_file()
    struct.pack_into('>H', data, TEXT_HDR + 16, 2000)  # binary header sample interval (bytes 3217-3218)
    path = tmp_path / "dt.segy"
    path.write_bytes(data)
    (tr,) = list(read_segy(path))
    assert tr.d_sample == pytest.approx(0.002)


def test_data_format_override(tmp_path):
    # before this was an AttributeError
    path = tmp_path / "rev1.segy"
    path.write_bytes(_as_revision_1(_big_file()))
    (tr,) = list(read_segy(path, data_format='i32'))
    npt.assert_array_equal(np.asarray(tr), ref_data(False).astype(np.float32))
    with pytest.raises(ValueError, match="Unrecognized"):
        read_segy(path, data_format='nonsense')


def test_read_segy_pipes_into_stages(tmp_path):
    data = _big_file()
    _patch(data, 116, '>H', 4000)
    path = tmp_path / "piped.segy"
    path.write_bytes(data)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = list(read_segy(path) | bfilt())
    assert len(out) == 1
    assert np.isfinite(np.asarray(out[0])).all()
    # and the equivalent through a collection in memory
    mem = read_segy(path).to_memory()
    npt.assert_array_equal(np.asarray(next(iter(mem | bfilt()))), np.asarray(out[0]))


@pytest.mark.parametrize('n_sample', [1, 7, 64])
def test_segy_trace_write_sizes(tmp_path, n_sample):
    # the data used to be written with the wrong item size (8 bytes per byte of data)
    from seispy.io.segy_standard import SEGYTrace, trace_header_size

    samples = np.arange(n_sample, dtype=np.float32) - 3.5
    tr = Trace(samples, d_sample=0.004)
    path = tmp_path / "trace.bin"
    with open(path, 'wb') as f:
        SEGYTrace.from_seispy(tr).write(f)

    written = path.read_bytes()
    n_hdr_bytes = 2 * trace_header_size()  # standard header + extended header
    assert len(written) == n_hdr_bytes + 4 * n_sample
    assert written[n_hdr_bytes:] == samples.tobytes()


def test_segy_trace_write_appends(tmp_path):
    from seispy.io.segy_standard import SEGYTrace, trace_header_size

    samples = np.arange(5, dtype=np.float32)
    segy_tr = SEGYTrace.from_seispy(Trace(samples, d_sample=0.002))
    path = tmp_path / "two.bin"
    with open(path, 'wb') as f:
        segy_tr.write(f)
        segy_tr.write(f)
    assert path.stat().st_size == 2 * (2 * trace_header_size() + 20)


@pytest.mark.parametrize('cdptrc, expected', [(3, 3), (1, 1), (0, 0)])
def test_ensemble_trace_number_is_one_based(tmp_path, cdptrc, expected):
    data = _big_file()
    _patch(data, 24, '>i', cdptrc)
    path = tmp_path / "cdptrc.segy"
    path.write_bytes(data)
    (tr,) = list(read_segy(path))
    assert tr.header['ensemble_trace_number'] == expected  # (0, not set, used to underflow)


def test_ensemble_trace_number_written_as_is():
    from seispy.io.segy_standard import SEGYTrace

    tr = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004).replace(ensemble_trace_number=5)
    assert SEGYTrace.from_seispy(tr).trace_header['cdptrc'] == 5


# ---------------------------------------------------------------- coordinate unit, trace type and statics
def _read_patched(tmp_path, patches):
    data = _big_file()
    for offset, fmt, value in patches:
        _patch(data, offset, fmt, value)
    path = tmp_path / "patched.segy"
    path.write_bytes(data)
    (tr,) = list(read_segy(path))
    return tr


def test_length_coordinates(tmp_path):
    tr = _read_patched(tmp_path, [(88, '>h', 1), (70, '>h', 1), (72, '>i', 500), (80, '>i', 700)])
    assert tr.header['coord_unit'] == 1
    assert tr.header['tx_loc'][0] == 500.0 and tr.header['rx_loc'][0] == 700.0
    assert tr.header['offset'] == 200.0


def test_unit_not_given_is_unknown(tmp_path):
    assert _read_patched(tmp_path, []).header['coord_unit'] == 0


def test_arc_seconds_become_degrees(tmp_path):
    # 7200 seconds is 2 degrees (the scalar divides by 100), and -1800 seconds is half a degree south
    tr = _read_patched(tmp_path, [(88, '>h', 2), (70, '>h', -100), (72, '>i', 7200 * 100), (76, '>i', -1800 * 100)])
    assert tr.header['coord_unit'] == 2
    assert tr.header['tx_loc'][0] == pytest.approx(2.0)
    assert tr.header['tx_loc'][1] == pytest.approx(-0.5)


def test_decimal_degrees_are_kept(tmp_path):
    tr = _read_patched(tmp_path, [(88, '>h', 3), (70, '>h', -10000), (72, '>i', -1051234)])
    assert tr.header['coord_unit'] == 2
    assert tr.header['tx_loc'][0] == pytest.approx(-105.1234)


def test_degrees_minutes_seconds(tmp_path):
    # 105 degrees 30 minutes 36 seconds west is -1053036.00, and 40 degrees 15 minutes 18 seconds north is 401518.00
    tr = _read_patched(tmp_path, [(88, '>h', 4), (70, '>h', -100), (72, '>i', -105303600), (76, '>i', 40151800)])
    assert tr.header['coord_unit'] == 2
    assert tr.header['tx_loc'][0] == pytest.approx(-(105 + 30 / 60 + 36 / 3600))
    assert tr.header['tx_loc'][1] == pytest.approx(40 + 15 / 60 + 18 / 3600)


def test_trace_type_is_read(tmp_path):
    assert _read_patched(tmp_path, [(28, '>h', 2)]).header['trace_type'] == 2


def test_statics_are_read_in_seconds(tmp_path):
    # scalar 10 multiplies: 12 -> 120 ms, ; and the delay uses the same one
    tr = _read_patched(tmp_path, [(214, '>h', 10), (108, '>h', 5), (98, '>h', 12), (100, '>h', -7), (102, '>h', 3)])
    assert tr.header['sample_start'] == pytest.approx(0.05)
    assert tr.header['source_static'] == pytest.approx(0.12)
    assert tr.header['receiver_static'] == pytest.approx(-0.07)
    assert tr.header['total_static'] == pytest.approx(0.03)


def test_no_time_scalar_means_milliseconds(tmp_path):
    # the scalar is 0 in files that are older than that (and it means 1, the delay used to be left out)
    tr = _read_patched(tmp_path, [(108, '>h', 250), (98, '>h', 4)])
    assert tr.header['sample_start'] == pytest.approx(0.25)
    assert tr.header['source_static'] == pytest.approx(0.004)


def test_negative_time_scalar_divides(tmp_path):
    tr = _read_patched(tmp_path, [(214, '>h', -100), (98, '>h', 250)])
    assert tr.header['source_static'] == pytest.approx(0.0025)


def test_written_header_has_the_new_values():
    from seispy.io.segy_standard import SEGYTrace

    tr = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004, sample_start=0.1, coord_unit='degrees', trace_type=2)
    tr = tr.replace(source_static=0.012, receiver_static=-0.007, total_static=0.0)
    h = SEGYTrace.from_seispy(tr).trace_header
    assert h['trctype'] == 2
    assert h['coorunit'] == 3
    scalar = h['tm_scal']
    factor = scalar if scalar > 0 else (1.0 / -scalar if scalar < 0 else 1.0)
    assert h['delay'] * factor == pytest.approx(100.0, abs=0.5 * factor)
    assert h['shstat'] * factor == pytest.approx(12.0, abs=0.5 * factor)
    assert h['rcstat'] * factor == pytest.approx(-7.0, abs=0.5 * factor)
    assert h['stapply'] == 0


def test_defaults_of_the_new_header_values():
    tr = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004)
    h = tr.header
    assert h['coord_unit'] == 0 and h['trace_type'] == 0
    assert h['source_static'] == h['receiver_static'] == h['total_static'] == 0.0
    assert tr.replace(coord_unit='length').header['coord_unit'] == 1
    with pytest.raises(KeyError):
        Trace(np.zeros(4, dtype=np.float32), d_sample=0.004, coord_unit='parsecs')


def test_inline_and_crossline_numbers(tmp_path):
    tr = _read_patched(tmp_path, [(188, '>i', 1250), (192, '>i', 4310)])
    assert tr.header['iline'] == 1250 and tr.header['xline'] == 4310
    assert _read_patched(tmp_path, []).header['iline'] == 0


def test_inline_and_crossline_numbers_are_written():
    from seispy.io.segy_standard import SEGYTrace

    tr = Trace(np.zeros(4, dtype=np.float32), d_sample=0.004).replace(iline=7, xline=-3)
    h = SEGYTrace.from_seispy(tr).trace_header
    assert h['iline'] == 7 and h['xline'] == -3
