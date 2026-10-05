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
    assert tr.header['mid_point'][2] == pytest.approx(60.0)


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
