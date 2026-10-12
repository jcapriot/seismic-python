import io

import numpy as np
import pytest

from seispy.container import Trace
from seispy.io import read_su, write_su, SUTraceWriter
from seispy.io import su as su_module


def make_traces(n=4, ns=32):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        tr = Trace(rng.standard_normal(ns).astype(np.float32), d_sample=0.004, sample_start=0.1).replace(
            ensemble_number=10 + i // 2,
            ensemble_trace_number=1 + i % 2,
            trace_type=1,
            tx_loc=[100.0 * i, 5.0, 2.0],
            rx_loc=[100.0 * i + 50.0 * (1 + i % 2), 5.0, 3.0],
            fold=3,
            extra={"fldr": 100 + i, "tracl": i + 1, "muts": 12},
        )
        out.append(tr)
    return out


def test_header_is_240_bytes():
    assert su_module._native_dtype().itemsize == 240


def test_round_trip_through_a_stream():
    traces = make_traces()
    buf = io.BytesIO()
    assert write_su(traces, buf) == 4
    assert len(buf.getvalue()) == 4 * (240 + 4 * 32)

    back = list(read_su(io.BytesIO(buf.getvalue())))
    assert len(back) == 4
    for a, b in zip(traces, back):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert b.header["d_sample"] == pytest.approx(0.004)
        assert b.header["sample_start"] == pytest.approx(0.1)
        assert b.header["ensemble_number"] == a.header["ensemble_number"]
        assert b.header["ensemble_trace_number"] == a.header["ensemble_trace_number"]
        assert b.header["fold"] == 3
        assert b.header["offset"] == pytest.approx(a.header["offset"])
        np.testing.assert_allclose(b.header["tx_loc"], a.header["tx_loc"])
        np.testing.assert_allclose(b.header["rx_loc"], a.header["rx_loc"])
        assert b.extra == {k: v for k, v in a.extra.items()}

    # and a second pass is byte identical
    buf2 = io.BytesIO()
    write_su(back, buf2)
    assert buf2.getvalue() == buf.getvalue()


@pytest.mark.parametrize("endian", ["big", "little"])
def test_endian(endian):
    traces = make_traces(2)
    buf = io.BytesIO()
    write_su(traces, buf, endian=endian)
    back = list(read_su(io.BytesIO(buf.getvalue()), endian=endian))
    np.testing.assert_array_equal(np.asarray(back[1]), np.asarray(traces[1]))
    assert back[1].extra["fldr"] == 101
    # the first word, tracl, is a big/little endian int32
    assert int.from_bytes(buf.getvalue()[:4], endian) == 1


def test_stage_changes_are_written_but_extras_are_kept():
    tr = make_traces(1)[0].replace(d_sample=0.002, sample_start=0.0)
    buf = io.BytesIO()
    write_su([tr], buf)
    (back,) = read_su(io.BytesIO(buf.getvalue()))
    assert back.header["d_sample"] == pytest.approx(0.002)
    assert back.header["sample_start"] == 0.0
    assert back.extra["fldr"] == 100


def test_variable_ns_and_file(tmp_path):
    rng = np.random.default_rng(0)
    traces = [Trace(rng.standard_normal(n).astype(np.float32), d_sample=0.001) for n in (10, 20, 5)]
    path = tmp_path / "a.su"
    write_su(traces, path)
    it = read_su(path)
    back = list(it)
    assert [len(t) for t in back] == [10, 20, 5]


def test_offset_without_coordinates_round_trips():
    header = np.zeros(1, dtype=su_module._native_dtype())
    header["ns"] = 8
    header["dt"] = 4000
    header["offset"] = 400
    raw = header.tobytes() + np.arange(8, dtype=np.float32).tobytes()
    (tr,) = read_su(io.BytesIO(raw))
    assert tr.header["offset"] == 400
    out = io.BytesIO()
    write_su([tr], out)
    assert out.getvalue() == raw


def test_truncated_stream_raises():
    buf = io.BytesIO()
    write_su(make_traces(2), buf)
    data = buf.getvalue()
    with pytest.raises(EOFError):
        list(read_su(io.BytesIO(data[:-4])))
    with pytest.raises(EOFError):
        list(read_su(io.BytesIO(data[:100])))
    assert list(read_su(io.BytesIO(b""))) == []


def test_short_reads_from_a_pipe_like_stream():
    class Dribble(io.RawIOBase):
        def __init__(self, data):
            self.data = data
            self.pos = 0

        def readable(self):
            return True

        def read(self, n=-1):
            chunk = self.data[self.pos : self.pos + min(n, 7)]
            self.pos += len(chunk)
            return chunk

    buf = io.BytesIO()
    write_su(make_traces(3), buf)
    back = list(read_su(Dribble(buf.getvalue())))
    assert len(back) == 3


def test_complex_is_rejected():
    tr = Trace(np.zeros(4, dtype=np.complex64), d_sample=0.1)
    with pytest.raises(NotImplementedError):
        write_su([tr], io.BytesIO())


def test_ns_too_long_is_rejected():
    tr = Trace(np.zeros(70000, dtype=np.float32), d_sample=0.001)
    with pytest.raises(OverflowError):
        write_su([tr], io.BytesIO())
