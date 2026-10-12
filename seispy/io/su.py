"""Reading and writing Seismic Unix (SU) trace data.

An SU file or stream is nothing but traces one after another: a 240 byte header (SU's ``segy`` struct, see
``segy.h``) followed by ``ns`` 32 bit floats. There is no file header, ``ns`` may differ between traces, and the
byte order is that of the machine that wrote it (or big endian, if SU was built with ``-DSUXDR``).

An SU header has about 90 words and a seispy ``Trace`` has about 20 header values, so words are handled in three ways:

* **mapped** words are converted to and from the header values of the trace (units, scalars and all) and are
  authoritative: a stage that changes ``d_sample`` changes ``dt`` in what is written. They are
  ``ns``, ``dt``, ``delrt``, ``trid``, ``tracl`` (the ``trace_id``, which stages use as SU programs use ``tracl``),
  ``cdp``, ``cdpt``, ``sx``, ``sy``, ``gx``, ``gy``, ``selev``, ``gelev``, ``offset``, ``nhs`` (the fold), ``sstat``,
  ``gstat``, ``tstat``, ``counit``, ``scalco`` and ``scalel``.
* every other non zero word is kept in ``trace.extra`` by its SU name, and written back as it was. Stages do not
  update these words (for example ``tracl``, ``fldr``, ``muts`` or ``d1``).
* words that SU has no place for (``iline``, ``xline``) and complex traces are not written.

Two things differ from what SU programs see in the headers, both on purpose:

* seispy keeps the offset as the distance between ``tx_loc`` and ``rx_loc``. Data made by SU's synthetic programs
  has an ``offset`` but zero coordinates; those get ``rx_loc = (offset, 0, 0)`` when read and are written back with
  zero coordinates again (until a stage gives the trace coordinates).
* when the ``offset`` of a trace disagrees with its coordinates, that value is kept (in ``extra``) and written.
* the ``offset`` word is an integer and is not scaled by ``scalco``: an offset of 0.1 (coordinates in km) is
  written as 0, with a warning. SU data is in meters (or feet).
"""
import os
import sys
import warnings
from contextlib import contextmanager

import numpy as np

__all__ = ["read_su", "write_su", "SUTraceWriter", "SU_HEADER_BYTES", "SU_HEADER_DTYPE"]

SU_HEADER_BYTES = 240

SU_HEADER_DTYPE = None  # (the native byte order structured dtype of the 240 bytes, made on first use)


def _native_dtype():
    global SU_HEADER_DTYPE
    if SU_HEADER_DTYPE is None:
        from .segy_standard import su_trace_header_dtype

        dtype = np.dtype(su_trace_header_dtype)
        if dtype.itemsize != SU_HEADER_BYTES:
            raise RuntimeError(f"The SU header is {dtype.itemsize} bytes, expected {SU_HEADER_BYTES}.")
        SU_HEADER_DTYPE = dtype
    return SU_HEADER_DTYPE


def _byte_order(endian):
    if endian in (None, "native"):
        return "="
    if endian in ("big", ">"):
        return ">"
    if endian in ("little", "<"):
        return "<"
    raise ValueError(f"endian must be None, 'native', 'big' or 'little', not {endian!r}")


# words that are turned into header values (all other non zero words go to trace.extra)
_MAPPED = frozenset(
    "ns dt delrt trid tracl cdp cdpt sx sy gx gy selev gelev offset nhs sstat gstat tstat counit".split()
)
_COORD_WORDS = ("sx", "sy", "gx", "gy")
_I16 = (-(2**15), 2**15 - 1)
_I32 = (-(2**31), 2**31 - 1)

# SU's coordinate units, as the CoordinateUnit names of Trace
_COUNIT_TO_NAME = {0: "unknown", 1: "length", 2: "degrees", 3: "degrees"}  # (2 is arc seconds)
_NAME_TO_COUNIT = {"unknown": 0, "length": 1, "degrees": 3}


def _scale(scalar):
    """The factor that a SEG-Y scalar (positive multiplies, negative divides, 0 means 1) stands for"""
    scalar = int(scalar)
    if scalar > 0:
        return float(scalar)
    if scalar < 0:
        return -1.0 / scalar
    return 1.0


def _fits(values, scalar, lo_hi):
    """Whether round(value / scale) are all within range, and reproduce the values to float precision"""
    scale = _scale(scalar)
    lo, hi = lo_hi
    for v in values:
        q = round(v / scale)
        if not lo <= q <= hi or abs(q * scale - v) > 1e-9 * max(1.0, abs(v)):
            return False
    return True


def _pick_scalar(values, preferred, lo_hi):
    """A SEG-Y scalar with which all the values are whole numbers, the preferred one if it works"""
    if _fits(values, preferred, lo_hi):
        return int(preferred)
    for scalar in (1, -10, -100, -1000, -10000):
        if _fits(values, scalar, lo_hi):
            return scalar
    # nothing is exact, take the finest scalar that does not overflow
    for scalar in (-10000, -1000, -100, -10, 1, 10, 100, 1000, 10000):
        scale = _scale(scalar)
        if all(lo_hi[0] <= round(v / scale) <= lo_hi[1] for v in values):
            return scalar
    raise OverflowError(f"Unable to store the values {list(values)} in SU header words.")


def su_header_to_trace(row, data):
    """A Trace from one SU header (a numpy void of the SU header dtype, native byte order) and its samples"""
    from ..container import Trace  # (not at the top, seispy.container imports seispy.io)

    w = {name: row[name].item() for name in row.dtype.names if name != "unass"}

    dt = w["dt"] * 1e-6
    if dt == 0.0:
        dt = float(w["d1"])  # (what SU programs do for data that has no sample interval in microseconds)
    sample_start = w["delrt"] * 1e-3
    if w["delrt"] == 0 and w["f1"] != 0.0 and w["dt"] == 0:
        sample_start = float(w["f1"])

    scalco = _scale(w["scalco"])
    scalel = _scale(w["scalel"])
    counit = w["counit"]
    coord_scale = scalco / 3600.0 if counit == 2 else scalco
    tx = [w["sx"] * coord_scale, w["sy"] * coord_scale, w["selev"] * scalel]
    rx = [w["gx"] * coord_scale, w["gy"] * coord_scale, w["gelev"] * scalel]

    extra = {}
    for name, value in w.items():
        if name not in _MAPPED and value != 0:
            extra[name] = value
    if counit >= 4:
        extra["counit"] = counit  # (no counterpart, keep it)
    for name in ("cdp", "cdpt"):
        if w[name] < 0:
            extra[name] = w[name]  # (an ensemble number is not negative for seispy)
    unass = row["unass"].tolist()
    if any(unass):
        extra["unass"] = tuple(unass)

    offset = w["offset"]
    if not any(w[name] for name in _COORD_WORDS):
        if offset != 0:
            rx[0] = float(offset)
            extra["_su_offset_only"] = 1
    else:
        # the offset is the distance of the coordinates for seispy
        dx, dy = rx[0] - tx[0], rx[1] - tx[1]
        distance = float(np.hypot(dx, dy))
        geometric = -distance if (dx < 0 or (dx == 0 and dy < 0)) else distance
        if abs(geometric - offset) > 0.5:
            extra["offset"] = offset

    tr = Trace(
        np.asarray(data, dtype=np.float32),
        d_sample=dt,
        sample_start=sample_start,
        tx_loc=tx,
        rx_loc=rx,
        coord_unit=_COUNIT_TO_NAME.get(counit, "unknown"),
        trace_type=w["trid"],
    )
    return tr.replace(
        trace_id=w["tracl"],
        ensemble_number=max(w["cdp"], 0),
        ensemble_trace_number=max(w["cdpt"], 0),
        fold=w["nhs"],
        source_static=w["sstat"] * 1e-3,
        receiver_static=w["gstat"] * 1e-3,
        total_static=w["tstat"] * 1e-3,
        extra=extra,
    )


def trace_to_su_header(trace, dtype=None):
    """A one element numpy array of the SU header of a Trace (``dtype`` selects the byte order)"""
    if dtype is None:
        dtype = _native_dtype()
    if trace.dtype != np.float32:
        raise NotImplementedError("Complex (Fourier domain) traces have no general representation in SU data.")
    hdr = trace.header
    extra = trace.extra
    out = np.zeros(1, dtype=dtype)[0:1]
    row = out[0]

    def put(name, value, lo_hi=_I16):
        value = int(value)
        if not lo_hi[0] <= value <= lo_hi[1]:
            raise OverflowError(f"{name}={value} does not fit in the SU header word.")
        row[name] = value

    # carried words first, then the mapped ones overwrite
    for name, value in extra.items():
        if name == "unass":
            row["unass"] = value
        elif name in dtype.names and name not in _MAPPED and not name.startswith("_"):
            row[name] = value

    n_sample = hdr["n_sample"]
    if n_sample > 65535:
        raise OverflowError(f"SU traces have at most 65535 samples (ns is unsigned short), this has {n_sample}.")
    row["ns"] = n_sample

    d_sample = hdr["d_sample"]
    dt = round(d_sample * 1e6)
    if 0 < dt <= 65535:
        row["dt"] = dt
    elif d_sample != 0.0:
        # too long or too short for SU's whole microseconds: programs then use d1 and f1
        row["dt"] = 0
        row["d1"] = d_sample
        row["f1"] = hdr["sample_start"]
    delrt = round(hdr["sample_start"] * 1e3)
    if _I16[0] <= delrt <= _I16[1]:
        row["delrt"] = delrt

    put("trid", hdr["trace_type"])
    put("tracl", hdr["trace_id"], _I32)
    put("cdp", hdr["ensemble_number"] or extra.get("cdp", 0), _I32)
    put("cdpt", hdr["ensemble_trace_number"] or extra.get("cdpt", 0), _I32)
    put("nhs", hdr["fold"])
    put("sstat", round(hdr["source_static"] * 1e3))
    put("gstat", round(hdr["receiver_static"] * 1e3))
    put("tstat", round(hdr["total_static"] * 1e3))

    tx, rx = hdr["tx_loc"], hdr["rx_loc"]
    row["counit"] = extra.get("counit", _NAME_TO_COUNIT[_coord_unit_name(hdr["coord_unit"])])

    # The coordinates of a trace read from data that had only an offset are made up (see su_header_to_trace), so
    # whatever a stage did with them (a stack puts them at the midpoint) is not a position: write no coordinates.
    offset_only = extra.get("_su_offset_only") and tx[1] == 0 and rx[1] == 0 and not tx[2] and not rx[2]
    if offset_only:
        row["scalco"] = int(extra.get("scalco", 0))
        offset = round(hdr["offset"])
        for name in _COORD_WORDS:
            row[name] = 0
        row["selev"] = row["gelev"] = 0
        row["scalel"] = int(extra.get("scalel", 0))
    else:
        values = [tx[0], tx[1], rx[0], rx[1]]
        scalco = _pick_scalar(values, extra.get("scalco", 0), _I32)
        row["scalco"] = scalco
        scale = _scale(scalco)
        for name, v in zip(_COORD_WORDS, values):
            row[name] = round(v / scale)
        elev = [tx[2], rx[2]]
        scalel = _pick_scalar(elev, extra.get("scalel", 0), _I32)
        row["scalel"] = scalel
        scale = _scale(scalel)
        row["selev"] = round(elev[0] / scale)
        row["gelev"] = round(elev[1] / scale)
        offset = extra.get("offset", round(hdr["offset"]))
    if abs(float(hdr["offset"]) - offset) > 1e-3 * max(1.0, abs(float(hdr["offset"]))) and "offset" not in extra:
        warnings.warn(
            "An offset does not fit in the integer offset word of SU and is written rounded (as 0 for less than "
            "half a unit): SU programs that use the offset will be wrong. Use smaller units for the coordinates "
            "(SU data is usually in meters, not km).",
            stacklevel=3,
        )
    row["offset"] = int(offset)
    return out


def _coord_unit_name(value):
    # CoordinateUnit enum values: 0 unknown, 1 length, 2 degrees
    return {0: "unknown", 1: "length", 2: "degrees"}.get(int(value), "unknown")


def _read_exactly(f, n):
    """n bytes from a binary file object (short reads are repeated), fewer only at the end of the file"""
    if n == 0:
        return b""
    chunks = []
    remaining = n
    while remaining:
        chunk = f.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


@contextmanager
def _open(source, mode):
    if isinstance(source, (str, os.PathLike)):
        with open(source, mode) as f:
            yield f
    else:
        yield source  # (the caller's stream, the caller closes it)


def _trace_generator(source, order):
    dtype = _native_dtype().newbyteorder(order)
    sample_dtype = np.dtype(np.float32).newbyteorder(order)
    with _open(source, "rb") as f:
        index = 0
        while True:
            raw = _read_exactly(f, SU_HEADER_BYTES)
            if len(raw) == 0:
                return
            if len(raw) != SU_HEADER_BYTES:
                raise EOFError(f"SU trace {index}: the stream ended inside a trace header ({len(raw)} bytes).")
            header = np.frombuffer(raw, dtype=dtype).astype(_native_dtype())[0]  # (native order from here)
            ns = int(header["ns"])
            raw = _read_exactly(f, 4 * ns)
            if len(raw) != 4 * ns:
                raise EOFError(f"SU trace {index}: the stream ended inside the samples ({len(raw)} of {4 * ns} bytes).")
            data = np.frombuffer(raw, dtype=sample_dtype).astype(np.float32)
            yield su_header_to_trace(header, data)
            index += 1


def _count_regular_file(path, order):
    """The number of traces in a file of traces of equal length, else None (without reading the data)"""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return 0
        with open(path, "rb") as f:
            raw = f.read(SU_HEADER_BYTES)
        if len(raw) != SU_HEADER_BYTES:
            return None
        dtype = _native_dtype().newbyteorder(order)
        ns = int(np.frombuffer(raw, dtype=dtype)[0]["ns"])
        trace_bytes = SU_HEADER_BYTES + 4 * ns
        if size % trace_bytes == 0:
            return size // trace_bytes  # (a file of unequal traces would be a coincidence, and iteration is exact)
    except OSError:
        pass
    return None


def read_su(source=None, endian=None):
    """Read SU traces lazily from a file or a stream.

    Parameters
    ----------
    source : str, path, or binary file object, optional
        The file to read. Defaults to the standard input (``sys.stdin.buffer``), which may be a pipe.
    endian : {None, 'native', 'big', 'little'}
        The byte order of the data. SU writes the byte order of its machine, unless built with ``-DSUXDR``, which
        is big endian.

    Returns
    -------
    A trace iterator (see ``seispy.container``) of the traces, which a stage can be applied to.
    """
    from ..container import from_iterable  # (not at the top, seispy.container imports seispy.io)

    order = _byte_order(endian)
    if source is None:
        source = sys.stdin.buffer
    n_traces = _count_regular_file(source, order) if isinstance(source, (str, os.PathLike)) else None
    return from_iterable(_trace_generator(source, order), n_traces)


class SUTraceWriter:
    """Writes traces one by one to a SU file or stream, without buffering them.

    Use as a context manager, or call ``close``. ``dest`` is a path or a binary file object (default: the standard
    output, ``sys.stdout.buffer``), which is flushed but not closed.
    """

    def __init__(self, dest=None, endian=None):
        self._order = _byte_order(endian)
        self._dtype = _native_dtype().newbyteorder(self._order)
        self._sample_dtype = np.dtype(np.float32).newbyteorder(self._order)
        if dest is None:
            dest = sys.stdout.buffer
        if isinstance(dest, (str, os.PathLike)):
            self._file = open(dest, "wb")
            self._owns = True
        else:
            self._file = dest
            self._owns = False
        self.n_written = 0

    def write(self, trace):
        header = trace_to_su_header(trace, self._dtype)
        self._file.write(header.tobytes())
        data = np.asarray(trace, dtype=np.float32)
        if self._order != "=":
            data = data.astype(self._sample_dtype)
        self._file.write(data.tobytes())
        self.n_written += 1

    def flush(self):
        self._file.flush()

    def close(self):
        try:
            self._file.flush()
        finally:
            if self._owns:
                self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_su(traces, dest=None, endian=None):
    """Write an iterable (or iterator) of traces to a SU file or stream, returning the number of traces written."""
    with SUTraceWriter(dest, endian) as writer:
        for trace in traces:
            writer.write(trace)
        return writer.n_written
