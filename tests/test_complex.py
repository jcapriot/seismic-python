import pickle

import numpy as np
import numpy.testing as npt
import pytest

from seispy import attributes as at
from seispy import operations as op
from seispy.amplitudes import ai2r, gain, nan, normalize, r2ai, weight, zero
from seispy.container import Trace, TraceCollection, from_iterable, join_complex, split_complex
from seispy.convolution import acor, conv, xcor
from seispy.filters import bfilt, filter as sufilter, frac, phase as filters_phase
from seispy.io.segy_standard import SEGYTrace
from seispy.parallel import pmap
from seispy.stretching import nmo, reduce, resamp, shift
from seispy.tapering import ramp, taper
from seispy.transforms import (
    amp, analytic, fft, hilb, ifft, imag, logamp, phase as complex_phase, real, zerophase,
)
from seispy.windowing import kill, mute, wind

DT = 0.004
RNG = np.random.default_rng(321)


def make(z, dt=DT, sample_start=0.0, **header):
    out = []
    for i, row in enumerate(np.atleast_2d(z)):
        t = Trace(np.asarray(row), d_sample=dt, sample_start=sample_start)
        values = {k: (v[i] if isinstance(v, (list, tuple, np.ndarray)) else v) for k, v in header.items()}
        out.append(t.replace(**values) if values else t)
    return out


def run(stage, trs):
    return list(from_iterable(list(trs)) | stage)


def arr(trs):
    return np.array([np.asarray(t) for t in trs])


N = 64
Z = (RNG.normal(size=(3, N)) + 1j * RNG.normal(size=(3, N))).astype(np.complex64) + (2 + 1j)
RE, IM = Z.real.copy(), Z.imag.copy()


# ------------------------------------------------------------------------------------------------------ container
def test_complex_trace_basics():
    # (a trace shares the memory of the array it is made from, as it does for real ones, so copy before writing)
    t = Trace(Z[0].copy(), d_sample=DT)
    assert t.dtype == np.complex64 and t.n_sample == N and len(t) == N
    view = np.asarray(t)
    assert view.dtype == np.complex64 and view.shape == (N,)
    npt.assert_array_equal(view, Z[0])
    assert memoryview(t).format == 'Zf' and memoryview(t).nbytes == 8 * N
    # a writable view of the trace's own samples
    view[3] = 7 - 2j
    assert np.asarray(t)[3] == 7 - 2j
    assert t.header['data_type'] == 1 and t.header['n_sample'] == N
    # real traces are as they were
    r = Trace(RE[0], d_sample=DT)
    assert r.dtype == np.float32 and r.header['data_type'] == 0 and memoryview(r).format == 'f'


def test_dtype_option():
    assert Trace(np.ones(4), d_sample=DT, dtype='complex64').dtype == np.complex64
    assert Trace(np.ones(4, dtype=np.complex128), d_sample=DT).dtype == np.complex64
    assert Trace(np.ones(4, dtype=np.float64), d_sample=DT, dtype=np.float32).dtype == np.float32
    npt.assert_array_equal(np.asarray(Trace([1, 2], d_sample=DT, dtype='complex64')), [1 + 0j, 2 + 0j])
    with pytest.raises(TypeError, match="complex64"):
        Trace(Z[0], d_sample=DT, dtype='float32')
    with pytest.raises(TypeError, match="float32 or complex64"):
        Trace(RE[0], d_sample=DT, dtype='float64')
    with pytest.raises(ValueError):
        Trace(np.ones((2, 2), dtype=np.complex64), d_sample=DT)


def test_replace_follows_the_data():
    t = Trace(Z[0], d_sample=DT, sample_start=0.2)
    assert t.replace().dtype == np.complex64
    npt.assert_array_equal(np.asarray(t.replace()), Z[0])
    real_out = t.replace(np.abs(np.asarray(t)))
    assert real_out.dtype == np.float32 and real_out.n_sample == N and real_out.header['sample_start'] == 0.2
    complex_out = Trace(RE[0], d_sample=DT).replace(np.asarray(t))
    assert complex_out.dtype == np.complex64
    with pytest.raises(TypeError, match="follows its data"):
        t.replace(data_type=0)
    # a different number of samples
    assert t.replace(Z[0][:10]).n_sample == 10


def test_split_and_join():
    t = Trace(Z[0], d_sample=DT, sample_start=0.1)
    re_part, im_part = split_complex(t)
    assert re_part.dtype == im_part.dtype == np.float32
    npt.assert_array_equal(np.asarray(re_part), RE[0])
    npt.assert_array_equal(np.asarray(im_part), IM[0])
    assert re_part.header['sample_start'] == 0.1
    joined = join_complex(re_part, im_part)
    npt.assert_array_equal(np.asarray(joined), Z[0])
    with pytest.raises(TypeError):
        split_complex(re_part)
    with pytest.raises(TypeError):
        join_complex(t, im_part)
    with pytest.raises(ValueError):
        join_complex(re_part, Trace(RE[0][:5], d_sample=DT))


def test_empty_complex_trace():
    t = Trace(np.zeros(0, dtype=np.complex64), d_sample=DT)
    assert t.n_sample == 0 and np.asarray(t).shape == (0,) and np.asarray(t).dtype == np.complex64
    assert pickle.loads(pickle.dumps(t)).dtype == np.complex64


def test_pickle_bytes_and_files(tmp_path):
    t = Trace(Z[0], d_sample=DT, sample_start=0.3, offset=12.0) if False else make(Z[:1], sample_start=0.3, offset=12.0)[0]
    again = pickle.loads(pickle.dumps(t))
    assert again.dtype == np.complex64 and again.header == t.header
    npt.assert_array_equal(np.asarray(again), Z[0])
    # through a file, with real traces next to them
    coll = TraceCollection(Z, d_sample=DT)
    coll.to_file(tmp_path / "complex.spy")
    back = TraceCollection.from_file(tmp_path / "complex.spy")
    out = list(back)
    assert len(out) == 3 and all(tr.dtype == np.complex64 for tr in out)
    npt.assert_array_equal(arr(out), Z)
    # a stream of bytes
    import io
    stream = io.BytesIO()
    coll.to_stream(stream)
    assert len(stream.getvalue()) > 3 * 8 * N


def test_collections_of_complex_arrays():
    coll = TraceCollection(Z, d_sample=DT)
    assert all(tr.dtype == np.complex64 for tr in coll)
    npt.assert_array_equal(arr(list(coll)), Z)


def test_segy_has_no_complex_traces():
    with pytest.raises(TypeError, match="complex"):
        SEGYTrace.from_seispy(Trace(Z[0], d_sample=DT))
    SEGYTrace.from_seispy(Trace(RE[0], d_sample=DT))  # (real ones are fine)


# ---------------------------------------------------------------------------------- stages that work as they are
def test_operations_defined_for_complex_numbers():
    trs = make(Z)
    # abs is the modulus, a real trace
    out = run(op.abs(), trs)
    assert all(t.dtype == np.float32 for t in out)
    npt.assert_allclose(arr(out), np.abs(Z), rtol=1e-6)
    # these stay complex
    for stage, expected in [
        (op.neg(), -Z),
        (op.nop(), Z),
        (op.sqr(), Z * Z),
        (op.sum(), np.cumsum(Z, axis=1)),
        (op.avg(), Z - Z.mean(axis=1, keepdims=True)),
        (op.d2m(), 1000 * Z),
        (op.inv(), 1 / Z),
        (op.exp(), np.exp(np.clip(Z.real, None, 5) + 1j * Z.imag)),
        (op.norm(), Z / np.abs(Z).max(axis=1, keepdims=True)),
        (op.cnorm(), Z / np.abs(Z)),
    ]:
        out = run(stage, make(np.clip(Z.real, None, 5) + 1j * Z.imag) if stage is not None and 'exp' in repr(stage) else trs)
        assert all(t.dtype == np.complex64 for t in out), repr(stage)
        if 'exp' in repr(stage):
            expected = np.exp(np.clip(Z.real, None, 5) + 1j * Z.imag)
        npt.assert_allclose(arr(out), expected, rtol=1e-4, atol=1e-5, err_msg=repr(stage))


def test_operations_on_complex_traces_more():
    trs = make(Z)
    for stage, expected in [
        (op.sin(), np.sin(Z)), (op.cos(), np.cos(Z)), (op.tanh(), np.tanh(Z)),
    ]:
        npt.assert_allclose(arr(run(stage, trs)), expected, rtol=1e-4, atol=1e-5, err_msg=repr(stage))
    # the rms amplitude (of the modulus) goes in the first sample
    out = arr(run(op.rmsamp(), trs))
    npt.assert_allclose(out[:, 0], np.sqrt((np.abs(Z) ** 2).mean(axis=1)), rtol=1e-5)
    npt.assert_array_equal(out[:, 1:], 0)
    # db is the level of the modulus
    (db_trace,) = run(op.db(), trs[:1])
    assert db_trace.dtype == np.float32
    npt.assert_allclose(np.asarray(db_trace), 20 * np.log10(np.abs(Z[0])), rtol=1e-4)
    # differences and means act on both parts
    npt.assert_allclose(arr(run(op.refl(), trs[:1]))[0, 1:], ((Z[0, 1:] - Z[0, :-1]) / (Z[0, 1:] + Z[0, :-1])), rtol=1e-4)
    npt.assert_allclose(
        arr(run(op.diff(), trs[:1]))[0, 5], (Z[0, 6] - Z[0, 4]) / (2 * DT), rtol=1e-4,
    )
    mean = arr(run(op.mean(nw=5), trs[:1]))[0]
    npt.assert_allclose(mean[10], Z[0, 8:13].mean(), rtol=1e-5)
    # the statistics of a window are real
    (std,) = run(op.std(nw=5), trs[:1])
    assert std.dtype == np.float32
    npt.assert_allclose(np.asarray(std)[10], Z[0, 8:13].std(), rtol=1e-4)


def test_operations_that_need_an_order_or_a_sign_say_so():
    trs = make(Z)
    for name in ('posonly', 'negonly', 'ssqrt', 'ssqr', 'sgn', 'sexp', 'slog', 'slog2', 'slog10', 'mod2pi',
                 'spike', 'lnza', 'saf', 'freq', 'despike'):
        with pytest.raises(TypeError, match="complex"):
            run(getattr(op, name)(), trs[:1])


def test_zero_weight_nan_normalize():
    trs = make(Z, offset=[0.0, 1000.0, 2000.0])
    out = run(zero(9, itmin=2, value=0.5), trs)
    assert all(t.dtype == np.complex64 for t in out)
    expected = Z.copy()
    expected[:, 2:10] = 0.5
    npt.assert_array_equal(arr(out), expected)
    # weights are real numbers
    out = arr(run(weight(), trs))
    npt.assert_allclose(out, Z * (1 + 0.0005 * np.array([0, 1000, 2000]))[:, None], rtol=1e-5)
    # NaNs in either part count
    bad = Z.copy()
    bad[0, 5] = complex(np.nan, 1.0)
    bad[0, 9] = complex(1.0, np.inf)
    out = arr(run(nan(value=3.0), make(bad)))
    assert np.isfinite(out).all() and out[0, 5] == 3 and out[0, 9] == 3
    interpolated = arr(run(nan(interp=True), make(bad)))
    npt.assert_allclose(interpolated[0, 5], (bad[0, 4] + bad[0, 6]) / 2, rtol=1e-6)
    # rms and max are of the modulus, and the level is real
    out = arr(run(normalize('rms'), trs))
    npt.assert_allclose(out, Z / np.sqrt((np.abs(Z) ** 2).mean(axis=1, keepdims=True)), rtol=1e-5)
    out = arr(run(normalize('max'), trs))
    npt.assert_allclose(out, Z / np.abs(Z).max(axis=1, keepdims=True), rtol=1e-5)
    # the median of complex numbers is not a thing
    for kind in ('med', 'balmed'):
        with pytest.raises(TypeError, match="median"):
            run(normalize(kind), trs)


def test_tapering_muting_windowing_moving():
    trs = make(Z, offset=[0.0, 1000.0, 2000.0], trace_id=[1, 2, 3])
    # tapers and ramps multiply by real numbers
    out = arr(run(taper(20.0, 12.0), trs))
    npt.assert_allclose(out[:, 0], 0, atol=1e-7)
    npt.assert_allclose(out[:, 10:20], Z[:, 10:20], rtol=1e-6)
    out = arr(run(taper(tr1=1, tr2=1), trs))
    npt.assert_allclose(out[0], 0, atol=1e-7)
    out = arr(run(ramp(tmin=0.02), trs))
    npt.assert_allclose(out[:, :5], Z[:, :5] * ((np.arange(5) + 1) / 5), rtol=1e-6)
    out = arr(run(mute([0.0, 2000.0], [0.0, 0.2], ntaper=3), trs))
    assert (out[1, :25] == 0).all() and out.dtype == np.complex64
    # windowing keeps the type (zeros are padded in)
    out = run(wind(itmin=4, nt=10), trs)
    assert all(t.dtype == np.complex64 and t.n_sample == 10 for t in out)
    npt.assert_array_equal(arr(out), Z[:, 4:14])
    out = run(wind(itmin=60, nt=10), trs)
    npt.assert_array_equal(arr(out)[:, 4:], 0)
    npt.assert_array_equal(arr(out)[:, :4], Z[:, 60:])
    out = run(kill('trace_id', 2), trs)
    assert out[1].dtype == np.complex64 and not np.asarray(out[1]).any() and np.asarray(out[0]).any()
    out = run(shift(tmin=-0.02, tmax=0.3, fill=1.0), trs)
    assert all(t.dtype == np.complex64 for t in out)
    npt.assert_array_equal(arr(out)[:, 5:5 + N], Z)
    assert (arr(out)[:, :5] == 1).all()
    out = arr(run(reduce(rv=2.0), make(np.tile(np.arange(100) * (1 + 1j), (1, 1)), offset=[200.0])))
    expected = np.zeros(100, dtype=complex)
    expected[:75] = np.arange(25, 100) * (1 + 1j)
    npt.assert_array_equal(out[0], expected)


def test_convolution_with_a_real_filter():
    out = run(conv([1.0, 2.0, 1.0]), make(Z))
    assert all(t.dtype == np.complex64 and t.n_sample == N + 2 for t in out)
    npt.assert_allclose(arr(out), np.array([np.convolve(row, [1, 2, 1]) for row in Z]), rtol=1e-5)


# ----------------------------------------------------------------- linear filters: the same on each part
def parts_equal(stage, trs, rtol=1e-5, atol=1e-6):
    """stage on complex traces is stage on the real parts, plus i times stage on the imaginary parts"""
    out = run(stage, make(trs))
    real_out = arr(run(stage, make(trs.real)))
    imag_out = arr(run(stage, make(trs.imag)))
    assert all(t.dtype == np.complex64 for t in out), repr(stage)
    npt.assert_allclose(arr(out), real_out + 1j * imag_out, rtol=rtol, atol=atol, err_msg=repr(stage))
    return arr(out)


def test_real_linear_filters_work_on_each_part():
    z = RNG.normal(size=(2, 200)) + 1j * RNG.normal(size=(2, 200))
    z = z.astype(np.complex64)
    parts_equal(bfilt(f_pass_low=10.0, f_stop_low=5.0), z)
    parts_equal(bfilt(f_pass_high=60.0, f_stop_high=80.0, zerophase=False), z)
    parts_equal(sufilter(f=[10, 20, 60, 80]), z, atol=1e-4)
    parts_equal(hilb(), z)
    parts_equal(resamp(rf=2.0), z)
    parts_equal(resamp(nt=50, dt=0.007), z)
    # (the gain that are linear, and a real bias added to the real part only)
    for kwargs in (dict(tpow=1.5), dict(epow=-2.0), dict(scale=3.0), dict(scale=2.0, norm=4.0)):
        parts_equal(gain(**kwargs), z)


def test_filter_of_a_complex_tone_keeps_the_sign_of_its_frequency():
    # a real filter does not tell positive from negative frequencies, but it keeps the complex tone as it was
    n = 400
    t = np.arange(n) * DT
    tone = np.exp(2j * np.pi * 25 * t).astype(np.complex64)
    (out,) = run(sufilter(f=[10, 20, 40, 50]), make(tone))
    npt.assert_allclose(np.asarray(out), tone, atol=2e-2)
    (low,) = run(sufilter(f=[1, 2, 5, 6]), make(tone))
    assert np.abs(np.asarray(low)).max() < 0.05


def test_gain_bias_is_a_real_number_added_to_the_real_part():
    out = arr(run(gain(bias=2.0), make(Z)))
    npt.assert_allclose(out, Z + 2.0, rtol=1e-6)
    out = arr(run(gain(bias=1.0, tpow=1.0, scale=2.0), make(Z)))
    t = np.arange(N) * DT
    t_factor = np.where(np.arange(N) == 0, 0.0, t)
    npt.assert_allclose(out.real, 2.0 * (Z.real + 1.0) * t_factor, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(out.imag, 2.0 * Z.imag * t_factor, rtol=1e-5, atol=1e-6)


def test_gain_options_that_are_not_defined_for_complex_numbers():
    trs = make(Z)
    for kwargs in (dict(agc=True), dict(gpow=0.5), dict(clip=1.0), dict(qclip=0.9), dict(pbal=True), dict(jon=True),
                   dict(trap=1.0), dict(mbal=True)):
        with pytest.raises(TypeError, match="complex"):
            run(gain(**kwargs), trs)
    with pytest.raises(TypeError, match="real"):
        run(gain(scale=2.0, panel=True), trs)


def test_nmo_on_complex_traces():
    nt = 400
    t = np.arange(nt) * DT

    def ricker(tc):
        a = (np.pi * 25 * (t - tc)) ** 2
        return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)

    offsets = [0.0, 400.0, 800.0]
    z = np.array([ricker(np.sqrt(0.8 ** 2 + (x / 2000.0) ** 2)) * (1 + 0.5j) for x in offsets]).astype(np.complex64)
    out = arr(run(nmo(vnmo=2000.0, sscale=False), make(z, offset=offsets)))
    assert out.dtype == np.complex64
    npt.assert_array_equal(np.argmax(np.abs(out), axis=1), 200)
    # the real and imaginary parts of the output are the real and imaginary parts of the input, flattened
    npt.assert_allclose(out.imag, 0.5 * out.real, atol=1e-5)


# ---------------------------------------------------------------------------- stages that are for real traces
def test_stages_that_are_for_real_traces_say_so():
    trs = make(Z[:1], offset=[0.0])
    for stage in (
        frac(power=1), filters_phase(a=90.0), zerophase(t0=0.1), acor(), xcor([1.0, 2.0]), ai2r(), r2ai(),
        at.amp(), at.freq(), at.phase(), at.q(), analytic(), fft(),
    ):
        with pytest.raises(TypeError, match="complex"):
            run(stage, trs)


# ------------------------------------------------------------------------------------------- the transforms
def test_analytic_trace():
    n = 400
    t = np.arange(n) * DT
    cos = np.cos(2 * np.pi * 25 * t).astype(np.float32)
    (out,) = run(analytic(), make(cos))
    assert out.dtype == np.complex64 and out.n_sample == n
    z = np.asarray(out)
    npt.assert_array_equal(z.real, cos)
    # SU's Hilbert transform has the opposite sign to the usual one, so this is cos - i sin
    npt.assert_allclose(z[50:-50], np.exp(-2j * np.pi * 25 * t)[50:-50], atol=5e-3)
    # the modulus is the envelope of the attributes, and so is the phase
    (envelope,) = run(analytic() | amp(), make(cos))
    (reference,) = run(at.amp(), make(cos))
    npt.assert_allclose(np.asarray(envelope), np.asarray(reference), rtol=1e-6)
    (ph,) = run(analytic() | complex_phase(), make(cos))
    (ph_ref,) = run(at.phase(), make(cos))
    npt.assert_allclose(np.asarray(ph), np.asarray(ph_ref), atol=1e-6)
    # a phase rotation
    (rotated,) = run(analytic(phaserot=90.0), make(cos))
    npt.assert_allclose(np.asarray(rotated).real, 0, atol=1e-6)
    npt.assert_allclose(np.asarray(rotated).imag, z.imag, atol=1e-6)


@pytest.mark.parametrize('n', [64, 65])
@pytest.mark.parametrize('sign', [1, -1])
def test_fft_and_back(n, sign):
    x = RNG.normal(size=(2, n)).astype(np.float32)
    spectra = run(fft(sign=sign), make(x, sample_start=0.1))
    n_fft = n + n % 2
    for s in spectra:
        assert s.dtype == np.complex64 and s.n_sample == n_fft // 2 + 1
        assert s.header['d_sample'] == pytest.approx(1.0 / (n_fft * DT))
        assert s.header['sample_start'] == 0.0 and s.header['sampling_domain'] == 2
    back = run(ifft(sign=-sign), spectra)
    for original, b in zip(x, back):
        assert b.dtype == np.float32 and b.n_sample == n_fft
        assert b.header['d_sample'] == pytest.approx(DT) and b.header['sampling_domain'] == 1
        npt.assert_allclose(np.asarray(b)[:n], original, atol=1e-5)
        if n % 2:
            assert np.asarray(b)[-1] == pytest.approx(0, abs=1e-5)  # the padding


def test_fft_values_and_signs():
    x = RNG.normal(size=(1, 32)).astype(np.float32)
    (plus,) = run(fft(sign=1), make(x))
    (minus,) = run(fft(sign=-1), make(x))
    # (the SU transform with sign = -1 is the usual one)
    npt.assert_allclose(np.asarray(minus), np.fft.rfft(x[0]), rtol=1e-5, atol=1e-5)
    npt.assert_allclose(np.asarray(plus), np.conj(np.fft.rfft(x[0])), rtol=1e-5, atol=1e-5)
    # not the inverse sign: the trace comes back turned around in time
    (wrong,) = run(ifft(sign=1), [plus])
    npt.assert_allclose(np.asarray(wrong), x[0][(-np.arange(32)) % 32], atol=1e-5)
    # dt for traces that do not say
    (no_dt,) = run(fft(dt=0.002), make(x, dt=0.0))
    assert no_dt.header['d_sample'] == pytest.approx(1 / (32 * 0.002))
    with pytest.raises(ValueError):
        fft(sign=0)
    with pytest.raises(ValueError):
        ifft(sign=2)
    with pytest.raises(TypeError, match="complex"):
        run(ifft(), make(x))


def test_real_imag_amp_logamp_phase():
    spectra = run(fft(), make(RNG.normal(size=(1, 64)).astype(np.float32)))
    z = np.asarray(spectra[0])
    for stage, expected in [
        (real(), z.real), (imag(), z.imag), (amp(), np.abs(z)), (complex_phase(), np.arctan2(z.imag, z.real)),
        (logamp(), np.log(np.abs(z))),
    ]:
        (out,) = run(stage, spectra)
        assert out.dtype == np.float32 and out.n_sample == z.shape[0], repr(stage)
        npt.assert_allclose(np.asarray(out), expected, rtol=1e-4, atol=1e-5, err_msg=repr(stage))
    # jack halves the zero frequency
    (jacked,) = run(amp(jack=True), spectra)
    assert np.asarray(jacked)[0] == pytest.approx(np.abs(z[0]) / 2)
    npt.assert_allclose(np.asarray(jacked)[1:], np.abs(z)[1:], rtol=1e-6)
    # the log of nothing is 0, and the phase of nothing is 0
    nothing = make(np.zeros((1, 4), dtype=np.complex64))
    assert not np.asarray(run(logamp(), nothing)[0]).any() and not np.asarray(run(complex_phase(), nothing)[0]).any()
    for stage in (real(), imag(), amp(), logamp(), complex_phase()):
        with pytest.raises(TypeError, match="complex"):
            run(stage, make(RE[:1]))


def test_complex_traces_in_threads_and_pickled_stages():
    z = (RNG.normal(size=(8, 128)) + 1j * RNG.normal(size=(8, 128))).astype(np.complex64)
    trs = make(z, offset=list(range(0, 800, 100)))
    for stage in (op.abs(), op.sum(), gain(scale=2.0), bfilt(), hilb(), weight(), taper(10.0, 10.0), fft() if False else op.nop()):
        expected = arr(run(stage, trs))
        got = arr(list(from_iterable(list(trs)) | pmap(stage, workers=3, chunk=2)))
        npt.assert_array_equal(got, expected)
        again = pickle.loads(pickle.dumps(stage))
        npt.assert_array_equal(arr(run(again, trs)), expected)
