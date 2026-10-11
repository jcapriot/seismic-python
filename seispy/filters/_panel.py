"""
Filters of a panel or of a gather of traces, done in the Fourier domain (``su/main/filters``): SUKFILTER, SUK1K2FILTER,
SUKFRAC, SUSMGAUSS2, SULFAF, SUCCFILT, SUDEGHOST, SUFWATRIM, SUSVDFILT, SUFWMIX (``su/main/operations``) and SUGROLL
(``su/main/noise``).

The traces of the stream are one panel (held in memory) for the filters of the wavenumber domain, and a gather (the run of
traces that have the same value of ``key``, by default ``ensemble_number``) for the others. The samples are transformed
with the FFT of seispy (``seispy._fft``, SU's prime-factor FFT), padded to the lengths that the programs use (``npfa``,
``npfar`` and ``npfaro`` of the SU library, so that edge effects are the same as those of the programs), with the sign of
the kernel that each program has: +1 forward in time (SUDEGHOST), and -1 forward and +1 inverse for the two dimensional
transforms of the filters of the wavenumber domain. The filters that are a polygon with sine squared tapers are made by
``polygonalFilter`` of the SU library (``_filter.design``).

Where the program plainly does not do what its documentation says (or what could not be intended), this does what the
documentation says, and the differences are listed with each stage. The output traces have the headers of the input
traces.
"""
import math
import warnings

import numpy as np

from ..container import as_trace_iterator, from_iterable
from ..parallel import group_by
from .. import _fft
from ._filter import design as _design

__all__ = ['_kfilter', '_k1k2filter', '_kfrac', '_smgauss2', '_lfaf', '_ccfilt', '_deghost', '_fwatrim', '_svdfilt', '_fwmix',
           '_groll']

# ratio of the default k1, k2, k3 and k4 to the Nyquist wavenumber
_FRACTIONS = (0.10, 0.15, 0.45, 0.50)


def _nint(x):
    """SU's NINT: to the nearest, halves away from zero"""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _real_panel(traces, name):
    """The samples of the traces as a float64 array (ntr, nt), after checking that they can be one"""
    n = traces[0].n_sample
    arrays = []
    for trace in traces:
        if trace.dtype.kind == 'c':
            raise TypeError(f"{name} works on real traces, but was given a complex one.")
        if trace.n_sample != n:
            raise ValueError("The traces must all have the same number of samples.")
        arrays.append(np.asarray(trace))
    return np.stack(arrays).astype(np.float64)


def _sample_interval(trace, dt, name):
    sample_dt = float(dt) if dt else trace.d_sample
    if not sample_dt:
        raise ValueError(f"{name} needs the sample interval: the trace has none, give dt.")
    return float(sample_dt)


def _polygon_parameters(k, amps, name_k, name_amps):
    """Check the wavenumbers and amplitudes of a polygonal filter, as the programs do"""
    if k is not None:
        k = np.atleast_1d(np.asarray(k, dtype=np.float64))
        if k.ndim != 1 or k.shape[0] == 0:
            raise ValueError(f"{name_k} must be a 1D array of wavenumbers.")
        if k.shape[0] < 2:
            warnings.warn(f"Only {k.shape[0]} value defining filter")
        if (k < 0.0).any() or (np.diff(k) < 0.0).any():
            raise ValueError("Bad filter parameters, the wavenumbers must be increasing and not negative.")
    if amps is not None:
        amps = np.atleast_1d(np.asarray(amps, dtype=np.float64))
        if amps.ndim != 1:
            raise ValueError(f"{name_amps} must be a 1D array.")
        if (amps < 0.0).any():
            raise ValueError("amps values must be positive")
        if not (amps > 0.0).any():
            raise ValueError("All amps values are zero")
        if (amps == amps[0]).all():
            warnings.warn("All amps values are the same")
        if k is not None and amps.shape[0] != k.shape[0]:
            raise ValueError("number of k values must = number of amps values")
        if k is None and amps.shape[0] != len(_FRACTIONS):
            raise ValueError(f"number of k values must = number of amps values (the default k has {len(_FRACTIONS)} values)")
    return k, amps


def _default_polygon(k, amps, nyquist):
    if k is None:
        k = np.array([fraction * nyquist for fraction in _FRACTIONS])
    if amps is None:
        # the default is a trapezoidal bandpass filter
        amps = np.ones(k.shape[0])
        amps[0] = amps[-1] = 0.0
    return k, amps


def _polygon(x, k, amps):
    """The amplitude, a polygon through the points (k, amps) with sine tapers, at the values x"""
    out = np.full(x.shape, amps[-1], dtype=np.float64)
    out[x < k[0]] = amps[0]
    for i in range(k.shape[0] - 1):
        lo, hi = k[i], k[i + 1]
        inside = (x >= lo) & (x <= hi)
        if not inside.any():
            continue
        u = (x[inside] - lo) / (hi - lo) if hi > lo else np.ones(int(inside.sum()))
        a0, a1 = amps[i], amps[i + 1]
        if a0 < a1:
            out[inside] = a0 + (a1 - a0) * np.sin(0.5 * np.pi * u)
        elif a0 > a1:
            out[inside] = a1 + (a0 - a1) * np.sin(0.5 * np.pi * (1.0 - u))
        else:
            out[inside] = a0
    return out


def _fft_lengths(nx1, nx2, name):
    nx1fft = _fft.good_length_within(nx1, 2 * nx1)
    nx2fft = _fft.good_length(nx2, real=False)
    if nx1fft >= 720720 or nx2fft >= 720720:
        raise ValueError(f"Padded nx1={nx1fft} or nx2={nx2fft}--too big")
    return nx1fft, nx2fft


def _apply_2d(data, nx1fft, nx2fft, response):
    """Filter the panel (nx2, nx1) with response (nx2fft, nx1fft // 2 + 1), the transforms made as SU's (padded with zeros,
    transformed in the first dimension as real to complex and in the second as complex to complex)"""
    nx2, nx1 = data.shape
    # (the sign of the kernel of these is -1 forward and +1 inverse, as the programs have it)
    spectrum = _fft.rfft(data, nx1fft, axis=1, sign=-1)
    spectrum = _fft.fft(spectrum, nx2fft, axis=0, sign=-1)
    spectrum *= response
    spectrum = _fft.ifft(spectrum, axis=0, sign=1)
    return _fft.irfft(spectrum, nx1fft, axis=1, sign=1)[:nx2, :nx1]


def _wavenumber_indices(nx1fft, nx2fft):
    """The k1 index (0 to nK1 - 1) of the columns and the |k2| index of the rows of the transform of a panel"""
    i1 = np.arange(nx1fft // 2 + 1)
    idx = np.arange(nx2fft)
    i2 = np.where(idx < nx2fft // 2 + 1, idx, nx2fft - idx)
    return i1, i2


def _panel_stage(upstream, name, make_output):
    """The traces of upstream as a panel, to make_output(traces, data) which gives the output array"""
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        data = _real_panel(panel, name)
        out = make_output(panel, data)
        for trace, row in zip(panel, out):
            yield trace.replace(np.ascontiguousarray(row, dtype=np.float32))

    return from_iterable(traces(), n_traces=source.n_traces)


# ------------------------------------------------------------------------------------------------------------ kfilter
def _kfilter(upstream, *, k=None, amps=None, d1=None, d2=None):
    k, amps = _polygon_parameters(k, amps, 'k', 'amps')

    def make_output(panel, data):
        nx2, nx1 = data.shape
        dx1 = float(d1) if d1 else (panel[0].d_sample or 1.0)
        dx2 = float(d2) if d2 else 1.0
        nx1fft, nx2fft = _fft_lengths(nx1, nx2, 'kfilter')
        nyquist = math.hypot(0.5 / dx1, 0.5 / dx2)
        kk, aa = _default_polygon(k, amps, nyquist)
        i1, i2 = _wavenumber_indices(nx1fft, nx2fft)
        # the wavenumber (cycles per unit) of every sample of the transform
        radius = np.hypot(i1[None, :] / (nx1fft * dx1), i2[:, None] / (nx2fft * dx2))
        # (the polygon is squared in the program: the amplitude of the taper is sine squared)
        response = _polygon(radius, kk, aa) ** 2
        return _apply_2d(data, nx1fft, nx2fft, response)

    return _panel_stage(upstream, 'kfilter', make_output)


# --------------------------------------------------------------------------------------------------------- k1k2filter
def _k1k2filter(upstream, *, k1=None, k2=None, amps1=None, amps2=None, d1=None, d2=None, quad=0):
    k1, amps1 = _polygon_parameters(k1, amps1, 'k1', 'amps1')
    k2, amps2 = _polygon_parameters(k2, amps2, 'k2', 'amps2')
    if quad not in (0, 1, 2):
        raise ValueError("quad is 0 (all four quadrants), 1 (quadrants 1 and 4) or 2 (quadrants 2 and 3).")

    def make_output(panel, data):
        nx2, nx1 = data.shape
        dx1 = float(d1) if d1 else (panel[0].d_sample or 1.0)
        dx2 = float(d2) if d2 else 1.0
        nx1fft, nx2fft = _fft_lengths(nx1, nx2, 'k1k2filter')
        kk1, aa1 = _default_polygon(k1, amps1, 0.5 / dx1)
        kk2, aa2 = _default_polygon(k2, amps2, 0.5 / dx2)
        k1filt = _design(kk1.astype(np.float32), aa1.astype(np.float32), nx1fft, np.float32(dx1))
        k2filt = _design(kk2.astype(np.float32), aa2.astype(np.float32), nx2fft, np.float32(dx2))
        nk2 = nx2fft // 2 + 1
        rows = np.arange(nx2fft)
        positive = rows < nk2
        response = k1filt[None, :] * k2filt[np.where(positive, rows, nx2fft - rows)][:, None]
        if quad == 1:
            response = response * positive[:, None]
        elif quad == 2:
            response = response * (~positive)[:, None]
        return _apply_2d(data, nx1fft, nx2fft, response)

    return _panel_stage(upstream, 'k1k2filter', make_output)


# -------------------------------------------------------------------------------------------------------------- kfrac
def _kfrac(upstream, *, power=0.0, sign=1.0, d1=None, d2=None, phasefac=0.0, angle=None):
    def make_output(panel, data):
        nx2, nx1 = data.shape
        dx1 = float(d1) if d1 else (panel[0].d_sample or 1.0)
        dx2 = float(d2) if d2 else 1.0
        nx1fft, nx2fft = _fft_lengths(nx1, nx2, 'kfrac')
        # the angular wavenumbers (the program had 1 / nK where 1 / nfft is the spacing: twice what they are)
        dk1 = 2.0 * math.pi / (nx1fft * dx1)
        dk2 = 2.0 * math.pi / (nx2fft * dx2)
        i1, i2 = _wavenumber_indices(nx1fft, nx2fft)
        k1 = i1[None, :] * dk1
        k2 = i2[:, None] * dk2
        if angle is not None:
            theta = math.radians(angle)
            k = np.sqrt((k1 * math.cos(theta)) ** 2 + (k2 * math.sin(theta)) ** 2)
        else:
            k = np.hypot(k1, k2)
        if power < 0:
            # (the program makes k huge at k = 0: the amplitude is 0 there)
            with np.errstate(divide='ignore'):
                amp = np.where(k == 0.0, 0.0, k ** power)
        else:
            amp = k ** power
        phase = sign * 0.5 * math.pi * power + phasefac * math.pi
        return _apply_2d(data, nx1fft, nx2fft, amp * np.exp(1j * phase))

    return _panel_stage(upstream, 'kfrac', make_output)


# ---------------------------------------------------------------------------------------------------------- smgauss2
def _smgauss2(upstream, *, a1=0.0, a2=0.0, slowness=True):
    if a1 < 0 or a2 < 0:
        raise ValueError("a1 and a2 must not be negative.")

    def make_output(panel, data):
        nx2, nx1 = data.shape
        nx1fft, nx2fft = _fft_lengths(nx1, nx2, 'smgauss2')
        if slowness:
            if (data <= 0.0).any():
                raise ValueError("The velocities must be positive.")
            field = 1.0 / data
        else:
            field = data
        i1, i2 = _wavenumber_indices(nx1fft, nx2fft)
        dk1 = 2.0 * math.pi / nx1fft
        dk2 = 2.0 * math.pi / nx2fft
        response = np.exp(-(a2 * i2 * dk2 / 2.0) ** 2)[:, None] * np.exp(-(a1 * i1 * dk1 / 2.0) ** 2)[None, :]
        smooth = _apply_2d(field, nx1fft, nx2fft, response)
        # (the amplitude lost to the zero padding is made up so that the maximum is the one of the input)
        peak = smooth.max()
        smooth = smooth * (field.max() / peak) if peak != 0.0 else smooth
        return 1.0 / smooth if slowness else smooth

    return _panel_stage(upstream, 'smgauss2', make_output)


# ------------------------------------------------------------------------------------------------------------- lfaf
def _mix_weights(ntr, mix, tri):
    """The matrix that mixes the traces: row i has the (normalized) weights of the traces in a window of mix traces
    centered on trace i, cut at the ends of the gather"""
    half = mix // 2
    weights = np.zeros((ntr, ntr))
    for i in range(ntr):
        start = max(i - half, 0)
        stop = min(i - half + mix, ntr)
        j = np.arange(start, stop)
        w = (half + 1 - np.abs(j - i)).astype(np.float64) if tri else np.ones(j.shape[0])
        weights[i, start:stop] = w / w.sum()
    return weights


def _lfaf(upstream, *, f1=3.0, f2=20.0, fr=5.0, vel=1000.0, dx=10.0, maxmix=None, adb=1.0, tri=False, dt=None,
          key='ensemble_number'):
    if f1 < 0 or f2 < f1 or fr < 0:
        raise ValueError("f1, f2 and fr must be increasing and not negative.")
    if vel <= 0 or dx <= 0:
        raise ValueError("vel and dx must be positive.")
    if maxmix is not None and maxmix < 1:
        raise ValueError("maxmix must be at least 1.")
    source = as_trace_iterator(upstream)

    def traces():
        for group in group_by(source, key):
            data = _real_panel(group, 'lfaf')
            ntr, nt = data.shape
            sample_dt = _sample_interval(group[0], dt, 'lfaf')
            nfft = _fft.good_length_within(nt, nt)
            nf = nfft // 2 + 1
            d_f = 1.0 / (nfft * sample_dt)
            if1, if2, ifr = _nint(f1 / d_f), _nint(f2 / d_f), _nint(fr / d_f)
            spectrum = _fft.rfft(data, nfft, axis=1)
            out = spectrum.copy()
            largest = ntr if maxmix is None else min(maxmix, ntr)
            weights = {}
            for iw in range(if1, min(if2 + ifr, nf)):
                f = iw * d_f
                # (the number of traces is the velocity over the wavelength in traces, as the documentation has it)
                mix = largest if f == 0.0 else min(_nint(vel / (f * dx)), largest)
                mix = max(mix, 1)
                if mix % 2 == 0:
                    mix -= 1
                if mix not in weights:
                    weights[mix] = _mix_weights(ntr, mix, tri)
                mixed = weights[mix] @ spectrum[:, iw]
                if iw < if2:
                    out[:, iw] = mixed * adb + spectrum[:, iw] * (1.0 - adb)
                else:
                    wh = 1.0 - (iw - if2) / ifr
                    out[:, iw] = (wh * mixed + spectrum[:, iw] * (1.0 - wh)) * adb + spectrum[:, iw] * (1.0 - adb)
            result = _fft.irfft(out, nfft, axis=1)[:, :nt]
            for trace, row in zip(group, result):
                yield trace.replace(np.ascontiguousarray(row, dtype=np.float32))

    return from_iterable(traces())


# ----------------------------------------------------------------------------------------------------------- ccfilt
def _ccfilt(upstream, *, cch=1.0, ccl=0.3, padd=25.0, key='ensemble_number'):
    source = as_trace_iterator(upstream)
    factor = np.float32(1.0) + np.float32(padd) / np.float32(100.0)

    def traces():
        for group in group_by(source, key):
            data = _real_panel(group, 'ccfilt')
            ntr, nt = data.shape
            if ntr < 2:
                raise ValueError("A gather needs at least 2 traces.")
            nfft = _fft.good_length(int(np.float32(nt) * factor))
            if nfft >= 720720:
                raise ValueError(f"Padded nt={nfft}--too big")
            nf = nfft // 2 + 1
            spectrum = _fft.rfft(data, nfft, axis=1)
            phase = np.angle(spectrum)
            # the correlation coefficient of the phases of neighbouring traces, in every frequency but the Nyquist
            cc = np.cos(phase[:-1, :nf - 1] - phase[1:, :nf - 1])
            kept = spectrum[:-1, :nf - 1]
            kept[(cc > cch) | (cc < ccl)] = 0.0
            result = _fft.irfft(spectrum, nfft, axis=1)[:, :nt]
            for trace, row in zip(group, result):
                yield trace.replace(np.ascontiguousarray(row, dtype=np.float32))

    return from_iterable(traces())


# ---------------------------------------------------------------------------------------------------------- deghost
def _deghost(upstream, *, h, v=1500.0, r=0.5, lambert=True, deghost=True, pnoise=1.0e-9, dt=None, p0=0.0, dp=0.0):
    if v <= 0:
        raise ValueError("v must be positive.")
    source = as_trace_iterator(upstream)

    def traces():
        cache = {}
        for number, trace in enumerate(source):
            if trace.dtype.kind == 'c':
                raise TypeError("deghost works on real traces, but was given a complex one.")
            nt = trace.n_sample
            # (as SU does, .004 if there is nothing at all)
            sample_dt = float(dt) if dt else (trace.d_sample or 0.004)
            nfft = _fft.good_length_within(nt, 2 * nt)
            if nfft >= 720720:
                raise ValueError(f"Padded nt={nfft}--too big")
            # the horizontal slowness: p0 for the first trace of the (tau, p) panel, dp more for each of the next
            p = p0 + number * dp
            key = (nt, sample_dt, p)
            if key not in cache:
                cache.clear()
                nf = nfft // 2 + 1
                f = np.arange(nf) / (nfft * sample_dt)
                vvpp = min(v * v * p * p, 1.0)  # (p v > 1 is not allowed)
                if lambert:
                    amp = max(r * math.sqrt(1.0 - vvpp), pnoise)
                else:
                    amp = r
                phi = 2.0 * math.pi * f * (2.0 * h * math.sqrt(1.0 - vvpp) / v)
                real = 1.0 - amp * np.cos(phi)
                imag = amp * np.sin(phi)
                if deghost:
                    response = (real + 1j * imag) / (real * real + imag * imag)
                else:
                    response = real - 1j * imag  # (the ghost is added: for modelling)
                cache[key] = response.astype(np.complex64)
            spectrum = _fft.rfft(np.asarray(trace), nfft)
            yield trace.replace(_fft.irfft(spectrum * cache[key], nfft)[:nt])

    return from_iterable(traces(), n_traces=source.n_traces)


# ---------------------------------------------------------------------------------------------------------- fwatrim
def _fwatrim(upstream, *, trim=15.0, padd=25.0, mark=None, key='ensemble_number'):
    if not 0.0 <= trim < 50.0:
        raise ValueError("trim is a percentage of the traces, from 0 to under 50.")
    source = as_trace_iterator(upstream)
    factor = np.float32(1.0) + np.float32(padd) / np.float32(100.0)

    def traces():
        for group in group_by(source, key):
            data = _real_panel(group, 'fwatrim')
            ntr, nt = data.shape
            nfft = _fft.good_length(int(np.float32(nt) * factor))
            if nfft >= 720720:
                raise ValueError(f"Padded nt={nfft}--too big")
            spectrum = _fft.rfft(data, nfft, axis=1)
            # in every frequency, the mean of the real parts and of the imaginary parts of the traces without the
            # smallest and the largest ones
            ist = _nint(ntr * (trim / 100.0))
            real = np.sort(spectrum.real, axis=0)[ist:ntr - ist]
            imag = np.sort(spectrum.imag, axis=0)[ist:ntr - ist]
            mean = (real.sum(axis=0) + 1j * imag.sum(axis=0)) / (ntr - 2 * ist)
            out = _fft.irfft(mean, nfft)[:nt]
            if mark is None:
                chosen = 0
            elif callable(mark):
                marked = [i for i, t in enumerate(group) if mark(t)]
                chosen = marked[0] if marked else 0
            else:
                chosen = int(mark)
                if not 0 <= chosen < ntr:
                    raise ValueError("mark is outside of the gather.")
            yield group[chosen].replace(np.ascontiguousarray(out, dtype=np.float32))

    return from_iterable(traces())


# ----------------------------------------------------------------------------------------------------------- svdfilt
def _svdfilt(upstream, *, ntrw=None, numpp=None, subtract=True):
    if ntrw is not None and ntrw < 3:
        raise ValueError("ntrw must be at least 3 traces.")
    if numpp is not None and numpp < 0:
        raise ValueError("numpp must not be negative.")
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        data = _real_panel(panel, 'svdfilt')
        ntr, nt = data.shape
        window = ntr if not ntrw else int(ntrw)
        if window > ntr:
            raise ValueError(f"ntrw={window} is larger than the number of traces, {ntr}.")
        planes = int(numpp) if numpp is not None else ntr // 100
        if planes == 0:
            warnings.warn("numpp=0 set to 1")
            planes = 1
        out = np.empty_like(data)
        # (the program did the first window only, and left the rest of its output unset: consecutive windows here)
        for start in range(0, ntr, window):
            block = data[start:start + window]
            u, s, vt = np.linalg.svd(block.T, full_matrices=False)
            m = min(planes, s.shape[0])
            planes_sum = (u[:, :m] * s[:m]) @ vt[:m]  # (nt, traces)
            out[start:start + window] = block - planes_sum.T if subtract else planes_sum.T
        for trace, row in zip(panel, out):
            yield trace.replace(np.ascontiguousarray(row, dtype=np.float32))

    return from_iterable(traces(), n_traces=source.n_traces)


# ----------------------------------------------------------------------------------------------------------- fwmix
def _fwmix(upstream, *, key, dx, mark=None, vf=False, vmin=5000.0, fac=1.0, padd=25.0, group='ensemble_number'):
    keys = [key] if (callable(key) or isinstance(key, str)) else list(key)
    spacing = np.atleast_1d(np.asarray(dx, dtype=np.float64))
    if not keys or spacing.shape[0] != len(keys):
        raise ValueError("key and dx must have the same number of values.")
    if (spacing == 0).any():
        raise ValueError("dx must not be 0.")
    source = as_trace_iterator(upstream)
    fac = max(float(fac), 1.0)
    factor = np.float32(1.0) + np.float32(padd) / np.float32(100.0)
    from ..stage import header_value

    def traces():
        for gather in group_by(source, group):
            data = _real_panel(gather, 'fwmix')
            ntr, nt = data.shape
            sample_dt = _sample_interval(gather[0], None, 'fwmix')
            nfft = _fft.good_length(int(np.float32(nt) * factor))
            if nfft >= 720720:
                raise ValueError(f"Padded nt={nfft}--too big")
            d_f = 1.0 / (nfft * sample_dt)
            spectrum = _fft.rfft(data, nfft, axis=1)
            if mark is None:
                centre = 0
            elif callable(mark):
                marked = [i for i, t in enumerate(gather) if mark(t)]
                centre = marked[0] if marked else 0
            else:
                centre = int(mark)
                if not 0 <= centre < ntr:
                    raise ValueError("mark is outside of the gather.")
            # the distance (in the units of dx) of each trace from the marked one, over the keys
            position = np.array([[header_value(t, k) for k in keys] for t in gather]) / spacing
            distance = np.sqrt(((position - position[centre]) ** 2).sum(axis=1))
            # (the program summed the traces without its own weight, and did not normalize: the documentation has a mix)
            weight = 1.0 / (distance / fac + 1.0)
            if vf:
                # a frequency dependent mix: the traces that are over a half wavelength of the slope vmin away do not
                # contribute to the frequency
                f = np.arange(spectrum.shape[1]) * d_f
                with np.errstate(divide='ignore'):
                    limit = np.where(f > 0, vmin / (2.0 * f), np.inf)
                use = limit[None, :] > distance[:, None]
                use[centre] = True
            else:
                use = np.ones(spectrum.shape, dtype=bool)
            w = weight[:, None] * use
            mixed = (spectrum * w).sum(axis=0) / w.sum(axis=0)
            out = _fft.irfft(mixed, nfft)[:nt]
            yield gather[centre].replace(np.ascontiguousarray(out, dtype=np.float32))

    return from_iterable(traces())


# ----------------------------------------------------------------------------------------------------------- groll
def _groll(upstream, *, nrot=3, sb=False):
    """SUGROLL: the ground roll of a gather is its first nrot principal components (the Karhunen-Loeve transform), which is
    what the first nrot singular vectors of the panel are"""
    if nrot < 0:
        raise ValueError("nrot must not be negative.")
    source = as_trace_iterator(upstream)

    def traces():
        panel = list(source)
        if not panel:
            return
        data = _real_panel(panel, 'groll')
        if nrot == 0:
            ground = np.zeros_like(data)
        else:
            u, s, vt = np.linalg.svd(data.T, full_matrices=False)
            m = min(nrot, s.shape[0])
            ground = ((u[:, :m] * s[:m]) @ vt[:m]).T
        out = ground if sb else data - ground
        for trace, row in zip(panel, out):
            yield trace.replace(np.ascontiguousarray(row, dtype=np.float32))

    return from_iterable(traces(), n_traces=source.n_traces)
