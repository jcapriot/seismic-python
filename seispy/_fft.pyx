# cython: embedsignature=True, language_level=3
"""
The Fourier transforms of seispy: the prime factor FFT of SU (``cwp/lib/pfafft.cpp``) and nothing else.

Every frequency domain operation of seispy, in Python and in the C++ of the SU library, goes through these transforms, so
that they all have the same lengths, sign convention, scaling and (single) precision. The conventions are those of SU:

* the transforms are of ``float32`` and ``complex64`` samples, and they are not normalized (``irfft`` and ``ifft`` divide by
  the length unless ``scale=False``);
* ``sign`` is the sign of the exponent of the kernel, as ``isign`` of SU. The default of the forward transform is ``+1``
  (``exp(+i w t)``, which is what the programs use from time to frequency; numpy's is ``-1``), and of the inverse ``-1``;
* the length of a transform is limited to products of the factors 2 to 16 that the algorithm has (``good_length`` gives the
  next valid one), and a real transform has an even length of which a half is valid for a complex one. A length that is not
  valid is an error: pad to one with ``good_length`` (the SU programs do), which the ``n`` argument does with zeros;
* the real transform of ``n`` samples has ``n // 2 + 1`` frequencies.

The functions work along any axis of an array of any number of dimensions.
"""
from .cimport su
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

_MAX_LENGTH = 720720


def good_length(int n, bint real=True):
    """The smallest valid length of at least n samples, for a real (or complex) transform"""
    if n < 1:
        raise ValueError("n must be at least 1")
    return su.npfar(n) if real else su.npfa(n)


def good_length_within(int n, int nmax, bint real=True):
    """A valid length of at least n samples that is not over nmax if there is one, the least expensive of them (SU's
    ``npfaro`` or ``npfao``)"""
    if n < 1:
        raise ValueError("n must be at least 1")
    return su.npfaro(n, nmax) if real else su.npfao(n, nmax)


def _check_length(int n, bint real):
    if n < 1 or n > _MAX_LENGTH:
        raise ValueError(f"The length {n} of the transform must be from 1 to {_MAX_LENGTH}.")
    valid = (su.npfar(n) if real else su.npfa(n)) == n
    if not valid:
        raise ValueError(
            f"{n} is not a valid length for a {'real' if real else 'complex'} transform (the next one is "
            f"{good_length(n, real)}; see good_length).")


def _sign(int sign):
    if sign != 1 and sign != -1:
        raise ValueError("sign must be 1 or -1")
    return sign


def _rows(a, int n_out, int length_axis_size=-1):
    """The array with its axis moved to the end, as a 2-D C contiguous array, and the shape of what is left"""
    shape = a.shape[:-1]
    return a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1), shape


def _restore(out, shape, int length, axis):
    """The 2-D result as an array with the transformed axis where it was (a copy only if that is not the last one)"""
    result = out.reshape(shape + (length,))
    ndim = result.ndim
    if axis % ndim == ndim - 1:
        return result
    return np.ascontiguousarray(np.moveaxis(result, -1, axis))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _rfft_rows(const float[:, ::1] x, float complex[:, ::1] out, int n, int sign) noexcept nogil:
    cdef:
        Py_ssize_t m = x.shape[0], nin = x.shape[1], i
        Py_ssize_t ncopy = nin if nin < n else n
        float *work = NULL
    if nin == n:
        # (the transform reads its input and writes the output: no copy of the input is needed)
        for i in range(m):
            su.pfarc(sign, n, <float *> &x[i, 0], <su.complex *> &out[i, 0])
        return
    work = <float *> malloc(sizeof(float) * n)
    for i in range(m):
        memcpy(work, &x[i, 0], sizeof(float) * ncopy)
        if ncopy < n:
            memset(work + ncopy, 0, sizeof(float) * (n - ncopy))
        su.pfarc(sign, n, work, <su.complex *> &out[i, 0])
    free(work)


def rfft(x, n=None, axis=-1, int sign=1):
    """The Fourier transform of real samples along an axis: ``n`` samples (zero padded, or cut) to ``n // 2 + 1`` complex
    ones, not normalized. The kernel is ``exp(sign i w t)``"""
    a = np.moveaxis(np.asarray(x), axis, -1)
    if a.ndim == 0:
        raise ValueError("x must be an array")
    if a.dtype.kind == 'c':
        raise TypeError("rfft takes real samples, use fft")
    a = np.ascontiguousarray(a, dtype=np.float32)
    nin = a.shape[-1]
    cdef int length = nin if n is None else int(n)
    _check_length(length, True)
    _sign(sign)
    rows, shape = _rows(a, length)
    out = np.empty((rows.shape[0], length // 2 + 1), dtype=np.complex64)
    cdef const float[:, ::1] rv = rows
    cdef float complex[:, ::1] ov = out
    cdef int s = sign
    with nogil:
        _rfft_rows(rv, ov, length, s)
    return _restore(out, shape, length // 2 + 1, axis)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _irfft_rows(const float complex[:, ::1] c, float[:, ::1] out, int n, int sign, float scale) noexcept nogil:
    cdef:
        Py_ssize_t m = c.shape[0], nin = c.shape[1], i, j
        Py_ssize_t nf = n // 2 + 1
        Py_ssize_t ncopy = nin if nin < nf else nf
        su.complex *work = NULL
    if nin == nf:
        # (the transform only reads its input)
        for i in range(m):
            su.pfacr(sign, n, <su.complex *> &c[i, 0], &out[i, 0])
            if scale != 1.0:
                for j in range(n):
                    out[i, j] = out[i, j] * scale
        return
    work = <su.complex *> malloc(sizeof(float) * 2 * (nf + 1))
    for i in range(m):
        memcpy(work, &c[i, 0], sizeof(float) * 2 * ncopy)
        if ncopy < nf:
            memset(<char *> work + sizeof(float) * 2 * ncopy, 0, sizeof(float) * 2 * (nf - ncopy))
        su.pfacr(sign, n, work, &out[i, 0])
        if scale != 1.0:
            for j in range(n):
                out[i, j] = out[i, j] * scale
    free(work)


def irfft(c, n, axis=-1, int sign=-1, bint scale=True):
    """The inverse of ``rfft``: ``n // 2 + 1`` complex frequencies to ``n`` real samples, divided by ``n`` unless
    ``scale`` is false. The imaginary parts of the zero and Nyquist frequencies are not used"""
    a = np.moveaxis(np.asarray(c), axis, -1)
    if a.ndim == 0:
        raise ValueError("c must be an array")
    a = np.ascontiguousarray(a, dtype=np.complex64)
    cdef int length = int(n)
    _check_length(length, True)
    _sign(sign)
    rows, shape = _rows(a, length)
    out = np.empty((rows.shape[0], length), dtype=np.float32)
    cdef const float complex[:, ::1] cv = rows
    cdef float[:, ::1] ov = out
    cdef int s = sign
    cdef float f = (1.0 / length) if scale else 1.0
    with nogil:
        _irfft_rows(cv, ov, length, s, f)
    return _restore(out, shape, length, axis)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fft_inplace_rows(float complex[:, ::1] z, int n, int sign, float scale) noexcept nogil:
    cdef Py_ssize_t m = z.shape[0], i, j
    for i in range(m):
        su.pfacc(sign, n, <su.complex *> &z[i, 0])
        if scale != 1.0:
            for j in range(n):
                z[i, j] = z[i, j] * scale


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _fft_rows(const float complex[:, ::1] x, float complex[:, ::1] out, int n, int sign, float scale) noexcept nogil:
    cdef:
        Py_ssize_t m = x.shape[0], nin = x.shape[1], i, j
        Py_ssize_t ncopy = nin if nin < n else n
    for i in range(m):
        memcpy(&out[i, 0], &x[i, 0], sizeof(float) * 2 * ncopy)
        if ncopy < n:
            memset(<char *> &out[i, 0] + sizeof(float) * 2 * ncopy, 0, sizeof(float) * 2 * (n - ncopy))
        su.pfacc(sign, n, <su.complex *> &out[i, 0])
        if scale != 1.0:
            for j in range(n):
                out[i, j] = out[i, j] * scale


def _complex_transform(x, n, axis, int sign, float scale):
    source = np.asarray(x)
    a = np.moveaxis(source, axis, -1)
    if a.ndim == 0:
        raise ValueError("x must be an array")
    a = np.ascontiguousarray(a, dtype=np.complex64)
    cdef int length = a.shape[-1] if n is None else int(n)
    _check_length(length, False)
    _sign(sign)
    rows, shape = _rows(a, length)
    cdef int s = sign
    cdef float f = scale
    cdef float complex[:, ::1] ov
    cdef const float complex[:, ::1] xv
    if length == a.shape[-1] and not np.may_share_memory(a, source):
        # (the input was converted or made contiguous: that array is ours, and is transformed where it is)
        out = rows
        ov = out
        with nogil:
            _fft_inplace_rows(ov, length, s, f)
    else:
        out = np.empty((rows.shape[0], length), dtype=np.complex64)
        xv = rows
        ov = out
        with nogil:
            _fft_rows(xv, ov, length, s, f)
    return _restore(out, shape, length, axis)


def fft(x, n=None, axis=-1, int sign=1):
    """The complex Fourier transform along an axis, ``n`` samples (zero padded, or cut), not normalized. The kernel is
    ``exp(sign i w t)``"""
    return _complex_transform(x, n, axis, sign, 1.0)


def ifft(c, n=None, axis=-1, int sign=-1, bint scale=True):
    """The complex transform with the opposite default sign of ``fft``, divided by the length unless ``scale`` is false"""
    a = np.asarray(c)
    length = a.shape[axis] if n is None else int(n)
    return _complex_transform(c, n, axis, sign, (1.0 / length) if scale else 1.0)
