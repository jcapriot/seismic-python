# cython: embedsignature=True, language_level=3
# cython: linetrace=True
"""
Kirchhoff-style modelling in V(X,Z) media: SUSYNVXZ (common offset) and SUSYNVXZCS (common shot) and SUKDSYN2D (demigration) with
the library functions of the SU fork (``su_synvxz_offset``, ``su_synvxzcs_shot``, ``su_kdsyn2d_*``, in ``su/main/synthetics_waveforms_testpatterns``).
"""
from .. cimport container as spyc
from .. cimport su
import math
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sqrtf
cimport cython


cdef void _make_the_table() noexcept:
    # addsinc makes a table of sinc coefficients (a file level static variable) the first time that it is called, which is not
    # safe to do from several threads at once. Once made it is only read, so make it once, while this module is being imported.
    su.su_addsinc_table()


_make_the_table()

DEFAULT_REFLECTORS = [[1.0, (1.0, 2.0), (4.0, 2.0)]]  # (SU's ref="1:1,2;4,2")


cdef class _Scene:
    """The reflectors (cut into diffracting segments), the Ricker wavelet and the half-derivative filter that the programs share"""
    cdef:
        su.Wavelet *w
        su.Reflector *r
        float *hd
        public int nr, lhd, nhd, nsegments

    def __cinit__(self):
        self.w = NULL
        self.r = NULL
        self.hd = NULL
        self.nr = 0

    def __dealloc__(self):
        cdef int i
        if self.w is not NULL:
            free(self.w.wv)
            free(self.w)
            self.w = NULL
        if self.r is not NULL:
            for i in range(self.nr):
                free(self.r[i].rs)
            free(self.r)
            self.r = NULL
        if self.hd is not NULL:
            free(self.hd)
            self.hd = NULL

    def __init__(self, reflectors, bint smooth, float dsmax, float fpeak, float dt):
        cdef:
            int ir, i_s, n_segments
            float *ar
            float **xr
            float **zr
            int *nxz
        if reflectors is False:
            # (no reflectors: only the wavelet and the filter, as kdsyn2d needs)
            self.nr = 0
            self.nsegments = 0
            su.makericker(fpeak, dt, &self.w)
            self.lhd = 20
            self.nhd = 1 + 2 * self.lhd
            self.hd = <float *> malloc(sizeof(float) * self.nhd)
            su.mkhdiff(dt, self.lhd, self.hd)
            return
        if reflectors is None:
            reflectors = DEFAULT_REFLECTORS
        self.nr = len(reflectors)
        if self.nr == 0:
            raise ValueError("At least one reflector is needed.")
        ar = <float *> malloc(sizeof(float) * self.nr)
        xr = <float **> malloc(sizeof(float*) * self.nr)
        zr = <float **> malloc(sizeof(float*) * self.nr)
        nxz = <int *> malloc(sizeof(int) * self.nr)
        for ir, ref in enumerate(reflectors):
            ref = list(ref)
            if not isinstance(ref[0], (tuple, list)):
                ar[ir] = ref[0]
                ref = ref[1:]
            else:
                ar[ir] = 1.0
            n_segments = len(ref)
            nxz[ir] = n_segments
            xr[ir] = <float *> malloc(sizeof(float) * n_segments)
            zr[ir] = <float *> malloc(sizeof(float) * n_segments)
            for i_s, (x, z) in enumerate(ref):
                xr[ir][i_s] = x
                zr[ir][i_s] = z
        if not smooth:
            su.breakReflectors(&self.nr, &ar, &nxz, &xr, &zr)
        # (deallocates ar, nxz, xr and zr, and allocates r)
        su.makeref(dsmax, self.nr, ar, nxz, xr, zr, &self.r)
        self.nsegments = 0
        for ir in range(self.nr):
            self.nsegments += self.r[ir].ns
        su.makericker(fpeak, dt, &self.w)
        # (SU's LHD = 20, NHD = 1 + 2 * LHD)
        self.lhd = 20
        self.nhd = 1 + 2 * self.lhd
        self.hd = <float *> malloc(sizeof(float) * self.nhd)
        su.mkhdiff(dt, self.lhd, self.hd)


def _vel_nx_nz(vel):
    """The velocity model as float32 [nx][nz] (z the fast axis, as SU has it), from the array (nz, nx) that is given"""
    v = np.asarray(vel)
    if v.ndim != 2:
        raise ValueError("The velocity must be a 2-D array (nz, nx).")
    if v.shape[0] < 2 or v.shape[1] < 2:
        raise ValueError("The velocity needs at least 2 samples in x and in z.")
    return np.ascontiguousarray(v.T, dtype=np.float32)


def _offsets(nxo, dxo, fxo, xo):
    if xo is not None:
        return np.ascontiguousarray(xo, dtype=np.float32)
    return (fxo + np.arange(nxo) * dxo).astype(np.float32)


cdef class synvxz(spyc.BaseTraceIterator):
    """SUSYNVXZ: synthetic seismograms of common-offset sections in V(X,Z) media, by Kirchhoff-style modelling (Bleistein, Geophysics
    34, 686-703, formula 58).

    ``vel`` is the velocity model, an array (nz, nx) on the grid ``fx + ix * dx``, ``iz * dz``. The traces are made for each of
    the offsets in turn (``nxo``, ``dxo`` and ``fxo``, or the list ``xo``; they are signed), for the ``nxm`` midpoints ``fxm + ixm *
    dxm``; the sources and the receivers have to be inside the grid. The traveltimes and amplitudes are calculated by finite
    differences in a band of ``nxb`` grid samples (the default is nx) about the midpoint, for one of every ``nxd`` midpoints, and
    interpolated for the others. ``reflectors`` is a list of ``[amplitude, (x1, z1), (x2, z2), ...]`` (the amplitude is optional;
    ``smooth`` makes them piecewise cubic splines), cut into segments of up to ``ndpfz`` for each Fresnel zone of the wavelet of
    peak frequency ``fpeak`` (the default is 0.2 / dt) at the time ``tmin``. The default reflector is that of SU's ref="1:1,2;4,2".
    ``ls`` makes line-source amplitudes.

    The traces have ``tx_loc[0]``, ``rx_loc[0]`` set, ``ensemble_number`` (the midpoint) and ``ensemble_trace_number`` (the offset).
    """
    cdef:
        _Scene scene
        float[:, ::1] vel
        float[::1] xo
        float[:, ::1] rows
        int nx, nz, nxb, nxd, nxm, nt, ls, ixo, ixm, tracl
        float dx, dz, fx, dt, ft, dxm, fxm

    def __init__(self, vel, *, float dx=100.0, float dz=100.0, float fx=0.0, nxb=None, int nxd=1,
                 int nt=101, float dt=0.04, float ft=0.0,
                 int nxo=1, float dxo=50.0, float fxo=0.0, xo=None,
                 int nxm=101, float dxm=50.0, float fxm=0.0,
                 fpeak=None, reflectors=None, bint smooth=False, bint ls=False, tmin=None, int ndpfz=5):
        cdef float fpeak_c, tmin_c, dsmax
        self.vel = _vel_nx_nz(vel)
        self.nx = self.vel.shape[0]
        self.nz = self.vel.shape[1]
        self.nxb = self.nx if nxb is None else nxb
        if self.nxb < 1 or nxd < 1 or nxm < 1 or nt < 1 or ndpfz < 1:
            raise ValueError("nxb, nxd, nxm, nt and ndpfz must be at least 1.")
        self.nxd = nxd
        self.nxm = nxm
        self.nt = nt
        self.ls = ls
        self.dx = dx
        self.dz = dz
        self.fx = fx
        self.dt = dt
        self.ft = ft
        self.dxm = dxm
        self.fxm = fxm
        self.xo = _offsets(nxo, dxo, fxo, xo)
        if self.xo.shape[0] == 0:
            raise ValueError("At least one offset is needed.")

        # the ranges of the shots and receivers, and the band
        cdef float ex = fx + (self.nx - 1) * dx
        for x_o in self.xo:
            xs = fxm + np.arange(nxm, dtype=np.float32) * dxm - 0.5 * x_o
            xg = xs + x_o
            if (xs < fx).any() or (xs > ex).any() or (xg < fx).any() or (xg > ex).any():
                raise ValueError("The shot or receiver lie outside of the specified (x,z) grid.")
            if abs(x_o) > self.nxb * dx:
                raise ValueError("The band nxb is too small for the offset.")

        fpeak_c = 0.2 / dt if fpeak is None else fpeak
        tmin_c = 10.0 * dt if tmin is None else tmin
        tmin_c = max(tmin_c, max(ft, dt))
        # (SU takes the velocity of the first sample as the minimum)
        dsmax = self.vel[0, 0] / (2 * ndpfz) * sqrtf(tmin_c / fpeak_c)
        self.scene = _Scene(reflectors, smooth, dsmax, fpeak_c, dt)
        self.rows = np.zeros((nxm, nt), dtype=np.float32)

        self.ixo = 0
        self.ixm = nxm  # (the first trace makes the first offset)
        self.tracl = 0
        self.hdr.n_traces = self.xo.shape[0] * nxm
        self.hdr.ensemble_type = spyc.EnsembleType.common_offset
        self.hdr.uniform_traces = True

    @property
    def n_segments(self):
        """The number of small reflecting segments"""
        return self.scene.nsegments

    cdef int _make_offset(self, int iox) except -1:
        cdef:
            int status
            _Scene sc = self.scene
        with nogil:
            status = su.su_synvxz_offset(
                &self.vel[0, 0], self.nx, self.nz, self.dx, self.dz, self.fx,
                self.nxb, self.nxd, self.xo[iox], self.nxm, self.dxm, self.fxm,
                self.ls, sc.w, sc.nr, sc.r, self.nt, self.dt, self.ft,
                sc.lhd, sc.nhd, sc.hd, &self.rows[0, 0]
            )
        if status == -1:
            raise ValueError("The shot or receiver lie outside of the specified (x,z) grid.")
        if status == -2:
            raise ValueError("The band nxb is too small for the offset.")
        if status != 0:
            raise ValueError("The parameters are not allowed.")
        return 0

    @cython.boundscheck(False)
    cdef spyc.Trace next_trace(self):
        if self.ixm == self.nxm:
            # the traces of the next offset
            if self.tracl > 0:
                self.ixo += 1
            if self.ixo == self.xo.shape[0]:
                raise StopIteration()
            self._make_offset(self.ixo)
            self.ixm = 0

        cdef:
            float xo = self.xo[self.ixo]
            float xm = self.fxm + self.ixm * self.dxm
            float xs = xm - 0.5 * xo
            float[::1] data = spyc.alloc_data(self.nt)
            spyc.spy_trace_header * hdr
        data[:] = self.rows[self.ixm]
        hdr = spyc.new_hdr(self.nt)
        hdr.trace_id = self.tracl
        hdr.coord_unit = spyc.CoordinateUnit.length
        hdr.trace_type = 1  # seismic data
        hdr.d_sample = self.dt
        hdr.sample_start = self.ft
        hdr.tx_loc[0] = xs
        hdr.rx_loc[0] = xs + xo
        hdr.ensemble_number = 1 + self.ixm
        hdr.ensemble_trace_number = 1 + self.ixo
        self.ixm += 1
        self.tracl += 1
        return spyc.Trace.from_trace(hdr, data, True)


cdef class synvxzcs(spyc.BaseTraceIterator):
    """SUSYNVXZCS: synthetic seismograms of common-shot gathers in V(X,Z) media, by Kirchhoff-style modelling (Bleistein, Geophysics
    34, 686-703, formula 58).

    ``vel`` is the velocity model, an array (nz, nx) on the grid ``fx + ix * dx``, ``iz * dz``. There are ``nxs`` shots ``fxs + ixs
    * dxs``, each recorded by ``nxg`` receivers ``fxg + ixg * dxg`` (the spread rolls along with the shot if ``cable``, and is
    fixed otherwise); they have to be inside the grid. The traveltimes and amplitudes are calculated by finite differences in a band
    of ``nxb`` grid samples (the default is nx // 2) about the receiver, for one of every ``nxd`` receivers, and interpolated for
    the others. The eikonal equation fails if a ray turns: smooth the velocity, reduce ``nxb``, or replace the velocity in the two
    upper corners (``nxc`` samples wide and ``nzc`` high) by extrapolation. The traveltimes are then corrected for the slowness
    perturbation ``pert``, an array (nz, nx) like the velocity, if there is one. ``reflectors``, ``smooth``, ``ls``, ``fpeak``,
    ``tmin`` and ``ndpfz`` are those of synvxz.

    The traces have ``tx_loc[0]``, ``rx_loc[0]`` set, ``ensemble_number`` (the shot) and ``ensemble_trace_number`` (the receiver).
    """
    cdef:
        _Scene scene
        float[:, ::1] vel
        float[:, ::1] pert
        float[:, ::1] rows
        bint has_pert, cable
        int nx, nz, nxb, nxd, nxc, nzc, nxs, nxg, nt, ls, ixs, ixg, tracl
        float dx, dz, fx, dt, ft, dxg, fxg, dxs, fxs

    def __init__(self, vel, *, float dx=50.0, float dz=50.0, float fx=0.0, nxb=None, int nxd=5, int nxc=0, int nzc=0, pert=None,
                 int nt=501, float dt=0.004, float ft=0.0,
                 int nxg=101, float dxg=15.0, float fxg=0.0, int nxs=1, float dxs=50.0, float fxs=0.0, bint cable=True,
                 fpeak=None, reflectors=None, bint smooth=False, bint ls=False, tmin=None, int ndpfz=5):
        cdef float fpeak_c, tmin_c, dsmax
        self.vel = _vel_nx_nz(vel)
        self.nx = self.vel.shape[0]
        self.nz = self.vel.shape[1]
        self.has_pert = pert is not None
        if self.has_pert:
            p = _vel_nx_nz(pert)
            if p.shape != (self.nx, self.nz):
                raise ValueError("The slowness perturbation must have the shape of the velocity.")
            self.pert = p
        self.nxb = self.nx // 2 if nxb is None else nxb
        if self.nxb < 1 or nxd < 1 or nxg < 1 or nxs < 1 or nt < 1 or ndpfz < 1 or nxc < 0 or nzc < 0:
            raise ValueError("nxb, nxd, nxg, nxs, nt and ndpfz must be at least 1, and nxc and nzc not negative.")
        self.nxd = nxd
        self.nxc = min(nxc, self.nxb)
        self.nzc = min(nzc, self.nz)
        self.nxg = nxg
        self.nxs = nxs
        self.nt = nt
        self.ls = ls
        self.cable = cable
        self.dx = dx
        self.dz = dz
        self.fx = fx
        self.dt = dt
        self.ft = ft
        self.dxg = dxg
        self.fxg = fxg
        self.dxs = dxs
        self.fxs = fxs

        # the ranges of the shots and receivers
        cdef float ex = fx + (self.nx - 1) * dx
        shots = fxs + np.arange(nxs, dtype=np.float32) * dxs
        if (shots < fx).any() or (shots > ex).any():
            raise ValueError("A shot lies outside of the specified (x,z) grid.")
        for ixs in range(nxs):
            g = (fxg + (ixs * dxs if cable else 0.0)) + np.arange(nxg, dtype=np.float32) * dxg
            if (g < fx).any() or (g > ex).any():
                raise ValueError("A receiver lies outside of the specified (x,z) grid.")

        fpeak_c = 0.2 / dt if fpeak is None else fpeak
        tmin_c = 10.0 * dt if tmin is None else tmin
        tmin_c = max(tmin_c, max(ft, dt))
        dsmax = self.vel[0, 0] / (2 * ndpfz) * sqrtf(tmin_c / fpeak_c)
        self.scene = _Scene(reflectors, smooth, dsmax, fpeak_c, dt)
        self.rows = np.zeros((nxg, nt), dtype=np.float32)

        self.ixs = 0
        self.ixg = nxg  # (the first trace makes the first shot)
        self.tracl = 0
        self.hdr.n_traces = nxs * nxg
        self.hdr.ensemble_type = spyc.EnsembleType.tx_gather
        self.hdr.uniform_traces = True

    @property
    def n_segments(self):
        """The number of small reflecting segments"""
        return self.scene.nsegments

    cdef int _make_shot(self, int ixs) except -1:
        cdef:
            int status
            _Scene sc = self.scene
            float xs = self.fxs + ixs * self.dxs
            float gx0 = self.fxg + (ixs * self.dxs if self.cable else 0.0)
            const float *pert = NULL
        if self.has_pert:
            pert = &self.pert[0, 0]
        with nogil:
            status = su.su_synvxzcs_shot(
                &self.vel[0, 0], pert, self.nx, self.nz, self.dx, self.dz, self.fx,
                self.nxb, self.nxd, self.nxc, self.nzc, xs, self.nxg, self.dxg, gx0,
                self.ls, sc.w, sc.nr, sc.r, self.nt, self.dt, self.ft,
                sc.lhd, sc.nhd, sc.hd, &self.rows[0, 0]
            )
        if status == -1:
            raise ValueError("The shot or a receiver lie outside of the specified (x,z) grid.")
        if status != 0:
            raise ValueError("The parameters are not allowed.")
        return 0

    @cython.boundscheck(False)
    cdef spyc.Trace next_trace(self):
        if self.ixg == self.nxg:
            # the traces of the next shot
            if self.tracl > 0:
                self.ixs += 1
            if self.ixs == self.nxs:
                raise StopIteration()
            self._make_shot(self.ixs)
            self.ixg = 0

        cdef:
            float xs = self.fxs + self.ixs * self.dxs
            float xg = self.fxg + (self.ixs * self.dxs if self.cable else 0.0) + self.ixg * self.dxg
            float[::1] data = spyc.alloc_data(self.nt)
            spyc.spy_trace_header * hdr
        data[:] = self.rows[self.ixg]
        hdr = spyc.new_hdr(self.nt)
        hdr.trace_id = self.tracl
        hdr.coord_unit = spyc.CoordinateUnit.length
        hdr.trace_type = 1  # seismic data
        hdr.d_sample = self.dt
        hdr.sample_start = self.ft
        hdr.tx_loc[0] = xs
        hdr.rx_loc[0] = xg
        hdr.ensemble_number = 1 + self.ixs
        hdr.ensemble_trace_number = 1 + self.ixg
        self.ixg += 1
        self.tracl += 1
        return spyc.Trace.from_trace(hdr, data, True)


cdef class kdsyn2d(spyc.BaseTraceIterator):
    """SUKDSYN2D: Kirchhoff depth synthesis of 2-D seismic data from a migrated section (demigration), in common-shot gathers.

    ``mig`` is the migrated section, an array (nz, nx) on the grid ``fz + iz * dz``, ``fx + ix * dx``. ``ttab`` are the traveltime
    tables of a background velocity model (from SU's rayt2d, for example): an array (ns, nzt, nxt) of the traveltimes from the ``ns``
    sources ``fs + is * ds`` on the grid ``fzt + izt * dzt``, ``fxt + ixt * dxt``; the migrated section has to be inside the tables,
    and the shots and receivers inside the range of the sources (which needs two or more). They are interpolated to the positions of
    the shots and receivers. There are ``nxs`` shots ``fxs + ixs * dxs``, each recorded by ``nxo`` receivers at the offsets ``fxo +
    ixo * dxo`` (``nxo=1`` is a common-offset section). The anti-aliasing filter takes ``fmax`` (the default is 1 / (4 dt)) as the
    maximum frequency of the section, ``aperx`` (the default is half of the width of the tables) is the lateral aperture of the
    modelling, ``angmax`` its angle from the vertical (degrees), ``v0`` the reference velocity at the surface and ``dvz`` its
    vertical gradient (the reference traveltimes are taken out of the tables, and put back as they are modelled). ``ls`` is the
    line-source flag.

    The defaults are those of the documentation of the program (which has other ones in the code for dxo and dxs: 50 and 15).

    The traces have ``tx_loc[0]``, ``rx_loc[0]`` set, ``ensemble_number`` (the shot) and ``ensemble_trace_number`` (the offset).
    """
    cdef:
        _Scene scene
        float[:, ::1] mig, migi
        float[:, :, ::1] ttab
        float[:, ::1] tb, pb, sigb, cosb, tsum, tt
        int nx, nz, nxt, nzt, ns, nr, nt, nxo, nxs, mzmax, ls, ixs, ixo, tracl
        float dx, dz, fx, fz, dxt, dzt, fxt, fzt, fs, ds, dt, ft, dxo, fxo, dxs, fxs
        float aperx, angmax, v0, fmax

    def __init__(self, mig, ttab, *, float dx, float dz, float dxt, float dzt, float fs, float ds,
                 float fx=0.0, float fz=0.0, float fxt=0.0, float fzt=0.0,
                 int nt=501, float dt=0.004, float ft=0.0,
                 int nxo=1, float dxo=25.0, float fxo=0.0, int nxs=101, float dxs=25.0, float fxs=0.0,
                 fmax=None, aperx=None, float angmax=60.0, float v0=1500.0, float dvz=0.0, bint ls=True):
        cdef:
            float ext, ezt, es, ex, ez, offmax, rmax, xs, xg
            int ixs_, ixo_
        self.mig = _vel_nx_nz(mig)
        self.nx = self.mig.shape[0]
        self.nz = self.mig.shape[1]
        t = np.asarray(ttab)
        if t.ndim != 3:
            raise ValueError("The traveltime tables must be a 3-D array (ns, nzt, nxt).")
        self.ns, self.nzt, self.nxt = t.shape
        if self.ns < 2 or self.nzt < 2 or self.nxt < 2:
            raise ValueError("The traveltime tables need at least two sources and two samples in x and in z.")
        # [ns][nxt][nzt], and a copy, as the reference traveltimes are taken out of it
        self.ttab = np.ascontiguousarray(np.transpose(t, (0, 2, 1)), dtype=np.float32).copy()
        if nt < 2 or dt <= 0.0 or dx <= 0.0 or dz <= 0.0 or dxt <= 0.0 or dzt <= 0.0 or ds <= 0.0 or nxo < 1 or nxs < 1:
            raise ValueError("nt, dt, the spacings, nxo and nxs have to be positive.")
        if angmax < 0.00001:
            raise ValueError("angmax must be positive!")
        self.nt = nt
        self.nxo = nxo
        self.nxs = nxs
        self.dx = dx
        self.dz = dz
        self.fx = fx
        self.fz = fz
        self.dxt = dxt
        self.dzt = dzt
        self.fxt = fxt
        self.fzt = fzt
        self.fs = fs
        self.ds = ds
        self.dt = dt
        self.ft = ft
        self.dxo = dxo
        self.fxo = fxo
        self.dxs = dxs
        self.fxs = fxs
        self.ls = ls
        self.v0 = v0
        self.angmax = angmax
        self.fmax = 1.0 / (4 * dt) if fmax is None else fmax

        ext = fxt + (self.nxt - 1) * dxt
        ezt = fzt + (self.nzt - 1) * dzt
        es = fs + (self.ns - 1) * ds
        ex = fx + (self.nx - 1) * dx
        ez = fz + (self.nz - 1) * dz
        # the ranges of the shots and receivers
        for ixs_ in range(nxs):
            xs = fxs + ixs_ * dxs
            for ixo_ in range(nxo):
                xg = xs + fxo + ixo_ * dxo
                if fs > xs or es < xs or fs > xg or es < xg:
                    raise ValueError("A shot or receiver lies outside of the specified (x,z) grid.")
        if fxt > fx or ext < ex or fzt > fz or ezt < ez:
            raise ValueError("The migration section is out of the traveltime table!")

        self.mzmax = <int> (dx * math.sin(angmax * math.pi / 180.0) / dz)
        if self.mzmax < 1:
            self.mzmax = 1
        self.aperx = 0.5 * self.nxt * dxt if aperx is None else aperx

        # the reference traveltime and slowness
        offmax = max(abs(fxo), abs(fxo + (nxo - 1) * dxo))
        rmax = max(es - fxt, ext - fs)
        rmax = min(rmax, 0.5 * offmax + self.aperx)
        self.nr = 2 + <int> (rmax / dx)
        self.tb = np.zeros((self.nr, self.nzt), dtype=np.float32)
        self.pb = np.zeros((self.nr, self.nzt), dtype=np.float32)
        self.sigb = np.zeros((self.nr, self.nzt), dtype=np.float32)
        self.cosb = np.zeros((self.nr, self.nzt), dtype=np.float32)
        su.su_kdsyn2d_reference(self.nr, self.nzt, dx, dzt, fzt, dvz, v0,
                                &self.tb[0, 0], &self.pb[0, 0], &self.sigb[0, 0], &self.cosb[0, 0])
        # the residual traveltimes
        su.su_kdsyn2d_residual(self.ns, fs, ds, self.nxt, fxt, dxt, self.nzt, self.nr, dx, &self.tb[0, 0], &self.ttab[0, 0, 0])
        # the integrated section, for the anti-aliasing filter
        self.migi = np.zeros((self.nx, self.nz + 2 * self.mzmax), dtype=np.float32)
        su.su_kdsyn2d_integrate(&self.mig[0, 0], self.nz, dz, self.nx, self.mzmax, &self.migi[0, 0])

        self.tsum = np.zeros((self.nxt, self.nzt), dtype=np.float32)
        self.tt = np.zeros((self.nxt, self.nzt), dtype=np.float32)
        # (SU's wavelet has the peak frequency 0.2 / dt)
        self.scene = _Scene(False, False, 0.0, 0.2 / dt, dt)

        self.ixs = 0
        self.ixo = 0
        self.tracl = 0
        self.hdr.n_traces = nxs * nxo
        self.hdr.ensemble_type = spyc.EnsembleType.tx_gather
        self.hdr.uniform_traces = True

    @cython.boundscheck(False)
    cdef spyc.Trace next_trace(self):
        if self.ixs == self.nxs:
            raise StopIteration()
        cdef:
            float xo = self.fxo + self.ixo * self.dxo
            float xs = self.fxs + self.ixs * self.dxs
            float xg = xs + xo
            float[::1] data = spyc.alloc_data(self.nt)
            _Scene sc = self.scene
            int status
            spyc.spy_trace_header * hdr
        with nogil:
            status = su.su_kdsyn2d_trace(
                &data[0], self.nt, self.ft, self.dt, xs, xg,
                &self.mig[0, 0], &self.migi[0, 0], self.aperx,
                self.nx, self.fx, self.dx, self.nz, self.fz, self.dz,
                self.mzmax, self.ls, self.angmax, self.v0, self.fmax, sc.w,
                &self.tb[0, 0], &self.pb[0, 0], &self.sigb[0, 0], &self.cosb[0, 0], self.nr,
                &self.ttab[0, 0, 0], self.ns, self.fs, self.ds, &self.tsum[0, 0], &self.tt[0, 0],
                self.nxt, self.fxt, self.dxt, self.nzt, self.fzt, self.dzt,
                sc.lhd, sc.nhd, sc.hd
            )
        if status != 0:
            raise ValueError("The parameters are not allowed.")
        hdr = spyc.new_hdr(self.nt)
        hdr.trace_id = self.tracl
        hdr.coord_unit = spyc.CoordinateUnit.length
        hdr.trace_type = 1  # seismic data
        hdr.d_sample = self.dt
        hdr.sample_start = self.ft
        hdr.tx_loc[0] = xs
        hdr.rx_loc[0] = xg
        hdr.ensemble_number = 1 + self.ixs
        hdr.ensemble_trace_number = 1 + self.ixo
        self.ixo += 1
        self.tracl += 1
        if self.ixo == self.nxo:
            self.ixo = 0
            self.ixs += 1
        return spyc.Trace.from_trace(hdr, data, True)
