# cython: embedsignature=True, language_level=3
# cython: linetrace=True
from .. cimport container as spyc
from .. cimport su
import numpy as np
from libc.stdlib cimport malloc, free
from libc.float cimport FLT_MAX
from libc.math cimport sqrtf, fabs
cimport cython

cdef class synlv(spyc.BaseTraceIterator):
    cdef:
        bint shots, ls, er, ob
        int ns, nr, nxo, nt

        float v00, dvdx, dvdz, ft, dt
        float[::1] ref_points
        float[::1] xo

        # iterator indices
        int ixsm, ixo, tracl
        int lhd, nhd

        su.Wavelet *w
        su.Reflector *r
        float *hd_filt

    def __dealloc__(self):
        # clear the memory used by wavelet and reflector objects
        if self.w is not NULL:
            free(self.w.wv)
            free(self.w)
            self.w = NULL
        if self.r is not NULL:
            for i in range(self.nr):
                free(self.r[i].rs)
            free(self.r)
            self.r = NULL
        if self.hd_filt is not NULL:
            free(self.hd_filt)
            self.hd_filt = NULL

    def __init__(
            self,
            int nt=101, float dt=0.04, float ft=0.0,
            int nxo=1, float dxo=0.05, float fxo=0.0, xo=None,
            int nxm=101, float dxm=0.05, float fxm=0.0,
            int nxs=0, float dxs=0.05, float fxs=0.0,
            float x0=0.0, float z0=0.0, float v00=2.0, float dvdx=0.0, float dvdz=0.0,
            fpeak=None, list reflectors=None,
            bint smooth=False, bint er=False, bint ls=False, int ob=True,
            tmin=None, int ndpfz=5, bint verbose=False,
    ):

        if tmin is None:
            tmin = 10.0 * dt
        if fpeak is None:
            fpeak = 0.2 / dt

        cdef float tmin_c = tmin
        cdef float fpeak_c = fpeak

        self.nt = nt

        # options:
        self.ls = ls
        self.er = er
        self.ob = ob

        self.ft = ft
        self.dt = dt

        self.shots = bool(nxs)
        midpoints = bool(nxm)
        if self.shots and midpoints:
            raise TypeError("Cannot specify both shot and midpoint sampling!")
        elif not self.shots and not midpoints:
            raise TypeError("Must specify one of shot or midpoint samplings!")
        if self.shots:
            self.ns = nxs
        else:
            self.ns = nxm


        if xo is None:
            self.xo = np.empty(nxo, dtype=np.float32)
            for ixo in range(nxo):
                self.xo[ixo] = fxo + ixo * dxo
        else:
            self.xo = np.require(xo, dtype=np.float32, flags='C')
        self.nxo = self.xo.shape[0]

        self.ref_points = np.empty(self.ns, dtype=np.float32)
        if self.shots:
            for ixsm in range(self.ns):
                self.ref_points[ixsm] = fxs + ixsm * dxs
        else:
            for ixsm in range(self.ns):
                self.ref_points[ixsm] = fxm + ixsm * dxm  # are actually the midpoints

        if reflectors is None:
            reflectors = [[1, (1, 4), (2, 2)]]
        # decode reflectors
        self.nr = len(reflectors)
        cdef:
            float *ar = <float *> malloc(sizeof(float) * self.nr)
            float **xr = <float **> malloc(sizeof(float*) * self.nr)
            float **zr = <float **> malloc(sizeof(float*) * self.nr)
            int *nxz = <int *> malloc(sizeof(int) * self.nr)
        for ir, ref in enumerate(reflectors):
            if not isinstance(ref[0], tuple):
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

        self.dvdx = dvdx
        self.dvdz = dvdz

        self.v00 = v00 - (dvdx * x0 + dvdz * z0)

        # determine minimum velocity and minimum reflection time
        cdef:
            float vmin = FLT_MAX
            float tminr = FLT_MAX
        for ir in range(self.nr):
            for ixz in range(nxz[ir]):
                x = xr[ir][ixz]
                z = zr[ir][ixz]
                v = v00 + dvdx * x + dvdz * z
                if v<vmin: vmin = v
                t = 2.0 * z / v
                if t<tmin_c: tminr = t

        # determine maximum reflector segment length
        tmin_c = max(tmin_c,max(ft,dt))
        cdef float dsmax = vmin/(2 * ndpfz) * sqrtf(tmin_c/fpeak_c)


        # will deallocate ar, nxz, xr, and zr
        # and allocate r
        su.makeref(dsmax, self.nr, ar, nxz, xr, zr, &self.r)

        # will allocate w
        su.makericker(fpeak_c, dt, &self.w)

        #init iterators
        self.ixo = 0
        self.ixsm = 0
        self.tracl = 0

        self.hdr.n_traces = self.nxo * self.ns
        if self.shots:
            self.hdr.ensemble_type = spyc.EnsembleType.tx_gather
        else:
            self.hdr.ensemble_type = spyc.EnsembleType.common_midpoint
        self.hdr.uniform_traces = True

        # from susynlv.c:
        # LHD = 20
        # NHD = 1 + 2 * LHD
        self.lhd = 20
        self.nhd = 1  +2 * self.lhd
        self.hd_filt = <float * > malloc(sizeof(float) * self.nhd)
        su.mkhdiff(self.dt, self.lhd, self.hd_filt)

    @cython.boundscheck(False)
    cdef spyc.Trace next_trace(self):
        if self.ixsm == self.ns:
            raise StopIteration()

        # susynlv_filltrace will fill with zeros
        cdef:
            # spy_trace *tr = new_hdr(self.nt, zero_fill=False)
            # spy_trace_header *hdr = &(tr.hdr)
            float xs, xr, xo
            float z = 0.0

        xs = self.ref_points[self.ixsm]
        xo = self.xo[self.ixo]
        if not self.shots:
            xs -= 0.5 * xo

        xr = xs + xo

        cdef float[::1] data = <float[:self.nt]> malloc(self.nt * sizeof(float))
        su.su_synlv(
            &data[0],
            xs, z, xr, z,
            self.nt, self.dt, self.ft,
            self.v00, self.dvdx, self.dvdz,
            self.ls, self.er, self.ob,
            self.w, self.nr, self.r,
            self.lhd, self.nhd, self.hd_filt
        )

        cdef spyc.spy_trace_header * hdr = spyc.new_hdr(self.nt)

        hdr.line_id = 1
        hdr.trace_id = self.tracl
        hdr.d_sample = self.dt
        hdr.sample_start = self.ft

        hdr.tx_loc[0] = xs
        hdr.rx_loc[0] = xr
        hdr.mid_point[0] = 0.5 * (xs + xr)
        hdr.offset = fabs(xo)
        hdr.ensemble_number = 1 + self.ixsm
        hdr.ensemble_trace_number = 1 + self.ixo

        # post update iters
        self.ixo += 1
        self.tracl += 1
        if self.ixo == self.nxo:
            self.ixo = 0
            self.ixsm += 1
        return spyc.Trace.from_trace(hdr, data,True)