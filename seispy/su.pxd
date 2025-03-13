from posix.types cimport off_t
from libc.stdio cimport FILE, fpos_t

cdef extern from "cwp.h" nogil:
    enum cwp_Bool:
        cwp_false, cwp_true
    ctypedef char* cwp_String
    enum FileType:
        BADFILETYPE = -1,
        TTY, DISK, DIRECTORY,TAPE, PIPE, FIFO, SOCKET, SYMLINK
    struct complex:
        float r,i
    struct dcomplex:
        double r,i

    # allocate and free multi-dimensional arrays
    void *alloc1(size_t n1, size_t size)
    void *realloc1(void *v, size_t n1, size_t size)
    void ** alloc2(size_t n1, size_t n2, size_t size)
    void ** *alloc3(size_t n1, size_t n2, size_t n3, size_t size)
    void ** ** alloc4(size_t n1, size_t n2, size_t n3, size_t n4, size_t size)
    void ** ** *alloc5(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t size)
    void ** ** ** alloc6(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6, size_t size)

    void free1(void *p)
    void free2(void ** p)
    void free3(void ** *p)
    void free4(void ** ** p)
    void free5(void ** ** *p)
    void free6(void ** ** ** p)
    int *alloc1int(size_t n1)
    int *realloc1int(int *v, size_t n1)
    int ** alloc2int(size_t n1, size_t n2)
    int ** *alloc3int(size_t n1, size_t n2, size_t n3)
    float *alloc1float(size_t n1)
    float *realloc1float(float *v, size_t n1)
    float ** alloc2float(size_t n1, size_t n2)
    float ** *alloc3float(size_t n1, size_t n2, size_t n3)

    float ** ** alloc4float(size_t n1, size_t n2, size_t n3, size_t n4)
    void free4float(float ** ** p)
    float ** ** *alloc5float(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
    void free5float(float ** ** *p)
    float ** ** ** alloc6float(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t n6)
    void free6float(float ** ** ** p)
    int ** ** alloc4int(size_t n1, size_t n2, size_t n3, size_t n4)
    void free4int(int ** ** p)
    int ** ** *alloc5int(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
    void free5int(int ** ** *p)
    unsigned short ** ** ** alloc6ushort(size_t n1, size_t n2, size_t n3, size_t n4,
                                         size_t n5, size_t n6)
    unsigned char ** ** *alloc5uchar(size_t n1, size_t n2, size_t n3, size_t n4,
                                     size_t n5)
    void free5uchar(unsigned char ** ** *p)
    unsigned short ** ** *alloc5ushort(size_t n1, size_t n2, size_t n3, size_t n4,
                                       size_t n5)
    void free5ushort(unsigned short ** ** *p)
    unsigned char ** ** ** alloc6uchar(size_t n1, size_t n2, size_t n3, size_t n4,
                                       size_t n5, size_t n6)
    void free6uchar(unsigned char ** ** ** p)
    unsigned short ** ** ** alloc6ushort(size_t n1, size_t n2, size_t n3, size_t n4,
                                         size_t n5, size_t n6)
    void free6ushort(unsigned short ** ** ** p)

    double *alloc1double(size_t n1)
    double *realloc1double(double *v, size_t n1)
    double ** alloc2double(size_t n1, size_t n2)
    double ** *alloc3double(size_t n1, size_t n2, size_t n3)
    complex *alloc1complex(size_t n1)
    complex *realloc1complex(complex *v, size_t n1)
    complex ** alloc2complex(size_t n1, size_t n2)
    complex ** *alloc3complex(size_t n1, size_t n2, size_t n3)

    dcomplex *alloc1dcomplex(size_t n1)
    dcomplex *realloc1dcomplex(dcomplex *v, size_t n1)
    dcomplex ** alloc2dcomplex(size_t n1, size_t n2)
    dcomplex ** *alloc3dcomplex(size_t n1, size_t n2, size_t n3)

    void free1int(int *p)
    void free2int(int ** p)
    void free3int(int ** *p)
    void free1float(float *p)
    void free2float(float ** p)
    void free3float(float ** *p)

    void free1double(double *p)
    void free2double(double ** p)
    void free3double(double ** *p)
    void free1complex(complex *p)
    void free2complex(complex ** p)
    void free3complex(complex ** *p)

    void free1dcomplex(dcomplex *p)
    void free2dcomplex(dcomplex ** p)
    void free3dcomplex(dcomplex ** *p)

    # complex number manipulation
    complex cadd(complex a, complex b)
    complex csub(complex a, complex b)
    complex cmul(complex a, complex b)
    complex cdiv(complex a, complex b)
    float rcabs(complex z)
    complex cmplx(float re, float im)
    complex conjg(complex z)
    complex cneg(complex z)
    complex cinv(complex z)
    complex cwp_csqrt(complex z)
    complex cwp_cexp(complex z)
    complex crmul(complex a, float x)

    # complex functions
    complex cipow(complex a, int p)
    complex crpow(complex a, float p)
    complex rcpow(float a, complex p)
    complex ccpow(complex a, complex p)
    complex cwp_ccos(complex a)
    complex cwp_csin(complex a)
    complex cwp_ccosh(complex a)
    complex cwp_csinh(complex a)
    complex cwp_cexp1(complex a)
    complex cwp_clog(complex a)

    # *double complex
    dcomplex dcadd(dcomplex a, dcomplex b)
    dcomplex dcsub(dcomplex a, dcomplex b)
    dcomplex dcmul(dcomplex a, dcomplex b)
    dcomplex dcdiv(dcomplex a, dcomplex b)
    double drcabs(dcomplex z)
    dcomplex dcmplx(double re, double im)
    dcomplex dconjg(dcomplex z)
    dcomplex dcneg(dcomplex z)
    dcomplex dcinv(dcomplex z)
    dcomplex dcsqrt(dcomplex z)
    dcomplex dcexp(dcomplex z)
    dcomplex dcrmul(dcomplex a, double x)

    # complex functions
    dcomplex dcipow(dcomplex a, int p)
    dcomplex dcrpow(dcomplex a, float p)
    dcomplex rdcpow(float a, dcomplex p)
    dcomplex dcdcpow(dcomplex a, dcomplex p)
    dcomplex dccos(dcomplex a)
    dcomplex dcsin(dcomplex a)
    dcomplex dccosh(dcomplex a)
    dcomplex dcsinh(dcomplex a)
    dcomplex dcexp1(dcomplex a)
    dcomplex dclog(dcomplex a)

    void chermite(int n, float x[], float y[], float yd[][4])

    # big matrix handler
    void *bmalloc(int nbpe, int n1, int n2)
    void bmfree(void *bm)
    void bmread(void *bm, int dir, int k1, int k2, int n, void *v)
    void bmwrite(void *bm, int dir, int k1, int k2, int n, void *v)

    # interpolation
    float fsinc(float x)
    double dsinc(double x)
    void mksinc(float d, int lsinc, float sinc[])
    void ints8r(int nxin, float dxin, float fxin, float yin[],
                float yinl, float yinr, int nxout, float xout[], float yout[])
    void ints8c(int nxin, float dxin, float fxin, complex yin[],
                complex yinl, complex yinr, int nxout, float xout[], complex yout[])
    void intt8r(int ntable, float table[][8],
                int nxin, float dxin, float fxin, float yin[],
                float yinl, float yinr, int nxout, float xout[], float yout[])
    void intt8c(int ntable, float table[][8],
                int nxin, float dxin, float fxin, complex yin[],
                complex yinl, complex yinr, int nxout, float xout[], complex yout[])
    void ress8r(int nxin, float dxin, float fxin, float yin[],
                float yinl, float yinr,
                int nxout, float dxout, float fxout, float yout[])
    void ress8c(int nxin, float dxin, float fxin, complex yin[],
                complex yinl, complex yinr,
                int nxout, float dxout, float fxout, complex yout[])
    void shfs8r(float dx, int nxin, float fxin, float yin[],
                float yinl, float yinr, int nxout, float fxout, float yout[])
    void xindex(int nx, float ax[], float x, int *index)
    void intl2b(int nxin, float dxin, float fxin,
                int nyin, float dyin, float fyin, unsigned char *zin,
                int nxout, float dxout, float fxout,
                int nyout, float dyout, float fyout, unsigned char *zout)
    void intlin(int nin, float xin[], float yin[], float yinl, float yinr,
                int nout, float xout[], float yout[])
    void intcub(int ideriv, int nin, float xin[], float ydin[][4],
                int nout, float xout[], float yout[])
    void cakima(int n, float x[], float y[], float yd[][4])
    void cmonot(int n, float x[], float y[], float yd[][4])
    void csplin(int n, float x[], float y[], float yd[][4])
    void yxtoxy(int nx, float dx, float fx, float y[],
                int ny, float dy, float fy, float xylo, float xyhi, float x[])
    void intlinc(int nin, float xin[], complex yin[], complex yinl, complex yinr,
                 int nout, float xout[], complex yout[])
    void intlirr2b(int nxin, float *xin,
                   int nyin, float dyin, float fyin, unsigned char *zin,
                   int nxout, float dxout, float fxout,
                   int nyout, float dyout, float fyout, unsigned char *zout)

    void linear_regression(float *y, float *x, int n, float coeff[4])

    void linfit(float *x, float *y, int ndata, float *sig, int mwt,
                float *a, float *b, float *siga, float *sigb,
                float *chi2, float *q)

    # Butterworth filters
    void bfhighpass(int npoles, float f3db, int n, float p[], float q[])
    void bflowpass(int npoles, float f3db, int n, float p[], float q[])
    void bfdesign(float fpass, float apass, float fstop, float astop,
                  int *npoles, float *f3db)

    # differentiator approximations
    void mkdiff(int n, float a, float h, int l, int m, float d[])
    void mkhdiff(float h, int l, float d[])
    void holbergd1(float e, int n, float d[])
    void differentiate(int n, float h, float *f, float *fprime)
    void ddifferentiate(int n, double h, double *f, double *fprime)

    #general signal processing
    void convolve_cwp(int lx, int ifx, float *x, int ly, int ify, float *y,
                      int lz, int ifz, float *z)
    void xcor(int lx, int ifx, float *x, int ly, int ify, float *y,
              int lz, int ifz, float *z)
    void hilbert(int n, float x[], float y[])
    void antialias(float frac, int phase, int n, float p[], float q[])

    # max and min
    int max_index(int n, float *a, int inc)
    int min_index(int n, float *a, int inc)

    # Abel transformer
    void *abelalloc(int n)
    void abelfree(void *at)
    void abel(void *at, float f[], float g[])

    # Hankel transformer
    void *hankelalloc(int nfft)
    void hankelfree(void *ht)
    void hankel0(void *ht, float f[], float h[])
    void hankel1(void *ht, float f[], float h[])

    # Hartley transforms
    void srfht(int *n, int *m, float *f)
    void r4fht(int n, int m, float *f)
    int nextpow2(int n)
    int nextpow4(int n)

    # Hartley transforms(double precision)
    void dsrfht(int *n, int *m, double *f)

    # sorting and searching
    void hpsort(int n, float a[])
    void qksort(int n, float a[])
    void qkfind(int m, int n, float a[])
    void qkisort(int n, float a[], int i[])
    void qkifind(int m, int n, float a[], int i[])

    # statistics
    float quest(float p, int n, float x[])
    void *questalloc(float p, int n, float x[])
    float questupdate(void *q, int n, float x[])
    void questfree(void *q)

    # PC byte swapping
    void swap_short_2(short *tni2)
    void swap_u_short_2(unsigned short *tni2)
    void swap_int_4(int *tni4)
    void swap_u_int_4(unsigned int *tni4)
    void swap_long_4(long *tni4)
    void swap_u_long_4(unsigned long *tni4)
    void swap_float_4(float *tnf4)
    void swap_double_8(double *tndd8)

    # Phase unwrapping
    void oppenheim_unwrap_phase(int n, int trend, int zeromean,
                                float df, float *xr, float *xi, float *phase)
    void simple_unwrap_phase(int n, int trend, int zeromean, float w,
                             float *phase)

    # Prime Factor FFTs
    int npfa(int nmin)
    int npfao(int nmin, int nmax)
    int npfar(int nmin)
    int npfaro(int nmin, int nmax)
    void pfacc(int isign, int n, complex z[])
    void pfarc(int isign, int n, float rz[], complex cz[])
    void pfacr(int isign, int n, complex cz[], float rz[])
    void pfa2cc(int isign, int idim, int n1, int n2, complex z[])
    void pfa2rc(int isign, int idim, int n1, int n2, float rz[], complex cz[])
    void pfa2cr(int isign, int idim, int n1, int n2, complex cz[], float rz[])
    void pfamcc(int isign, int n, int nt, int k, int kt, complex z[])

    # Prime Factor FFTs(double version)
    int npfa_d(int nmin)
    int npfao_d(int nmin, int nmax)
    int npfar_d(int nmin)
    int npfaro_d(int nmin, int nmax)
    void pfacc_d(int isign, int n, dcomplex z[])
    void pfacr_d(int isign, int n, dcomplex cz[], double rz[])
    void pfarc_d(int isign, int n, double rz[], dcomplex cz[])
    void pfamcc_d(int isign, int n, int nt, int k, int kt, dcomplex z[])
    void pfa2cc_d(int isign, int idim, int n1, int n2, dcomplex z[])
    void pfa2cr_d(int isign, int idim, int n1, int n2, dcomplex cz[],
                  double rz[])
    void pfa2rc_d(int isign, int idim, int n1, int n2, double rz[],
                  dcomplex cz[])

    # BLAS(Basic Linear Algebra Subroutines adapted from LINPACK FORTRAN)
    int isamax(int n, float *sx, int incx)
    float sasum(int n, float *sx, int incx)
    void saxpy(int n, float sa, float *sx, int incx, float *sy, int incy)
    void scopy(int n, float *sx, int incx, float *sy, int incy)
    float sdot(int n, float *sx, int incx, float *sy, int incy)
    float snrm2(int n, float *sx, int incx)
    void sscal(int n, float sa, float *sx, int incx)
    void sswap(int n, float *sx, int incx, float *sy, int incy)
    int idamax(int n, double *sx, int incx)
    double dasum(int n, double *sx, int incx)
    void daxpy(int n, double sa, double *sx, int incx, double *sy, int incy)
    void dcopy(int n, double *sx, int incx, double *sy, int incy)
    double ddot(int n, double *sx, int incx, double *sy, int incy)
    double dnrm2(int n, double *sx, int incx)
    void dscal(int n, double sa, double *sx, int incx)
    void dswap(int n, double *sx, int incx, double *sy, int incy)

    # LINPACK functions(adapted from LINPACK FORTRAN)
    void sgeco(float ** a, int n, int *ipvt, float *rcond, float *z)
    void sgefa(float ** a, int n, int *ipvt, int *info)
    void sgesl(float ** a, int n, int *ipvt, float *b, int job)
    void sqrdc(float ** x, int n, int p, float *qraux, int *jpvt,
               float *work, int job)
    void sqrsl(float ** x, int n, int k, float *qraux,
               float *y, float *qy, float *qty,
               float *b, float *rsd, float *xb, int job, int *info)
    void sqrst(float ** x, int n, int p, float *y, float tol,
               float *b, float *rsd, int *k,
               int *jpvt, float *qraux, float *work)
    void dgeco(double ** a, int n, int *ipvt, double *rcond, double *z)
    void dgefa(double ** a, int n, int *ipvt, int *info)
    void dgesl(double ** a, int n, int *ipvt, double *b, int job)

    # other linear system solvers
    void stoepd(int n, double r[], double g[], double f[], double a[])
    void stoepf(int n, float r[], float g[], float f[], float a[])
    void vanded(int n, double v[], double b[], double x[])
    void vandef(int n, float v[], float b[], float x[])
    void tridif(int n, float a[], float b[], float c[], float r[], float u[])
    void tridid(int n, double a[], double b[], double c[], double r[], double u[])
    void tripd(float *d, float *e, float *b, int n)
    void tripp(int n, float *d, float *e, float *c, float *b)

    # root finding
    int mnewt(int maxiter, float ftol, float dxtol, int n, float *x, void *aux,
              void (*fdfdx)(int n, float *x, float *f, float ** dfdx, void *aux))

    # transform rectangular => polar and polar => to rectangular coordinates
    void recttopolar(int nx, float dx, float fx, int ny, float dy,
                     float fy, float ** p, int na, float da, float fa, int nr, float dr,
                     float fr, float ** q)
    void polartorect(int na, float da, float fa, int nr, float dr,
                     float fr, float ** q, int nx, float dx, float fx, int ny, float dy,
                     float fy, float ** p)

    # graphics utilities
    void rfwtva(int n, float z[], float zmin, float zmax, float zbase,
                int yzmin, int yzmax, int xfirst, int xlast,
                int wiggle, int nbpr, unsigned char *bits, int endian)
    void rfwtvaint(int n, float z[], float zmin, float zmax, float zbase,
                   int yzmin, int yzmax, int xfirst, int xlast,
                   int wiggle, int nbpr, unsigned char *bits, int endian)
    void scaxis(float x1, float x2, int *nxnum, float *dxnum, float *fxnum)
    int yclip(int nx, float dx, float fx, float y[], float ymin, float ymax,
              float xc[], float yc[])

    # special functions
    float airya(float x)
    float airyb(float x)
    float airyap(float x)
    float airybp(float x)

    # timers
    float cpusec()

    float cputime()

    float wallsec()

    float walltime()


    # pseudo-random numbers
    float franuni()

    void sranuni(int seed)
    float frannor()

    void srannor(int seed)

    # Ax = b routines
    void LU_decomposition(int nrows, float ** matrix, int *index, float *d)
    void backward_substitution(int nrows, float ** matrix, int *index, float *b)
    void inverse_matrix(int nrows, float ** matrix)
    void inverse_matrix_multiply(int nrows1, float ** matrix1, int ncols2,
                                 int nrows2, float ** matrix2, float ** out_matrix)

    # Conjugate gradient
    void simple_conj_gradient(int n, float *x, int m, float *b,
                              float ** a, int niter)

    # singular value decomposition routines
    int compute_svd(float ** a, int m, int n, float w[], float ** v)
    void svd_backsubstitute(float ** u, float w[], float ** v,
                            int m, int n, float b[], float x[])
    void svd_sort(float ** u, float *w, float ** v, int n, int m)

    # symmetric matrix eigenvalue routines
    void eig_jacobi(float ** a, float d[], float ** v, int n)
    void sort_eigenvalues(float d[], float ** v, int n)

    # waveforms
    void ricker1_wavelet(int nt, float dt, float fpeak, float *wavelet)
    void ricker2_wavelet(int hlw, float dt, float period, float ampl,
                         float distort, float *wavelet)
    void akb_wavelet(int nt, float dt, float fpeak, float *wavelet)
    void spike_wavelet(int nt, int tindex, float *wavelet)
    void unit_wavelet(int nt, float *wavelet)
    void zero_wavelet(int nt, float *wavelet)
    void berlage_wavelet(int nt, float dt, float fpeak, float ampl, float tn,
                         float decay, float ipa, float *wavelet)
    void gaussian_wavelet(int nt, float dt, float fpeak, float *wavelet)
    void gaussderiv_wavelet(int nt, float dt, float fpeak, float *wavelet)
    void deriv_n_gauss(double dt, int nt, double t0, float fpeak, int n, double *w,
                       int sign, int verbose)

    # orthogonal polynomials
    void hermite_n_polynomial(double *h, double *h0, double *h1,
                         double *t, int nt, int n, double sigma)

    # windowing functions
    void hanningnWindow(int n, float *w)

    # wrap
    void wrapArray(void *base, size_t nmemb, size_t size, int f)

    # miscellaneous
    void pp1d(FILE *fp, char *title, int lx, int ifx, float x[])
    void pplot1(FILE *fp, char *title, int nx, float ax[])
    FILE *temporary_stream(const char *prefix)
    char *temporary_filename(char *prefix)
    void zasc(char *ainput, char *aoutput, int nchar)
    void zebc(char *ainput, char *aoutput, int nchar)
    void IBMFLT(float *inn, float *out, int *nwds, int *idirec)

    dcomplex *alloc1dcomplex(size_t n1)
    dcomplex *realloc1dcomplex(dcomplex *v, size_t n1)
    dcomplex ** alloc2dcomplex(size_t n1, size_t n2)
    dcomplex ** *alloc3dcomplex(size_t n1, size_t n2, size_t n3)

    void free1dcomplex(dcomplex *p)
    void free2dcomplex(dcomplex ** p)
    void free3dcomplex(dcomplex ** *p)

    # Prime Factor FFTs(double version)
    int npfa_d(int nmin)
    int npfao_d(int nmin, int nmax)
    int npfar_d(int nmin)
    int npfaro_d(int nmin, int nmax)
    void pfacc_d(int isign, int n, dcomplex z[])
    void pfacr_d(int isign, int n, dcomplex cz[], double rz[])
    void pfarc_d(int isign, int n, double rz[], dcomplex cz[])
    void pfamcc_d(int isign, int n, int nt, int k, int kt, dcomplex z[])
    void pfa2cc_d(int isign, int idim, int n1, int n2, dcomplex z[])
    void pfa2cr_d(int isign, int idim, int n1, int n2, dcomplex cz[],
                  double rz[])
    void pfa2rc_d(int isign, int idim, int n1, int n2, double rz[],
                  dcomplex cz[])

    dcomplex *alloc1dcomplex(size_t n1)
    dcomplex *realloc1dcomplex(dcomplex *v, size_t n1)
    dcomplex ** alloc2dcomplex(size_t n1, size_t n2)
    dcomplex ** *alloc3dcomplex(size_t n1, size_t n2, size_t n3)

    void free1dcomplex(dcomplex *p)
    void free2dcomplex(dcomplex ** p)
    void free3dcomplex(dcomplex ** *p)

    # Prime Factor FFTs(double version)
    int npfa_d(int nmin)
    int npfao_d(int nmin, int nmax)
    int npfar_d(int nmin)
    int npfaro_d(int nmin, int nmax)
    void pfacc_d(int isign, int n, dcomplex z[])
    void pfacr_d(int isign, int n, dcomplex cz[], double rz[])
    void pfarc_d(int isign, int n, double rz[], dcomplex cz[])
    void pfamcc_d(int isign, int n, int nt, int k, int kt, dcomplex z[])
    void pfa2cc_d(int isign, int idim, int n1, int n2, dcomplex z[])
    void pfa2cr_d(int isign, int idim, int n1, int n2, dcomplex cz[],
                  double rz[])
    void pfa2rc_d(int isign, int idim, int n1, int n2, double rz[],
                  dcomplex cz[])

    # manipulation
    char *cwp_strdup(char *str)
    void strchop(char *s, char *t)

cdef extern from "par.h" nogil:
    # GLOBAL DECLARATIONS
    extern int xargc
    extern char ** xargv

    ctypedef int ssize_t

    # define structures for Hale's modeling
    struct ReflectorSegmentStruct:
        float x # x coordinate of segment midpoint
        float z # z coordinate of segment midpoint
        float s # x component of unit-normal-vector
        float c # z component of unit-normal-vector
    ctypedef ReflectorSegmentStruct ReflectorSegment
    struct ReflectorStruct:
        int ns # number of reflector segments
        float ds # segment length
        float a # amplitude of reflector
        ReflectorSegment * rs # array[ns] of reflector segments
    ctypedef ReflectorStruct Reflector

    struct WaveletStruct:
        int lw # length of wavelet
        int iw # index of first wavelet sample
        float * wv # wavelet sample values
    ctypedef WaveletStruct Wavelet

    # DEFINES

    # getpar macros
    void MUSTGETPARINT(char *name, int *p)
    void MUSTGETPARFLOAT(char *name, float *p)
    void MUSTGETPARSTRING(char *name, char **p)
    void MUSTGETPARDOUBLE(char *name, double *p)

    int STDIN # 0
    int STDOUT # 1
    int STDERR # 2

    # FUNCTION PROTOTYPES

    # getpar parameter parsing
    void initargs(int argc, char** argv)
    int getparint(char *name, int *p)
    int getparuint(char *name, unsigned int *p)
    int getparshort(char *name, short *p)
    int getparushort(char *name, unsigned short *p)
    int getparlong(char *name, long *p)
    int getparulong(char *name, unsigned long *p)
    int getparfloat(char *name, float *p)
    int getpardouble(char *name, double *p)
    int getparstring(char *name, char ** p)
    int getparstringarray(char *name, char ** p)
    int getnparint(int n, char *name, int *p)
    int getnparuint(int n, char *name, unsigned int *p)
    int getnparshort(int n, char *name, short *p)
    int getnparushort(int n, char *name, unsigned short *p)
    int getnparlong(int n, char *name, long *p)
    int getnparulong(int n, char *name, unsigned long *p)
    int getnparfloat(int n, char *name, float *p)
    int getnpardouble(int n, char *name, double *p)
    int getnparstring(int n, char *name, char ** p)
    int getnparstringarray(int n, char *name, char ** p)
    int getnpar(int n, char *name, char *type, void *ptr)
    int countparname(char *name)
    int countparval(char *name)
    int countnparval(int n, char *name)
    void checkpars()


    # For ProMAX
    void getPar(char *name, char *type, void *ptr)

    # errors and warnings
    void err(char *fmt, ...)
    void syserr(char *fmt, ...)
    void warn(char *fmt, ...)

    # self documentation
    void pagedoc(char *sdoc[])
    void requestdoc(int i, char *sdoc[])

    # system  subroutine calls with error trapping
    FILE *efopen(const char *file, const char *mode)
    FILE *efreopen(const char *file, const char *mode, FILE *stream1)
    FILE *efdopen(int fd, const char *mode)
    FILE *epopen(char *command, char *type)
    int efclose(FILE *stream)
    int epclose(FILE *stream)
    int efflush(FILE *stream)
    int eremove(const char *file)
    int erename(const char *oldfile, const char * newfile)
    int efseeko(FILE *stream, off_t offset, int origin)
    int efseek(FILE *stream, off_t offset, int origin)
    long eftell(FILE *stream)
    off_t eftello(FILE *stream)
    void erewind(FILE *stream)
    FILE *etmpstream(char *prefix)
    FILE *etmpfile()

    char *emkstemp(char *namebuffer)
    void *emalloc(size_t size)
    void *erealloc(void *memptr, size_t size)
    void *ecalloc(size_t count, size_t size)
    size_t efread(void *bufptr, size_t size, size_t count, FILE *stream)
    size_t efwrite(void *bufptr, size_t size, size_t count, FILE *stream)

    int efgetpos(FILE *stream, fpos_t *position)
    int efsetpos(FILE *stream, const fpos_t *position)

    # allocation with error trapping
    void * ealloc1(size_t n1, size_t size)
    void *erealloc1(void *v, size_t n1, size_t size)
    void ** ealloc2(size_t n1, size_t n2, size_t size)
    void ** *ealloc3(size_t n1, size_t n2, size_t n3, size_t size)
    void ** ** ealloc4(size_t n1, size_t n2, size_t n3, size_t n4, size_t size)
    void ** ** ealloc4(size_t n1, size_t n2, size_t n3, size_t n4, size_t size)
    void ** ** *ealloc5(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5, size_t size)
    void ** ** ** ealloc6(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5,
                          size_t n6, size_t size)

    int *ealloc1int(size_t n1)
    int *erealloc1int(int *v, size_t n1)
    int ** ealloc2int(size_t n1, size_t n2)
    int ** *ealloc3int(size_t n1, size_t n2, size_t n3)
    float *ealloc1float(size_t n1)
    float *erealloc1float(float *v, size_t n1)
    float ** ealloc2float(size_t n1, size_t n2)
    float ** *ealloc3float(size_t n1, size_t n2, size_t n3)

    int ** ** ealloc4int(size_t n1, size_t n2, size_t n3, size_t n4)
    int ** ** *ealloc5int(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
    float ** ** ealloc4float(size_t n1, size_t n2, size_t n3, size_t n4)
    float ** ** *ealloc5float(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5)
    float ** ** ** ealloc6float(size_t n1, size_t n2, size_t n3, size_t n4, size_t n5,
                                size_t n6)

    unsigned short ** ** *ealloc5ushort(size_t n1, size_t n2,
                                        size_t n3, size_t n4, size_t n5)
    unsigned char ** ** *ealloc5uchar(size_t n1, size_t n2,
                                      size_t n3, size_t n4, size_t n5)
    unsigned short ** ** ** ealloc6ushort(size_t n1, size_t n2,
                                          size_t n3, size_t n4, size_t n5, size_t n6)

    double *ealloc1double(size_t n1)
    double *erealloc1double(double *v, size_t n1)
    double ** ealloc2double(size_t n1, size_t n2)
    double ** *ealloc3double(size_t n1, size_t n2, size_t n3)
    complex *ealloc1complex(size_t n1)
    complex *erealloc1complex(complex *v, size_t n1)
    complex ** ealloc2complex(size_t n1, size_t n2)
    complex ** *ealloc3complex(size_t n1, size_t n2, size_t n3)

    # string to numeric conversion with error checking
    short eatoh(char *s)
    unsigned short eatou(char *s)
    int eatoi(char *s)
    unsigned int eatop(char *s)
    long eatol(char *s)
    unsigned long eatov(char *s)
    float eatof(char *s)
    double eatod(char *s)

    # file type checking
    FileType filestat(int fd)
    char *printstat(int fd)

    # Hale's modeling code
    void decodeReflectors(int *nrPtr,
                          float ** aPtr, int ** nxzPtr, float ** *xPtr, float ** *zPtr)
    int decodeReflector(char *string,
                        float *aPtr, int *nxzPtr, float ** xPtr, float ** zPtr)
    void breakReflectors(int *nr, float ** ar,
                         int ** nu, float ** *xu, float ** *zu)
    void makeref(float dsmax, int nr, float *ar,
                 int *nu, float ** xu, float ** zu, Reflector ** r)
    void raylv2(float v00, float dvdx, float dvdz,
                float x0, float z0, float x, float z,
                float *c, float *s, float *t, float *q)
    void addsinc(float time, float amp,
                 int nt, float dt, float ft, float *trace)
    void makericker(float fpeak, float dt, Wavelet ** w)

    # upwind eikonal stuff
    void eikpex(int na, float da, float r, float dr,
           float sc[], float uc[], float wc[], float tc[],
           float sn[], float un[], float wn[], float tn[])
    void ray_theoretic_sigma(int na, float da, float r, float dr,
                             float uc[], float wc[], float sc[],
                             float un[], float wn[], float sn[])
    void ray_theoretic_beta(int na, float da, float r, float dr,
                            float uc[], float wc[], float bc[],
                            float un[], float wn[], float bn[])
    void eiktam(float xs, float zs,
                int nz, float dz, float fz, int nx, float dx, float fx, float ** vel,
                float ** time, float ** angle, float ** sig, float ** beta)

    # smoothing routines
    void dlsq_smoothing(int nt, int nx, int ift, int ilt, int ifx, int ilx,
                   float r1, float r2, float rw, float ** traces)
    void SG_smoothing_filter(int np, int nl, int nr, int ld, int m, float *filter)
    void rwa_smoothing_filter(int flag, int nl, int nr, float *filter)
    void gaussian2d_smoothing(int nx, int nt, int nsx, int nst, float ** data)
    void gaussian1d_smoothing(int ns, int nsr, float *data)
    void smooth_histogram(int nintlh, float *pdf)
    void smooth_segmented_array(float *index, float *val, int n, int sm, int inc, int m)
    void smooth_1(float *x, float *z, float r, int n)

    # function minimization
    void bracket_minimum(float *ax, float *bx, float *cx, float *fa,
                    float *fb, float *fc, float (*func)(float))
    float golden_bracket(float ax, float bx, float cx,
                         float (*f)(float), float tol, float *xmin)
    float brent_bracket(float ax, float bx, float cx,
                        float (*f)(float), float tol, float *xmin)

    void linmin(float p[], float xi[], int n, float *fret, float (*func)())
    void powell_minimization(float p[], float ** xi, int n,
                             float ftol, int *iter, float *fret, float (*func)())

    # fractals
    float hausdorff_dimension(float *ar, int n, int minl, int maxl, int dl)

    # lincoeff -- linearized reflection coefficients
    # type definitions


    struct ErrorFlag:
        float iso[5]
        float upper[2]
        float lower[2]
        float glob "global"[4]
        float angle[4]

    # prototypes for functions defined

    float lincoef_Rp(float ang, float azim, float kappa, float *rpp, ErrorFlag *rp_1st, ErrorFlag *rp_2nd,
               int count)

    float lincoef_Rs(float ang, float azim, float kappa, float *rps1, float *rps2,
                     float *sv, float *sh, float *cphi, float *sphi, int i_hsp,
                     ErrorFlag *rsv_1st, ErrorFlag *rsv_2nd, ErrorFlag *rsh_1st, ErrorFlag *rsh_2nd, int count)

    float Iso_exact(int type, float vp1, float vs1, float rho1,
                    float vp2, float vs2, float rho2, float ang)

    int Phi_rot(float *rs1, float *rs2, int iso_plane, float pb_x, float pb_y, float pb_z, float gs1_x, float gs1_y,
                float gs1_z, float gs2_x, float gs2_y, float gs2_z, float *CPhi1, float *SPhi1, float *CPhi2, float
                *SPhi2)

cdef extern from "su.h" nogil:
    union Value: # storage for arbitrary type *
        char s[8];
        short h;
        unsigned short u;
        long l;
        unsigned long v;
        int i;
        unsigned int p;
        float f;
        double d;
        unsigned int U;
        unsigned int P;

    int READ_OK
    int WRITE_OK
    int EXEC_OK
    int FILE_OK

    int IS_DEPTH(char *str)
    int IS_COORD(char *str)

    # valpkge
    int vtoi(cwp_String type, Value val);
    long vtol(cwp_String type, Value val);
    float vtof(cwp_String type, Value val);
    double vtod(cwp_String type, Value val);
    int valcmp(cwp_String type, Value val1, Value val2);
    void printfval(cwp_String type, Value val);
    void fprintfval(FILE *stream, cwp_String type, Value val);
    void scanfval(cwp_String type, Value *valp);
    void atoval(cwp_String type, cwp_String keyval, Value *valp);
    void atoval(cwp_String type, cwp_String keyval, Value *valp);
    void getparval(cwp_String name, cwp_String type, int n, Value *valp);
    Value valtoabs(cwp_String type, Value val);

    # segy coordinate scalar utilities
    short elco_scalar(int ncoords, double c[]);
    double from_segy_elco_multiplier(short segy_scalar);
    double to_segy_elco_multiplier(short segy_scalar); # reciprocal of from

    ##### SU Main functions
    # filters
    void su_bfhighpass(int zerophase, int npoles, float f3db, size_t nt, float *data_in, float *data_out);
    void su_bflowpass(int zerophase, int npoles, float f3db, size_t nt, float *data_in, float *data_out);

    #Synthetics
    void su_synlv(float *data,
                  float xs, float zs, float xg, float zg,
                  size_t nt, float dt, float ft,
                  float v00, float dvdx, float dvdz,
                  int ls, int er, int ob, Wavelet *w, int nr, Reflector *r, int lhd, int nhd, float *hd
              );