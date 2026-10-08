# cython: embedsignature=True, language_level=3
"""
The slant stacks (tau-p transforms) of the SU library that SUTAUP uses: ``fwd_FK_sstack``, ``fwd_tx_sstack``,
``inv_FK_sstack`` and ``inv_tx_sstack`` (``par/lib/taup.c``).
"""
from libc.stdlib cimport malloc, free
from .. cimport su
import numpy as np


# the F-K routines make a table of sinc coefficients the first time that they are used, which is not safe to do from several
# threads at once. Once made it is only read, so make it once, here.
su.su_taup_tables()


def slant_stack(int option, float[:, ::1] traces, float dt, int nx, float xmin, float dx, int n_slopes, float pmin,
                float dp, float fmin, int npoints):
    """Slant stack of the traces, as SUTAUP does: option 1 is the forward F-K transform, 2 the forward t-x transform, 3
    the inverse F-K transform and 4 the inverse t-x transform.

    The forward transforms take (at least) nx traces, and give n_slopes. The inverse ones take (at least) n_slopes traces
    and give nx.
    """
    cdef:
        int nt = traces.shape[1]
        int n_in = nx if option in (1, 2) else n_slopes
        int n_out = n_slopes if option in (1, 2) else nx
        int i
        float **rows_in = NULL
        float **rows_out = NULL
        float[:, ::1] out
    if option not in (1, 2, 3, 4):
        raise ValueError("option flag has to be between 1 and 4")
    if traces.shape[0] < n_in:
        raise ValueError(f"The transform needs {n_in} traces.")
    if nt < 1 or n_in < 1 or n_out < 1:
        raise ValueError("There is nothing to transform.")
    if n_slopes < 2:
        raise ValueError("np must be at least 2.")
    out = np.zeros((n_out, nt), dtype=np.float32)
    rows_in = <float **> malloc(n_in * sizeof(float *))
    rows_out = <float **> malloc(n_out * sizeof(float *))
    try:
        if rows_in == NULL or rows_out == NULL:
            raise MemoryError()
        for i in range(n_in):
            rows_in[i] = &traces[i, 0]
        for i in range(n_out):
            rows_out[i] = &out[i, 0]
        with nogil:
            if option == 1:
                su.fwd_FK_sstack(dt, nt, nx, xmin, dx, n_slopes, pmin, dp, fmin, rows_in, rows_out)
            elif option == 2:
                su.fwd_tx_sstack(dt, nt, nx, xmin, dx, n_slopes, pmin, dp, rows_in, rows_out)
            elif option == 3:
                su.inv_FK_sstack(dt, nt, nx, xmin, dx, n_slopes, pmin, dp, fmin, rows_in, rows_out)
            else:
                su.inv_tx_sstack(dt, nt, nx, npoints, xmin, dx, n_slopes, pmin, dp, rows_in, rows_out)
    finally:
        free(rows_in)
        free(rows_out)
    return np.asarray(out)
