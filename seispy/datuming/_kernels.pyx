# cython: embedsignature=True, language_level=3
"""
The library functions of the datuming programs of SU (``su_datumfd``, in ``su/main/datuming``), on arrays.
"""
from .. cimport su
import numpy as np


def datumfd(const float[:, :, ::1] data, const float[:, ::1] gx, const float[:, ::1] vel1, const float[:, ::1] vel2,
            float dt, float dz, int nz, float dx, float x_0, int buff, int tap_len):
    """The shot gathers (nsx, nx, nt) datumed downward nz steps of dz by the finite difference scheme, with the
    velocities (mx, mz) of the thin lens and of the diffraction term, and the x of each receiver (nsx, nx)"""
    cdef:
        int nsx = data.shape[0]
        int nx = data.shape[1]
        int nt = data.shape[2]
        int mx = vel1.shape[0]
        int mz = vel1.shape[1]
        float[:, :, ::1] out = np.zeros((nsx, nx, nt), dtype=np.float32)
        int status
    if vel2.shape[0] != mx or vel2.shape[1] != mz:
        raise ValueError("The two velocity models must have the same shape.")
    if gx.shape[0] != nsx or gx.shape[1] != nx:
        raise ValueError("gx must have a value for every trace.")
    if nz > mz:
        raise ValueError("nz is more than the number of depth samples of the velocity models.")
    if nsx == 0 or nx < 4 or nt < 2:
        raise ValueError("There must be at least 4 receivers, 2 samples and one shot.")
    with nogil:
        status = su.su_datumfd(nt, dt, nx, nsx, nz, dz, dx, x_0, buff, tap_len, mx, mz, &vel1[0, 0], &vel2[0, 0],
                               &gx[0, 0], &data[0, 0, 0], &out[0, 0, 0])
    return np.asarray(out), status
