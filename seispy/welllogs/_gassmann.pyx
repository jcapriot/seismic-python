# cython: embedsignature=True, language_level=3
"""
The rock and fluid physics of SUGASSMANN (``su/main/well_logs/sugassmann.c``): Gassmann's fluid substitution, with the Batzle
and Wang properties of brine, oil and gas. The functions of the SU library (``gassmann_*``) are called for every row of an
array of arguments.
"""
from .. cimport su
import numpy as np

# the names of the functions, with the names of their arguments in order
FUNCTIONS = {
    'rock_twt': ('h', 'kmin', 'mumin', 'rmin', 'phi', 'a', 'b', 'c', 'api', 'g', 's', 'so', 'sg', 't', 'p'),
    'rock_reflectivity': ('v1', 'r1', 'kmin', 'mumin', 'rmin', 'phi', 'a', 'b', 'c', 'api', 'g', 's', 'so', 'sg', 't', 'p'),
    'rock_impedance': ('kmin', 'mumin', 'rmin', 'phi', 'a', 'b', 'c', 'api', 'g', 's', 'so', 'sg', 't', 'p'),
    'rock_velocity': ('kmin', 'mumin', 'rmin', 'phi', 'a', 'b', 'c', 'api', 'g', 's', 'so', 'sg', 't', 'p'),
    'rock_density': ('rmin', 'phi', 'api', 'g', 's', 'so', 'sg', 't', 'p'),
    'fluid_velocity': ('api', 'g', 's', 'so', 'sg', 't', 'p'),
    'fluid_modulus': ('api', 'g', 's', 'so', 'sg', 't', 'p'),
    'fluid_density': ('api', 'g', 's', 'so', 'sg', 't', 'p'),
    'brine_velocity': ('s', 't', 'p'),
    'brine_modulus': ('s', 't', 'p'),
    'brine_density': ('s', 't', 'p'),
    'oil_modulus': ('api', 'g', 't', 'p'),
    'oil_velocity': ('api', 'g', 't', 'p'),
    'oil_density': ('api', 'g', 't', 'p'),
    'gas_velocity': ('g', 't', 'p'),
    'gas_modulus': ('g', 't', 'p'),
    'gas_density': ('g', 't', 'p'),
}
_IDS = {name: i for i, name in enumerate(FUNCTIONS)}


cdef float _call(int which, const float *x) noexcept nogil:
    if which == 0:
        return su.gassmann_rockT0(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14])
    elif which == 1:
        return su.gassmann_rockR0(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14],
                                  x[15])
    elif which == 2:
        return su.gassmann_rockIp(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13])
    elif which == 3:
        return su.gassmann_rockVp(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13])
    elif which == 4:
        return su.gassmann_rockDensity(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8])
    elif which == 5:
        return su.gassmann_fluidVelocity(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    elif which == 6:
        return su.gassmann_fluidModulus(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    elif which == 7:
        return su.gassmann_fluidDensity(x[0], x[1], x[2], x[3], x[4], x[5], x[6])
    elif which == 8:
        return su.gassmann_brineVelocity(x[0], x[1], x[2])
    elif which == 9:
        return su.gassmann_brineModulus(x[0], x[1], x[2])
    elif which == 10:
        return su.gassmann_brineDensity(x[0], x[1], x[2])
    elif which == 11:
        return su.gassmann_oilModulus(x[0], x[1], x[2], x[3])
    elif which == 12:
        return su.gassmann_oilVelocity(x[0], x[1], x[2], x[3])
    elif which == 13:
        return su.gassmann_oilDensity(x[0], x[1], x[2], x[3])
    elif which == 14:
        return su.gassmann_gasVelocity(x[0], x[1], x[2])
    elif which == 15:
        return su.gassmann_gasModulus(x[0], x[1], x[2])
    return su.gassmann_gasDensity(x[0], x[1], x[2])


def evaluate(str name, const float[:, ::1] args):
    """The function ``name`` for every row of ``args`` (rows, number of arguments of the function): a float32 array"""
    cdef:
        int which = _IDS[name]
        Py_ssize_t n = args.shape[0]
        Py_ssize_t i
        float[::1] out = np.empty(n, dtype=np.float32)

    if args.shape[1] != len(FUNCTIONS[name]):
        raise ValueError(f"{name} takes {len(FUNCTIONS[name])} arguments.")
    with nogil:
        for i in range(n):
            out[i] = _call(which, &args[i, 0])
    return np.asarray(out)
