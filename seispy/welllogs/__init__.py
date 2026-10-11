"""
Well logs: LAS2SU, SUBACKUS, SUBACKUSH, SULPRIME, SUWELLRF and SUGASSMANN (``su/main/well_logs``).

A log is a trace whose samples are the values at the depths of a well. See ``_logs.py`` for the notes on the programs and for
their parameters.
"""
from . import _logs
from ._logs import backus, backush, gassmann, las2su, lprime, rock_functions, wellrf

# The functions of Gassmann's fluid substitution and of the fluid properties, from SUGASSMANN, with the names of their
# kind: rock_twt, rock_reflectivity, rock_impedance, rock_velocity, rock_density, fluid_velocity, fluid_modulus,
# fluid_density, brine_*, oil_* and gas_* (velocity, modulus, density).
globals().update(rock_functions)

__all__ = ['backus', 'backush', 'lprime', 'wellrf', 'las2su', 'gassmann'] + sorted(rock_functions)
