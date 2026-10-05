from ._bandpass import butterworth_bandpass as _butterworth_bandpass
from ..stage import stage

# Stages are only ever chained with `|`: `source | bfilt(...)`.
# The iterator class they build is an implementation detail.
# (named after the SU program that does the same job, subfilt, without the `su` prefix)
bfilt = stage(_butterworth_bandpass, parallelism='trace', name='bfilt')

__all__ = ['bfilt']
