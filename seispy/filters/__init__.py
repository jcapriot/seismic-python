from ._bandpass import butterworth_bandpass as _butterworth_bandpass
from ._filter import _filter
from ..stage import stage

# Stages are only ever chained with `|`: `source | bfilt(...)`.
# The iterator classes they build are implementation details.

# Butterworth filters, named after the SU program that does the same job, subfilt, without the `su` prefix.
bfilt = stage(_butterworth_bandpass, parallelism='trace', name='bfilt')

# A zero-phase, sine-squared tapered filter (SUFILTER), `source | filter(f=[10, 20, 40, 50])`
#
# f : array of frequencies (Hz) that define the filter, default .10, .15, .45, .50 times the Nyquist frequency
# amps : array of amplitudes at those frequencies, default 0, 1, ..., 1, 0 (a trapezoid-like bandpass)
# dt : sample interval (s) for traces that do not have one, default .004
# inplace : overwrite the samples of the incoming traces instead of making new traces.
#
# Bandpass:   filter(f=[10, 20, 40, 50])
# Bandreject: filter(f=[10, 20, 30, 40], amps=[1, 0, 0, 1])
# Lowpass:    filter(f=[10, 20, 40, 50], amps=[1, 1, 0, 0])
# Highpass:   filter(f=[10, 20, 40, 50], amps=[0, 0, 1, 1])
# Notch:      filter(f=[10, 12.5, 35, 50, 60], amps=[1, .5, 0, .5, 1])
filter = stage(_filter, parallelism='trace', name='filter', validate=True)

__all__ = ['bfilt', 'filter']
