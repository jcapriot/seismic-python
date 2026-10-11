from ._bandpass import butterworth_bandpass as _butterworth_bandpass
from ._fftfilter import _filter_stage as _filter
from ._moveout import median, medmix
from . import _panel
from ._dipfilt import dipfilt
from ._phasefilters import _frac, _phase
from ._spectral import _minphase, _tvband
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

# A general (fractional) time derivative or integral, plus a phase shift (SUFRAC): `source | frac(power=1)`. See
# _phasefilters.py for the parameters.
frac = stage(_frac, parallelism='trace', name='frac', validate=True)

# Phase manipulation by a linear transformation of the phase spectrum (SUPHASE): `source | phase(a=90)`
phase = stage(_phase, parallelism='trace', name='phase', validate=True)

# The minimum phase equivalent of each trace (SUMINPHASE), see _spectral.py for the parameters.
minphase = stage(_minphase, parallelism='trace', name='minphase', validate=True)

# A time-variant bandpass filter (SUTVBAND): `source | tvband(tf=[.2, 1.5], f=[[10, 12.5, 40, 50], [10, 12.5, 30, 40]])`
tvband = stage(_tvband, parallelism='trace', name='tvband', validate=True)

# ---------------------------------------------------------------------------------------- panels and gathers (_panel.py)
# The traces of the stream are one panel (held in memory) for the filters of the wavenumber domain; a gather (the traces that
# have the same value of ``key``, ``ensemble_number`` by default) for the others. See _panel.py for the parameters.

# A radially symmetric, sin^2 tapered polygonal filter of the (k1, k2) wavenumbers of a panel (SUKFILTER):
# `source | kfilter(k=[.05, .1, .2, .25], amps=[0, 1, 1, 0], d1=10, d2=25)`. d1 is the trace's sample interval by default, d2 is 1.
kfilter = stage(_panel._kfilter, parallelism='serial', name='kfilter', validate=True)

# A box-like filter of (k1, k2): the product of two sin^2 tapered polygons (SUK1K2FILTER); quad=1 keeps quadrants 1 and 4,
# quad=2 quadrants 2 and 3: `source | k1k2filter(k1=[...], amps1=[...], k2=[...], amps2=[...], d1=10, d2=25)`
k1k2filter = stage(_panel._k1k2filter, parallelism='serial', name='k1k2filter', validate=True)

# A fractional power of i|k| with a phase shift, in the (k1, k2) domain of a panel (SUKFRAC): `source | kfrac(power=1)`.
# `angle` (degrees) makes it a directional derivative.
kfrac = stage(_panel._kfrac, parallelism='serial', name='kfrac', validate=True)

# Smoothing of a panel of velocities (nz samples per trace) with a Gaussian filter of correlation lengths a1 and a2 (samples)
# (SUSMGAUSS2). As its documentation says, the velocities are made slowness first (`slowness=False` smooths the velocities,
# which is what the program does).
smgauss2 = stage(_panel._smgauss2, parallelism='serial', name='smgauss2', validate=True)

# Low frequency array forming: a trace mix of f1 to f2 Hz (ramped over fr Hz) over vel / (f dx) traces (SULFAF), per gather
lfaf = stage(_panel._lfaf, parallelism='ensemble', name='lfaf', validate=True)

# Frequencies of a gather that do not correlate in phase with those of the next trace are zeroed (SUCCFILT), per gather
ccfilt = stage(_panel._ccfilt, parallelism='ensemble', name='ccfilt', validate=True)

# A deghosting (or ghosting, with deghost=False) filter, in (t, x), or in (tau, p) with p0 and dp (SUDEGHOST). h is the
# depth of the source or receiver.
deghost = stage(_panel._deghost, parallelism='serial', name='deghost', validate=True)

# An alpha trimmed mean in the frequency domain of the traces of every gather, one trace per gather (SUFWATRIM), with the
# header of the trace that ``mark`` picks (an index in the gather, or a function of a trace; the first by default)
fwatrim = stage(_panel._fwatrim, parallelism='ensemble', name='fwatrim', validate=True)

# An SVD (eigenimage) filter: the first numpp principal planes, subtracted from the traces or, with subtract=False, kept
# (SUSVDFILT). The traces are taken in windows of ntrw traces (all of them by default).
svdfilt = stage(_panel._svdfilt, parallelism='serial', name='svdfilt', validate=True)

# A weighted mix in the frequency domain of the traces of every gather, one trace per gather (SUFWMIX): the weight of a trace
# is 1 / (d / fac + 1), d the distance from the marked trace (an index in the gather or a function of a trace; the first by
# default) in the units of dx, over the header values key (names, or functions of a trace). `vf=True` leaves a trace out of a
# frequency if it is over a half wavelength of the slope vmin from the marked one.
fwmix = stage(_panel._fwmix, parallelism='ensemble', name='fwmix', validate=True)

# Ground roll suppression with the Karhunen-Loeve transform (SUGROLL): the first nrot principal components of the traces of
# the stream (one panel) are the ground roll, which is subtracted (or kept, with sb=True).
groll = stage(_panel._groll, parallelism='serial', name='groll', validate=True)

__all__ = ['bfilt', 'filter', 'frac', 'phase', 'minphase', 'tvband', 'median', 'medmix', 'dipfilt', 'kfilter', 'k1k2filter', 'kfrac',
           'smgauss2', 'lfaf', 'ccfilt', 'deghost', 'fwatrim', 'svdfilt', 'fwmix', 'groll']
