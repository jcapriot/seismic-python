"""The programs that are compared against SU. Add one ``Case`` for each program (and each notable parameter set)."""
from harness import Case

# Shot-like data from SU's own plane-wave generator, with the headers that the stages read set the way SU does it:
# 8 traces per cdp (offsets 100..800 m), cdps 1..4.
PLANE = "suplane npl=2 ntr=32 nt=256 dt=0.004 | sushw key=cdp a=1 b=0 c=1 j=8 | sushw key=offset a=100 b=100 j=8"

# the same, with the source and receiver positions (sx=0, gx=offset) as well as the offset
COORDS = PLANE + " | sushw key=sx a=0 | sushw key=gx a=100 b=100 j=8"

CASES = [
    Case("sugain-tpow", "sugain", PLANE, "sugain tpow=2.0", (("amplitudes.gain", dict(tpow=2.0)),)),
    Case("sugain-agc", "sugain", PLANE, "sugain agc=1 wagc=0.2", (("amplitudes.gain", dict(agc=True, wagc=0.2)),)),
    Case(
        "sugain-gpow-clip", "sugain", PLANE, "sugain gpow=0.5 qclip=0.95",
        (("amplitudes.gain", dict(gpow=0.5, qclip=0.95)),),
    ),
    Case(
        "subfilt", "subfilt", PLANE, "subfilt locut=1 hicut=1 f3dblo=5 npoleslo=2 f3dbhi=60 npoleshi=4",
        (("filters.bfilt", dict(f3db_low=5.0, n_poles_low=2, f3db_high=60.0, n_poles_high=4)),),
        rtol=1e-4,
    ),
    Case(
        "sufilter", "sufilter", PLANE, "sufilter f=10,20,60,80 amps=0,1,1,0",
        (("filters.filter", dict(f=[10, 20, 60, 80], amps=[0, 1, 1, 0])),),
        rtol=1e-4,
    ),
    Case(
        "sunmo", "sunmo", PLANE, "sunmo tnmo=0,0.5,1.0 vnmo=2000,2300,2600",
        (("stretching.nmo", dict(tnmo=[0.0, 0.5, 1.0], vnmo=[2000.0, 2300.0, 2600.0])),),
        rtol=1e-4,
    ),
    Case("sustack", "sustack", PLANE, "sustack key=cdp", (("stacking.stack", dict(key="ensemble_number")),)),
    Case(
        "sustack-coords", "sustack", COORDS, "sustack key=cdp", (("stacking.stack", dict(key="ensemble_number")),),
        note="real source and receiver coordinates (not just an offset)",
    ),
    Case(
        "suwind", "suwind", PLANE, "suwind key=cdp min=2 max=3 tmin=0.2 tmax=0.6",
        (("windowing.wind", dict(key="ensemble_number", min=2, max=3, tmin=0.2, tmax=0.6)),),
    ),
    Case(
        "sushift", "sushift", PLANE, "sushift tmin=-0.1 tmax=0.7",
        (("stretching.shift", dict(tmin=-0.1, tmax=0.7)),),
    ),
    Case(
        "suresamp", "suresamp", PLANE, "suresamp nt=128 dt=0.008",
        (("stretching.resamp", dict(nt=128, dt=0.008)),),
        rtol=1e-4,
    ),
    Case(
        "sumute", "sumute", PLANE, "sumute xmute=0,800 tmute=0.1,0.5 key=offset ntaper=4",
        (("windowing.mute", dict(xmute=[0.0, 800.0], tmute=[0.1, 0.5], key="offset", ntaper=4)),),
    ),
    Case(
        "sumute-below", "sumute", PLANE, "sumute xmute=0,800 tmute=0.5,0.9 key=offset ntaper=4 mode=1",
        (("windowing.mute", dict(xmute=[0.0, 800.0], tmute=[0.5, 0.9], key="offset", ntaper=4, mode=1)),),
    ),
    Case(
        "sutaper", "sutaper", PLANE, "sutaper tbeg=0.1 tend=0.1 ntr=32",
        (("tapering.taper", dict(tbeg=0.1, tend=0.1)),),
    ),
]
