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

    Case("suop-abs", "suop", PLANE, "suop op=abs", (("operations.abs", {}),)),
    Case("suop-sqr", "suop", PLANE, "suop op=sqr", (("operations.sqr", {}),)),
    Case("suop-ssqrt", "suop", PLANE, "suop op=ssqrt", (("operations.ssqrt", {}),)),
    Case("suop-sgn", "suop", PLANE, "suop op=sgn", (("operations.sgn", {}),)),
    Case("suop-slog10", "suop", PLANE, "suop op=slog10", (("operations.slog10", {}),)),
    Case("suop-db", "suop", PLANE, "suop op=db", (("operations.db", {}),)),
    Case("suop-diff", "suop", PLANE, "suop op=diff", (("operations.diff", {}),)),
    Case("suop-integ", "suop", PLANE, "suop op=integ", (("operations.integ", {}),)),
    Case("suop-norm", "suop", PLANE, "suop op=norm", (("operations.norm", {}),)),
    Case("suop-mean", "suop", PLANE, "suop op=mean nw=11", (("operations.mean", dict(nw=11)),)),
    Case("suzero", "suzero", PLANE, "suzero itmin=10 itmax=50", (("amplitudes.zero", dict(itmin=10, itmax=50)),)),
    Case("sunormalize-rms", "sunormalize", PLANE, "sunormalize norm=rms", (("amplitudes.normalize", dict(norm='rms')),)),
    Case(
        "sunormalize-rms-first-trace", "sunormalize", "suplane npl=2 ntr=1 nt=256 dt=0.004", "sunormalize norm=rms",
        (("amplitudes.normalize", dict(norm="rms")),),
        note="one trace: the accumulated rms of SU is only right for the first trace (see sunormalize-rms)",
    ),
    Case("sunormalize-max", "sunormalize", PLANE, "sunormalize norm=max t0=0.2 t1=0.8", (("amplitudes.normalize", dict(norm='max', t0=0.2, t1=0.8)),)),
    Case("suweight", "suweight", PLANE, "suweight key=offset a=1 b=0.0005", (("amplitudes.weight", dict(key='offset', a=1.0, b=0.0005)),)),
    Case("sudivcor", "sudivcor", PLANE, "sudivcor trms=0,1 vrms=1500,2500", (("amplitudes.divcor", dict(trms=[0.0, 1.0], vrms=[1500.0, 2500.0])),)),
    Case("supgc", "supgc", PLANE, "supgc file={tmp} ntrscan=32 lwindow=0.5", (("amplitudes.pgc", dict(ntrscan=32, lwindow=0.5)),)),
    Case("suhilb", "suhilb", PLANE, "suhilb", (("transforms.hilb", {}),), rtol=0.0001),
    Case("sutsq", "sutsq", PLANE, "sutsq tmin=0.1 dt=0.004", (("stretching.tsq", dict(tmin=0.1, dt=0.004)),), rtol=0.0001),
    Case("sureduce", "sureduce", PLANE, "sureduce rv=2.0", (("stretching.reduce", dict(rv=2.0)),)),
    Case("suacor", "suacor", PLANE, "suacor ntout=51", (("convolution.acor", dict(ntout=51)),), rtol=0.0001),
    Case("suramp", "suramp", PLANE, "suramp tmin=0.1 tmax=0.9", (("tapering.ramp", dict(tmin=0.1, tmax=0.9)),)),
    Case("sugausstaper", "sugausstaper", PLANE, "sugausstaper key=offset x0=400 xw=200", (("tapering.gausstaper", dict(key='offset', x0=400.0, xw=200.0)),)),
    Case("sukill", "sukill", PLANE, "sukill min=3 count=2", (("windowing.kill", dict(min=3, count=2)),)),
    Case("sudivstack", "sudivstack", PLANE, "sudivstack key=cdp winlen=0.1", (("stacking.divstack", dict(key='ensemble_number', winlen=0.1)),)),
    Case("supws", "supws", PLANE, "supws key=cdp pwr=1.0", (("stacking.pws", dict(key='ensemble_number', pwr=1.0)),), rtol=0.0001),
    Case("sufrac", "sufrac", PLANE, "sufrac power=0.5 sign=1", (("filters.frac", dict(power=0.5, sign=1.0)),), rtol=0.0001),
    Case("suphase", "suphase", PLANE, "suphase a=45", (("filters.phase", dict(a=45.0)),), rtol=0.0001),
    Case("sutvband", "sutvband", PLANE, "sutvband tf=0.2,0.8 f=10,12.5,40,50 f=10,12.5,30,40", (("filters.tvband", dict(tf=[0.2, 0.8], f=[[10, 12.5, 40, 50], [10, 12.5, 30, 40]])),), rtol=0.0001),
    Case("supef", "supef", PLANE, "supef maxlag=0.1", (("decon.pef", dict(maxlag=0.1)),), rtol=0.0001),
    Case("suaddnoise", "suaddnoise", PLANE, "suaddnoise sn=20 seed=1", (("noise.addnoise", dict(sn=20, seed=1)),), rtol=0.0001),

    # sources: the program makes the traces (no input)
    Case("suplane", "suplane", "", "suplane", (("synthetics.plane", {}),)),
    Case(
        "suplane-params", "suplane", "", "suplane nt=128 ntr=16 dt=0.002 offset=200 taper=1",
        (("synthetics.plane", dict(nt=128, ntr=16, dt=0.002, offset=200.0, taper=True)),),
    ),
    Case("suspike", "suspike", "", "suspike", (("synthetics.spike", {}),)),
    Case(
        "suspike-params", "suspike", "", "suspike nt=100 ntr=8 dt=0.002 offset=100 nspk=2 ix1=2 it1=10 ix2=5 it2=60",  # (1 based; seispy: 0 based)
        (("synthetics.spike", dict(nt=100, ntr=8, dt=0.002, offset=100.0, spikes=[(1, 9), (4, 59)])),),
    ),
    Case("sunull", "sunull", "", "sunull nt=64 ntr=5", (("synthetics.null", dict(nt=64, ntr=5)),)),
    Case(
        "surandspike", "surandspike", "", "surandspike n1=100 n2=8 nspk=5 seed=1",
        (("synthetics.randspike", dict(n1=100, n2=8, nspk=5, seed=1)),),
    ),
    Case(
        "susynlv", "susynlv", "",
        "susynlv nt=256 dt=0.004 nxo=4 dxo=0.1 fxo=0.1 nxs=2 dxs=0.2 fxs=0 nvq=1 v00=2.0 ob=0",
        (("synthetics.synlv", dict(nt=256, dt=0.004, nxo=4, dxo=0.1, fxo=0.1, nxm=0, nxs=2, dxs=0.2, fxs=0.0, v00=2.0, ob=0)),),
        rtol=1e-4,
    ),
]
