# Post-hoc: errors at lambda/h = 10 (N = 40, MECOM resolution) and 20, 32 vs N = 512, per amplitude (worst seed); throughput.
import csv, sys, json
A = sys.argv[1]; lam = 0.25
def rd(e, s, N, f):
    return {float(r[list(r)[0]]): r for r in csv.DictReader(l for l in open(f"{A}/raw/v2/v2_e{e}_s{s}_N{N}/{f}") if not l.startswith("#"))}
for e, s2 in (("10", 1), ("15", 2.25), ("20", 4)):
    out = {}
    for N in (40, 80, 128):
        wsT = wG = waT = waL = wvt = 0.0
        for s in ("3101", "3102", "3103"):
            p, p5 = rd(e, s, N, "moments_planes.csv"), rd(e, s, 512, "moments_planes.csv")
            t, t5 = rd(e, s, N, "moments_time.csv"), rd(e, s, 512, "moments_time.csv")
            st = lambda q, x: (float(q[x]["var_x2"]) + float(q[x]["var_x3"])) / 2
            wsT = max(wsT, abs(st(p, 16.0) / st(p5, 16.0) - 1))
            wvt = max(wvt, max(abs(float(p[x]["var_tau"]) / float(p5[x]["var_tau"]) - 1) for x in p))
            aT = lambda q: (st(q, 16.0) - st(q, 4.0)) / 24 / lam
            waT = max(waT, abs(aT(p) - aT(p5)))
            aL = lambda q: (float(q[16.0]["var_x"]) - float(q[8.0]["var_x"])) / 16 / lam
            waL = max(waL, abs(aL(t) / aL(t5) - 1))
        out[N] = f"lam/h {N//4}: sigT2(16) {wsT*100:.1f}% | var_tau(any plane) {wvt*100:.1f}% | aT-rate abs err {waT:.1e} lambda | aL-rate {waL*100:.1f}%"
    print(f"sigma^2 {s2} (worst of 3 seeds):"); [print("   " + v) for v in out.values()]
# throughput: particle-lambda per second in the plane pass (65536 particles to x1 = 16 = 64 lambda)
for e in ("10", "20"):
    for N in (128, 256, 512):
        tj = json.load(open(f"{A}/raw/v2/v2_e{e}_s3102_N{N}/track.json"))
        tp = tj["timings_s"]["plane_kernel"]
        print(f"eps{e} N{N} (lam/h {N//4}): plane pass {tp:.1f} s -> {65536*64/tp:,.0f} particle-lambda/s; per (lambda/h) normalised {65536*64/tp*(N/4):,.0f}")
