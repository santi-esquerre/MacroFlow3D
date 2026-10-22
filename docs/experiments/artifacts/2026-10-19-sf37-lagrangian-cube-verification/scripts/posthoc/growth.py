# Post-hoc (NOT pre-registered) reading of committed SF-37 v2 moments: growth part of sigma_T^2, e_v vs x1.
import csv, sys, math
A = sys.argv[1] + "/raw/v2"
def load(e, s, N):
    rows = list(csv.DictReader(open(f"{A}/v2_e{e}_s{s}_N{N}/moments_planes.csv")))
    return {float(r["x1"]): r for r in rows}
Ns = [32, 40, 64, 80, 128, 256, 512]
for e in (10, 15, 20):
    for s in (3101, 3102, 3103):
        d = {N: load(e, s, N) for N in Ns}
        st = {N: {x: (float(r["var_x2"]) + float(r["var_x3"])) / 2 for x, r in d[N].items()} for N in Ns}
        x0 = 0.25
        print(f"eps{e} s{s}: sigT2(512) x1=0.25 {st[512][0.25]:.4f} x1=4 {st[512][4.0]:.4f} x1=16 {st[512][16.0]:.4f} | "
              f"var_x2 16: {float(d[512][16.0]['var_x2']):.4f} var_x3 16: {float(d[512][16.0]['var_x3']):.4f}")
        # growth G(x)=sigT2(x)-sigT2(0.25); relative diff vs 512 at x1=16 and x1=4
        for x in (4.0, 16.0):
            g512 = st[512][x] - st[512][x0]
            rel = {N: (st[N][x] - st[N][x0] - g512) / g512 for N in Ns[:-1]}
            relQ = {N: (st[N][x] - st[512][x]) / st[512][x] for N in Ns[:-1]}
            print(f"   x1={x:>4}: growth512={g512:.4f} ({g512/st[512][x]*100:.0f}% of sigT2) | rel err of growth: " +
                  " ".join(f"{N}:{rel[N]*100:+.1f}%" for N in Ns[:-1]) + " | rel err sigT2: " +
                  " ".join(f"{N}:{relQ[N]*100:+.1f}%" for N in Ns[:-1]))
        print("   rms_ev x1=4/16 at N=128: %.3g / %.3g ; N=512: %.3g / %.3g" % (
            float(d[128][4.0]["rms_ev"]), float(d[128][16.0]["rms_ev"]), float(d[512][4.0]["rms_ev"]), float(d[512][16.0]["rms_ev"])))
