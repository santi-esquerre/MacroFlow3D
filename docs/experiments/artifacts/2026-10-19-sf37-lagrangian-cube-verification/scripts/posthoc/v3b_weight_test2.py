# Same test at N = 512 with the face-ownership rule of pollock_init_state: on a lower face (r == 0) with face
# velocity <= 0 the lower cell is taken, so d_a = +h/2 if v_a(face) > 0 else -h/2. Sign proxy: v_s at the seed
# (seed_v.npy of the V1 run with the same field, N and seed lattice).
import sys, numpy as np
ROOT = sys.argv[1]; M = 256; N = 512; h = 1.0 / N; PL = [0.25, 0.5, 1, 2, 4, 8, 16]
def wvar(w, x):
    ok = np.isfinite(x); w = w[ok] / w[ok].sum(); m = (w * x[ok]).sum(); return (w * (x[ok] - m) ** 2).sum()
def sT2(w, X, k): return 0.5 * (wvar(w, X[:, k, 0]) + wvar(w, X[:, k, 1]))
for e in ("10", "15", "20"):
    for s in (3101, 3102, 3103):
        d = f"{ROOT}/v3b/v3b_e{e}_s{s}_N{N}"
        sd = np.load(f"{d}/seeds.npy"); wo = np.load(f"{d}/weights.npy"); wp = np.load(f"{d}/pollock_weights.npy")
        sv = np.load(f"{ROOT}/v1/v1_e{e}_s{s}_N{N}_tol1e-8_h8/seed_v.npy")
        sd1 = np.load(f"{ROOT}/v1/v1_e{e}_s{s}_N{N}_tol1e-8_h8/seeds.npy"); assert np.array_equal(sd, sd1)
        wo_r = wo / wo.mean(); wp_r = wp / wp.mean(); dw = wp_r - wo_r
        W = wo_r.reshape(M, M)
        g2 = ((np.roll(W, -1, 0) - np.roll(W, 1, 0)) * (M / 2.0)).ravel()
        g3 = ((np.roll(W, -1, 1) - np.roll(W, 1, 1)) * (M / 2.0)).ravel()
        d2 = np.where(sv[:, 1] > 0, h / 2, -h / 2); d3 = np.where(sv[:, 2] > 0, h / 2, -h / 2)
        pred = d2 * g2 + d3 * g3
        beta = (dw @ pred) / (pred @ pred); corr = np.corrcoef(dw, pred)[0, 1]
        X = np.load(f"{d}/planes_x23.npy")
        act = [sT2(wp_r, X, k) - sT2(wo_r, X, k) for k in range(7)]
        prd = [sT2(wo_r + pred, X, k) - sT2(wo_r, X, k) for k in range(7)]
        print(f"eps{e} s{s} N512: frac v2>0 {np.mean(sv[:,1]>0):.2f} v3>0 {np.mean(sv[:,2]>0):.2f} | rms(dw) {np.sqrt(np.mean(dw**2)):.2e} "
              f"rms(pred) {np.sqrt(np.mean(pred**2)):.2e} slope {beta:.3f} corr {corr:.3f} rms(dw-pred) {np.sqrt(np.mean((dw-pred)**2)):.2e}\n"
              "      WP actual/pred: " + " ".join(f"{PL[k]:g}:{act[k]:+.2e}/{prd[k]:+.2e}" for k in range(7)))
