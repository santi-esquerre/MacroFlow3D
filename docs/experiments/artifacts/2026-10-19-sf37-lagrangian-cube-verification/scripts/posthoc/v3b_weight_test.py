# Post-hoc, read-only test of the V3b weight-offset hypothesis on existing SF-37 outputs (no new runs).
# Prediction: Pollock weight = RT0 x1-velocity of the cell chosen at the seed = face flux centred at the cell
# centre, so dw = w_P - w_O6 ~ d2 dw/dx2 + d3 dw/dx3 with d = cell_centre - seed (d = +h/2 at N = 512, 0 at 256).
import sys, os, numpy as np
ROOT = sys.argv[1]; M = 256; PL = [0.25, 0.5, 1, 2, 4, 8, 16]
def wvar(w, x):
    ok = np.isfinite(x); w = w[ok] / w[ok].sum(); m = (w * x[ok]).sum(); return (w * (x[ok] - m) ** 2).sum()
def sT2(w, X, k): return 0.5 * (wvar(w, X[:, k, 0]) + wvar(w, X[:, k, 1]))
cells = np.load(f"{ROOT}/v3b/v3b_e20_s3101_N512/pollock_cells.npy"); print("pollock_cells", cells.dtype, cells.shape, cells[:3].tolist())
for e in ("10", "15", "20"):
    for s in (3101, 3102, 3103):
        out = []
        for N in (64, 128, 256, 512):
            d = f"{ROOT}/v3b/v3b_e{e}_s{s}_N{N}"
            sd = np.load(f"{d}/seeds.npy"); wo = np.load(f"{d}/weights.npy"); wp = np.load(f"{d}/pollock_weights.npy")
            wo_r = wo / wo.mean(); wp_r = wp / wp.mean(); dw = wp_r - wo_r
            W = wo_r.reshape(M, M)  # p = m2 M + m3
            g2 = ((np.roll(W, -1, 0) - np.roll(W, 1, 0)) * (M / 2.0)).ravel()
            g3 = ((np.roll(W, -1, 1) - np.roll(W, 1, 1)) * (M / 2.0)).ravel()
            c2 = np.floor(sd[:, 1] * N); c3 = np.floor(sd[:, 2] * N)
            d2 = (c2 + 0.5) / N - sd[:, 1]; d3 = (c3 + 0.5) / N - sd[:, 2]
            pred = d2 * g2 + d3 * g3
            rms = np.sqrt(np.mean(dw ** 2)); rp = np.sqrt(np.mean(pred ** 2))
            beta = (dw @ pred) / (pred @ pred) if rp > 0 else float("nan")
            corr = np.corrcoef(dw, pred)[0, 1] if rp > 0 else float("nan")
            resid = np.sqrt(np.mean((dw - (pred if rp > 0 else 0)) ** 2))
            item = f"N{N}: d2 in [{d2.min()*N:+.3f},{d2.max()*N:+.3f}]h rms(dw) {rms:.2e} rms(pred) {rp:.2e} slope {beta:.3f} corr {corr:.3f} rms(dw-pred) {resid:.2e}"
            if N == 512:
                X = np.load(f"{d}/planes_x23.npy"); wpred = wo_r + pred
                act = [sT2(wp_r, X, k) - sT2(wo_r, X, k) for k in range(7)]
                prd = [sT2(wpred, X, k) - sT2(wo_r, X, k) for k in range(7)]
                item += "\n      WP(actual) vs WP(pred from offset): " + " ".join(f"{PL[k]:g}:{act[k]:+.2e}/{prd[k]:+.2e}" for k in range(7))
            out.append(item)
        print(f"eps{e} s{s}\n   " + "\n   ".join(out))
