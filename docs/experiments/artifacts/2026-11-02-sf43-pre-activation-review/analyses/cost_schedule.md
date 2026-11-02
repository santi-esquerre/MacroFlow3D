Every committed number reproduces. The full admissible design does not fit before the Mon 11-09 deadline once the measured +10.1 % overhead is applied. The trimmed variant fits with overhead (no +20 %) up to a Wed 14:00 launch; the displacement-basis variants fit at every launch time.

All computation ran in memory with `python3 -B -c`, calling the committed functions on the committed data. No file was written: read-only mode blocked writing to the scratch directory, so the simulator code is at the end of this message.

**Path aliases**
- N = `ROOT/docs/experiments/2026-10-30-sf42-confirmatory-tooling.md`
- N39 = `ROOT/docs/experiments/2026-10-27-sf39-macrodispersion-study.md`
- P = `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/pilots`
- S = `…/2026-10-30-sf42-confirmatory-tooling/scripts`
- A39 = `ROOT/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study`

**Which value each arm uses (column "basis")**
- **D(n):** directly measured at the same grid, σ² and M; n = number of pilot realizations, from `P/J*/*/RUN.json`, field `seconds.total`.
- **I:** interpolated linearly in M² between two J2 lattice sizes at the same σ².
- **Dm:** measured at the same σ² but another M, adjusted with the refit's tracing coefficient (4.472e-7, N:1312). That coefficient reproduces the J2 M-steps on 1040x240 (5.72 s predicted vs 5.93 s measured; 9.52 vs 9.64).
- **A:** the committed anchored value (refit × geometry anchor, `S/pilots/report.py:197`). Used where the grid was measured only at another σ².
- **E:** estimated from measured stage times.

**Pessimistic column, by evidence**
- Multi-realization measurement (D, I): ×1.07, the largest gap between two pilot runs of one configuration (J2 27.84 s vs J3 26.05 s).
- Single realization or M-adjusted: ×1.153, one refit rms (N:1312).
- σ²-extrapolated (A): the larger of value ×1.153 and refit × the worst observed under-prediction for the grid class. That is ×1.375 for shallow-multigrid grids (`c_s4_2496x288`, residual −0.273, N:1370) and ×1.106 for deep grids (N:1388).
- Par2: ×1.07 where measured, ×1.301 where estimated (N:1358).

## 1. Seconds per realization (own time; "×1.101" adds the J2 job overhead)

| grid @σ² (M) | used by | basis | refit | committed anchored | planning | ×1.101 | pessimistic | pess ×1.101 |
|---|---|---|---|---|---|---|---|---|
| 1040x240 @0.25 (128) | production, Gate-5 consumer | D(14) | 23.0 | 27.4 | **26.4** | 29.1 | 28.3 | 31.1 |
| 1040x240 @1 (128) | production, Gate-5 consumer | D(14) | 23.4 | 27.9 | **26.8** | 29.5 | 28.7 | 31.6 |
| 1664x384 @2.25 (86) | production | I(58–100) | 61.3 | 54.9 | **53.1** | 58.5 | 56.9 | 62.6 |
| 1664x384 @2.25 (100) | production (alternative M) | D(6) | 63.2 | 56.6 | 54.0 | 59.4 | 57.7 | 63.6 |
| 3328x384 @4 (68) | production | I(58–100) | 122.4 | 128.9 | **128.1** | 141.1 | 137.1 | 150.9 |
| 3328x384 @4 (100) | production (alternative M) | D(6) | 130.4 | 137.4 | 131.5 | 144.8 | 140.7 | 154.9 |
| 520x120 @0.25 / @1 (128) | E_h | A / Dm | 8.9 / 9.0 | 12.0 / 12.1 | 12.0 / 11.5 | 13.2 / 12.6 | 13.8 / 13.2 | 15.2 / 14.6 |
| 2080x480 @0.25 (128) | E_h | A | 112.2 | 138.1 | **138.1** | 152.0 | 159.2 | 175.3 |
| 2080x480 @1 (128) | E_h | A | 115.4 | 142.0 | **142.0** | 156.3 | 163.7 | 180.3 |
| 2080x480 @2.25 (86) | E_h | Dm | 112.4 | 138.2 | 140.2 | 154.3 | 161.6 | 177.9 |
| 832x192 @2.25 (86) | E_h and E_W | A | 12.9 | 13.6 | 13.6 | 15.0 | 15.7 | 17.3 |
| 416x96 @2.25 (86) | alternative third E_h rung | Dm (J1B) | 5.9 | 5.9 | 4.7 | 5.1 | 5.4 | 5.9 |
| 832x96 @4 (68) | E_h and E_W | Dm | 7.1 | 7.6 | 7.7 | 8.5 | 8.9 | 9.8 |
| 1664x192 @4 (68) | E_h, E_W, Gate-5 consumer | Dm(9) | 21.1 | 23.0 | 23.7 | 26.0 | 27.3 | 30.0 |
| 1040x480 @0.25 (128) | E_W | A | 58.0 | 79.3 | **79.3** | 87.3 | 91.4 | 100.6 |
| 1040x480 @1 (128) | E_W | A | 59.5 | 81.5 | **81.5** | 89.7 | 93.9 | 103.4 |
| 832x384 @2.25 (86) | E_W | Dm (J1B) | 32.5 | 32.5 | 29.0 | 32.0 | 33.5 | 36.9 |
| 1040x240 @2.25 (86) | E_W | Dm | 19.8 | 23.7 | 27.6 | 30.3 | 31.8 | 35.0 |
| 1040x480 @2.25 (86) | E_W | Dm | 58.0 | 79.4 | 80.9 | 89.1 | 93.3 | 102.7 |
| 832x192 @4 (68) | E_W | Dm | 12.4 | 13.0 | 13.1 | 14.4 | 15.1 | 16.6 |
| 1664x384 @4 (68) | E_W | A ¹ | 63.1 | 56.5 | 56.5 | 62.2 | 69.7 | 76.7 |
| h_max, 3 passes, 1040x240 @0.25 / @1 (128) | h_max | D + 6×tracing | 68.7 / 69.1 | 82.0 / 82.5 | 72.1 / 72.5 | 79.4 / 79.9 | 81.0 / 81.4 | 89.2 / 89.6 |
| h_max, 3 passes, 1664x384 @2.25 (86 / 100) | h_max | I / D + tracing | 94.3 / 107.9 | 84.5 / 96.6 | 86.2 / 98.6 | 94.9 / 108.6 | 94.9 / 109.2 | 104.5 / 120.2 |
| h_max, 3 passes, 3328x384 @4 (68 / 100) | h_max | I / D + tracing | 163.7 / 219.7 | 172.4 / 231.4 | 169.4 / 220.8 | 186.5 / 243.1 | 184.7 / 243.7 | 203.4 / 268.3 |
| 1248x288 @2.25 (86) | Gate-5 consumer | Dm(9) | 29.8 | 34.6 | 35.8 | 39.4 | 41.2 | 45.4 |
| Par2 horizon 3.6, 1040x240 @0.25 | Gate-5 Par2 | E ² | 13.2 | 13.8 | 12.8 | 14.1 | 16.7 | 18.4 |
| Par2 horizon 3.6, 1040x240 @1 / 1248x288 @2.25 / 1664x192 @4 | Gate-5 Par2 | D(8) J3B | 13.2 / 22.7 / 13.5 | 13.8 / 19.2 / 17.5 | 13.8 / 19.2 / 17.5 | 15.2 / 21.1 / 19.3 | 14.7 / 20.5 / 18.7 | 16.2 / 22.6 / 20.6 |
| G7 Par2 cell mean @0.25 / @1 | G7 | E ³ | 13.2 | 13.8 | 15.7 / 16.3 | 17.2 / 17.9 | 20.4 / 21.2 | 22.4 / 23.3 |
| 2080x240 @1 (128) | distance | D(1) J1B | 43.0 | 43.0 | **54.0** | 59.5 | 62.3 | 68.6 |
| 3328x384 @2.25 (86, λ/h 16) | distance | A ¹ | 118.9 | 125.3 | 125.3 | 137.9 | 144.4 | 159.0 |

Seconds per Gate-5 pair (consumer + Par2, planning): 39.2 / 40.6 / 55.0 / 41.2 at σ² 0.25 / 1 / 2.25 / 4.

**Notes on the table**
1. SF-39 measured these two exact geometries at M = 100 (`A39/raw/campaign/extend/walls.tsv` and `width/walls.tsv`):
   - Distance arm at 2.25: `s225_B` took 4288 s for 42 realizations, 102.1 s each, about 98 s at M = 86.
   - E_W at 4: `s4_w24` took 1492 s for 30 realizations, 49.7 s each, about 45.7 s at M = 68.
   - The refit over-predicts both, by 20 % and 35 %. So these two committed anchored values are conservative.
2. J3 at horizon 1.2 measured 11.37 s. I added the transport increment measured at σ² = 1 (J3B 3.09 s − J3 1.63 s).
3. **G7 cells.** The base is the Gate-5 Par2 configuration. Increments are:
   - G (Par2's own generator, SF-39's 10,000 modes): 8.55 s versus 2.04–2.16 s for the SF-18 field, so +6.4 to +6.5 s. The 1.43e-7 s per cell comes from the SF-39 profile, K_generation 19146 ms over 1.34e8 cells (`A39/raw/campaign/pilot/logs/par2_s1.log`, end of file).
   - dt/2: +2.18 s, from transport ≈ a + 1.574e-5 s per step (J3 vs J3B).
   - cg_rtol 1e-6: −0.4 × solve time, i.e. −3.0 / −3.6 s. **Assumption:** iterations scale with log(1/rtol).
   - Injection: no cost effect.
   - Cells at σ² = 1, in order GITS: −−−− 13.8, −−++ 12.4, −+−+ 10.2, −++− 16.0, +−−+ 16.6, +−+− 22.4, ++−− 20.2, ++++ 18.8 s.
   - With 1000 modes the cell means drop to 11.8 / 12.5 s.

**Likely conservative (not measured).** E_W 1040x480 and E_h 2080x480 at σ² 0.25 and 1 rest on one σ² = 2.25 run each (90 PCG iterations). On 1040x240 the same σ² change scaled time by 0.83–0.85 (26.4 / 26.8 / 31.7 s). Applying that ratio would remove about 9.6 GPU-h. Running 2–3 realizations of each at σ² = 1 in the smoke test (about 0.2 GPU-h) would settle it.

## 2. Memory class (own-process GPU peak, MiB; the V100 has 32510, N:1648)

| class | grids (peak MiB) |
|---|---|
| **Needs a near-empty GPU** | 2080x480: 28203 in J1 (single realization); estimated 29229 for multi-realization runs ⁴ · 3328x384: 29881 (J2) |
| **Half** | 1664x384: 15703 (J2) · 1040x480: 14365 in J1, estimated 15391 multi ⁴ · Par2 1248x288: 13673 (borderline) |
| **Small** | 2496x288 12471 · Par2 1664x192 8335 · Par2 1040x240 8157 · 832x384 7595 (est. 8621) · 1248x288 7517 · 2080x240 7445 (est. 8471) · 1664x192 5077 · 1040x240 4997 · 832x192 2275 · 520x120 1757 · 832x96 1701 · 416x96 1655 |

⁴ On three grids, multi-realization runs peak exactly 1026 MiB above the single-realization J1 run (1040x240: 3971 → 4997; 1248x288: 6491 → 7517; 1664x192: 4051 → 5077). Finding (e) quotes only the J1 value for 2080x480. The peak does not depend on M (N:1647).

**Two half-class jobs on one GPU:** the memory alone would fit on an otherwise empty V100 (15391 + 15703 = 31094 MiB, 1.4 GB spare; 2 × 15703 = 31406, 1.1 GB spare). But the second job cannot pass the 30000-MiB-free start check, there is no room for a colleague's process, and two processes time-sharing one GPU gain no throughput. The plan assumes **no sharing**.

## 3. Reproduction and re-sizing

Committed family values reproduce exactly (refit model, N:1429): production 20.62, E_h 24.53, E_W 49.63, h_max 4.59, Gate-5 consumer 3.17, Gate-5 Par2 2.08, G7 3.51, distance 4.50, reserve 5.63, total **118.24**. Committed anchored: **147.35**. With overhead: **162.24**.

GPU-h by family:

| family | refit (committed) | committed anchored | planning | pessimistic |
|---|---|---|---|---|
| production | 20.62 | 23.91 | 23.05 | 24.67 |
| E_h | 24.53 | 30.24 | 30.25 | 34.88 |
| E_W | 49.63 | 66.95 | 67.02 | 77.31 |
| h_max | 4.59 | 4.89 | 4.62 | 5.11 |
| Gate-5 consumer / Par2 | 3.17 / 2.08 | 3.50 / 2.50 | 3.60 / 2.49 | 4.13 / 2.68 |
| G7 | 3.51 | 3.68 | 4.26 | 5.54 |
| distance | 4.50 | 4.67 | 4.98 | 5.74 |
| subtotal | 112.61 | 140.34 | 140.27 | 160.06 |
| +5 % reserve | 118.24 | 147.35 | 147.28 | 168.06 |
| ×1.101 (with reserve) | 130.19 | 162.24 | 162.16 | 185.04 |

GPU-h per 100 realizations (planning / pessimistic):

| family | σ² 0.25 | σ² 1 | σ² 2.25 | σ² 4 |
|---|---|---|---|---|
| production | 0.734 / 0.785 | 0.745 / 0.797 | 1.476 / 1.579 | 3.559 / 3.808 |
| E_h, coarse rung | 520x120: 0.333 / 0.384 | 520x120: 0.319 / 0.368 | 832x192: 0.378 / 0.435 (416x96: 0.130 / 0.150) | 832x96: 0.214 / 0.247 |
| E_h, fine rung | 2080x480: 3.835 / 4.422 | 2080x480: 3.944 / 4.548 | 2080x480: 3.894 / 4.489 | 1664x192: 0.657 / 0.758 |
| E_W | 1040x480: 2.202 / 2.539 | 1040x480: 2.263 / 2.609 | 832x192 0.378 / 0.435 · 832x384 0.807 / 0.930 · 1040x240 0.766 / 0.883 · 1040x480 2.247 / 2.591 | 832x96 0.214 / 0.247 · 832x192 0.364 / 0.420 · 1664x192 0.657 / 0.758 · 1664x384 1.569 / 1.936 |
| h_max (3 passes) | 2.004 / 2.249 | 2.015 / 2.261 | 2.393 / 2.637 | 4.706 / 5.131 |
| Gate-5 pair | 1.090 / 1.248 | 1.128 / 1.207 | 1.526 / 1.715 | 1.144 / 1.279 |
| G7 | 0.435 / 0.566 | 0.452 / 0.589 | – | – |
| distance | – | 1.501 / 1.731 | 3.479 / 4.011 | – |

## 4. Displacement basis primary (production and E_W rows re-sized)

The variance ratios (SD_disp/SD_pos)² are 7.25e-4 / 0.0103 / 0.0724 / 0.775 (`P/hostref.json`). Scaled counts before floors would be:
- production: 0.8 / 17.0 / 8.0 / 23.2
- E_W, per arm: the same factors applied to each arm's planned count.

Every arm therefore lands on its floor. `S/pilots/analyze.py:125-127` never raises an arm above its planned count, so the σ² = 4 arms stay at 30. All other rows are unchanged.

| variant | production n | E_W n per arm | refit | committed anchored | planning | +5 % | ×1.101 | ×1.101 ×1.2 | pess ×1.101 |
|---|---|---|---|---|---|---|---|---|---|
| floor 30 | 30/30/30/30 | 30 everywhere | 47.2 | 54.9 | **55.6** | 58.4 | 61.2 | 73.5 | 70.7 |
| floor 50 | 50/50/50/30 | 50 (0.25, 1, 2.25); 30 (σ² 4) | 49.1 | 57.2 | **57.9** | 60.8 | 63.8 | 76.5 | 73.6 |
| floor 100 | 100/100/100/30 | 100 (0.25, 1, 2.25); 30 (σ² 4) | 53.9 | 63.1 | **63.7** | 66.9 | 70.2 | 84.2 | 80.8 |

In these variants E_h (30.3 GPU-h) becomes the largest block. Its N of 300 at σ² 0.25 and 1 was kept as specified, but it was sized against the position-basis noise.

## 5. Two-stage schedule

Rules used:
- **Heavy** = consumer arms on the four grids above 14 GB (2080x480, 3328x384, 1664x384, 1040x480): 100.9 GPU-h planning in the full design.
- **Stage 1:** two lanes, longest-first; no item may end after 30 h.
- **Stage 2:** one lane.
- **One resume:** 2 h of latency (one helper GPU-wait cycle, an assumption) plus re-running the realization in flight (at most 0.06 h).

All times are hours.

| variant | margin | stage-1 wall | stage-1 GPU-h | stage-2 wall | total GPU-h | total wall | wall with resume |
|---|---|---|---|---|---|---|---|
| (a) full | planning | 30.0 | 60.0 | 80.3 | 140.3 | 110.3 | 112.3 |
| | ×1.101 | 30.0 | 60.0 | 94.4 | 154.4 | 124.4 | 126.5 |
| | ×1.101 ×1.2 | 30.0 | 60.0 | 125.3 | 185.3 | 155.3 | 157.4 |
| (a′) full with J3B Gate-5 n | ×1.101 | 30.0 | 60.0 | 91.6 | 151.6 | 121.6 | 123.6 |
| (b) displacement, floor 30 | plan / ×1.101 / ×1.32 | 18.9 / 20.8 / 24.9 | 37.7 / 41.5 / 49.8 | 17.9 / 19.7 / 23.6 | 55.6 / 61.2 / 73.5 | 36.7 / 40.5 / 48.5 | 38.8 / 42.5 / 50.6 |
| (b) displacement, floor 100 | plan / ×1.101 / ×1.32 | 21.7 / 23.9 / 28.7 | 43.4 / 47.8 / 57.4 | 20.3 / 22.3 / 26.8 | 63.7 / 70.2 / 84.2 | 42.0 / 46.3 / 55.5 | 44.0 / 48.3 / 57.5 |
| (c) trimmed | plan / ×1.101 / ×1.32 | 30.0 | 60.0 | 51.1 / 62.3 / 86.8 | 111.1 / 122.3 / 146.7 | 81.1 / 92.3 / 116.8 | 83.1 / 94.3 / 118.8 |
| (b100 + c) | plan / ×1.101 / ×1.32 | 18.9 / 20.8 / 24.9 | 37.8 / 41.6 / 49.9 | 18.8 / 20.7 / 24.8 | 56.5 / 62.3 / 74.7 | 37.7 / 41.5 / 49.8 | 39.7 / 43.5 / 51.8 |

- Floor 50 sits between floors 30 and 100 (57.9 / 63.8 / 76.5 GPU-h).
- **Fill policy** (stage 1 continues with light items until 30 h) uses the same GPU-h and shortens the displacement variants' wall: floor 100 drops to 33.7 / 40.2 / 54.2 h.
- Cutting the queue into blocks of 10 realizations changes nothing (stage 1 29.97 h, stage 2 80.45 h).

## 6. Critical path (local, UTC-3)

| milestone | base | pessimistic |
|---|---|---|
| Phase 1 consult ready | Mon 11-02 20:00 | same |
| owner answer + merge | Tue 11-03 10:00 | Tue 11-03 18:00 |
| Phase 2 done (10–16 h, including a 2–4 h smoke test) | Tue 20:00 to Wed 02:00 | Wed 04:00 to Wed 10:00 |
| stage 1 ends (launch + 30 h), for launches Tue 22:00 / Wed 06:00 / Wed 14:00 / Thu 00:00 | Thu 04:00 / Thu 12:00 / Thu 20:00 / Fri 06:00 | |

**Proposed clock cutoff: Sun 11-08 16:00** (deadline minus 8 h; minus 12 h, Sun 12:00, is the conservative option). The 8 h break down as:
- Finish in-flight realizations: 0.1 h.
- Reduction tail: ≤0.25 h, assuming reduction runs in the background during the job (J2 reduced at about 5.8 s per realization, N:250/263 and RUN.json walls).
- Recompute and reconcile: 0.1 h (13 s and 3 s in SF-42, N:215-216).
- mcstat analysis: 1.5–2 h. Bootstrap B is 2000–5000 (`budget.py:125`, `gate5.py:90`, `intervals.py:434`); about 1 ms per replicate (`_util.py:94`); the host is 4.4–9.1× slower (N:794). SF-39's whole analysis, including raw reduction, took 1081 s (N39:144).
- Report and prune: 0.5 h. Pull and commit: 1 h.
- Margin: one analysis re-run plus about 1 h.

If reduction is left to the end instead, add 2.5–14 h (8572 consumer realizations at 1–5.8 s each).

Slack in hours to the Sun 16:00 cutoff, after one resume, for margins planning / ×1.101 / ×1.32. Subtract 4 h for the Sun 12:00 cutoff. Any GPU wait (2 h per helper attempt) comes out of the slack.

| launch | wall available | GPU-h available | max planning GPU-h that fits (×1 / ×1.101 / ×1.32) | full | trimmed | displacement, floor 100 | floor 100 + trimmed |
|---|---|---|---|---|---|---|---|
| Tue 22:00 | 114 h | 144 | 142 / 129 / 107.5 | +1.7 / −12.5 / −43.4 | +30.9 / +19.7 / **−4.8** | +78.3 / +71.8 / +57.8 | +83.7 / +79.7 / +67.3 |
| Wed 06:00 | 106 h | 136 | 134 / 122 / 101 | −6.3 / −20.5 / −51.4 | +22.9 / +11.7 / −12.8 | +70.3 / +63.8 / +49.8 | +75.7 / +71.7 / +59.3 |
| Wed 14:00 | 98 h | 128 | 126 / 114 / 95 | −14.3 / −28.5 / −59.4 | +14.9 / +3.7 / −20.8 | +62.3 / +55.8 / +41.8 | +67.7 / +63.7 / +51.3 |
| Thu 00:00 | 88 h | 118 | 116 / 105 / 88 | −24.3 / −38.5 / −69.4 | +4.9 / −6.3 / −30.8 | +52.3 / +45.8 / +31.8 | +57.7 / +53.7 / +41.3 |

The full design with overhead (154.4 GPU-h) also exceeds the 150 GPU-h budget even if time allowed.

## 7. Host disk

Raw bytes per realization ≈ 56 B × M² × (planes + time snapshots) for one pass, and 120 B for three passes.
- The 56 B fit SF-39's measured bytes (`A39/raw/campaign/checkpoint/bytes.json`) to about 0.1 %: s1_B 296.3 MB ↔ 197 planes + 332 times; 3-pass s1_A 635.0 MB = 120 B.
- The arrays the reducer reads (32 B of the 56) match the SF-42 manifests: 120.4 MB per realization at σ² = 1, M = 128 (`P/J2/reduced/*.manifest.json`).
- Time snapshots come from the pilot `TWIN` log lines: 180 / 159 / 130 / 100 at σ² 0.25 / 1 / 2.25 / 4.

| arm | MB per realization |
|---|---|
| σ² 0.25, M 128 | 229.4 (3 passes: 491.5) |
| σ² 1, M 128 | 210.1 (3 passes: 450.2) |
| σ² 2.25, M 86 | 82.8 (3 passes: 177.5; M 100: 112.0) |
| σ² 4, M 68 | 44.0 (3 passes: 94.3; M 100: 95.2) |
| distance @1 | 463.3 (assumes 170 planes, 335 times) |
| distance @2.25 | 178.1 (assumes about 260 times) |
| Par2 | ≥0.02 (crossings CSV) |

Totals with no pruning:

| variant | raw total |
|---|---|
| full | **1.69 TB** |
| full with J3B Gate-5 n | 1.68 TB |
| trimmed | 1.37 TB |
| displacement, floor 30 / 50 / 100 | 0.48 / 0.51 / 0.57 TB |
| floor 100 + trimmed | 0.49 TB |

- **Write rate per lane:** production 1040x240 at 25.6–28.4 GB/h (about 57 GB/h if two lanes run it); E_W 1040x480 at 8.4–9.5 GB/h.
- **Cadence:** reduce and prune at least hourly in the background, keeping the backlog under about 60 GB. The retained subset (≥5 per arm, N:1584) is about 20–25 GB.
- **Not determined:**
  - Free disk on the host.
  - Whether SF-39's ~176 GB tree (N39:541) is still there.
  - The size of Par2's full output directory.
  - The distance arms' final plane and time counts.
  - Whether any time-snapshot or Pollock arrays must be kept after reduction.

## 8. Ranking (planning, full design)

| rank | family | GPU-h | share | main blocks |
|---|---|---|---|---|
| 1 | E_W | 67.0 | 48 % | 1040x480: @1 37.3, @0.25 24.2 |
| 2 | E_h | 30.3 | 22 % | 2080x480 rungs 27.6 |
| 3 | production | 23.1 | 16 % | |
| 4 | distance | 5.0 | | |
| 5 | h_max | 4.6 | | |
| 6 | G7 | 4.3 | | |
| 7 | Gate-5 consumer | 3.6 | | σ² 4 alone 2.8 |
| 8 | Gate-5 Par2 | 2.5 | | |

**Stage 1** (60 GPU-h) is the 2080x480 E_h rungs at 27.6 GPU-h (46 %) and E_W 1040x480 at σ² 1 at 25.2 GPU-h (42 %), plus 3328x384 items at 6.0 (10 %).

Putting all near-empty-GPU items (33.6 GPU-h) on lane A means lane B only needs a GPU with about 16 GB free, with no wall-time cost. This needs stage 2 to keep lane A's GPU.

## Inconsistencies in the committed numbers

1. **162.3** is 147.4 (already rounded) × 1.101; unrounded it is 162.24.
2. **Gate-5 n vs cost.** The admissible design uses Gate-5 n = 7 / 16 / 63 / 426, which come from J3 at horizon 1.2, but costs Par2 at horizon 3.6. J3B with the note's own floor of 30 gives 30 / 30 / 30 / 205 (N:1284-1286, Finding (h)), which is −2.7 GPU-h.
3. **Anchors ignore σ².**
   - E_W 2.25 on 1040x240 gets 23.7 s; the only same-σ² measurement gives 27.6 s.
   - The distance arm at 2.25 and E_W 1664x384 at 4 borrow anchors from another σ² and λ/h; the SF-39 measurements of those exact arms are lower.
4. **h_max anchored** multiplies the whole three-pass model by the anchor: 82.0 s, versus 72.1 s from the measured single pass plus tracing.
5. **The 10.1 % overhead** was measured on a job that reduced inline (N:250). It splits into 1.6 % process start-up (4947.6 s of process walls vs 4871.8 s own time) and 8.6 % between-process reduction. With background reduction and blocks of 10 or more realizations per process it would be about 1 %. Dispatching one realization per process would instead add about 4 s each, about +15 % on production.
6. **Memory:** J1's single-realization peaks understate multi-realization peaks by 1026 MiB (section 2).
7. **G7** is costed at the matched Gate-5 Par2 cost (13.15 s) for all eight cells; the cell mean is 15.7–16.3 s.
8. **Note's variant B (45.3)** uses the non-admissible M_plan. On the admissible design it is 47.2 (refit) / 55.6 (planning).
9. **Not in any projection:**
   - σ² = 4 solves hitting the PCG cap (2000 iterations, `P/hostref.json`; SF-39 had 3 fallbacks).
   - SF-39's 2× slowdown on the shared host (`s4_tw` 42.2 s vs 21.9 s).
   - Finding (j): Gate-5 TOST does not qualify at n ≤ 170, so Gate-5 n may grow.

## Simulator (ran inline; not saved)

```python
def two_stage(items, heavy, cap_h=30.0, fill=False):
    """items: [(seconds, arm)]; heavy(arm)->bool. Stage 1: 2 lanes, longest first, nothing ends after cap; stage 2: 1 lane."""
    H = sorted((i for i in items if heavy(i[1])), key=lambda x: -x[0])
    L = sorted((i for i in items if not heavy(i[1])), key=lambda x: -x[0])
    lanes, left, cap = [0.0, 0.0], [], cap_h * 3600
    for d, a in H + (L if fill else []):
        k = 0 if lanes[0] <= lanes[1] else 1
        if lanes[k] + d <= cap: lanes[k] += d
        else: left.append((d, a))
    s1, s2 = max(lanes), sum(d for d, _ in left + ([] if fill else L))
    return dict(s1_h=s1/3600, s1_gpuh=2*s1/3600, s2_h=s2/3600, gpuh=(2*s1+s2)/3600, wall_h=(s1+s2)/3600)
# items = [(sec_per_real * margin, arm)] * n per arm; heavy = arm grid in {2080x480, 3328x384, 1664x384, 1040x480}
# resume = +2 h latency + max(item) re-run
```

### Critical files for implementation
- `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/pilots/cost.py` (`plan`, the arm list)
- `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/pilots/report.py` (anchor and variants, lines 186-200 and 297-319)
- `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/mcstat/costmodel.py`
- `ROOT/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/pilots/cost_projection.json` and `P/J*/*/RUN.json`
- `ROOT/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/scripts/run_sf39.sh` (model for `run_sf43.sh`)

Not checked: host free disk, GPU availability and the real Par2 G7 configuration; the solver-tolerance and 10,000-mode generator costs are assumptions. `__pycache__` directories appeared in `S/mcstat` and `S/seedlattice` at 10:44–10:50. They are not from my runs: my runs also imported `pilots`, `sf42red`, `costmodel` and `gate5`, and none of those have `.pyc` files. They are gitignored, git status is clean, and I left them in place.

Main risk: the full design only fits with no margin at a Tue 22:00 launch with zero GPU wait. Two near-empty GPUs at once, and reduction running in the background during the job, are hard requirements.