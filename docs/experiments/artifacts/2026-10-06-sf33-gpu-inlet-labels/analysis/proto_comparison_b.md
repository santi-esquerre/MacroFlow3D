# SF-33 N6b — prototype reproduction (step 9a) with the coarse-corrected solver (C3 driver defaults)

V100 re-run of the N6 matrix after N7a/N7b/N7c, mirror `~/MacroFlow3D-SF-33` synced from commit `bfe4efb`
(chain N0..N7c + C3; index and job records: `README.md`). Facts first; the reading is in the last section.

## Sources

- GPU: `logs/proto_b/<case>.log` + `raw/proto_b/<case>.json` (job `sf33b-proto`; driver defaults of C3, no extra
  option: `--coarse mult --coarse-profiles 2 --coarse-assembly colored --coarse-factor banded --psitc on --forcing ew
  --restart 100 --max-inner 6000 --max-newton 120 --lin-tol 1e-12 --newton-tol 1e-13 --bisect 4
  --gmres-stagnation-factor 0.9`; the SOLVER line of every log and `solver_config` of every JSON record them).
  `--solution <exports>/solutions/<case>_i1o4` was passed wherever that directory exists on the host: every 16^3
  case with a saved SF-29 solution and the five 24^3 references of N6 (`sf33-proto-ref24`, `~/sf33_ref24`).
- Prototype, full precision: as N6 (`solution.json` of the saved 16^3 solutions and of the five 24^3 references).
  Prototype, 4 significant digits: the sweep2 cell logs (32^3 eps 0.25, `control2d:0.25:24`, `generic3d:1:20/24`).
- New in N6b: `gauss:0.5:32` and `gauss_ch:0.5:32` (the gate's 32^3 point at eps 0.5). **No prototype reference
  exists** (SF-29 sweep2 timed out there); exports produced by `sf33b-export` (`export_proto.py gauss:0.5:32
  gauss_ch:0.5:32`; the 32^3 oracle caches were written to the SF-29 `raw/cache/` on the host, gitignored). These
  rows are judged on convergence, r_F and the oracle-ceiling sanity only (GPU metrics of the exported oracle labels,
  `cand=oracle_fd4`, vs the exporter's `ref_metrics.json`).
- Comparison: `scripts/compare_proto.py ../logs/proto_b/*.log --exports ../exports --out
  ../raw/proto_b/compare_proto_b.md` on the host (job `sf33b-compare`; per-key 17-digit output in
  `logs/jobs/sf33b-compare.log`; exit 1 = some case FAIL, by design). Newton/GMRES digest:
  `scripts/digest_newton.py ../raw/proto_b/*.json > ../raw/proto_b/digest_b.txt` (same job).

## Reading rule (unchanged from N6)

- `PASS`: GPU vs prototype at full precision, `|gpu - proto| / |proto| <= 1e-6` (16^3 cases and the five 24^3 cases
  with a reference).
- `PASS(4dig)`: prototype value printed with 4 significant digits; the band is the rounding bound of the printed
  value; a 1e-6 agreement cannot be resolved and is not claimed (32^3 eps 0.25, `control2d:0.25:24`).
- `roundoff`: both sides below 1e-10: not gated. `n/a`: no prototype value (32^3 eps 0.5: **no prototype
  reference**).
- `overall` gates every CASE key of both `cand=i1o4` and the ceiling `cand=oracle_fd4`, `STATUS converged`,
  `r_F <= 1e-10`, `FIELDDIFF joint <= 1e-8`. PATH equality is informational (since N7b).

## Table produced by `compare_proto.py` (`raw/proto_b/compare_proto_b.md`, verbatim)

| case | status | r_F | PATH equal (info) | e_v gpu | e_v proto | e_v verdict | e_psi gpu | e_psi proto | e_psi verdict | ceiling e_v verdict | FIELDDIFF joint | r_F hist digits | GMRES max / median (target) | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control2d:0.25:16 | converged | 5.03e-14 | yes | 3.067326e-03 | 3.067326e-03 | PASS | 3.220421e-02 | 3.220421e-02 | PASS | PASS | 8.58e-13 | 15.9 | 9 / 3.5 | PASS |
| control2d:0.25:24 | converged | 2.86e-15 | yes | 5.246774e-04 | 5.247000e-04 | PASS(4dig) | 2.666897e-03 | 2.667000e-03 | PASS(4dig) | PASS | - | - | 7 / 3.0 | PASS |
| control2d:0.5:16 | converged | 2.65e-15 | yes | 7.686791e-03 | 7.686791e-03 | PASS | 3.183307e-02 | 3.183307e-02 | PASS | PASS | 8.90e-15 | 14.2 | 11 / 5.0 | PASS |
| control2d:0.5:24 | converged | 5.40e-15 | yes | 1.618042e-03 | 1.618042e-03 | PASS | 1.065001e-02 | 1.065001e-02 | PASS | PASS | 7.91e-15 | 15.7 | 10 / 4.0 | PASS |
| gauss:0.25:16 | converged | 9.64e-14 | yes | 5.505929e-03 | 5.505929e-03 | PASS | 7.726616e-02 | 7.726616e-02 | PASS | PASS | 7.05e-13 | -4.1 | 30 / 6.5 | PASS |
| gauss:0.25:24 | converged | 6.79e-15 | no (`0.25` vs `0.25(fail)->0.125->0.25`) | 1.808087e-03 | 1.808087e-03 | PASS | 2.886058e-02 | 2.886058e-02 | PASS | PASS | 2.14e-14 | -6.4 | 49 / 8.5 | PASS |
| gauss:0.25:32 | converged | 9.85e-15 | no (`0.25` vs `0.25(fail)->0.125->0.25`) | 8.014281e-04 | 8.014000e-04 | PASS(4dig) | 1.302715e-02 | 1.303000e-02 | PASS(4dig) | PASS | - | - | 77 / 8.5 | PASS |
| gauss:0.5:16 | converged | 6.31e-15 | no (`0.25->0.5` vs `0.25->0.5(fail)->0.375->0.5`) | 1.908385e-02 | 1.908385e-02 | PASS | 1.402589e-01 | 1.402589e-01 | PASS | PASS | 6.14e-15 | -4.2 | 74 / 26.0 | PASS |
| gauss:0.5:24 | converged | 4.74e-14 | no (`0.25->0.5` vs `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`) | 8.473961e-03 | 8.473961e-03 | PASS | 6.831467e-02 | 6.831467e-02 | PASS | PASS | 7.58e-14 | -4.4 | 400 / 55.0 | PASS |
| gauss:0.5:32 | continuation_floor | 1.41e-02 | no (`0.25->0.5(fail)->0.375->0.5(fail)->0.4375(fail)->0.40625->0.4375(fail)->0.421875->0.4375(fail)->0.5(final)` vs `None`) | 1.387424e-02 | - | n/a | 1.150721e-01 | - | n/a | PASS | - | - | 400 / 26.0 | FAIL |
| gauss_ch:0.25:16 | converged | 4.20e-15 | no (`0.25` vs `0.25(fail)->0.125->0.25`) | 5.287384e-03 | 5.287384e-03 | PASS | 6.780355e-02 | 6.780355e-02 | PASS | PASS | 4.57e-15 | -4.6 | 24 / 6.0 | PASS |
| gauss_ch:0.25:24 | converged | 7.32e-15 | no (`0.25` vs `0.25(fail)->0.125->0.25`) | 1.783041e-03 | 1.783041e-03 | PASS | 2.482687e-02 | 2.482687e-02 | PASS | PASS | 2.09e-14 | -6.4 | 45 / 7.0 | PASS |
| gauss_ch:0.25:32 | converged | 1.22e-14 | no (`0.25` vs `0.25(fail)->0.125->0.25`) | 7.657190e-04 | 7.657000e-04 | PASS(4dig) | 1.086056e-02 | 1.086000e-02 | PASS(4dig) | PASS | - | - | 86 / 8.5 | PASS |
| gauss_ch:0.5:16 | converged | 7.62e-15 | no (`0.25->0.5` vs `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`) | 1.784597e-02 | 1.784597e-02 | PASS | 1.231619e-01 | 1.231619e-01 | PASS | PASS | 1.27e-14 | -4.1 | 74 / 26.0 | PASS |
| gauss_ch:0.5:24 | converged | 4.00e-14 | no (`0.25->0.5` vs `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5`) | 8.419261e-03 | 8.419261e-03 | PASS | 6.221403e-02 | 6.221403e-02 | PASS | PASS | 9.32e-14 | -5.5 | 500 / 74.0 | PASS |
| gauss_ch:0.5:32 | continuation_floor | 9.66e-03 | no (`0.25->0.5(fail)->0.375->0.5(fail)->0.4375->0.5(fail)->0.46875(fail)->0.453125(fail)->0.5(final)` vs `None`) | 1.074118e-02 | - | n/a | 7.022528e-02 | - | n/a | PASS | - | - | 700 / 73.0 | FAIL |
| generic3d:0.25:16 | converged | 4.55e-15 | yes | 3.423212e-03 | 3.423212e-03 | PASS | 4.633915e-02 | 4.633915e-02 | PASS | PASS | 1.79e-14 | -5.8 | 24 / 8.0 | PASS |
| generic3d:0.5:16 | converged | 6.13e-15 | yes | 1.459883e-02 | 1.459883e-02 | PASS | 1.017169e-01 | 1.017169e-01 | PASS | PASS | 1.90e-14 | -7.1 | 77 / 19.0 | PASS |
| generic3d:1:16 | continuation_floor | 4.39e-02 | no (`0.25->0.5->1(fail)->0.75->1(fail)->0.875->1(fail)->0.9375(fail)->0.90625(fail)->1(final)` vs `0.25->0.5->1(fail)->0.75->1(fail)->0.875->1(fail)->0.9375->1`) | 9.211381e-02 | 1.033824e-01 | FAIL | 3.283959e-01 | 3.468169e-01 | FAIL | PASS | 1.83e-01 | -1.9 | 800 / 95.0 | FAIL |
| generic3d:1:20 | continuation_floor | 4.60e-02 | no (`0.25->0.5->1(fail)->0.75->1(fail)->0.875(fail)->0.8125(fail)->0.78125(fail)->1(final)` vs `0.25->0.5->1(fail)->0.75->1(fail)->0.875(fail)->0.8125->0.875->1`) | 8.056217e-02 | 5.885000e-02 | FAIL(4dig) | 3.429685e-01 | 2.342000e-01 | FAIL(4dig) | PASS | - | - | 400 / 33.0 | FAIL |
| generic3d:1:24 | continuation_floor | 8.07e-02 | no (`0.25->0.5->1(fail)->0.75(fail)->0.625->0.75(fail)->0.6875(fail)->0.65625->0.6875(fail)->1(final)` vs `0.25->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)->0.875->1(fail)->0.9375->1`) | 1.033788e-01 | 4.400000e-02 | FAIL(4dig) | 4.450555e-01 | 1.782000e-01 | FAIL(4dig) | PASS | - | - | 400 / 12.0 | FAIL |

## Per-case values, both sides (from `logs/jobs/sf33b-compare.log`, `cand=i1o4`)

Cell = `gpu / prototype (rel diff, verdict)`, `%.10e`; last columns: maximum relative difference over every gated
key of both candidates (`i1o4` and `oracle_fd4`), and the keys that FAIL. The two 32^3 eps 0.5 rows have **no
prototype reference**: their maximum is over the ceiling keys only (GPU ceiling metrics vs the exporter's
`ref_metrics.json`), and their `i1o4` values belong to an unconverged state.

| case | GPU status | prototype source | e_v gpu / proto (rel, verdict) | e_psi | e_psi1 | e_psi2 | max rel diff over gated keys (both cands) | FAIL keys |
|---|---|---|---|---|---|---|---|---|
| control2d:0.25:16 | converged | solution.json (full precision) | 3.0673264314e-03 / 3.0673264314e-03 (4.78e-12, PASS) | 3.2204212467e-02 / 3.2204212468e-02 (3.84e-11, PASS) | 3.2204212467e-02 / 3.2204212468e-02 (3.84e-11, PASS) | 5.1749706216e-12 / 5.1749940265e-12 (nan, roundoff) | 3.84e-11 | none |
| control2d:0.25:24 | converged | i1o4-control2d-0.25-N24.txt (4 significant digits) | 5.2467742300e-04 / 5.2470000000e-04 (4.30e-05, PASS(4dig)) | 2.6668969883e-03 / 2.6670000000e-03 (3.86e-05, PASS(4dig)) | 2.6668969883e-03 / 2.6670000000e-03 (3.86e-05, PASS(4dig)) | 5.2614329201e-12 / 5.2610000000e-12 (nan, roundoff) | 4.62e-04 | none |
| control2d:0.5:16 | converged | solution.json (full precision) | 7.6867914868e-03 / 7.6867914868e-03 (3.84e-15, PASS) | 3.1833069452e-02 / 3.1833069452e-02 (2.79e-14, PASS) | 3.1833069452e-02 / 3.1833069452e-02 (2.79e-14, PASS) | 2.3338651383e-13 / 2.3342565259e-13 (nan, roundoff) | 2.79e-14 | none |
| control2d:0.5:24 | converged | solution.json (full precision) | 1.6180422779e-03 / 1.6180422779e-03 (1.35e-14, PASS) | 1.0650007214e-02 / 1.0650007214e-02 (3.32e-14, PASS) | 1.0650007214e-02 / 1.0650007214e-02 (3.32e-14, PASS) | 2.3471424199e-13 / 2.3467844175e-13 (nan, roundoff) | 3.32e-14 | none |
| gauss:0.25:16 | converged | solution.json (full precision) | 5.5059288178e-03 / 5.5059288178e-03 (7.40e-13, PASS) | 7.7266164257e-02 / 7.7266164256e-02 (1.75e-12, PASS) | 7.7266164257e-02 / 7.7266164256e-02 (1.75e-12, PASS) | 5.8594649382e-02 / 5.8594649382e-02 (6.70e-14, PASS) | 1.75e-12 | none |
| gauss:0.25:24 | converged | solution.json (full precision) | 1.8080867184e-03 / 1.8080867184e-03 (6.66e-14, PASS) | 2.8860579641e-02 / 2.8860579641e-02 (8.04e-14, PASS) | 2.8860579641e-02 / 2.8860579641e-02 (8.04e-14, PASS) | 2.1059960187e-02 / 2.1059960187e-02 (4.27e-14, PASS) | 8.04e-14 | none |
| gauss:0.25:32 | converged | i1o4-gauss-0.25-N32.txt (4 significant digits) | 8.0142813632e-04 / 8.0140000000e-04 (3.51e-05, PASS(4dig)) | 1.3027149708e-02 / 1.3030000000e-02 (2.19e-04, PASS(4dig)) | 1.3027149708e-02 / 1.3030000000e-02 (2.19e-04, PASS(4dig)) | 9.9516667508e-03 / 9.9520000000e-03 (3.35e-05, PASS(4dig)) | 3.97e-04 | none |
| gauss:0.5:16 | converged | solution.json (full precision) | 1.9083846132e-02 / 1.9083846132e-02 (1.82e-16, PASS) | 1.4025886950e-01 / 1.4025886950e-01 (1.78e-15, PASS) | 1.4025886950e-01 / 1.4025886950e-01 (1.78e-15, PASS) | 9.9536390148e-02 / 9.9536390148e-02 (1.30e-14, PASS) | 1.50e-13 | none |
| gauss:0.5:24 | converged | solution.json (full precision) | 8.4739608321e-03 / 8.4739608321e-03 (9.91e-14, PASS) | 6.8314673933e-02 / 6.8314673933e-02 (2.66e-14, PASS) | 6.8314673933e-02 / 6.8314673933e-02 (2.66e-14, PASS) | 4.4045449389e-02 / 4.4045449389e-02 (9.34e-14, PASS) | 1.35e-13 | none |
| gauss:0.5:32 | continuation_floor | none (4 significant digits) | 0.013874240723479217 / - (nan, n/a) | 0.11507214383625569 / - (nan, n/a) | 0.10966921998879973 / - (nan, n/a) | 0.11507214383625569 / - (nan, n/a) | 4.70e-15 | none |
| gauss_ch:0.25:16 | converged | solution.json (full precision) | 5.2873838165e-03 / 5.2873838165e-03 (6.23e-15, PASS) | 6.7803545078e-02 / 6.7803545078e-02 (1.13e-14, PASS) | 6.7803545078e-02 / 6.7803545078e-02 (1.13e-14, PASS) | 4.1388179361e-02 / 4.1388179361e-02 (1.09e-14, PASS) | 1.24e-14 | none |
| gauss_ch:0.25:24 | converged | solution.json (full precision) | 1.7830408700e-03 / 1.7830408700e-03 (1.17e-14, PASS) | 2.4826874602e-02 / 2.4826874602e-02 (5.73e-15, PASS) | 2.4826874602e-02 / 2.4826874602e-02 (5.73e-15, PASS) | 1.5026994982e-02 / 1.5026994982e-02 (3.09e-14, PASS) | 3.09e-14 | none |
| gauss_ch:0.25:32 | converged | i1o4-gauss_ch-0.25-N32.txt (4 significant digits) | 7.6571896425e-04 / 7.6570000000e-04 (2.48e-05, PASS(4dig)) | 1.0860563406e-02 / 1.0860000000e-02 (5.19e-05, PASS(4dig)) | 1.0860563406e-02 / 1.0860000000e-02 (5.19e-05, PASS(4dig)) | 6.7772368188e-03 / 6.7770000000e-03 (3.49e-05, PASS(4dig)) | 1.34e-04 | none |
| gauss_ch:0.5:16 | converged | solution.json (full precision) | 1.7845965058e-02 / 1.7845965058e-02 (3.11e-15, PASS) | 1.2316189898e-01 / 1.2316189898e-01 (9.01e-15, PASS) | 1.2316189898e-01 / 1.2316189898e-01 (9.01e-15, PASS) | 6.0805706746e-02 / 6.0805706746e-02 (2.04e-14, PASS) | 1.75e-13 | none |
| gauss_ch:0.5:24 | converged | solution.json (full precision) | 8.4192608082e-03 / 8.4192608082e-03 (7.19e-14, PASS) | 6.2214029352e-02 / 6.2214029352e-02 (8.59e-15, PASS) | 6.2214029352e-02 / 6.2214029352e-02 (8.59e-15, PASS) | 3.2592440913e-02 / 3.2592440913e-02 (6.09e-14, PASS) | 7.19e-14 | none |
| gauss_ch:0.5:32 | continuation_floor | none (4 significant digits) | 0.010741178741796704 / - (nan, n/a) | 0.070225283611563591 / - (nan, n/a) | 0.070225283611563591 / - (nan, n/a) | 0.067964439278858091 / - (nan, n/a) | 2.77e-15 | none |
| generic3d:0.25:16 | converged | solution.json (full precision) | 3.4232120896e-03 / 3.4232120896e-03 (8.61e-15, PASS) | 4.6339148648e-02 / 4.6339148648e-02 (3.47e-14, PASS) | 2.7939156021e-02 / 2.7939156021e-02 (3.25e-14, PASS) | 4.6339148648e-02 / 4.6339148648e-02 (3.47e-14, PASS) | 9.01e-13 | none |
| generic3d:0.5:16 | converged | solution.json (full precision) | 1.4598831034e-02 / 1.4598831034e-02 (2.97e-15, PASS) | 1.0171694625e-01 / 1.0171694625e-01 (7.78e-15, PASS) | 7.7984919249e-02 / 7.7984919249e-02 (1.82e-14, PASS) | 1.0171694625e-01 / 1.0171694625e-01 (7.78e-15, PASS) | 4.40e-13 | none |
| generic3d:1:16 | continuation_floor | solution.json (full precision) | 9.2113809289e-02 / 1.0338238572e-01 (1.09e-01, FAIL) | 3.2839588504e-01 / 3.4681690946e-01 (5.31e-02, FAIL) | 3.2839588504e-01 / 3.4681690946e-01 (5.31e-02, FAIL) | 2.8915929792e-01 / 2.9004096633e-01 (3.04e-03, FAIL) | 1.59e-01 | e_v,e_psi,e_psi1,e_psi2,e_i1,e_i2,e_div,min_c,p0.1,p1,p5,p50 |
| generic3d:1:20 | continuation_floor | i1o4-generic3d-1-N20.txt (4 significant digits) | 8.0562173017e-02 / 5.8850000000e-02 (3.69e-01, FAIL(4dig)) | 3.4296847295e-01 / 2.3420000000e-01 (4.64e-01, FAIL(4dig)) | 3.1481818796e-01 / 2.3420000000e-01 (3.44e-01, FAIL(4dig)) | 3.4296847295e-01 / 1.6850000000e-01 (1.04e+00, FAIL(4dig)) | 1.04e+00 | e_v,e_psi,e_psi1,e_psi2,e_i1,e_i2,e_div,min_c,p0.1,p1,p5,p50 |
| generic3d:1:24 | continuation_floor | i1o4-generic3d-1-N24.txt (4 significant digits) | 1.0337882207e-01 / 4.4000000000e-02 (1.35e+00, FAIL(4dig)) | 4.4505553749e-01 / 1.7820000000e-01 (1.50e+00, FAIL(4dig)) | 3.7936063170e-01 / 1.7820000000e-01 (1.13e+00, FAIL(4dig)) | 4.4505553749e-01 / 1.2340000000e-01 (2.61e+00, FAIL(4dig)) | 2.61e+00 | e_v,e_psi,e_psi1,e_psi2,e_i1,e_i2,e_div,min_c,p0.1,p1,p5,p50 |

## Continuation outcome of the cases that did not converge (`raw/proto_b/<case>.json`, `continuation`)

| case | status | last accepted amplitude | bisections used | r_F at stop | GPU e_v at stop (unconverged) | ceiling e_v (oracle_fd4) | solve [s] |
|---|---|---|---|---|---|---|---|
| gauss:0.5:32 | continuation_floor | 0.421875 | 4 | 1.41e-02 | 1.3874e-02 | 2.9987e-03 | 270.3 |
| gauss_ch:0.5:32 | continuation_floor | 0.4375 | 4 | 9.66e-03 | 1.0741e-02 | 2.9117e-03 | 329.8 |
| generic3d:1:16 | continuation_floor | 0.875 | 4 | 4.39e-02 | 9.2114e-02 | 1.6341e-01 | 46.0 |
| generic3d:1:20 | continuation_floor | 0.75 | 4 | 4.60e-02 | 8.0562e-02 | 1.3497e-01 | 52.6 |
| generic3d:1:24 | continuation_floor | 0.65625 | 4 | 8.07e-02 | 1.0338e-01 | 1.1355e-01 | 87.5 |

Oracle-ceiling sanity at 32^3 eps 0.5: the GPU metrics of the exported oracle labels agree with the exporter's
`ref_metrics.json` to 4.7e-15 (gauss) / 2.8e-15 (gauss_ch) over every ceiling key; the ceiling e_v at 32^3
(3.00e-3 / 2.91e-3) lies below the 24^3 ceiling (7.28e-3 / 6.77e-3), as expected under refinement. The exports
themselves are therefore sane; the failure is the solver's.

## Reading

- The 14 step-9a cases of N6 (`gauss`, `gauss_ch`, `control2d` x eps 0.25, 0.5 x N 16, 24; N 32 at eps 0.25) all
  end `STATUS converged` with r_F <= 9.9e-14 and **no bisection at all** (GPU PATHs `0.25` / `0.25->0.5`; the
  prototype PATHs contain stage failures the GPU does not have — informational). Full-precision agreement
  (16^3, 24^3): maximum relative difference over all gated keys 1.75e-12 (gauss:0.25:16; N6 had ~1e-14 there with
  fixed forcing: the inexact-Newton iterate differs at the ~1e-12 level, FIELDDIFF 7.05e-13); FIELDDIFF joint
  <= 8.6e-13 for every case with a solution. 32^3 eps 0.25 and `control2d:0.25:24`: `PASS(4dig)`. The N6 FAILs
  `gauss:0.5:24` and `gauss_ch:0.5:24` now PASS (e_v rel diff 9.9e-14 / 7.2e-14).
- The two new 32^3 eps 0.5 points (`gauss:0.5:32`, `gauss_ch:0.5:32`, no prototype reference) **FAIL**:
  `continuation_floor`; every failed stage is `linear_failure` (GMRES restart stagnation at true relative residual
  0.24-0.81 against eta <= 0.1, and one inner cap at 1.6e-4 against eta 6.5e-5); see `preconditioner_gate_b.md`.
- `generic3d` eps 0.25 / 0.5 at 16^3 converge and PASS at full precision; `generic3d` eps 1 at 16/20/24 still end
  `continuation_floor` (informative, outside 9a; the N7c limitation, unchanged in kind).
