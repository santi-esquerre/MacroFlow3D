# SF-33 N6 — prototype reproduction (step 9a): GPU `inlet_slab --proto` vs the SF-29 CPU prototype

Campaign A, V100 host, mirror `~/MacroFlow3D-SF-33` synced from commit `681b039` (index: `README.md`).
Facts first; the reading is in the last section.

## Sources

- GPU: `logs/proto/<case>.log` + `raw/proto/<case>.json` (job `sf33-proto`, `run_proto.sh`, driver defaults
  `--lin-tol 1e-12 --restart 100 --max-inner 6000 --newton-tol 1e-13 --max-newton 40 --bisect 4 --prec pa`);
  `logs/proto_24ref/` + `raw/proto_24ref/` (job `sf33-proto-24ref`: the five 24^3 cases re-run with the same
  defaults plus `--solution` against the full-precision 24^3 prototype references); `logs/proto_r200/` +
  `raw/proto_r200/` (job `sf33-proto-r200`: the whole matrix with `--restart 200 --max-inner 12000`, additional
  run for the preconditioner gate, see `preconditioner_gate.md`).
- Prototype, full precision: the saved 16^3 solutions `raw/sweep2/solutions/*_16_i1o4.npz` of the SF-29 artifact
  (converted by `export_proto.py --solutions`) and five 24^3 references produced in this node with the prototype
  itself (job `sf33-proto-ref24`: `candidate_i.py <case> --direct-max 32 --reuse-lu 0 --save ~/sf33_ref24`,
  sparse direct Newton solves; the five cases ran as five concurrent processes with 3 BLAS threads each, the
  sweep2 setting, instead of one sequential 16-thread process; logs `logs/jobs/candidate_i_24_*.log`, concatenated
  in `logs/jobs/candidate_i_24.log`): `gauss:0.25:24`, `gauss_ch:0.25:24`, `control2d:0.5:24`, `gauss:0.5:24`,
  `gauss_ch:0.5:24`. Their status / its / r_F / PATH equal the sweep2 record: converged; PATHs
  `0.25(fail)->0.125->0.25` (gauss, gauss_ch 0.25), `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5` (gauss,
  gauss_ch 0.5), `0.25->0.5` (control2d 0.5); r_F 6.363e-15, 8.914e-15, 6.582e-15, 1.224e-14, 1.613e-14 (sweep2
  values). No prototype-side discrepancy.
- Prototype, 4 significant digits: the sweep2 cell logs `i1o4-<field>-<eps>-N<N>.txt` (32^3, `control2d:0.25:24`,
  `generic3d:1:20/24`).
- Comparison: `scripts/compare_proto.py`, run on the host (job `sf33-compare`; full per-key output with 17-digit
  values in `logs/jobs/sf33-compare.log`; tables `raw/proto/compare_proto_fullref.md`,
  `raw/proto_24ref/compare_proto.md`, `raw/proto_r200/compare_proto.md`). `raw/proto/compare_proto.md` is the table
  `run_proto.sh` wrote before the 24^3 references existed (24^3 read at 4 digits there); it is superseded by
  `compare_proto_fullref.md`.

## Reading rule

- `PASS`: GPU vs prototype at full precision, `|gpu - proto| / |proto| <= 1e-6` (all 16^3 cases and the five 24^3
  cases of `sf33-proto-ref24`).
- `PASS(4dig)`: prototype value printed with 4 significant digits (`%.3e`); the band is the rounding bound of the
  printed value; a 1e-6 agreement cannot be resolved and is not claimed (32^3, `control2d:0.25:24`).
- `roundoff`: both sides below 1e-10 (control2d `e_psi2`, oracle `e_psi*`): not gated.
- `overall` also gates every other CASE key (`e_i1`, `e_i2`, `e_div`, `min_c`, `p0.1`, `p1`, `p5`, `p50`, both for
  `cand=i1o4` and the ceiling `cand=oracle_fd4`), `r_F <= 1e-10`, PATH equality and `FIELDDIFF joint <= 1e-8`
  (`max |u_gpu - u_proto| / max |u_proto|` over planes 0..N, both fields).

## Per-case table, driver defaults (step-9a matrix + generic3d), both sides

Rows from `logs/proto/`, except the five 24^3 rows with a full-precision reference, which come from
`logs/proto_24ref/` (same solver and settings plus `--solution`; their metrics equal those of `logs/proto/` to all
17 digits, and they add FIELDDIFF). Values `%.10e`; cell = `gpu / prototype (rel diff, verdict)`.

| case | GPU status | r_F gpu / proto | its gpu / proto | PATH | prototype source | e_v gpu / proto (rel diff, verdict) | e_psi gpu / proto | e_psi1 gpu / proto | e_psi2 gpu / proto | FIELDDIFF joint | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control2d:0.25:16 | converged | 1.217e-15 / 1.532e-15 | 2 / 2 | EQUAL | solution.json (full precision) | 3.0673264314e-03 / 3.0673264314e-03 (1.84e-15, PASS) | 3.2204212468e-02 / 3.2204212468e-02 (2.82e-14, PASS) | 3.2204212468e-02 / 3.2204212468e-02 (2.82e-14, PASS) | 5.1749754823e-12 / 5.1749940265e-12 (nan, roundoff) | 6.312e-15 | PASS |
| control2d:0.25:24 | converged | 2.790e-15 / 3.393e-15 | 2 / 2 | EQUAL | i1o4-control2d-0.25-N24.txt (4 significant digits) | 5.2467742300e-04 / 5.2470000000e-04 (4.30e-05, PASS(4dig)) | 2.6668969883e-03 / 2.6670000000e-03 (3.86e-05, PASS(4dig)) | 2.6668969883e-03 / 2.6670000000e-03 (3.86e-05, PASS(4dig)) | 5.2614230691e-12 / 5.2610000000e-12 (nan, roundoff) | - | PASS |
| control2d:0.5:16 | converged | 2.445e-15 / 2.980e-15 | 2 / 2 | EQUAL | solution.json (full precision) | 7.6867914868e-03 / 7.6867914868e-03 (2.82e-15, PASS) | 3.1833069452e-02 / 3.1833069452e-02 (2.46e-14, PASS) | 3.1833069452e-02 / 3.1833069452e-02 (2.46e-14, PASS) | 2.3339324825e-13 / 2.3342565259e-13 (nan, roundoff) | 7.443e-15 | PASS |
| gauss:0.25:16 | converged | 8.628e-14 / 8.631e-14 | 10 / 10 | EQUAL | solution.json (full precision) | 5.5059288178e-03 / 5.5059288178e-03 (4.41e-15, PASS) | 7.7266164256e-02 / 7.7266164256e-02 (5.75e-15, PASS) | 7.7266164256e-02 / 7.7266164256e-02 (5.75e-15, PASS) | 5.8594649382e-02 / 5.8594649382e-02 (6.87e-15, PASS) | 2.625e-15 | PASS |
| gauss:0.25:32 | converged | 8.961e-15 / 1.108e-14 | 8 / 8 | EQUAL | i1o4-gauss-0.25-N32.txt (4 significant digits) | 8.0142813632e-04 / 8.0140000000e-04 (3.51e-05, PASS(4dig)) | 1.3027149708e-02 / 1.3030000000e-02 (2.19e-04, PASS(4dig)) | 1.3027149708e-02 / 1.3030000000e-02 (2.19e-04, PASS(4dig)) | 9.9516667508e-03 / 9.9520000000e-03 (3.35e-05, PASS(4dig)) | - | PASS |
| gauss:0.5:16 | converged | 4.576e-15 / 5.431e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 1.9083846132e-02 / 1.9083846132e-02 (3.82e-15, PASS) | 1.4025886950e-01 / 1.4025886950e-01 (4.35e-15, PASS) | 1.4025886950e-01 / 1.4025886950e-01 (4.35e-15, PASS) | 9.9536390148e-02 / 9.9536390148e-02 (2.79e-15, PASS) | 5.236e-15 | PASS |
| gauss_ch:0.25:16 | converged | 2.781e-15 / 3.407e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 5.2873838165e-03 / 5.2873838165e-03 (8.69e-15, PASS) | 6.7803545078e-02 / 6.7803545078e-02 (1.47e-14, PASS) | 6.7803545078e-02 / 6.7803545078e-02 (1.47e-14, PASS) | 4.1388179361e-02 / 4.1388179361e-02 (1.89e-14, PASS) | 5.411e-15 | PASS |
| gauss_ch:0.25:32 | converged | 1.104e-14 / 1.396e-14 | 8 / 8 | EQUAL | i1o4-gauss_ch-0.25-N32.txt (4 significant digits) | 7.6571896425e-04 / 7.6570000000e-04 (2.48e-05, PASS(4dig)) | 1.0860563406e-02 / 1.0860000000e-02 (5.19e-05, PASS(4dig)) | 1.0860563406e-02 / 1.0860000000e-02 (5.19e-05, PASS(4dig)) | 6.7772368188e-03 / 6.7770000000e-03 (3.49e-05, PASS(4dig)) | - | PASS |
| gauss_ch:0.5:16 | converged | 5.777e-15 / 7.014e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 1.7845965058e-02 / 1.7845965058e-02 (1.17e-15, PASS) | 1.2316189898e-01 / 1.2316189898e-01 (1.01e-15, PASS) | 1.2316189898e-01 / 1.2316189898e-01 (1.01e-15, PASS) | 6.0805706746e-02 / 6.0805706746e-02 (1.84e-14, PASS) | 4.677e-15 | PASS |
| control2d:0.5:24 | converged | 5.306e-15 / 6.582e-15 | 2 / 2 | EQUAL | solution.json (full precision) | 1.6180422779e-03 / 1.6180422779e-03 (1.37e-14, PASS) | 1.0650007214e-02 / 1.0650007214e-02 (3.14e-14, PASS) | 1.0650007214e-02 / 1.0650007214e-02 (3.14e-14, PASS) | 2.3469802529e-13 / 2.3467844175e-13 (nan, roundoff) | 9.670e-15 | PASS |
| gauss:0.25:24 | converged | 5.147e-15 / 6.363e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 1.8080867184e-03 / 1.8080867184e-03 (1.09e-14, PASS) | 2.8860579641e-02 / 2.8860579641e-02 (1.25e-14, PASS) | 2.8860579641e-02 / 2.8860579641e-02 (1.25e-14, PASS) | 2.1059960187e-02 / 2.1059960187e-02 (1.45e-14, PASS) | 5.215e-15 | PASS |
| gauss:0.5:24 | continuation_floor | 5.182e-02 / 1.224e-14 | 2 / 8 | DIFFERENT | solution.json (full precision) | 1.7755636228e-02 / 8.4739608321e-03 (1.10e+00, FAIL) | 1.5035349746e-01 / 6.8314673933e-02 (1.20e+00, FAIL) | 1.5035349746e-01 / 6.8314673933e-02 (1.20e+00, FAIL) | 1.1314020746e-01 / 4.4045449389e-02 (1.57e+00, FAIL) | 1.369e-01 | FAIL |
| gauss_ch:0.25:24 | converged | 7.617e-15 / 8.914e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 1.7830408700e-03 / 1.7830408700e-03 (1.65e-14, PASS) | 2.4826874602e-02 / 2.4826874602e-02 (2.61e-14, PASS) | 2.4826874602e-02 / 2.4826874602e-02 (2.61e-14, PASS) | 1.5026994982e-02 / 1.5026994982e-02 (8.20e-15, PASS) | 6.282e-15 | PASS |
| gauss_ch:0.5:24 | continuation_floor | 1.008e-01 / 1.613e-14 | 2 / 8 | DIFFERENT | solution.json (full precision) | 2.4825801532e-02 / 8.4192608082e-03 (1.95e+00, FAIL) | 1.7624233259e-01 / 6.2214029352e-02 (1.83e+00, FAIL) | 1.7624233259e-01 / 6.2214029352e-02 (1.83e+00, FAIL) | 1.1599196460e-01 / 3.2592440913e-02 (2.56e+00, FAIL) | 1.614e-01 | FAIL |
| generic3d:0.25:16 | converged | 2.142e-15 / 2.688e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 3.4232120896e-03 / 3.4232120896e-03 (5.32e-15, PASS) | 4.6339148648e-02 / 4.6339148648e-02 (4.49e-15, PASS) | 2.7939156021e-02 / 2.7939156021e-02 (1.14e-14, PASS) | 4.6339148648e-02 / 4.6339148648e-02 (4.49e-15, PASS) | 4.529e-15 | PASS |
| generic3d:0.5:16 | converged | 4.532e-15 / 5.521e-15 | 7 / 7 | EQUAL | solution.json (full precision) | 1.4598831034e-02 / 1.4598831034e-02 (7.49e-15, PASS) | 1.0171694625e-01 / 1.0171694625e-01 (4.91e-15, PASS) | 7.7984919249e-02 / 7.7984919249e-02 (2.21e-14, PASS) | 1.0171694625e-01 / 1.0171694625e-01 (4.91e-15, PASS) | 6.805e-15 | PASS |
| generic3d:1:16 | continuation_floor | 4.728e-01 / 1.580e-14 | 2 / 9 | DIFFERENT | solution.json (full precision) | 1.6718875240e-01 / 1.0338238572e-01 (6.17e-01, FAIL) | 4.8850538836e-01 / 3.4681690946e-01 (4.09e-01, FAIL) | 4.3764737523e-01 / 3.4681690946e-01 (2.62e-01, FAIL) | 4.8850538836e-01 / 2.9004096633e-01 (6.84e-01, FAIL) | 6.010e-01 | FAIL |
| generic3d:1:20 | continuation_floor | 6.604e-01 / 1.969e-14 | 2 / 14 | DIFFERENT | i1o4-generic3d-1-N20.txt (4 significant digits) | 2.1653989182e-01 / 5.8850000000e-02 (2.68e+00, FAIL(4dig)) | 5.6524676405e-01 / 2.3420000000e-01 (1.41e+00, FAIL(4dig)) | 5.5034739055e-01 / 2.3420000000e-01 (1.35e+00, FAIL(4dig)) | 5.6524676405e-01 / 1.6850000000e-01 (2.35e+00, FAIL(4dig)) | - | FAIL |
| generic3d:1:24 | continuation_floor | 8.159e-01 / 2.835e-14 | 2 / 9 | DIFFERENT | i1o4-generic3d-1-N24.txt (4 significant digits) | 2.6111496129e-01 / 4.4000000000e-02 (4.93e+00, FAIL(4dig)) | 6.6169970043e-01 / 1.7820000000e-01 (2.71e+00, FAIL(4dig)) | 6.6169970043e-01 / 1.7820000000e-01 (2.71e+00, FAIL(4dig)) | 6.2841132320e-01 / 1.2340000000e-01 (4.09e+00, FAIL(4dig)) | - | FAIL |

PATHs of the five failing rows (GPU defaults vs prototype):

| case | GPU PATH | prototype PATH |
|---|---|---|
| gauss:0.5:24 | `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5(fail)->0.4375(fail)->0.40625->0.4375(fail)->0.5(final)` | `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5` |
| gauss_ch:0.5:24 | `0.25(fail)->0.125->0.25->0.5(fail)->0.375(fail)->0.3125->0.375->0.5(fail)->0.4375(fail)->0.5(final)` | `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5` |
| generic3d:1:16 | `0.25->0.5->1(fail)->0.75(fail)->0.625(fail)->0.5625->0.625(fail)->0.59375->0.625(fail)->1(final)` | `0.25->0.5->1(fail)->0.75->1(fail)->0.875->1(fail)->0.9375->1` |
| generic3d:1:20 | `0.25->0.5->1(fail)->0.75(fail)->0.625(fail)->0.5625(fail)->0.53125->0.5625(fail)->1(final)` | `0.25->0.5->1(fail)->0.75->1(fail)->0.875(fail)->0.8125->0.875->1` |
| generic3d:1:24 | `0.25->0.5(fail)->0.375->0.5(fail)->0.4375->0.5(fail)->0.46875->0.5(fail)->0.484375->0.5(fail)->1(final)` | `0.25->0.5->1(fail)->0.75(fail)->0.625->0.75->1(fail)->0.875->1(fail)->0.9375->1` |

FIELDDIFF (joint) of every case with a saved prototype solution: 16^3 — control2d 0.25 / 0.5: 6.31e-15 / 7.44e-15;
gauss 0.25 / 0.5: 2.62e-15 / 5.24e-15; gauss_ch 0.25 / 0.5: 5.41e-15 / 4.68e-15; generic3d 0.25 / 0.5 / 1:
4.53e-15 / 6.80e-15 / 6.01e-01. 24^3 — gauss 0.25: 5.21e-15; gauss_ch 0.25: 6.28e-15; control2d 0.5: 9.67e-15;
gauss 0.5: 1.37e-01; gauss_ch 0.5: 1.61e-01. Every converged case: <= 9.7e-15 (expected <= 1e-8).

r_F history agreement (minimum correct digits of the GPU `HISTORY_FULL` vs the prototype `hist`, entries with
r_F > 1e-10, target stage): control2d 15.7-17.0; converged gauss / gauss_ch / generic3d 5.6-7.1. The minimum sits
at the last pre-converged entry (e.g. gauss:0.25:16: `6.466291e-08` vs `6.466305e-08`): the GMRES steps solved to
`lin_tol = 1e-12` and the prototype's direct steps (step residual ~1e-14-1e-13) differ at ~1e-12 relative, and
Newton's quadratic phase amplifies that difference in r_F; the earlier entries agree to all 7 printed digits and the
converged states agree to ~1e-14 (FIELDDIFF).

## Additional run `--restart 200 --max-inner 12000` (`raw/proto_r200/compare_proto.md`)

The same 14 cases PASS with the same agreement (FIELDDIFF <= 9.2e-15, PATH equal); the same five cases fail
(`continuation_floor`, r_F 8.8e-3, 3.2e-2, 1.75, 2.96, 0.76), PATHs:

- gauss:0.5:24: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5(fail)->0.4375->0.5(fail)->0.46875->0.5(fail)`
- gauss_ch:0.5:24: `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5(fail)->0.4375->0.5(fail)->0.46875(fail)->0.5(final)`
- generic3d:1:16: `0.25->0.5->1(fail)->0.75(fail)->0.625->0.75(fail)->0.6875->0.75(fail)->0.71875->0.75(fail)->1(final)`
- generic3d:1:20: `0.25->0.5->1(fail)->0.75(fail)->0.625->0.75(fail)->0.6875(fail)->0.65625(fail)->1(final)`
- generic3d:1:24: `0.25->0.5->1(fail)->0.75(fail)->0.625(fail)->0.5625(fail)->0.53125->0.5625(fail)->1(final)`

## Verdict (acceptance (a); N6 criteria 1 and 3)

- Step-9a matrix (14 cases): **12 PASS, 2 FAIL**. PASS: the eight 16^3 cases (full precision; metrics rel diff
  <= 3e-14, FIELDDIFF <= 7.4e-15), `gauss:0.25:24`, `gauss_ch:0.25:24`, `control2d:0.5:24` (full precision, rel
  diff <= 3.2e-14, FIELDDIFF <= 9.7e-15), `control2d:0.25:24` and the two 32^3 cases (`PASS(4dig)`, PATH equal,
  its equal, r_F <= 1.1e-14). FAIL: `gauss:0.5:24` and `gauss_ch:0.5:24` end `continuation_floor` (r_F 5.2e-2 /
  1.0e-1) with the default settings and with `--restart 200 --max-inner 12000`.
- `generic3d` eps 1 at 16 / 20 / 24 (criterion 3): **FAIL** on all three (`continuation_floor`; the prototype
  converged). `generic3d` eps 0.25 / 0.5 at 16: PASS.
- Every failure is a linear-solver failure of P-A-preconditioned GMRES in a continuation stage that the prototype
  (direct solves) accepted; `preconditioner_gate.md` lists each. Where both converge, the GPU and the prototype
  agree to roundoff (metrics ~1e-14 relative, fields ~1e-14): the GPU solves the same discrete problem. What it does
  not reproduce is the prototype's Newton path at `eps = 0.5, N = 24` and at `eps = 1`, because the linear solves
  do not reach `lin_tol` there. The defect is on the linear solver / preconditioner side (`src/`), not in the
  prototype, the exports or the campaign scripts; it is a finding for a corrective node.
