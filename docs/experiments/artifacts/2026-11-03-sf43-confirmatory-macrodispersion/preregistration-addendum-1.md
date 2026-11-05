# SF-43 addendum 1 — pre-registration of the exploratory arms J1, J3, J9 (S0, J8)

Decision record: `docs/decisions/2026-11-04-sf43-addendum-1.md` (owner decisions of 2026-11-04, verbatim there). Machine-readable
companion: `prereg_addendum.json` (same directory; the analysis reads it; where the two differ this text governs and the difference is a
defect). Committed and pushed to PR #87 before any script, check or run of the addendum (workflow 8A; no force-push).

Everything here is EXPLORATORY and DESCRIPTIVE, outside the confirmatory Holm family (`m = 8` unchanged). No frozen file changes
(`preregistration.md`, `prereg.json`, `plan.json`, `scripts/sf43ana`, `scripts/sf43camp`, `results/` ...); every new file is new. The
confirmatory verdict `gate5:4` (`disagree`, equivalence not established) is not recomputed. The owner's text of the rules is reproduced
verbatim in Appendix A; this operational version may only TIGHTEN it, and every tightening is marked **[T]**.

Abbreviations: `A` = this artifact directory; `PRE` = `preregistration.md`; frozen arms are those of `plan.json` / `prereg.json` of
pre-registration `cbbd69da`; "IC90" = two-sided 90 % interval; "within [a, b]" means `lo >= a` and `hi <= b`.

## 1. Arms, seeds, settings

Every arm is an exact copy of its origin arm's runner dict (`par2` or `consumer`) and `reduce` block in `plan.json` (including
`snapshots_every`, `cg_rtol`, `cg_max_iter`, window, `np_side`, `transport_seed`, `hmax_div`, `tol`, planes, `x0_ell`, `twin`,
`pollock_bridge`, solver) except the factor named below. The plan generator derives each arm from the frozen `plan.json` and asserts the
difference is exactly the named factor **[T]**.

| arm | origin | factor (only change) | seeds | group, reserves | priority |
|---|---|---|---|---|---|
| `j1_base10` | `s4_g5p` | none (reproduction check) | 434001-434010 | `a1_j1_base`, 438201-438203 | 0 |
| `j1_H3` | `s4_g5p` | `hz` 3.6 -> 10.8 | 434001-434060 | `a1_j1_H3`, 438211-438215 | 1 |
| `j1_D` | `s4_g5p` | `dt_half` false -> true (`dt/2`, exactly `2 x n_steps`) | 434001-434060 | `a1_j1_D`, 438221-438225 | 2 |
| `j1_DH3` | `s4_g5p` | `hz` 10.8 and `dt_half` true | 434001-434060 | `a1_j1_DH3`, 438231-438235 | 3 |
| `j1_H2` | `s4_g5p` | `hz` 3.6 -> 7.2 (dose-response) | 434001-434060 | `a1_j1_H2`, 438241-438245 | 4 |
| `j3_l5` | `s1_dist` | grid 2080x240x240 -> 1040x120x120 (`lambda/h` 10 -> 5; `ell` 1/24, L = 208, W = 24 `lambda` unchanged) | 436001-436100 | `a1_j3`, 438301-438305 | 5 |
| `j9_w48` | `j3_l5` | grid y, z 120 -> 240, `ell` 1/24 -> 1/48, window `0.1,0.9,0.1,0.9` -> `0.3,0.7,0.3,0.7` (W = 48 `lambda`; source 0.4 x 48 = 19.2 `lambda`, as `s1_w48`; `lambda/h` 5, L = 208) | 438001-438150 | `a1_j9`, 438151-438165 | 6 |
| `j8_H3`, `j8_D`, `j8_DH3` | `j1_H3`, `j1_D`, `j1_DH3` | none (more seeds; conditional, section 5) | 434061-434100 | `a1_j8_*`, 438251-438265 | second job |

- **Paired reuse, declared:** J1 (and J8) reuse field seeds of `s4_g5p` WITH its transport seeds (Par2 transport stream `420800001 +
  1000 s` for field seed `s`, independent of segmentation): `j1_H3` repeats the first `n_steps` of `s4_g5p` exactly and continues them;
  `j1_D` uses the same stream with half steps. J3 reuses the field seeds of `s1_dist`. These reuses are the design (common random
  numbers), not collisions. J9's block 438001-438150 and every reserve block are new and disjoint: `sf43plan.build.check_seeds` on the
  frozen `seeds_registry.json` plus the new blocks must pass (`seeds_registry_addendum.json`).
- One arm per CRN group with distinct priorities, so that the engine runs the arms in the owner's order **[T: the order is strict]**:
  `j1_base10 -> j1_H3 -> j1_D -> j1_DH3 -> j1_H2 -> j3_l5 -> j9_w48`. A replaced seed (an unusable solve) has no partner and drops out of
  every paired contrast; the analysis uses common seeds only and reports `n`.
- `unit_size` 10 (as `s4_g5p` and `s1_dist`); `primary: false`; `no_reestimation: true` (no blinded re-estimation: `N` fixed);
  `retained`: the first two seeds of each arm (`min_retained` 2); failure policy as `plan.json` (`halt_rate` 0.25, `min_for_rate` 20).
- `est_seconds`, `raw_gb`, `min_free_mib` provisional until S0 (section 4), then set from its measurements.
- Not run (owner): the 832 `lambda` box, `sigma^2 = 0.0625`, a generator contrast with Par2, any tracker run on the random-spectral
  generator.

## 2. Responses and inference (all arms)

- **Responses.** J1 (and J8): the Gate 5 estimator of `prereg.json` (`key = alpha_LP`), Par2 rows `weights = unif`, `set = perplane`,
  family `plane`, window `main` [25, 94], per-realization values as `sf43ana.loader` reads them; the Pollock reference is the frozen arm
  `s4_g5pol` (Pollock rows of `s4_l16`, uniform weights, as in `prereg.json`). Par2 crossing completeness per realization =
  `n_last / n_traced` of the main-window row (H12 definition). J3, J9: `alpha_TP:disp` (displacement basis, flux weights, `perplane`,
  family `plane`, consumer code), windows `main` [25, 94] and `far` [94, 194], as for `s1_dist` (reduce block unchanged).
- **Frozen inputs.** The frozen reduced tables (`s4_g5p`, `s4_l16` Pollock rows, `s1_dist`) are read read-only from
  `~/sf43_campaign/reduced` on the host, or from the committed per-arm archives `A/results/reduced/`; every file read is verified
  against the freeze manifest (sha256 `250395ed5319052d...` of `~/sf43_campaign.freeze.reduced.sha256`, bitácora row 2026-11-04T13:20Z) **[T]**.
- **Paired ratio** of arms `X`, `Y` (owner: ratio of means per seed, on the common seeds): `theta = log(mean X / mean Y)` over the seeds
  present and finite in both; IC90 = `mcstat.gate5.gate5_paired(X, Y, level=0.90, method="bootstrap-t", B=2000, seed=20261104)`
  (studentized seed-pair bootstrap of the linearized log ratio; the method of `gate5_paired`). Its Student interval (`method="student"`,
  as the frozen decomposition) is reported beside it, never used by a rule.
- **Paired ratio of ratios** (`m_far / m_main`, J3): `theta = log(mean J3_far / mean D_far) - log(mean J3_main / mean D_main)` on the
  seeds common to the four series; per-seed linearized contrast `e = J3_far/mean - D_far/mean - J3_main/mean + D_main/mean`, `se =
  sd(e, ddof 1)/sqrt(n)`, studentized seed bootstrap with `B = 2000`, seed 20261104 and the rank rule of `gate5_paired`. With two series
  this procedure must reproduce `gate5_paired` exactly (unit test) **[T]**.
- **Unpaired ratio** (`F48/F5`; the `W48/W24` levels): `theta = theta_1 - theta_2`, `se = sqrt(se_1^2 + se_2^2)` from the two
  linearizations, Welch-Satterthwaite `nu`, IC90 = `theta +- t_{0.95, nu} se` ("Welch sobre el log").
- Point estimates in a rule are `exp(theta)`. Everything is descriptive and outside Holm.

## 3. Rules

### 3.1 J1 — Par2 at `sigma^2 = 4`, horizon against step

Quantities (paired, common seeds among 434001-434060):
`r_H = Par2(j1_H3)/Par2(s4_g5p)`, `r_D = Par2(j1_D)/Par2(s4_g5p)`, and descriptively `r_H2`, `r_DH3` (same form);
`q_X = Pollock(s4_g5pol)/Par2(X)` for `X` in {`base` = `s4_g5p` on the same seeds, `j1_H2`, `j1_H3`, `j1_D`, `j1_DH3`};
completeness per arm (mean, fraction of realizations with `n_last = n_traced`, quantiles);
representativity: `q_base` on these seeds against the frozen 1.164 [1.131, 1.198] (all 250 seeds, Student).

1. **Reproduction (gate).** `j1_base10` against `s4_g5p` on seeds 434001-434010: every numeric column of every reduced row present in
   both (all families, windows, weights; except `source`, `r` and path or timing columns) equal within 1e-12 relative (1e-15 absolute
   where the value is 0); the Par2 crossing CSVs are also compared by sha256 where the frozen ones are on the host (bitwise identity is
   the expectation; reported). Fail: J1 is NOT classified; the cause is diagnosed; nothing is re-run with other settings without a new
   owner decision. S0 pre-check **[T]**: the smoke's `j1_base10` seed 434001 must already pass the same comparison, otherwise J1 is
   removed from the production job.
2. **Manipulation check.** On `j1_H3`: mean completeness `>= 0.999` AND fraction of complete realizations `>= 0.90`. If it fails, the
   horizon is "no saturado", nothing is attributed to the horizon, and **[T]** the J1 class is `abierto (horizonte no saturado)` (classes
   1-3 are not assigned).
3. **Classification** (first match, if 1 and 2 pass):
   1. `horizonte dominante`: `r_H >= 1.10`, `abs(r_D - 1) <= 0.03` and IC90(`q_H3`) within [1/1.10, 1.10];
   2. `paso dominante`: `r_D >= 1.05` and `r_H < 1.05`;
   3. `mixto`: `r_H >= 1.03`, `r_D >= 1.03` and IC90(`q_DH3`) within [1/1.10, 1.10];
   4. `abierto`: any other case, or lower limit of IC90(`q_DH3`) above 1.10.
4. `q_H3` and `q_DH3` are ALWAYS reported with their IC90 as the residual; the rule does not attribute it (about 5 % expected from the
   post-hoc completeness strata). Acceptance D stays the owner's decision; this class is an input.

### 3.2 J3 — `lambda/h = 5` twin of `s1_dist`

`m_main = J3/s1_dist` (main window, paired), `m_far` likewise (far window), `F5 = far/main` within `j3_l5` (paired per realization),
`m_far/m_main` (paired ratio of ratios).
- `lambda/h = 5 equivalente` iff IC90(`m_main`) within [0.97, 1.03] AND IC90(`m_far`) within [0.97, 1.03] AND IC90(`m_far/m_main`)
  within [0.98, 1.02].
- Otherwise `lambda/h = 5 no equivalente`: J9 is reported with the grid bias as a caveat and NO width conclusion (its class is
  reported with that qualification). If `j3_l5` did not run, the same applies.

### 3.3 J9 — width 48 `lambda` at 208 `lambda`

`F48 = far/main` within `j9_w48` (paired per realization), IC90 as section 2; compared with `F5` (J3; unpaired, Welch on the log) and
with `F24` = far/main of `s1_dist` (frozen secondary 1.021, percentile [0.976, 1.068]; recomputed with this section's method, reported).
Classes, first match:
0. **[T]** `inconcluso por N` if fewer than 100 `j9_w48` realizations completed (this takes precedence);
1. `meseta robusta al ancho` iff IC90(`F48`) within [0.90, 1.10] AND IC90(`F48/F5`) within [0.90, 1.10];
2. `decae con el ancho` iff the upper limit of IC90(`F48`) is below 0.95;
3. `inconcluso` otherwise.
Descriptive, no class: the level `W48/W24` at `lambda/h = 5` (`j9_w48` against `j3_l5`), in main and in far (unpaired, Welch on the
log).

## 4. Smoke S0 (structure and cost only)

Plan section `smoke` (campaign `~/sf43_addendum_smoke`), one job: `j3_l5` 1 realization (436001), `j9_w48` 1 (438001), `j1_H3` 2
(434001-434002), `j1_DH3` 2 (434001-434002), `j1_base10` 1 (434001); all smoke seeds retained (nothing pruned, so that `raw_gb` is
measured). Read: wall seconds per realization, peak device MiB (`peak_mib`), CG iterations (consumer `solver_b.json`, Par2 `iters=`), GB
per realization (`du` of the realization directories), solver verdicts, the structure check. Never an `alpha`.
- `j9_w48` is removed (and reported) if: seconds per realization > 180 (2 x the estimate of 90 s, the upper end of "40-90 s"), or peak
  > 20 GiB (20480 MiB), or CG iterations > 1500 (fixed cap `cg_max_iter_b` = 2000, `EnsembleRunner.cuh`), or the runner rejects the
  geometry.
- J1 is stopped if any J1 smoke unit fails structurally (runner error, non-finite or missing crossings, unusable solve) or the
  reproduction pre-check of 3.1.1 fails **[T]**. `j3_l5` is removed if it fails structurally **[T]**.
- Any need to change C++/CUDA or the runners removes that arm from the addendum (future increment).
- From S0 the plan is finalized before the production job, by the plan generator (no hand edit), in a bitácora row with the new
  sha256: `est_seconds` = 1.25 x measured seconds per realization, `raw_gb` = measured, `min_free_mib` = measured peak + 25 % (at least
  the frozen origin arm's), `budget_gpu_h` = 8 - (S0 GPU-h held) - 0.5 (start-up, reduction, analysis tail), `disk_min_free_gb` = host free
  GB at finalization - 60 (so the engine prunes, then stops, before the campaign holds more than 60 GB).

## 5. Production job, J8, cutoff, budget

- One detached job, one GPU: `REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-43 run sf43add-prod --
  "bash A/scripts/run_sf43.sh build/v100-release all --plan A/plan_addendum.json"` (never `--gpu`, `CUDA_VISIBLE_DEVICES`,
  `REMOTE_GPU_COUNT`). Same for S0 (`--smoke`) and J8. Binaries: the md5 of `binaries.json` (rebuild of the same sources; verified on the
  host before `blinded start`).
- Engine `cutoff_utc` 2026-11-07T02:30:00Z with `grace_minutes` 30: no unit runs after 2026-11-07T03:00Z. The engine cuts (budget,
  cutoff, disk); results never cut anything. Plan B (owner): if the shared GPU does not let J1 finish, what exists is reported and the
  owner decides D with the frozen decomposition; J3 and J9 are cut before J1 (priority order).
- Hard cap 8 GPU-h (GPUs held x wall from the grant, helper log) for S0 + production + J8.
- **J8** (owner answer of 2026-11-04): NOT in the production job. After the freeze and the J1 analysis at `n = 60`, J8 runs (a second
  short job, its own `blinded start`, freeze row and `blinded stop`) iff ALL of: (a) J1 passed 3.1.1; (b) at least one IC90 (bootstrap-t)
  of `r_H`, `r_D`, `q_H3`, `q_DH3` contains in its interior a threshold the rule compares that quantity with (`r_H`: 1.03, 1.05, 1.10;
  `r_D`: 0.97, 1.03, 1.05; `q_H3`, `q_DH3`: 1/1.10, 1.10); (c) GPU-h used by the addendum + 1.2 x the projected J8 cost (measured
  per-realization costs) <= 8; (d) the J8 plan's `cutoff_utc + grace_minutes` <= 2026-11-07T03:00Z and the projection ends before it.
  Seeds 434061-434100 in H3, D and DH3, identical settings. Reported: the class at `n = 60` (primary) and at `n = 100` (the same rule;
  secondary).

## 6. Analysis, blinded phase, freeze

- Engine hooks of the production plan: `analysis.command` = the frozen addendum analysis (`python3 -m sf43add run ...`), writing
  `{campaign_dir}/analysis/addendum_summary.json` and the reproduction report, printing ONLY structure (arms, counts, `n` common seeds,
  finite flags, errors/warnings counts); then `recompute43` and `reconcile43` on the addendum campaign (frozen tools; retained
  realizations covered at 1e-10 relative, the rest NOT_COVERED by design).
- Blinded phase (workflow 8A) from `blinded start` (after the host build and md5 check, before S0) to the freeze: only costs, solver
  verdicts, counts and structure are read. Freeze row (UTC, analysis code commit, sha256 of the addendum reduced-table manifest,
  `addendum_summary.json`, ledger, REPORT, job log, `exec-blinded.log`) committed and pushed before any addendum result is read; then
  `blinded stop`. After the freeze only this pre-registered analysis runs; anything else is post hoc and labelled.
- Report (experiment note, section "Addendum 1 (exploratory)"): a table with every rule — numbers, IC90, class — and GPU-h per job and
  in total.

## 7. Scripts frozen by sha256 (list; hashes in the bitácora row of the integration, before S0)

`A/plan_addendum.json`, `A/seeds_registry_addendum.json`, `A/prereg_addendum.json`, this file, `A/scripts/sf43add/` (plan generator,
analysis, statistics, reproduction check, structure check; with tests), and every frozen dependency they import (`sf43ana`, `mcstat`,
`sf42red`, `sf43camp`, `recompute43`, `reconcile43`: unchanged, hashed in the plan's `analysis.sha256_list`). Post hoc (not rules, but
hashed before their outputs are read): `A/scripts/posthoc/` with `A/scripts/posthoc/prereg_posthoc_g7.json`. Acceptance F material:
`A/scripts/review_f/` and its runbook.

## 8. Deviations

As `PRE` section 13: any deviation is recorded with its reason and date; no threshold, margin, level, estimator or rule is changed after
data. A rule found mis-specified is reported as a deviation and not rewritten. A problem of code, data or pairing found by the analysis
goes to a corrective node and re-audit, never to a relaxed rule.

## Appendix A — owner's text of the rules (2026-11-04), verbatim

```text
## 3. Brazos GPU del addendum: reglas a preregistrar ANTES de cualquier corrida (texto fijo; sólo podés endurecerlas)

Reglas generales:
- Cada brazo es una copia exacta de la configuración del brazo de origen salvo el factor manipulado. Eso incluye `snapshots_every`, rtol, ventana, np_side, transport_seed y solver.
- Respuesta y base: las mismas del bloque congelado correspondiente. Para J1, el estimador, la base y los pesos de `gate5`; para J3 y J9, `alpha_TP:disp` con pesos de flujo.
- Inferencia: cocientes de medias por semilla; si es pareado, sobre semillas comunes. IC90 por bootstrap del log con B=2000 y semilla fija. Mismo método que `mcstat`/`gate5_paired`. Todo descriptivo y fuera de Holm (m=8 no cambia).

**J1. Par2 a σ²=4: horizonte frente a paso.** Origen: `s4_g5p` (1664×192², sf18_medium_b, mg_cg rtol 1e-10, dt=0.0244 dx, hz 3.6). Brazos, en este orden de prioridad:

| Orden | Brazo | Cambio | Semillas | Uso |
|---|---|---|---|---|
| 1 | `j1_base10` | ninguno | 434001-434010 | chequeo de reproducción |
| 2 | `j1_H3` | `hz 10.8` | 434001-434060 | |
| 3 | `j1_D` | `dt_half` | 434001-434060 | |
| 4 | `j1_DH3` | ambos | 434001-434060 | |
| 5 | `j1_H2` | `hz 7.2` | 434001-434060 | dosis-respuesta |

Las semillas 434001-434060 se reusan pareadas con `s4_g5p` y `s4_g5pol`, con el mismo `transport_seed`.

- **Reproducción:** j1_base10 tiene que ser idéntico bit a bit (o ≤1e-12 relativo) a s4_g5p en las 10 semillas. Si no lo es, J1 se detiene y se diagnostica. No se re-corre con otra configuración sin una nueva decisión del dueño.
- **Manipulación:** en H3, completitud media ≥0.999 y ≥90 % de realizaciones completas. Si no se cumple, el horizonte queda "no saturado" y no hay atribución al horizonte.
- **Cantidades:**
  - r_H = Par2(H3)/Par2(s4_g5p)
  - r_D = Par2(D)/Par2(s4_g5p)
  - q_X = Pollock(s4_g5pol)/Par2(X) para X ∈ {base, H2, H3, D, DH3}
  - completitud por brazo
  - representatividad: q_base de las 60 semillas frente a 1.164 [1.131, 1.198] (n completo)
- **Clasificación**, la primera que se cumpla:
  1. "horizonte dominante": r_H ≥ 1.10, |r_D−1| ≤ 0.03 y IC90(q_H3) ⊂ [1/1.10, 1.10].
  2. "paso dominante": r_D ≥ 1.05 y r_H < 1.05.
  3. "mixto": r_H ≥ 1.03, r_D ≥ 1.03 e IC90(q_DH3) ⊂ [1/1.10, 1.10].
  4. "abierto": cualquier otro caso, o cota inferior del IC90 de q_DH3 > 1.10.

  **Siempre** se informan q_H3 y q_DH3 con su IC como residuo. La regla no atribuye ese residuo (≈5 % esperado, §1b).
- **Extensión J8:** sólo si sobra presupuesto y un IC de la regla cruza un umbral. Son +40 semillas (434061-434100) en H3, D y DH3, y se informan n=60 y n=100.

**J3. Gemelo λ/h=5 de s1_dist.** Grid 1040×120×120: L=208λ, W=24λ (ell 1/24), σ²=1 (eps 1). Mismos planos y ventanas que s1_dist: main [25,94] y far [94,194]. Semillas 436001-436100, pareadas con s1_dist.
- m_main = J3/s1_dist en main; m_far, lo mismo en far; F5 = far/main dentro de J3.
- "λ/h=5 equivalente" si IC90(m_main) e IC90(m_far) ⊂ [0.97, 1.03] y el IC90 pareado de m_far/m_main ⊂ [0.98, 1.02]. Si no se cumple, J9 se informa con el sesgo de malla como caveat, sin conclusión de ancho.

**J9. Ancho W=48λ a 208λ.** Grid 1040×240×240: ell 1/48, λ/h=5, σ²=1. Fuente fija de 19.2λ, igual que s1_w48 (ventana "0.3,0.7,0.3,0.7"; verificalo en plan.json). Mismos planos y ventanas que J3. N=150 semillas nuevas en un bloque disjunto, verificado con `check_seeds`.
- F48 = far/main dentro de cada realización (pareado), con IC90.
- Se compara con F5 (J3) y con F24 de s1_dist: 1.021 [0.976, 1.068].
- "meseta robusta al ancho" si IC90(F48) ⊂ [0.90, 1.10] e IC90(F48/F5), no pareado, ⊂ [0.90, 1.10].
- "decae con el ancho" si la cota superior del IC90 de F48 es < 0.95.
- En cualquier otro caso, "inconcluso". También es "inconcluso por N" si se completan menos de 100 realizaciones.
- Descriptivo: nivel W48/W24 a λ/h=5 en main y en far (no pareado, Welch sobre el log). No asigna clase.

**S0. Smoke** (en `~/sf43_addendum_smoke`, vía la sección `smoke` del plan). Corre 1 realización de J3, 1 de J9, 2 semillas de H3, 2 de DH3 y 1 de base. Mide:
- tiempo por realización;
- pico de memoria;
- iteraciones del CG (el tope fijo `cg_max_iter_b=2000` está en `EnsembleRunner.cuh:102`);
- GB por realización.

Con esos datos se fijan `est_seconds` y `raw_gb` del plan. Sólo se miran métricas estructurales (`check-finite`), nunca α.

**Corte por el smoke:**
- Se elimina J9 (y se informa) si pasa cualquiera de estas cosas:
  - costo por realización > 2× la estimación (J9 ≈ 40-90 s);
  - pico > 20 GB;
  - iteraciones > 1500;
  - el runner rechaza la geometría.
- Si J1 falla en el smoke, se detiene J1.
- Cualquier necesidad de cambiar C++/CUDA o el runner saca ese brazo del addendum y lo pasa a un incremento futuro.

Costos esperados (con el ledger): S0 ≈0.3, J1 ≈1.8 (cota 3.8), J3 ≈0.4, J9 ≈1-1.5 GPU-h. **Es un solo job de producción**, con este orden de prioridad:

j1_base10 → H3 → D → DH3 → H2 → J3 → J9 → (J8 si corresponde)

`budget_gpu_h` y `cutoff_utc` van en el plan nuevo; el motor corta, nunca los resultados.

```

## Amendment A1 (pre-run clarification, 2026-11-04T18:30Z; before any addendum script ran on data and before any addendum run)

J1 class 4 (section 3.1.3) reads "any other case, or lower limit of IC90(`q_DH3`) above 1.10". Under the first-match order the second
clause would be redundant (class 3 already requires IC90(`q_DH3`) within [1/1.10, 1.10], and "any other case" covers the rest). It is
therefore applied as an OVERRIDE **[T]**: if the lower limit of IC90(`q_DH3`) is above 1.10, the J1 class is `abierto` even when class 1
(`horizonte dominante`) or class 2 (`paso dominante`) matched. This can only make an attribution rarer (a tightening). The class that
would have matched without the override is stored and reported beside it. Machine-readable: `prereg_addendum.json`
`j1.classes[order 4].override_if` and `amendments[A1]`. Nothing else changes.
