# SF-39 pre-registration addendum: the amplitude `sigma^2 = 0.25` (`s025`)

- **What this file is.** The fixed settings, rules and readings for the amplitude `sigma^2 = 0.25`. It is committed before any
  `s025` check or run; there is no rebuild (RR1). Its sha256 is cited in the bitácora row of the commit that adds it.
- **Authority.** `docs/decisions/2026-10-29-sf39-sigma025-addendum.md`: owner decisions of 2026-10-29, after three independent
  reviews of this text.
- **Relation to `preregistration.md`.** That file is not changed (sha256
  `88fe1fa67eab28a209778cf5a240e270ab4ed1e5f44777c3d0f7cecc89ad7cb7`). Each of its sections applies to `s025` with `c = 0.25`,
  except where this addendum says otherwise.
- **What it does not change.** No setting, threshold, margin, CI level or target, branch, geometry, estimator or result of `s1`,
  `s225` or `s4`.

**Disclosure.**

- `s025` was decided after the results at 1, 2.25 and 4 were known.
- Every item below is fixed before the first `s025` run. Apart from `--eps` and the Par2 variance, each is identical to the
  corresponding `s1` item.
- **Common random numbers.** `s025` reuses the `s1` seeds (5001+, check seed 4901, Par2 9001/9021 with the same transport seeds)
  on the `s1` geometry. Each `s025` log-field of the Lagrangian runner is therefore exactly 0.5 times the `s1` field of the same
  seed (`ln K = eps Y`). The Par2 fields are expected to scale the same way; this is not verified.
  - `s025` and `s1` estimates are positively correlated.
  - No test treats them as independent. Any cross-amplitude statement (ratios, exponents) is descriptive and paired by seed, with
    no new fit.
  - No seed is chosen, dropped or reordered.
  - The preregistration's statement that seeds 5001+ are unused by earlier runs does not hold for `s025`.

## A1. Arms (as section 1; runner flags as section 1, `--sigma2 1 --eps 0.5` = variance 0.25)

| arm | role | `--grid` | `lambda/h` | `--ell` | `--eps` | `--seed0` | `--nr` | `--hmax-div` | planes |
|---|---|---|---|---|---|---|---|---|---|
| `s025_A` | production, directory A | `2048,256,256` | 10 | `0.0390625` | `0.5` | `5001` | pilot 20, then `N_A` | `8,16,32` | P1 |
| `s025_B` | production, directory B | `2048,256,256` | 10 | `0.0390625` | `0.5` | `5001+N_A` | `N_run - N_A` | `8` | P1 |
| `s025_n05` / `s025_n10` / `s025_n20` | narrow rungs 1/2, 1, 2 | `1024,64,64` / `2048,128,128` / `4096,256,256` | 5 / 10 / 20 | `0.078125` | `0.5` | `5001` | `N_narrow` | `8` | P4 |
| `s025_tw` | coarse twin | `1024,128,128` | 5 | `0.0390625` | `0.5` | `5001` | `N_twin` | `8` | P1 |
| `chk_s025_n05`, `chk_s025_n10`, `chk_s025_n20`, `chk_s025_tw` | solve checks only, never data | as the arm | as the arm | as the arm | `0.5` | `4901` | 1 | `8` | as the arm |

- Production counts, directories A/B, `N_run`, `N_A` and pairing by seed follow section 1 as written. The `s4` floor of 30 does
  not apply: it serves F10, which is not run at 0.25.
- **Checks (RR3).** The `s025` checks run:
  - RR3 (i), the unit tests of the integrated scripts tree;
  - RR3 (ii), the four `chk_s025_*` solve checks;
  - RR3 (iv), the reducer smoke test.

  They do not re-run:
  - RR2 (regression references): the RR1 base build is reused, md5-gated, with no rebuild;
  - RR3 (iii) (byte check): it is amplitude-independent and passed at the `s4_A` geometry.

  An unusable `chk_s025_*` narrow solve stops that `s025` ladder until the owner decides. This is procedural: the runner halts only
  the check arm.

## A2. Planes (section 2)

The plane list depends only on `(nx, ny, lambda/h)`, so `s025` uses the existing strings:

- P1 for `s025_A`, `s025_B`, `s025_tw`: 197 planes, sha256 `0bf748196fff9c61acca5f9db5973eaf350f4364a8f3d313e66cf50b563340fa`.
- P4 for `s025_n05`, `s025_n10`, `s025_n20`: 197 planes, sha256 `b03effae1b1c6172d6a13d12efca504a93a2ec2939c1a9dde7690615d953059f`.

No new list is added.

## A3. Par2 (Gate 5 baseline)

- **Pilot config** `configs/par2_s025.yaml`: a copy of `apps/config_pipeline_par2_sf38_s1.yaml`, the same physical domain
  2048x256x256 at `lambda/h` 10. It changes only:
  - `stochastic.sigma2: 1.0 -> 0.25`;
  - `stochastic.seed: 9001`;
  - analysis `NR: 20`;
  - `output.output_dir: raw/par2_s025`;
  - a header naming the `s1` source and the variance change.

  Everything else is unchanged. `n_steps`, `dt`, `output_every`, `inject_x` and the snapshot cadence depend only on `J`, `Lx`,
  `lambda` and `dt`. No `apps/` file is added.
- **Extension** `configs/par2_s025_ext.yaml`: by the rule of section 1 / R15 (`stochastic.seed: 9021`, `transport.seed: 123476789`,
  `NR: N_R - 20`, `output_dir: raw/par2_s025_ext`), with `python3 -m sf39.par2 mkext`. It is created only if `N_R > 20`.
- **Field statistics.** `par2_field_stats` runs on each config used.

## A4. Estimators, intervals, verdicts (sections 3-5)

**As written** for `c = 0.25`:

- A, with margins `+-max(0.05 |alpha|, 1.25e-3)` for `alpha_TAM` and `alpha_TP`, and `+-0.025 alpha_LAM`;
- B, C;
- D: rule D per ladder, branches as written, and the amplitude label;
- F4, F6, F7, F8 and F11.

The nominal variance for the Gate 5 ANCOVA sensitivity is 0.25. F10 is not run.

**Pre-stated reading at 0.25.** These are interpretation notes; no threshold changes.

1. **What `converged` means.** With `alpha_T` of order 1e-3 `lambda`, `converged` means a grid effect below 1.25e-3 `lambda`
   (narrow) and 3.54e-3 `lambda` (twin, `2^1.5 m`), not agreement within 5 %.
   - `numerical` (rule D branch 2) needs a downward grid effect above the floor. It is practically unreachable, so a `physical`
     label does not exclude a mostly numerical `alpha_T`.
   - The note reports these beside each `s025` label, as fractions of the production mean, with mean and 90 % CI:
     `m / |alpha|`, `D_f` and `D_tw / 2^1.5`.
   - If a refinement CI excludes 0, the label carries "refinement trend of x % detected".
   - `alpha_LAM` keeps its relative margin. It is the only A verdict whose meaning is unchanged.
2. **Anticipated outcome (as F9 did for `sigma^2 = 4`).** This is an extrapolation from 1, 2.25 and 4, not a measurement.
   - `nr_rule` is expected to give `N_R` of about 20, because the B-rule floor 2.5e-3 exceeds `alpha_T`.
   - The 90 % half-width of `alpha_T` is expected to be comparable to `alpha_T`.
   - `physical (positive)`, `physical (bounded)` (rule D branch 4, F6; `U <= 2.5e-3`) and `unresolved (estimator-discordant)` are
     each plausible from sampling alone. Discordance between the two estimators is not, by itself, evidence that they disagree.
   - "B met" does not mean that the sign is resolved.
   - `N` is not raised after any `s025` result. The owner chose `nr_rule` unchanged after being shown this.
3. **B&dD.**
   - B&dD report `alpha_TA < 1e-3` at 0.25 (Table 2), itself an upper bound. The descriptive one-sided comparison uses 1e-3.
   - "Below" (our one-sided 95 % upper limit `< 1e-3`) is read as consistent with B&dD's bound. "Not below" is read as
     uninformative.
   - A disagreement is stated only if the one-sided 95 % lower limit, which is recorded, exceeds 1e-3.
   - B&dD's conditions are stated as at 1 and 2.25.
4. **Gate 5.**
   - Gate 5 at 0.25 uses `n = N_R` per arm and is not sized for its own margin. If the `s1` offset (Lagrangian 8.8 % below Par2)
     persists, `inconclusive` is anticipated.
   - An unexplained `disagree` falls under RR6 for all of SF-39.
   - The ANCOVA sensitivity has little leverage, because the realized-variance spread is small. It is expected to equal the primary
     verdict.
   - The Par2 `dt/2` sensitivity (SF-38 E, D12) is not available at 0.25.
5. **Geometry.** The `sigma^2 = 1` geometry is conservative for the absolute error in `lambda`. The relative error of `alpha_T`
   is not bounded by the resolution-vs-variance argument.

## A5. Pilot, checkpoint, `N`, budget (section 6)

- **Pilot.** 20 realizations per `s025` arm (`s025_A`, the three narrow rungs, `s025_tw`) and the Par2 pilot (20).
- **Checkpoint for `s025` only**, run as a separate case:
  - It writes everything under `checkpoint/s025/`:
    - the write-once `tau_b_star_s025.json` (same schema and content rules as `tau_b_star.json`: `tau_b*`, twin and Par2 frozen
      times, `tau_c`);
    - the sealed `s025` pilot estimates;
    - the `nr_rule` input and output;
    - costs, bytes;
    - the blinded report.
  - The existing checkpoint outputs (`tau_b_star.json`, `sealed/`, `costs.json`, `bytes.json`, `checkpoint_report.*`) are not
    touched. The part bookkeeping files are appended to or recomputed.
  - The report follows RR4: `tau_b*`, `N`, GPU-s per realization, GPU-h, wall time, bytes. It has no mean, sd, margin, verdict or
    `alpha`.
- **`N`.** `nr_rule.py` runs on the `s025` pilot only, with the section 6 rules (TOST power 0.80, B rule, cap 500).
- **Budget.** The owner budget of 2026-10-28 applies.
  - If the `s025` projection (GPU-h and bytes) is within it and within the host's free disk, then f = 1 and the printed `N` are
    final, with no stop.
  - Otherwise the section 6 budget rule applies, and anything beyond it goes to the owner.
  - The `N` are added as the `s025` entry of `configs/budget.json` (existing entries unchanged). They are committed, with the new
    sha256, in a bitácora row and synced before any `s025` extension run.

## A6. Reading rules (section 7), the analysis, the regression of the existing amplitudes

- **RR0-RR6 apply to `s025`.**
- **RR4** applies with "owner's budget recorded" read as "`s025` `N` recorded". The readable files are RR4's list plus
  `checkpoint/s025/tau_b_star_s025.json` and the `s025` report. The results at 1, 2.25 and 4 are already known, as disclosed above.
- **Code freeze.** All `s025` script changes, including the `s025` entries of the analysis (nominal variance 0.25, B&dD bound 1e-3),
  are integrated and their tree recorded before the `s025` checks. Any later change is a disclosed corrective.
- **Analysis.** The analysis part is re-run over the four amplitudes, on the same `$SF39_OUT_ROOT` data and the same host Python /
  numpy / scipy (versions recorded before the run).
- **Regression of `s1`, `s225`, `s4`.** It must hold against evidence commit `4dbf31f2`:
  - (a) parsed-JSON equality of `summary.json` `amplitudes.{s1,s225,s4}` and `budget.{s1,s225,s4}`;
  - (b) byte equality of the 53 `reduced/*.csv` files and of `series/alpha_t_{s1,s225,s4}.csv`;
  - (c) byte equality of the ordered `decisions.csv` lines whose amplitude is `s1`, `s225` or `s4`, and of the existing `all`
    process rows. Additions are listed.

  The comparison command is committed with the evidence. Any difference is a defect and stops the work; it is never accepted as new.
- **Evidence import.** New bookkeeping is imported beside the committed one (`raw/campaign_s025/<part>/`, `raw/par2_s025[_ext]/`)
  without overwriting `raw/campaign/`. The analysis output replaces `analysis/` only after the regression above passes.
- **Decision points.** The section 10 table gains the `s025` rows:
  - primary: Gate 4 per estimator and the amplitude label; Gate 5 with both sensitivities;
  - key secondary: A and B, with the A4 readings beside them;
  - estimation: F7, F8, F11;
  - no F10 rows.
