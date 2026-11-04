# sf43add.ana: analysis of SF-43 addendum 1 (J1, J3, J9; conditional J8)

Purpose: the pre-registered analysis of the exploratory arms of `preregistration-addendum-1.md` / `prereg_addendum.json` (same artifact
directory; the text governs): J1 (Par2 at `sigma^2 = 4`, horizon against step), J3 (`lambda/h = 5` twin of `s1_dist`), J9 (width 48 `lambda`),
the J8 trigger and the reproduction check that gates J1. Descriptive, outside the Holm family (`m = 8`). Nothing frozen is modified: the data go
through the FROZEN `sf43ana.loader` (imported) and the statistics through the frozen SF-42 `mcstat` (`gate5_paired`). stdlib + numpy + scipy;
Python 3.11-compatible. Run from the repository root with `PYTHONPATH=<A>/scripts:<SF42>/scripts` (`A` = this artifact directory).

```bash
python3 -W default -m sf43add.ana run --prereg A/prereg_addendum.json --campaign C --frozen-reduced R --frozen-manifest M --frozen-ledger L \
        [--frozen-segments S] [--j8-campaign J] --out O          # O/addendum_summary.json, addendum_summary.md, reproduction.json
python3 -W default -m sf43add.ana check --prereg A/prereg_addendum.json --campaign C --out O         # S0 structure check -> O/smoke_check.json
python3 -W default -m sf43add.ana repro --campaign C --arm j1_base10 --seeds 434001[-434010] --frozen-reduced R --frozen-manifest M [--frozen-segments S]
python3 -W default -m sf43add.ana j8-trigger --summary O/addendum_summary.json --gpu-h-used X --j8-gpu-h-projected Y --j8-end-utc 2026-11-06T22:00:00Z
python3 -W error -m unittest discover -s A/scripts/sf43add/ana/tests -t A/scripts                     # 79 tests, about 15 s, no GPU, no host
```

## Blinded phase: what reaches the log

`run` and `check` print STRUCTURE ONLY: arms and realizations per arm with finite counts, common-seed counts per contrast, whether the
reproduction check was evaluated (not its outcome), frozen files verified, warnings / errors counts, `STRUCTURE_OK` | `STRUCTURE_FAIL`. No ratio, interval,
class, alpha or rule outcome is printed; `BLOCK_ERROR` lines carry the block and the exception type only (a unit test injects a sentinel value and asserts
it is absent from stdout and stderr, and that no decimal number appears). `repro` prints one line, `REPRO PASS|FAIL n_seeds n_rows max_rel csv_sha_equal=k/n`
(`max_rel` is a discrepancy, not an alpha); its stderr names columns, seeds and counts, never values. Results are in the files, read after the freeze.
Exit codes: `run` 0 analysis completed (a "not evaluable" rule is a result), 1 a block raised (recorded in the summary, `STRUCTURE_FAIL`), 2 an input was refused
(a frozen file whose sha256 differs from the freeze manifest, a manifest other than the pre-registered `250395ed...`, no campaign ledger, ...); `repro` 0 PASS, 1 FAIL, 2 refused.

## Data

* Frozen arms `s4_g5p`, `s4_g5pol` (Pollock rows of `s4_l16`, uniform weights) and `s1_dist`: the entries of the frozen `prereg.json` validated by
  `sf43ana.config`, read from `--frozen-reduced` with `--frozen-ledger`; EVERY file read is verified against `--frozen-manifest` (`<sha256>  ./<arm>/<segment>/<file>`)
  before it is parsed (`data.Frozen`); a missing entry or a mismatch is an error. The frozen arms must have their planned sizes (250, 250, 100).
* Addendum arms: entries built by `data.arm_cfgs` from `prereg_addendum.json` (`j1_*`, `j8_*`: like `s4_g5p`, code par2, weights unif; `j3_l5`, `j9_w48`: like `s1_dist`),
  read from `C/ledger.jsonl` and `C/reduced/<arm>/<segment>/...`. The analysis set of an arm is the loader's (longest complete prefix after the group-wide
  replacements): a replaced seed has no partner and drops out of every paired contrast; every `n` is stored. Crossing CSVs: `C/segments/<arm>/<segment>/out/crossings/`.
* Responses: J1 `alpha_LP` (position column, the `gate5` key), window `main` [25, 94]; J3 / J9 `alpha_TP:disp`, windows `main` and `far` [94, 194] (the loader reads every window
  with the same selection). Completeness = `n_last / n_traced` of the Par2 main-window row; complete = `n_last == n_traced`.

## Statistics (`stats.py`; preregistration section 2)

* Paired ratio `mean X / mean Y` on the seeds finite in both (sorted by seed): IC90 = `gate5_paired(method="bootstrap-t", B=2000, seed=20261104)`; the Student interval is reported
  beside it and used by no rule. Every rule uses the bootstrap-t one; point estimates are `exp(theta)`; "within [a, b]" is `lo >= a and hi <= b`.
* Paired ratio of ratios (`m_far / m_main`): `log_contrast` = the same studentized seed-pair bootstrap (same stream, same rank rule) for `sum_k s_k log mean x_k` with the
  linearized per-seed contrast; with two series it IS `gate5_paired` bit for bit (unit test).
* Unpaired (`F48/F5`, `F48/F24`, the `W48/W24` levels): `theta_1 - theta_2`, `se = sqrt(se_1^2 + se_2^2)`, Welch-Satterthwaite `nu`, `theta +- t(0.95, nu) se`.
* A quantity with fewer than 3 common finite seeds, an absent arm or a non-positive mean is `{"evaluable": false, "n", "reason"}`, never an exception.

## Rules (`rules.py`; thresholds read from `prereg_addendum.json`)

Every condition of every class is evaluated and stored with its value and truth; the first class whose conditions hold wins; the fall-through (`abierto`, `inconcluso`) is last.
* J1: reproduction gate (not evaluated / failed: NOT classified, reason stored) -> manipulation check on `j1_H3` (failed: `abierto (horizonte no saturado)`; not evaluable: not
  classified) -> classification (needs `r_H, r_D, q_H3, q_DH3` evaluable, else "not evaluable"). `q_H3`, `q_DH3`, completeness per arm, `q_base` against the frozen 1.164 [1.131, 1.198]
  and the J8 trigger inputs (which IC90 contain which threshold in their interior) are always stored. With `--j8-campaign` the same at `n = 100` (`S.j1.n100`, the 60 seeds
  of each arm plus 434061-434100; no `j8` arm for H2); `n = 60` (`S.j1.n60`) is primary.
* J3: `m_main`, `m_far`, `F5`, `m_far/m_main`; `lambda/h = 5 equivalente` iff the three IC90 are within [0.97, 1.03], [0.97, 1.03], [0.98, 1.02]; otherwise (also with `j3_l5` missing)
  `no equivalente`.
* J9: `F48`, `F48/F5`, `F48/F24`, `F24` recomputed (frozen percentile value beside it), `W48/W24` main and far (descriptive); classes in order `inconcluso por N` (< 100 realizations
  with finite main and far values: precedence), `meseta robusta al ancho`, `decae con el ancho`, `inconcluso`; the J3 qualification is attached (`j3_equivalent`, caveat text).
* `j8-trigger`: (a) J1 reproduction passed, (b) some IC90 contains a threshold, (c) used + 1.2 x projected <= cap (8 GPU-h), (d) projected end <= `hard_end_utc`; reads only the summary.

## Reproduction check (`repro.py`; preregistration 3.1.1)

Rows matched by `seed, family, weights, basis, set, d_lo, d_hi, hmax_div, code` (Par2 seed = `seed0 + r`, `seed0` from the segment directory `u<unit>_s<seed0>`); every column except
`source`, `r` and path / timing columns within 1e-12 relative (1e-15 absolute where a value is 0). Strict (a tightening): a row on one side only, different column sets, a repeated row or a
requested seed absent on one side is a FAIL. The Par2 crossing CSV of each seed is compared by sha256 against the frozen segment (`--frozen-segments`), else against the `files` list of the
frozen `.manifest.json`; equality is the expectation and is reported (`csv_sha_equal`), it does not gate.

## Files

`stats.py` inference, `data.py` loading and views, `rules.py` conditions / J1 / J3 / J9 / J8 inputs, `repro.py` reproduction check, `summary.py` `run` (+ markdown), `check.py` smoke
structure check, `trigger.py` `j8-trigger`, `__main__.py` CLI, `tests/` (`synth.py` builds synthetic frozen and addendum worlds with the frozen test fixture and the REAL `prereg.json` /
`prereg_addendum.json` entries; `test_fidelity.py` reads the committed `A/results/` archives into a scratch directory and checks the published numbers).
