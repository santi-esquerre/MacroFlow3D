# SF-43 addendum 1: results of the production job `sf43add-prod` (frozen, n = 60)

Frozen outputs of the exploratory arms J1, J3, J9 (outside the Holm family), copied from the host campaign `~/sf43_addendum` after
the analysis freeze (bitácora row 2026-11-05T09:15Z, commit `786448b9`). Interpretation: section 3.5 of
`docs/experiments/2026-11-03-sf43-confirmatory-macrodispersion.md`. Pre-registration: `../preregistration-addendum-1.md` (Amendment A1)
and `../prereg_addendum.json`. Layout mirrors `../results/`.

| path | content |
|---|---|
| `analysis/addendum_summary.json`, `.md`, `analysis/reproduction.json` | the pre-registered analysis (the results; `sf43add.ana run`) |
| `REPORT.txt`, `ledger.jsonl`, `DONE` | engine report, solver-verdict and failure ledger, completion record |
| `logs/sf43add-prod.log`, `logs/sf43add-s0.log` | job logs of the production job and of the S0 smoke |
| `logs/analysis.log`, `logs/recompute.log`, `logs/reconcile.log` | in-job analysis (structure only), `recompute43`, `reconcile43` |
| `logs/exec-blinded.log` | blinded-phase log: lines 1-21 are the frozen version (sha256 `cddcb837...`), line 22 a structure read after the job, line 23 the `blinded-stop` (2026-11-05T09:15:16Z) |
| `reduced/tables/<arm>.tar.xz` | per-realization estimator tables (`*.csv` that are not `*.planes.csv`) and `*.manifest.json`; members `<arm>/<segment>/<file>` |
| `reduced/planes/<arm>.tar.xz` | per-plane moment tables (`*.planes.csv`) |
| `reduced/freeze.reduced.sha256` | sha256 of the 225 reduced files at the freeze (manifest sha256 `63bb2def4238224aae9d460ba2a75aad3b05ea8769e0819a2c3b0db9d07f0dd6`); paths are `reduced/<arm>/<segment>/<file>` relative to the campaign root |
| `SHA256SUMS` | sha256 of every other file in this directory |

Archives are deterministic (`tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner -cJf`, run from `reduced/`).

Verify the files:

```bash
cd docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/addendum1-results
sha256sum -c SHA256SUMS
```

Verify the reduced data against the freeze (225 files):

```bash
R=$PWD
mkdir -p /tmp/a1x/reduced && for f in $R/reduced/tables/*.tar.xz $R/reduced/planes/*.tar.xz; do tar -xJf $f -C /tmp/a1x/reduced; done
cd /tmp/a1x && sha256sum -c $R/reduced/freeze.reduced.sha256     # must be run from the parent of the extracted reduced/
```

The raw segment outputs (plane arrays, Pollock arrays; about 54 GB of unpruned `j9_w48` arrays among them, deviation AD3) stay on the
host only.

## `j8/`: the conditional J8 extension (seeds 434061-434100 in `j8_H3`, `j8_D`, `j8_DH3`; job `sf43add-j8`)

Same layout as above, for the J8 campaign `~/sf43_addendum_j8` (plan `A/plan_addendum_j8.json`, sha256 `40ef528d...`; freeze row commit
`188054d4`, blinded stop 2026-11-05T10:34:41Z). `j8/analysis/addendum_summary.{json,md}` is the J8 job's frozen analysis: its `n60`
block is identical to the production one and it adds the `J1 (n100)` block (secondary). Differences with the production part:

- `j8/recompute.txt` and `j8/reconcile.txt` are the independent-check outputs (`sf43_addendum_j8.recompute.txt` / `.reconcile.txt` on the
  host); `j8/logs/` has the job log `sf43add-j8.log` and `exec-blinded.log` in its 29-line version: **lines 1-27 are the frozen version**
  (sha256 `87bbdaa2...`), line 28 a structure read after the job, line 29 the `blinded-stop`.
- `j8/reduced/` has 36 files in 3 arms; `j8/reduced/freeze.reduced.sha256` (manifest sha256
  `9e0d3d037c202cace16ed522c3a6a39b3c5f76941cd76807557c0a5f25d516e3`) is checked the same way (`sha256sum -c` from the parent of the
  extracted `reduced/`; 36/36).
- The raw J8 segment outputs (retention 22/2, 22/2, 32/2; 13 GB) stay on the host.

`SHA256SUMS` (this directory only, one file) covers every file here including `j8/`; there is no `j8/SHA256SUMS`.
