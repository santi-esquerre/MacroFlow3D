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

The conditional J8 extension (seeds 434061-434100) is recorded in a sibling `j8/` directory when it exists.
