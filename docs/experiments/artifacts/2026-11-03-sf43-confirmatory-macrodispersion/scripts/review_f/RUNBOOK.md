# Acceptance F: independent reduction of four realizations (owner runbook)

**What this is.** `indep_reduce_f.py` re-computes `alpha_TP`, `alpha_TAP`, `alpha_LP` (flux weights, set `perplane`, displacement basis, window
`d` in [25, 94]) of one production realization directly from its raw arrays (`seeds`, `weights`, `planes_x23`, `planes_tau`, `track.json`),
using only the definitions of `preregistration.md` section 4, and compares them with the frozen reduced table. It is Python 3 + numpy and imports
nothing from the project (no `sf43ana`, `sf43camp`, `sf42red`, `mcstat`). It was written by an agent, not by a human: it is a **weaker**
reduction than one you write yourself; the comparison and the verdict are yours. File-format conventions it assumes are listed in the script header.

**The four realizations** (field seed; segment and index are found by the script through `seedmap.json`):

| arm | seed | segment (index) | pass |
|---|---|---|---|
| `s025_prod` | 431001 | `u000350_s431001` (r 0) | multi-pass, production `hmax_8` |
| `s1_prod` | 432080 | `u000365_s432071` (r 9) | single |
| `s225_prod` | 433140 | `u000185_s433131` (r 9) | single |
| `s4_prod` | 434001 | `u000006_s434001` (r 0) | multi-pass, production `hmax_8` |

**Commands** (one per realization; `A` = `docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion`, written in full):

```bash
scripts/remote --increment SF-43 exec -- "cd ~/MacroFlow3D-SF-43 && python3 docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/review_f/indep_reduce_f.py --campaign ~/sf43_campaign --arm s025_prod --seed 431001"
scripts/remote --increment SF-43 exec -- "cd ~/MacroFlow3D-SF-43 && python3 docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/review_f/indep_reduce_f.py --campaign ~/sf43_campaign --arm s1_prod --seed 432080"
scripts/remote --increment SF-43 exec -- "cd ~/MacroFlow3D-SF-43 && python3 docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/review_f/indep_reduce_f.py --campaign ~/sf43_campaign --arm s225_prod --seed 433140"
scripts/remote --increment SF-43 exec -- "cd ~/MacroFlow3D-SF-43 && python3 docs/experiments/artifacts/2026-11-03-sf43-confirmatory-macrodispersion/scripts/review_f/indep_reduce_f.py --campaign ~/sf43_campaign --arm s4_prod --seed 434001"
```

(`--hmax-div 8` is the default, so it is omitted; it is also the table label for single-pass realizations; it selects `hmax_8/` only when the realization directory has
that sub-directory.) Each is one short `exec` (estimated: reads tens of MB, a few seconds; not measured on the host).

**Preconditions.** Run it only when no `scripts/remote sync` is in progress and, preferably, no SF-43 job is `running` (the data are read-only
and frozen, but a sync replaces the mirror tree that contains the script). The mirror must already hold this commit. No CUDA, no GPU, nothing
is written on the host: output is on stdout only. The script stops with exit 2 and a message on stderr (never a guess) if the seed is in no or
several segments, the realization has no `DONE`, both single- and multi-pass layouts exist, the requested pass is absent, or arrays are inconsistent.

**Reading the result.** Each run prints the located paths, `ell`, `x0`, the window (expect 70 planes, `d` 25 .. 94), then a table
(independent value, table value, relative difference) for `alpha_TP`, `alpha_TAP`, `alpha_LP`, `U_r` and the particle counts (`n_traced`,
`n_first`, `n_last`, `n_min`, `n_max`), and a last line:

* `F PASS` (exit 0): every relative difference is <= 1e-10 (absolute 1e-15 where the table value is 0), the tolerance fixed by the
  pre-registration (section 10) and the specification for recomputation. Typical agreement is 1e-14 or better.
* `F FAIL` (exit 1): at least one quantity is beyond tolerance (marked `<-- BEYOND TOLERANCE`) or is not finite.

**What to record in the review.** The four complete outputs (arm, seed, segment, the table of differences and the `F PASS/FAIL` line), the commit of
this script and the host path/date of the runs. A `FAIL` is not a verdict on the science: first check the printed layout/pass and your own
reading of the definitions, then report it as found. The comparison and the acceptance decision belong to the owner.
