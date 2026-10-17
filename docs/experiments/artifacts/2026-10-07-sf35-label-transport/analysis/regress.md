# SF-35 regression G: `--production` 32^3 eps 0.25 vs SF-33 `ladder_0.25_c4/N32.log`

- new log: `docs/experiments/artifacts/2026-10-07-sf35-label-transport/logs/regress/prod32.log`
- reference log: `docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/logs/ladder_0.25_c4/N32.log`
- compared lines (first token in CASE, ORACLE, ORACLE_SUMMARY, EXTRA): new 102, reference 102
- masked fields (timing only): `t=`, `t_trace=`, `t_labels=`, `t_lu=`, `t_apply=`, `t_fact=`, `t_lin=`

**Verdict G (production lines): identical** (all 102 lines equal after masking the timing fields)

Informational (not part of G): ORACLE_CONFIG/ORACLE_RUN/ORACLE_LABELDIFF lines: new 6, reference 6, identical
