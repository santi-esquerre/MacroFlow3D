#!/usr/bin/env bash
# SF-42 R1, acceptance A (extra byte checks): base binaries against head binaries, same device, same job.
#   bytes.sh <base_bin_dir> <head_bin_dir> <out_dir>      (run from the root of the mirror = head tree)
# (a) one default `lagrangian_ensemble --medium b` realization with SF-39-style flags (no new flag);
# (b) a shortened copy of the SF-39 Par2 config `par2_s1.yaml` (no new key).
# Each binary runs in its own empty cwd with identical relative arguments, so command lines and relative paths match.
# Verdicts: treecmp.py (every file byte for byte; only timings.json skipped; manifest.json structural after dropping
# `timestamp` and `build.git_hash`, with the raw byte equality printed too).
set -u
BASE=$(cd "$1" && pwd)
HEAD=$(cd "$2" && pwd)
OUT=$3
REPO=$(pwd)
V=$REPO/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/verification
SF39=$REPO/docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
bad=0

# ---- (a) lagrangian_ensemble medium (b), default options ----------------------------------------------------------------
# 512x64x64, ell = 10/64 (lambda/h = 10), sigma^2 = 1 (eps 1), seed 420201 (SF-42 reserved range), M = 100 passes RR12 on ny = 64
RUNNER_ARGS=(--medium b --grid 512,64,64 --ell 0.15625 --sigma2 1 --eps 1 --seed0 420201 --nr 1 --window 0.1,0.9,0.1,0.9
    --np-side 100 --times auto --tol 1e-8 --fused-splines --single-launch --twin on --pollock-bridge on --out out)
for tag in base head; do
    bin=$BASE
    [ "$tag" = head ] && bin=$HEAD
    mkdir -p "$OUT/a_$tag"
    (cd "$OUT/a_$tag" && echo "CMD lagrangian_ensemble ${RUNNER_ARGS[*]}" && date -u +START_UTC=%FT%TZ &&
        "$bin/lagrangian_ensemble" "${RUNNER_ARGS[@]}"; echo "RC=$?"; date -u +END_UTC=%FT%TZ) > "$OUT/a_$tag.log" 2>&1
    grep -q '^RC=0$' "$OUT/a_$tag.log" || { echo "a_$tag: runner failed"; bad=$((bad + 1)); }
done
echo "== (a) lagrangian_ensemble medium b: base vs head"
python3 "$V/treecmp.py" "$OUT/a_base/out" "$OUT/a_head/out" | tee "$OUT/a_treecmp.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || bad=$((bad + 1))

# ---- (b) Par2, SF-39 config shortened, no new key -------------------------------------------------------------------------
python3 - "$SF39/configs/par2_s1.yaml" "$OUT/par2_tiny.yaml" << 'PY'
import re, sys
t = open(sys.argv[1]).read()
for pat, rep in [(r"nx: 2048", "nx: 256"), (r"ny: 256", "ny: 32"), (r"nz: 256", "nz: 32"),
                 (r"value: 100\.0", "value: 12.5"), (r"mg_levels: 6", "mg_levels: 4"),
                 (r"n_particles: 10000", "n_particles: 200"), (r"n_steps: 95749", "n_steps: 300"),
                 (r"output_every: 319 ", "output_every: 30 "), (r"NR: 20", "NR: 2"), (r"every: 957", "every: 100")]:
    t, n = re.subn(pat, rep, t)
    assert n >= 1, pat
open(sys.argv[2], "w").write(t)
PY
for tag in base head; do
    bin=$BASE
    [ "$tag" = head ] && bin=$HEAD
    mkdir -p "$OUT/b_$tag"
    cp "$OUT/par2_tiny.yaml" "$OUT/b_$tag/par2_tiny.yaml"
    (cd "$OUT/b_$tag" && echo "CMD macroflow3d_pipeline par2_tiny.yaml" && date -u +START_UTC=%FT%TZ &&
        "$bin/macroflow3d_pipeline" par2_tiny.yaml; echo "RC=$?"; date -u +END_UTC=%FT%TZ) > "$OUT/b_$tag.log" 2>&1
    grep -q '^RC=0$' "$OUT/b_$tag.log" || { echo "b_$tag: pipeline failed"; bad=$((bad + 1)); }
done
echo "== (b) Par2 (SF-39 s1 config, shortened): base vs head"
python3 "$V/treecmp.py" "$OUT/b_base/raw" "$OUT/b_head/raw" | tee "$OUT/b_treecmp.txt"
[ "${PIPESTATUS[0]}" -eq 0 ] || bad=$((bad + 1))

echo "BYTES_DONE bad=$bad"
[ "$bad" -eq 0 ]
