#!/usr/bin/env bash
# SF-42 R1, acceptances B and C: field identity (Par2 sf18_medium_b against lagrangian_ensemble --medium b), injection window,
# plane crossings (standalone brute-force test), seed-lattice probe (standalone).
#   identity.sh <bin_dir> <out_dir>      (run from the root of the mirror)
set -u
BIN=$(cd "$1" && pwd)
OUT=$2
REPO=$(pwd)
V=$REPO/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/verification
SCR=$REPO/docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
bad=0

# ---- B: field identity --------------------------------------------------------------------------------------------------
while read -r name nx ny nz ell eps seed0; do
    d=$OUT/B_$name
    mkdir -p "$d/par2" "$d/consumer"
    python3 "$V/identity.py" config "$name" "$d/par2/config.yaml" out
    {
        echo "== $name: grid $nx,$ny,$nz ell $ell eps $eps seed0 $seed0 (sigma2 1)"
        # M = 10 passes RR12 on ny = 16 and ny = 24 for the window [0.1, 0.9]^2 (seedlattice.rule)
        echo "CMD lagrangian_ensemble --medium b --grid $nx,$ny,$nz --ell $ell --sigma2 1 --eps $eps --seed0 $seed0 --nr 1 --np-side 10 --times auto --tol 1e-6 --dump-field fields --out out"
        (cd "$d/consumer" && "$BIN/lagrangian_ensemble" --medium b --grid "$nx,$ny,$nz" --ell "$ell" --sigma2 1 --eps "$eps" --seed0 "$seed0" \
            --nr 1 --np-side 10 --times auto --tol 1e-6 --dump-field fields --out out > run.log 2>&1)
        echo "consumer rc=$?"
        echo "CMD macroflow3d_pipeline config.yaml"
        (cd "$d/par2" && "$BIN/macroflow3d_pipeline" config.yaml > run.log 2>&1)
        echo "par2 rc=$?"
        python3 "$V/identity.py" fields "$d/consumer/fields" "$d/par2/out"
    } > "$OUT/B_$name.txt" 2>&1
    cat "$OUT/B_$name.txt"
    grep -q '^FIELD_IDENTITY PASS' "$OUT/B_$name.txt" || bad=$((bad + 1))
done << 'EOF'
p2_e1 64 16 16 0.25 1 420211
p2_e2 64 16 16 0.25 2 420212
f3_e1 96 24 24 0.25 1 420213
f3_e2 96 24 24 0.25 2 420214
EOF

# ---- C: injection window ----------------------------------------------------------------------------------------------------
w=$OUT/C_window
mkdir -p "$w"
python3 "$V/identity.py" window "$w/config.yaml" out
{
    echo "CMD macroflow3d_pipeline config.yaml (inject_window, plane_crossings dump_injection)"
    (cd "$w" && "$BIN/macroflow3d_pipeline" config.yaml > run.log 2>&1)
    echo "par2 rc=$?"
    python3 "$V/identity.py" injection "$w/out"
} > "$OUT/C_window.txt" 2>&1
cat "$OUT/C_window.txt"
grep -q '^INJECTION_WINDOW PASS' "$OUT/C_window.txt" || bad=$((bad + 1))

# ---- C: plane crossings against the brute-force reference (the ctest par2_sf42_options binary, standalone) --------------------
echo "== par2_sf42_options_tests (standalone; contains B.crossings_equal_bruteforce_1e-12)"
"$BIN/par2_sf42_options_tests" "$BIN/macroflow3d_pipeline" > "$OUT/C_par2_sf42_options.txt" 2>&1
rc=$?
grep -E 'CHECK B\.(crossings|injection|Y_|K_|var_Y)|^(PASS|FAIL|ALL)|passed|failed' "$OUT/C_par2_sf42_options.txt" | head -40
echo "par2_sf42_options_tests rc=$rc"
[ "$rc" -eq 0 ] || bad=$((bad + 1))

# ---- C: seed-lattice probe against the Python rule (standalone) -------------------------------------------------------------
echo "== seedlattice verify_probe (standalone)"
(cd "$SCR" && python3 -W error -m seedlattice.verify_probe "$BIN/seedlattice_probe") > "$OUT/C_seedlattice_probe.txt" 2>&1
rc=$?
tail -5 "$OUT/C_seedlattice_probe.txt"
echo "seedlattice verify_probe rc=$rc"
[ "$rc" -eq 0 ] || bad=$((bad + 1))

echo "IDENTITY_DONE bad=$bad"
[ "$bad" -eq 0 ]
