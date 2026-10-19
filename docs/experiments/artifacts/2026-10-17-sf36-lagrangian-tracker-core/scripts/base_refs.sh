#!/usr/bin/env bash
# SF-36 orchestrator job sf36-base-refs: base-commit (040e3ce) references for criteria F and G.
# Runs on the SF-36 mirror synced from a clean detached worktree of the base. bash 4.2-safe
# (no empty-array expansion under set -u; no backslashes in pattern replacements).
set -o pipefail
M=~/MacroFlow3D-SF-36
B=~/sf36_base_refs
cd "$M" || exit 2
echo "BASE_TREE_COMMIT=$(cat "$M/.sf36_base_commit" 2>/dev/null)"
cmake --preset v100-release > /dev/null && cmake --build build/v100-release -j 2>&1 | tail -n 5
echo "BUILD_EXIT=$?"
ctest --test-dir build/v100-release -N | tail -n 2
rm -rf "$B"
mkdir -p "$B/bin" "$B/trio" "$B/trace" "$B/mem"
cp build/v100-release/inlet_slab build/v100-release/macroflow3d_pipeline "$B/bin/"
md5sum "$B/bin/inlet_slab" "$B/bin/macroflow3d_pipeline"

echo "== SF-27 trio (base)"
cd "$B/trio" || exit 2
for c in config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation; do
    "$B/bin/macroflow3d_pipeline" "$M/apps/$c.yaml" > "$c.log" 2>&1
    echo "$c exit=$?"
done

echo "== SF-35 --construct trace (base): F at 32/64 x eps 0.5/1.0 x host/gpu; eps 2 gpu at 64/128"
cd "$B/trace" || exit 2
F="--construct trace --sigma2 1 --ell 0.25 --seed 3001 --threads 32"
run_case() { # name args...
    local name=$1
    shift
    /usr/bin/time -v -o "$name.time" "$B/bin/inlet_slab" "$@" --save-labels "$B/trace/$name" --summary "$B/trace/$name.json" > "$name.log" 2>&1
    echo "$name rc=$?"
}
for N in 32 64; do
    for e in 0.5 1.0; do
        for t in host gpu; do
            run_case "N${N}_e${e}_${t}" $F --n "$N" --eps "$e" --tracer "$t"
        done
    done
done
for N in 64 128; do
    run_case "N${N}_e2.0_gpu" $F --n "$N" --eps 2 --tracer gpu
done

echo "== SF-33 production regression input (base): --production 32^3 eps 0.25 --oracle-ladder"
"$B/bin/inlet_slab" --production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder > "$B/trace/prod32.log" 2>&1
echo "prod32 rc=$?"

echo "== G: device peak, base build, 256^3 eps 2 gpu (nvidia-smi 200 ms, physical GPU ${CUDA_VISIBLE_DEVICES})"
cd "$B/mem" || exit 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" -lms 200 > N256_e2.mem &
SAMPLER=$!
/usr/bin/time -v -o N256_e2.time "$B/bin/inlet_slab" $F --n 256 --eps 2 --tracer gpu --summary "$B/mem/N256_e2.json" > N256_e2.log 2>&1
echo "N256_e2 rc=$?"
kill "$SAMPLER" 2> /dev/null
echo "BASE_PEAK_MIB_N256_E2=$(sort -n N256_e2.mem | tail -n 1)"
echo "SF36_BASE_REFS_END"
