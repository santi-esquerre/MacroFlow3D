#!/usr/bin/env bash
# SF-37 orchestrator job sf37-base-refs: base-commit (44d4697) references for criterion R
# (SF-35 --construct trace and SF-36 --track forward outputs byte-identical at the head with default
# options; SF-33 --production lines; SF-27 trio). Runs on the SF-37 mirror synced from a clean detached
# worktree of the base BEFORE any SF-37 source reaches it. Regression runs only (seed 3001, small N):
# not experiment runs of SF-37 (the fresh seeds 3101-3103 are never touched here).
# bash 4.2-safe (no empty-array expansion under set -u; no backslashes in pattern replacements).
set -o pipefail
M=~/MacroFlow3D-SF-37
B=~/sf37_base_refs
cd "$M" || exit 2
echo "BASE_TREE_COMMIT=$(cat "$M/.sf37_base_commit" 2>/dev/null) ACT_COMMIT=$(cat "$M/.sf37_act_commit" 2>/dev/null)"
echo "GPU=${CUDA_VISIBLE_DEVICES:-unset} $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}")"
cmake --preset v100-release > /dev/null && cmake --build build/v100-release -j 2>&1 | tail -n 5
echo "BUILD_EXIT=$?"
ctest --test-dir build/v100-release -N | tail -n 2
rm -rf "$B"
mkdir -p "$B/bin" "$B/trio" "$B/trace" "$B/fwd" "$B/mem"
cp build/v100-release/inlet_slab build/v100-release/macroflow3d_pipeline "$B/bin/"
md5sum "$B/bin/inlet_slab" "$B/bin/macroflow3d_pipeline"

echo "== SF-27 trio (base)"
cd "$B/trio" || exit 2
for c in config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation; do
    "$B/bin/macroflow3d_pipeline" "$M/apps/$c.yaml" > "$c.log" 2>&1
    echo "$c exit=$?"
done

echo "== SF-35 --construct trace (base): 32/64 x eps 0.5/1.0 x host/gpu; eps 2 gpu at 64"
cd "$B/trace" || exit 2
F="--construct trace --sigma2 1 --ell 0.25 --seed 3001 --threads 32"
run_trace() { # name args...
    local name=$1
    shift
    /usr/bin/time -v -o "$name.time" "$B/bin/inlet_slab" "$@" --save-labels "$B/trace/$name" --summary "$B/trace/$name.json" > "$name.log" 2>&1
    echo "$name rc=$?"
}
for N in 32 64; do
    for e in 0.5 1.0; do
        for t in host gpu; do
            run_trace "N${N}_e${e}_${t}" $F --n "$N" --eps "$e" --tracer "$t"
        done
    done
done
run_trace "N64_e2.0_gpu" $F --n 64 --eps 2 --tracer gpu

echo "== SF-33 production regression input (base): --production 32^3 eps 0.25 --oracle-ladder"
"$B/bin/inlet_slab" --production --n 32 --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder > "$B/trace/prod32.log" 2>&1
echo "prod32 rc=$?"

echo "== SF-36 --track forward (base): default options"
cd "$B/fwd" || exit 2
run_fwd() { # name args...
    local name=$1
    shift
    /usr/bin/time -v -o "$name.time" "$B/bin/inlet_slab" --track forward "$@" --save-particles "$B/fwd/$name" --summary "$B/fwd/$name.json" > "$name.log" 2>&1
    echo "$name rc=$?"
}
G="--sigma2 1 --ell 0.25 --seed 3001"
run_fwd smoke_g3d_n16 --analytic generic3d --n 16 --eps 0.25 --np-side 16 --planes 0.25,0.5,1 --times 0.5,1 --covectors
run_fwd g3d_n32_cov --analytic generic3d --n 32 --eps 0.25 --np-side 32 --planes 0.5,1 --times 1 --covectors
run_fwd gauss_n64_e1_cov $G --n 64 --eps 1 --np-side 64 --planes 0.25,0.5,1,2 --times 0.5,1,2 --covectors
run_fwd gauss_n64_e2 $G --n 64 --eps 2 --np-side 64 --planes 0.25,0.5,1,2,4 --times 0.5,1,2,4
run_fwd gauss_n64_e2_tol $G --n 64 --eps 2 --np-side 64 --planes 0.5,1,2 --times 1,2 --tol 1e-10 --hmax-div 32 --max-arclength 1000
run_fwd gauss_n40_e1 $G --n 40 --eps 1 --np-side 32 --planes 0.5,1 --times 1
run_fwd gauss_n80_e2 $G --n 80 --eps 2 --np-side 32 --planes 0.5,1 --times 1

echo "== recorded: device peak, base build, --track forward 256^3 eps 2 (nvidia-smi 200 ms)"
cd "$B/mem" || exit 2
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" -lms 200 > fwd_N256_e2.mem &
SAMPLER=$!
/usr/bin/time -v -o fwd_N256_e2.time "$B/bin/inlet_slab" --track forward $G --n 256 --eps 2 --np-side 256 --planes 0.25,0.5,1,2,4 --times 0.5,1,2,4 --summary "$B/mem/fwd_N256_e2.json" > fwd_N256_e2.log 2>&1
echo "fwd_N256_e2 rc=$?"
kill "$SAMPLER" 2> /dev/null
echo "BASE_PEAK_MIB_FWD_N256_E2=$(sort -n fwd_N256_e2.mem | tail -n 1)"
echo "== recorded: 512^3 forward feasibility (base build; eps 2, M = 256, planes/times to 16, cap 1000; nvidia-smi 200 ms)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" -lms 200 > fwd_N512_e2.mem &
SAMPLER=$!
/usr/bin/time -v -o fwd_N512_e2.time "$B/bin/inlet_slab" --track forward $G --n 512 --eps 2 --np-side 256 --planes 0.25,0.5,1,2,4,8,16 --times 0.25,0.5,1,2,4,8,16 --max-arclength 1000 --summary "$B/mem/fwd_N512_e2.json" > fwd_N512_e2.log 2>&1
echo "fwd_N512_e2 rc=$?"
kill "$SAMPLER" 2> /dev/null
echo "BASE_PEAK_MIB_FWD_N512_E2=$(sort -n fwd_N512_e2.mem | tail -n 1)"
echo "SF37_BASE_REFS_END"
