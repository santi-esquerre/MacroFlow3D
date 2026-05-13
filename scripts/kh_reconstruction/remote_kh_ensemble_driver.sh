#!/usr/bin/env bash
set -euo pipefail

count="${1:-100}"
phase="${2:-full}"

root="artifacts/kh_reconstruction"
mkdir -p "${root}/config" "${root}/logs" "${root}/raw" "${root}/summary" "${root}/plots"

cmake --preset v100-release
cmake --build build/v100-release -j

write_config() {
  local seed="$1"
  local backend="$2"
  local mode="$3"
  local out_dir="$4"
  local cfg="$5"

  cat > "$cfg" <<EOF
run_mode: single_run

grid:
  nx: 64
  ny: 32
  nz: 32
  dx: 5.0

stochastic:
  sigma2: 1.0
  corr_length: 50.0
  n_modes: 1000
  covariance_type: 0
  seed: ${seed}
  K_mean: 1.0

flow:
  solver: mg_cg
  mg_levels: 4
  mg_pre_smooth: 4
  mg_post_smooth: 4
  mg_coarse_iters: 50
  mg_max_cycles: 20
  cg_max_iter: 200
  cg_rtol: 1.0e-6
  cg_check_every: 10
  rtol: 1.0e-6
  pin:
    mode: off
  bc:
    west:   { type: dirichlet, value: 100.0 }
    east:   { type: dirichlet, value: 0.0 }
    south:  { type: periodic }
    north:  { type: periodic }
    bottom: { type: periodic }
    top:    { type: periodic }

transport:
  method: par2
  velocity_eval_mode: ${mode}
  n_particles: 1000
  dt: 1.0
  n_steps: 500
  porosity: 1.0
  diffusion: 0.0
  alpha_l: 0.0
  alpha_t: 0.0
  seed: 123456789
  output_every: 100
  snapshot_every: 0
  inject_x: 50.0

analysis:
  macrodispersion:
    enabled: true
    NR: 1
    lambda: 50.0
    vmean_norm: 0.3125
    sample_every: 10
    var_estimator: biased
  snapshots:
    enabled: false
    every: 0
    legacy_format: true
    include_time: false
    include_status: false
    include_wrap_counts: false
    include_unwrapped: false
    stride: 1
    max_particles: -1
    precision: 15

diagnostics:
  velocity_field: true

output:
  output_dir: ${out_dir}
  save_K: false
  save_head: false
  save_velocity: false
  save_particles: true
  format: binary
EOF
}

run_one() {
  local seed="$1"
  local backend="$2"
  local mode="$3"
  local seed_tag
  seed_tag="$(printf 'seed_%03d' "$seed")"
  local run_dir="${root}/raw/${seed_tag}/${backend}"
  local cfg="${root}/config/${seed_tag}_${backend}.yaml"
  local log="${run_dir}/log.txt"

  if [[ -e "$run_dir" ]]; then
    local archived="${run_dir}.previous.$(date +%Y%m%d%H%M%S)"
    mv "$run_dir" "$archived"
    echo "[kh] archived previous run_dir=${run_dir} -> ${archived}"
  fi
  mkdir -p "$run_dir"
  write_config "$seed" "$backend" "$mode" "$run_dir" "$cfg"

  echo "[kh] phase=${phase} seed=${seed} backend=${backend} mode=${mode}"
  ./build/v100-release/macroflow3d_pipeline "$cfg" > "$log" 2>&1

  cp "${run_dir}/effective_config.yaml" "${run_dir}/config_used.yaml"
  if [[ -f "${run_dir}/analysis/macrodispersion.csv" ]]; then
    cp "${run_dir}/analysis/macrodispersion.csv" "${run_dir}/alpha_timeseries.csv"
  fi
}

for ((seed = 0; seed < count; ++seed)); do
  run_one "$seed" face FACE_TRILINEAR
  run_one "$seed" kh KH_POTENTIAL_RECONSTRUCTION
done

bash scripts/kh_reconstruction/collect_kh_ensemble_results.sh "$root"
