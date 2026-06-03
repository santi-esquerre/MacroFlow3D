#!/usr/bin/env bash
set -euo pipefail

count="${1:-10}"
phase="${2:-gaussian_smooth}"

root="artifacts/kh_higher_order"
mkdir -p "${root}/config" "${root}/logs" "${root}/raw" "${root}/summary" "${root}/plots"

grid_nx=64
grid_ny=32
grid_nz=32
dx=5.0
sigma2=1.0
corr_length=50.0
n_modes=1000
covariance_type=1
flow_solver="mg_cg"
west_bc=100.0
east_bc=0.0
n_particles=1000
dt=1.0
n_steps=500
output_every=100
sample_every=10
vmean_norm=0.3125
macro_lambda=50.0

case "${phase}" in
  smoke)
    covariance_type=1
    n_steps=5
    output_every=1
    sample_every=1
    ;;
  gaussian_smooth)
    covariance_type=1
    ;;
  dreuzy_gaussian_reduced)
    covariance_type=1
    ;;
  exponential_previous)
    covariance_type=0
    ;;
  *)
    echo "[kh_higher_order] unknown phase: ${phase}" >&2
    exit 2
    ;;
esac

backend_names=(face kh_linear kh_cubic kh_logk_cubic)
backend_modes=(
  FACE_TRILINEAR
  KH_LINEAR
  KH_CUBIC_POTENTIAL_RECONSTRUCTION
  KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION
)

if [[ "${KH_SKIP_BUILD:-0}" != "1" ]]; then
  cmake --preset v100-release
  cmake --build build/v100-release --target macroflow3d_pipeline kh_potential_reconstruction_tests -j
  ctest --test-dir build/v100-release --output-on-failure -R kh_potential_reconstruction_tests
fi

write_config() {
  local seed="$1"
  local backend="$2"
  local mode="$3"
  local out_dir="$4"
  local cfg="$5"

  cat > "$cfg" <<EOF
run_mode: single_run

grid:
  nx: ${grid_nx}
  ny: ${grid_ny}
  nz: ${grid_nz}
  dx: ${dx}

stochastic:
  sigma2: ${sigma2}
  corr_length: ${corr_length}
  n_modes: ${n_modes}
  covariance_type: ${covariance_type}
  seed: ${seed}
  K_mean: 1.0

flow:
  solver: ${flow_solver}
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
    west:   { type: dirichlet, value: ${west_bc} }
    east:   { type: dirichlet, value: ${east_bc} }
    south:  { type: periodic }
    north:  { type: periodic }
    bottom: { type: periodic }
    top:    { type: periodic }

transport:
  method: par2
  velocity_eval_mode: ${mode}
  n_particles: ${n_particles}
  dt: ${dt}
  n_steps: ${n_steps}
  porosity: 1.0
  diffusion: 0.0
  alpha_l: 0.0
  alpha_t: 0.0
  seed: 123456789
  output_every: ${output_every}
  snapshot_every: 0
  inject_x: 50.0

analysis:
  macrodispersion:
    enabled: true
    NR: 1
    lambda: ${macro_lambda}
    vmean_norm: ${vmean_norm}
    sample_every: ${sample_every}
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
  local run_dir="${root}/raw/${phase}/${seed_tag}/${backend}"
  local cfg="${root}/config/${phase}_${seed_tag}_${backend}.yaml"
  local log="${run_dir}/log.txt"

  if [[ -e "$run_dir" ]]; then
    local archived="${run_dir}.previous.$(date +%Y%m%d%H%M%S)"
    mv "$run_dir" "$archived"
    echo "[kh_higher_order] archived ${run_dir} -> ${archived}"
  fi

  mkdir -p "$run_dir"
  write_config "$seed" "$backend" "$mode" "$run_dir" "$cfg"

  echo "[kh_higher_order] phase=${phase} seed=${seed} backend=${backend} mode=${mode}"
  ./build/v100-release/macroflow3d_pipeline "$cfg" > "$log" 2>&1

  cp "${run_dir}/effective_config.yaml" "${run_dir}/config_used.yaml"
  if [[ -f "${run_dir}/analysis/macrodispersion.csv" ]]; then
    cp "${run_dir}/analysis/macrodispersion.csv" "${run_dir}/alpha_timeseries.csv"
  fi
}

for ((seed = 0; seed < count; ++seed)); do
  for idx in "${!backend_names[@]}"; do
    run_one "$seed" "${backend_names[$idx]}" "${backend_modes[$idx]}"
  done
done

if [[ -x scripts/kh_higher_order/collect_kh_higher_order_results.sh ]]; then
  bash scripts/kh_higher_order/collect_kh_higher_order_results.sh "$root"
fi
