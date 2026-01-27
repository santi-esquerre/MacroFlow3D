#!/bin/bash
set -euo pipefail

rm -rf ./output/* 2>/dev/null || true
mkdir -p ./output/log

# --- Cola GPU 1 ---
commands_gpu1=(
  "./run_flow2.out 1 1 > ./output/log/var_1.log 2>&1"
  "./run_flow2.out 1 4 > ./output/log/var_4.log 2>&1"
)

# --- Cola GPU 0 ---
commands_gpu0=(
  "./run_flow2.out 0 0.25 > ./output/log/var_0.25.log 2>&1"
  "./run_flow2.out 0 2.25 > ./output/log/var_2.25.log 2>&1"
  "./run_flow2.out 0 6.25 > ./output/log/var_6.25.log 2>&1"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_queue_gpu1() {
  for cmd in "${commands_gpu1[@]}"; do
    echo "[$(ts)] [GPU1] Ejecutando: $cmd"
    eval "$cmd"
    echo "[$(ts)] [GPU1] Finalizado: $cmd"
  done
  echo "[$(ts)] [GPU1] Cola completada."
}

run_queue_gpu0() {
  for cmd in "${commands_gpu0[@]}"; do
    echo "[$(ts)] [GPU0] Ejecutando: $cmd"
    eval "$cmd"
    echo "[$(ts)] [GPU0] Finalizado: $cmd"
  done
  echo "[$(ts)] [GPU0] Cola completada."
}

# Ejecutar ambas colas en paralelo
run_queue_gpu1 &
pid1=$!
run_queue_gpu0 &
pid0=$!

wait $pid1
wait $pid0

echo "[$(ts)] Todas las colas finalizaron."
