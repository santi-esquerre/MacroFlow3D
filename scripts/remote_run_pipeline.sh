#!/usr/bin/env bash
set -euo pipefail

# Run a pipeline config on the remote V100 server.
#
# Compatibility shim. Prefer `scripts/remote run <job> -- "<command>"`.
#
# Example:
#   scripts/remote_run_pipeline.sh apps/config_pspta_small.yaml
#   BUILD_DIR=build/v100-release scripts/remote_run_pipeline.sh apps/config_pipeline_pspta.yaml

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config-path-relative-to-repo>" >&2
  exit 1
fi

CONFIG_PATH="$1"
BUILD_DIR="${BUILD_DIR:-build/v100-release}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$script_dir/remote" exec -- "./$BUILD_DIR/macroflow3d_pipeline $CONFIG_PATH"
