#!/usr/bin/env bash
set -euo pipefail

# Run a pipeline config on the remote V100 server.
#
# Example:
#   scripts/remote_run_pipeline.sh apps/config_pspta_small.yaml
#   BUILD_DIR=build/v100-release scripts/remote_run_pipeline.sh apps/config_pipeline_pspta.yaml

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <config-path-relative-to-repo>" >&2
  exit 1
fi

CONFIG_PATH="$1"
REMOTE_HOST="${REMOTE_HOST:-v100}"
REMOTE_DIR="${REMOTE_DIR:-~/MacroFlow3D}"
BUILD_DIR="${BUILD_DIR:-build/v100-release}"

ssh "$REMOTE_HOST" "
  set -euo pipefail
  cd $REMOTE_DIR
  ./$BUILD_DIR/macroflow3d_pipeline $CONFIG_PATH
"
