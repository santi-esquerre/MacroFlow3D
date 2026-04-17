#!/usr/bin/env bash
set -euo pipefail

# Compatibility shim. Prefer `scripts/remote sync`.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/remote" sync "$@"
