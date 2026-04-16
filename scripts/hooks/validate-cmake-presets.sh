#!/usr/bin/env bash
# Validate CMakePresets.json is valid JSON and has required fields.
set -euo pipefail

PRESETS_FILE="CMakePresets.json"

if [[ ! -f "$PRESETS_FILE" ]]; then
    echo "ERROR: $PRESETS_FILE not found." >&2
    exit 1
fi

# Check valid JSON
if ! python3 -c "import json, sys; json.load(open(sys.argv[1]))" "$PRESETS_FILE" 2>/dev/null; then
    echo "ERROR: $PRESETS_FILE is not valid JSON." >&2
    exit 1
fi

# Check required structure
python3 -c "
import json, sys
data = json.load(open(sys.argv[1]))
assert 'version' in data, 'Missing version field'
assert 'configurePresets' in data, 'Missing configurePresets'
presets = {p['name'] for p in data['configurePresets']}
required = {'wsl-debug', 'wsl-release', 'v100-release', 'v100-petsc'}
missing = required - presets
if missing:
    print(f'ERROR: Missing required presets: {missing}', file=sys.stderr)
    sys.exit(1)
" "$PRESETS_FILE"

echo "CMakePresets.json: OK"
