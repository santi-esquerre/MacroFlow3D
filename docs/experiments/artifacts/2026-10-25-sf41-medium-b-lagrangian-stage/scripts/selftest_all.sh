#!/usr/bin/env bash
# Runs the --selftest of every verdict script and of the copied SF-38 tools (synthetic inputs; seconds; python3 + numpy; no GPU, no network).
#   bash selftest_all.sh        exit 0 iff every self-test prints SELFTEST OK
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
PY=${PYTHON:-python3}
bad=0
for s in a_verdict b_verdict c_verdict d_verdict e_verdict f_verdict g_compare h_verdict pid_peak mem_verdict d_checks ab_decision; do
    out=$("$PY" -I "$s.py" --selftest 2>&1 | tail -n 3)
    echo "$s: $out"
    echo "$out" | grep -q "SELFTEST OK" || bad=$((bad + 1))
done
echo "SELFTEST_ALL failing_scripts=$bad"
[ "$bad" -eq 0 ]
