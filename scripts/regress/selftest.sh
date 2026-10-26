#!/usr/bin/env bash
# Self-test of scripts/regress (no GPU, no binaries, seconds): the python helpers and compare.sh on synthetic output trees.
#   bash scripts/regress/selftest.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
FAIL=0

expect() { # expect <name> <condition-rc>
    if [ "$2" -eq 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAIL=$((FAIL + 1)); fi
}

mk() { # mk <dir>: a minimal run.sh-shaped output
    local d=$1
    mkdir -p "$d/pipe/a" "$d/r/fwd/x" "$d/log" "$d/g_ab"
    printf 'pipe_a\t0\nr_fwd_x\t0\ng_ab\t0\n' > "$d/exit_codes.tsv"
    printf 'abc v1\n' > "$d/pipe/a/out.bin"
    printf '{"timestamp":"T1","build":{"git_hash":"h1"},"k":1.5}\n' > "$d/pipe/a/manifest.json"
    printf '{"t":1}\n' > "$d/pipe/a/timings.json"
    printf '1,2,3\n' > "$d/r/fwd/x/p.csv"
    printf 'STATUS ok t=1.0s\nnoise line\n' > "$d/log/r_fwd_x.log"
    printf 'data\n' > "$d/g_ab/f.txt"
    echo "REGRESS_DONE unexpected_rc=0" > "$d/REGRESS_DONE"
    echo "deadbeef  macroflow3d_pipeline" > "$d/binaries.md5"
}

cmpr() { bash "$HERE/compare.sh" "$@" 2>&1; }
last() { tail -n 1; }

# --- python helpers ---
printf '{"a":{"timestamp":"x","v":1.0},"l":[1,2]}' > "$W/a.json"
printf '{"a":{"timestamp":"y","v":1.0},"l":[1,2]}' > "$W/b.json"
printf '{"a":{"timestamp":"y","v":1.0000000001},"l":[1,2]}' > "$W/c.json"
"$PY" "$HERE/json_strip_cmp.py" "$W/a.json" "$W/b.json" --drop a.timestamp > /dev/null
expect "json_strip_cmp: dropped key ignored" $?
"$PY" "$HERE/json_strip_cmp.py" "$W/a.json" "$W/b.json" > /dev/null
expect "json_strip_cmp: undropped key differs" $(($? == 1 ? 0 : 1))
"$PY" "$HERE/json_strip_cmp.py" "$W/b.json" "$W/c.json" --drop a.timestamp > /dev/null
expect "json_strip_cmp: floats compared exactly" $(($? == 1 ? 0 : 1))
printf 'STATUS ok t=1.0s\nx\n' > "$W/l1"
printf 'STATUS ok t=9.5s\ny\n' > "$W/l2"
"$PY" "$HERE/lines_cmp.py" "$W/l1" "$W/l2" STATUS > /dev/null
expect "lines_cmp: timing masked, other lines unselected" $?
printf 'STATUS bad t=9.5s\n' > "$W/l3"
"$PY" "$HERE/lines_cmp.py" "$W/l1" "$W/l3" STATUS > /dev/null
expect "lines_cmp: content difference" $(($? == 1 ? 0 : 1))

# --- compare.sh on synthetic trees ---
mk "$W/ref"
mk "$W/head"
printf '{"timestamp":"T2","build":{"git_hash":"h2"},"k":1.5}\n' > "$W/head/pipe/a/manifest.json"
printf '{"t":2}\n' > "$W/head/pipe/a/timings.json"
printf 'STATUS ok t=7.7s\nother noise\n' > "$W/head/log/r_fwd_x.log"
out=$(cmpr "$W/ref" "$W/head")
[ "$(echo "$out" | last)" = "REGRESS IDENTICAL" ]
expect "identical (manifest timestamp/hash, timings.json, log timing differ)" $?

mk "$W/h2"
printf 'abc v1\n' | sed 's/c/C/' > "$W/h2/pipe/a/out.bin"
out=$(cmpr "$W/ref" "$W/h2")
[ "$(echo "$out" | last)" = "REGRESS DIFF n=1" ]
expect "one changed byte -> DIFF n=1" $?

mk "$W/h3"
printf 'abc v2\n' > "$W/h3/pipe/a/out.bin"
out=$(cmpr "$W/ref" "$W/h3")
[ "$(echo "$out" | last)" = "REGRESS DIFF n=1" ]
expect "expected diff, no filter -> DIFF" $?
out=$(cmpr "$W/ref" "$W/h3" --ref-filter 'pipe/*/out.bin=s/v[0-9]/vN/')
[ "$(echo "$out" | last)" = "REGRESS IDENTICAL" ]
expect "expected diff declared by --ref-filter -> IDENTICAL" $?

mk "$W/h4"
rm -rf "$W/h4/g_ab"
out=$(cmpr "$W/ref" "$W/h4")
[ "$(echo "$out" | last | cut -c1-12)" = "REGRESS DIFF" ]
expect "head case listed but output missing -> DIFF" $?

mk "$W/h5"
echo "other" > "$W/h5/g_ab/extra.txt"
out=$(cmpr "$W/ref" "$W/h5")
[ "$(echo "$out" | last | cut -c1-12)" = "REGRESS DIFF" ]
expect "extra head file -> DIFF" $?

mk "$W/h6"
printf 'pipe_a\t0\nr_fwd_x\t0\ng_ab\t0\npipe_new\t0\n' > "$W/h6/exit_codes.tsv"
out=$(cmpr "$W/ref" "$W/h6")
[ "$(echo "$out" | last | cut -c1-12)" = "REGRESS DIFF" ]
expect "case absent in ref -> DIFF" $?

mk "$W/h7"
printf 'pipe_a\t0\nr_fwd_x\t0\ng_ab\t3\n' > "$W/h7/exit_codes.tsv"
out=$(cmpr "$W/ref" "$W/h7")
[ "$(echo "$out" | last)" = "REGRESS DIFF n=1" ]
expect "exit code differs -> DIFF" $?

mk "$W/h8"
mk "$W/ref2"
printf 'pipe_a\t0\nr_fwd_x\t0\ng_ab\t0\ng_only_ref\t0\n' > "$W/ref2/exit_codes.tsv"
out=$(cmpr "$W/ref2" "$W/h8")
[ "$(echo "$out" | last)" = "REGRESS IDENTICAL" ] && echo "$out" | grep -q '^NOT_COMPARED g_only_ref'
expect "case only in ref -> NOT_COMPARED, IDENTICAL" $?

mk "$W/h9"
rm "$W/h9/REGRESS_DONE"
out=$(cmpr "$W/ref" "$W/h9")
[ "$(echo "$out" | last | cut -c1-12)" = "REGRESS DIFF" ]
expect "head without REGRESS_DONE -> DIFF" $?

[ -x "$HERE/run.sh" ] || chmod +x "$HERE/run.sh" 2> /dev/null
bash "$HERE/run.sh" --list /nonexistent_bin_dir "$W/none" > /dev/null 2>&1
expect "run.sh refuses a missing bin_dir" $(($? == 2 ? 0 : 1))

if [ "$FAIL" -eq 0 ]; then echo "SELFTEST PASS"; else echo "SELFTEST FAIL n=$FAIL"; fi
[ "$FAIL" -eq 0 ]
