#!/usr/bin/env bash
# Deterministic test of gpu_select (scripts/remote) on canned nvidia-smi CSV.
# Run: bash scripts/tests/remote_gpu_select_test.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/../remote"   # main is not run when sourced

fails=0
# check <name> <min_free> <req> <expected_rc> <expected_CHOSEN_line> [count] [min_free_2] <<< csv
check() {
  local name="$1" minfree="$2" req="$3" want_rc="$4" want="$5" count="${6-}" minfree2="${7-}" procs="${8-}" allfree="${9-0}" out rc=0 got
  if [[ -n "$procs" ]]; then out="$(gpu_select "$minfree" "$req" "$count" "${minfree2:-$minfree}" "$procs" "$allfree")" || rc=$?
  elif [[ -n "$minfree2" ]]; then out="$(gpu_select "$minfree" "$req" "$count" "$minfree2")" || rc=$?
  elif [[ -n "$count" ]]; then out="$(gpu_select "$minfree" "$req" "$count")" || rc=$?
  else out="$(gpu_select "$minfree" "$req")" || rc=$?; fi
  got="$(grep -E '^(CHOSEN|ERROR)' <<<"$out" || true)"
  if [[ "$rc" == "$want_rc" && "$got" == "$want" ]]; then
    echo "PASS $name (rc=$rc, $got)"
  else
    echo "FAIL $name: rc=$rc want_rc=$want_rc got='$got' want='$want'"; echo "$out" | sed 's/^/    /'
    fails=$((fails + 1))
  fi
}
rep5() { for _ in 1 2 3 4 5; do printf '%s\n' "$@"; done; }

# live case: GPU0 colleague 84 %, 1281 MiB; GPU1 idle, 4 MiB
check live-case 0 auto 0 "CHOSEN 1" < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# equal utilization: more free memory wins (GPU1 has less used)
check equal-util-more-free 0 auto 0 "CHOSEN 1" < <(rep5 "0, 9000, 32768, 10" "1, 4000, 32768, 10")
# full tie: lower index
check full-tie-lower-index 0 auto 0 "CHOSEN 0" < <(rep5 "0, 100, 32768, 10" "1, 100, 32768, 10")
# lowest utilization but memory-ineligible (free 3768 < 10000): skipped
check mem-ineligible-skipped 10000 auto 0 "CHOSEN 1" < <(rep5 "0, 29000, 32768, 0" "1, 2000, 32768, 90")
# min free is over samples: one bad sample makes GPU0 ineligible
check min-free-over-samples 20000 auto 0 "CHOSEN 1" < <(rep5 "0, 1000, 32768, 0" "1, 1000, 32768, 50"; echo "0, 30000, 32768, 0")
# none eligible
check none-eligible 40000 auto 1 "CHOSEN none" < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# pin: honoured even if busier, threshold respected
check pin-ok 0 0 0 "CHOSEN 0" < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
check pin-below-threshold 10000 0 1 "CHOSEN none" < <(rep5 "0, 29000, 32768, 0" "1, 4, 32768, 0")
check pin-unlisted 0 7 2 "ERROR pinned GPU 7 not listed by nvidia-smi" < <(rep5 "0, 1, 32768, 0" "1, 1, 32768, 0")
# unparseable / empty -> rc 2, never a pick
check garbage 0 auto 2 "ERROR unparseable or empty nvidia-smi output" < <(echo "No devices were found")
check na-field 0 auto 2 "ERROR unparseable or empty nvidia-smi output" < <(rep5 "0, 1, 32768, [N/A]" "1, 1, 32768, 0")
check empty 0 auto 2 "ERROR unparseable or empty nvidia-smi output" < /dev/null

# count 2: best two ranked by utilization (GPU1 idle first), rc 0
check c2-two-eligible 0 auto 0 "CHOSEN 1,0" 2 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# count 2 tie rules: equal util -> more free memory first; full tie -> lower index first
check c2-equal-util-more-free 0 auto 0 "CHOSEN 1,0" 2 < <(rep5 "0, 9000, 32768, 10" "1, 4000, 32768, 10")
check c2-full-tie 0 auto 0 "CHOSEN 0,1" 2 < <(rep5 "0, 100, 32768, 10" "1, 100, 32768, 10")
# count 2 with 3 GPUs: the two least loaded, best first
check c2-three-gpus 0 auto 0 "CHOSEN 2,0" 2 < <(rep5 "0, 100, 32768, 20" "1, 100, 32768, 90" "2, 100, 32768, 5")
# count 2, one eligible: names it, rc 3 (launcher applies REMOTE_GPU_COUNT_MIN)
check c2-one-eligible 10000 auto 3 "CHOSEN 1" 2 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0"; echo "0, 31000, 32768, 0")
# count 2, second GPU memory-ineligible (free 3768 < 10000): rc 3, only GPU1
check c2-second-mem-ineligible 10000 auto 3 "CHOSEN 1" 2 < <(rep5 "0, 29000, 32768, 0" "1, 2000, 32768, 90")
# count 2, none eligible: rc 1
check c2-none-eligible 40000 auto 1 "CHOSEN none" 2 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# count 1 given explicitly equals the default
check c1-explicit 0 auto 0 "CHOSEN 1" 1 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")

# two thresholds (count 2): first needs free >= 30000, second >= 17000
# first meets 30000, second meets only the lower threshold: both, first then second
check t2-first-high-second-low 30000 auto 0 "CHOSEN 0,1" 2 17000 < <(rep5 "0, 100, 32768, 50" "1, 15000, 32768, 10")
# the first is the BEST GPU meeting the high threshold, not the one with most memory: GPU1 (free 32668, util 10) first
check t2-first-best-high 30000 auto 0 "CHOSEN 1,0" 2 17000 < <(rep5 "0, 14000, 32768, 0" "1, 100, 32768, 10")
# only GPU1 meets the high threshold: first = GPU1; GPU0 (free 18768) meets the low one -> second
check t2-low-ok 30000 auto 0 "CHOSEN 1,0" 2 17000 < <(rep5 "0, 14000, 32768, 0" "1, 100, 32768, 90")
# second below both thresholds (free 2768): rc 3 with the first only
check t2-second-below-both 30000 auto 3 "CHOSEN 0" 2 17000 < <(rep5 "0, 100, 32768, 50" "1, 30000, 32768, 0")
# nobody meets the high threshold, one meets the low: none (the first must meet the high one)
check t2-none-high 30000 auto 1 "CHOSEN none" 2 17000 < <(rep5 "0, 20000, 32768, 0" "1, 29000, 32768, 0")
# second threshold above the first: a GPU meeting only the first threshold cannot be second
check t2-second-stricter 10000 auto 3 "CHOSEN 0" 2 20000 < <(rep5 "0, 100, 32768, 0" "1, 20000, 32768, 0")
# min_free_2 is ignored for count 1
check t1-ignores-second 30000 auto 0 "CHOSEN 0" 1 17000 < <(rep5 "0, 100, 32768, 50" "1, 10000, 32768, 0")

# sharing policy (owner decision of 2026-11-02): args ... <count> <min_free_2> <procs> <allfree>
# count 1: the free GPU0 (util 84) beats the less-utilized busy GPU1
check p-count1-free-first 0 auto 0 "CHOSEN 0" 1 "" "1=1" < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# count 1, none free: old order (GPU1 least loaded)
check p-count1-none-free 0 auto 0 "CHOSEN 1" 1 "" "0=2 1=1" < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# count 2 with allfree=0 (COUNT_MIN=2): both, free first
check p-c2-nofree-flag 0 auto 0 "CHOSEN 0,1" 2 "" "1=1" 0 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# count 2 with allfree=1, both free: both
check p-c2-allfree-both 0 auto 0 "CHOSEN 1,0" 2 "" "0=0" 1 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# count 2 with allfree=1, GPU0 busy: GPU1 only, rc 3, BUSY line names GPU 0
check p-c2-allfree-degrades 0 auto 3 "CHOSEN 1" 2 "" "0=3" 1 < <(rep5 "0, 4, 32768, 0" "1, 1281, 32768, 84")
# count 2 with allfree=1, none free: single least-loaded
check p-c2-allfree-none-free 0 auto 3 "CHOSEN 1" 2 "" "0=1 1=1" 1 < <(rep5 "0, 1281, 32768, 84" "1, 4, 32768, 0")
# a busy but memory-ineligible GPU does not degrade (normal rc 3 path), nor does a busy ineligible one block a free pair
check p-c2-busy-ineligible 10000 auto 3 "CHOSEN 1" 2 "" "0=1" 1 < <(rep5 "0, 29000, 32768, 0" "1, 2000, 32768, 90")
out="$(gpu_select 0 auto 2 0 "0=3" 1 < <(rep5 "0, 4, 32768, 0" "1, 1281, 32768, 84") || true)"
grep -qx 'BUSY 0 3' <<<"$out" && echo "PASS busy-line" || { echo "FAIL busy-line: $out"; fails=$((fails + 1)); }

(( fails == 0 )) && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
