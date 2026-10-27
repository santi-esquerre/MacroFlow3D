#!/usr/bin/env bash
# Deterministic test of gpu_select (scripts/remote) on canned nvidia-smi CSV.
# Run: bash scripts/tests/remote_gpu_select_test.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$here/../remote"   # main is not run when sourced

fails=0
# check <name> <min_free> <req> <expected_rc> <expected_CHOSEN_line> <<< csv
check() {
  local name="$1" minfree="$2" req="$3" want_rc="$4" want="$5" out rc=0 got
  out="$(gpu_select "$minfree" "$req")" || rc=$?
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

(( fails == 0 )) && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
