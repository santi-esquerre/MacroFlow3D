#!/usr/bin/env bash
# Fake-host test of REMOTE_GPU_COUNT / REMOTE_GPU_COUNT_MIN in the `run` launcher
# of scripts/remote. The host is this machine: `ssh` evaluates the remote command
# locally, and stubs on PATH replace nvidia-smi (canned CSV per sampling round),
# tmux (runs the launcher synchronously) and sleep (no-op); flock is real.
# Run: bash scripts/tests/remote_gpu_count_test.sh   (no network, no GPU, no remote host)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export HOME="$tmp/home"; mkdir -p "$HOME" "$tmp/bin" "$tmp/repo" "$tmp/nv"
# bash -lc -> no login profile, so the stubs stay first on PATH inside the launcher and the job
printf '#!/bin/bash\n[ "$1" = -lc ] && shift && set -- -c "$@"\nexec /bin/bash --noprofile --norc "$@"\n' >"$tmp/bin/bash"
# nvidia-smi: one canned csv per sampling round (5 samples); the last file repeats
cat >"$tmp/bin/nvidia-smi" <<'EOF'
#!/bin/bash
case "$*" in
  *query-gpu=index,memory.used*)
    n=$(cat "$NV_DIR/n" 2>/dev/null || echo 0); n=$((n + 1)); echo "$n" >"$NV_DIR/n"
    f="$NV_DIR/csv.$(( (n - 1) / 5 + 1 ))"
    [ -f "$f" ] || f="$(ls "$NV_DIR"/csv.* | sort -V | tail -1)"
    cat "$f" ;;
  *query-gpu=index,uuid*) printf '0, GPU-a\n1, GPU-b\n' ;;
  *) ;;
esac
EOF
printf '#!/bin/bash\nexit 0\n' >"$tmp/bin/sleep"
# tmux: new-session runs the launcher (last argument) synchronously; the rest is "no sessions"
cat >"$tmp/bin/tmux" <<'EOF'
#!/bin/bash
if [ "$1" = new-session ]; then eval "\"${@: -1}\"" || true; fi
exit 0
EOF
chmod +x "$tmp"/bin/*
export NV_DIR="$tmp/nv"
export REMOTE_INCREMENT=T1
export REMOTE_REPO_DIR="$tmp/repo"
export REMOTE_STATE_ROOT="$tmp/state"
export REMOTE_LOCK_ROOT="$tmp/lock"
export REMOTE_SSH_RETRIES=0 REMOTE_SSH_RETRY_DELAY=0
# shellcheck disable=SC1091
source "${REMOTE_SCRIPT:-$here/../remote}"

ssh() { shift; ( PATH="$tmp/bin:$PATH"; eval "$1" ); }

fails=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
expect_eq() { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1: got [$2] want [$3]"; fi; }

LOGD="$REMOTE_STATE_ROOT/logs"; STD="$REMOTE_STATE_ROOT/status"
# csv <round> <line>...   one canned sampling round
csv() { local r="$1"; shift; printf '%s\n' "$@" >"$NV_DIR/csv.$r"; }
reset_nv() { rm -f "$NV_DIR"/*; }
# jlog <job>: job log without the leading UTC timestamps
jlog() { sed -E 's/^\[[0-9TZ:-]+\] //' "$LOGD/$1.log"; }
jstate() { ( set +u; source "$STD/$1.env"; printf '%s %s %s' "$STATE" "$EXIT_CODE" "$GPU" ); }
CMD='echo CVD=$CUDA_VISIBLE_DEVICES GRANTED=${REMOTE_GPU_COUNT_GRANTED-}; ( . "$REMOTE_LOCK_ROOT/gpu.holder"; echo HOLDER_GPU=$HOLDER_GPU ); touch '"$tmp"'/ran'
launch() { # <job> ; uses the environment of the caller subshell
  rm -f "$tmp/ran"
  run_command "$1" auto "$CMD" >/dev/null
}
ONE_BUSY=("0, 1281, 32768, 84" "1, 4, 32768, 0")     # both eligible at min_free 30000, GPU 1 better
ONE_ELIG=("0, 29000, 32768, 0" "1, 4, 32768, 0")     # GPU 0 has 3768 MiB free -> ineligible at 30000
NONE_ELIG=("0, 29000, 32768, 0" "1, 29000, 32768, 0")

# --- 1. count 1 (default): exactly the historical log lines (verified equal to the pre-change script
#        apart from the new GRANTED=1 in the job environment), one id in CVD -----------
reset_nv; csv 1 "${ONE_BUSY[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=1 REMOTE_GPU_COUNT_MIN=1; launch c1 )
want="job=c1 GPU snapshot (5 samples, min_free_required=30000 MiB, request=auto):
  gpu 0: mean_util=84.0% min_free=31487/32768 MiB compute_procs=0
  gpu 1: mean_util=0.0% min_free=32764/32768 MiB compute_procs=0
job=c1 chose GPU 1 (auto: least-loaded)
job=c1 session=macroflow3d-T1-c1 started
GPU=1 (CUDA_VISIBLE_DEVICES=1, request=auto)
cwd=$tmp/repo
command=$CMD
CVD=1 GRANTED=1
HOLDER_GPU=1
job=c1 finished exit=0"
got="$(jlog c1)"
expect_eq count1-golden-log "$got" "$want"
expect_eq count1-status "$(jstate c1)" "succeeded 0 1"
[[ -e "$tmp/ran" ]] && ok count1-ran || bad count1-ran
# default (unset count) behaves as count 1: remote.env defaults
( unset REMOTE_GPU_COUNT REMOTE_GPU_COUNT_MIN; source "$here/../remote.env"; expect_eq env-defaults "$REMOTE_GPU_COUNT/$REMOTE_GPU_COUNT_MIN" "1/1" )

# --- 2. count 2, two eligible: both granted, ranked best first ------------------------
reset_nv; csv 1 "${ONE_BUSY[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch c2 )
got="$(jlog c2)"
grep -qxF 'job=c2 chose GPUs 1,0 (auto: least-loaded, 2 of 2; first=GPU 1 min_free>=30000 MiB, second=GPU 0 min_free>=30000 MiB)' <<<"$got" && ok count2-chose-line || bad "count2-chose-line: $got"
grep -qx 'CVD=1,0 GRANTED=2' <<<"$got" && ok count2-cvd-granted || bad "count2-cvd-granted: $got"
grep -qx 'HOLDER_GPU=1,0' <<<"$got" && ok count2-holder || bad "count2-holder: $got"
grep -q 'GPU=1,0 (CUDA_VISIBLE_DEVICES=1,0, request=auto, gpu_count=2 granted=2)' <<<"$got" && ok count2-gpu-line || bad "count2-gpu-line"
grep -q 'degraded' <<<"$got" && bad count2-not-degraded || ok count2-not-degraded
expect_eq count2-status "$(jstate c2)" "succeeded 0 1,0"

# --- 2b. two thresholds: first meets 30000, second only the lower REMOTE_GPU_MIN_FREE_MIB_2 ---
LOWER2=("0, 100, 32768, 50" "1, 15000, 32768, 10")   # GPU0 free 32668 (high), GPU1 free 17768 (low only)
reset_nv; csv 1 "${LOWER2[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=17000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch t2 )
got="$(jlog t2)"
grep -qxF 'job=t2 chose GPUs 0,1 (auto: least-loaded, 2 of 2; first=GPU 0 min_free>=30000 MiB, second=GPU 1 min_free>=17000 MiB)' <<<"$got" \
  && grep -qx 'CVD=0,1 GRANTED=2' <<<"$got" && grep -qx 'HOLDER_GPU=0,1' <<<"$got" && ok two-thresholds-both-granted-in-order || bad "two-thresholds-both-granted-in-order: $got"
# without _2 the same host state is degraded (second needs 30000 too)
reset_nv; csv 1 "${LOWER2[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch t2n )
grep -qx 'CVD=0 GRANTED=1' <<<"$(jlog t2n)" && ok two-thresholds-default-second-equals-first || bad two-thresholds-default-second-equals-first
# second below both thresholds (free 2768): MIN=1 degrades to the first only, MIN=2 fails 75
BELOW=("0, 100, 32768, 50" "1, 30000, 32768, 10")
reset_nv; csv 1 "${BELOW[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=17000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch t2b )
got="$(jlog t2b)"
grep -q 'GPU_COUNT degraded: granted 1 of 2 (REMOTE_GPU_COUNT_MIN=1); chose GPU 0 (auto: least-loaded; min_free>=30000 MiB; second GPU needed min_free>=17000 MiB, none eligible)' <<<"$got" \
  && grep -qx 'CVD=0 GRANTED=1' <<<"$got" && ok second-below-both-degraded || bad "second-below-both-degraded: $got"
reset_nv; csv 1 "${BELOW[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=17000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=2; launch t2c )
expect_eq second-below-both-strict-75 "$(jstate t2c)" "failed 75 "
[[ ! -e "$tmp/ran" ]] && ok second-below-both-strict-not-run || bad second-below-both-strict-not-run
# the first must meet the HIGH threshold: only the low one met by both -> 75 even with MIN=1
reset_nv; csv 1 "0, 20000, 32768, 0" "1, 29000, 32768, 0"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_MIN_FREE_MIB_2=3000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch t2d )
expect_eq first-must-meet-high-75 "$(jstate t2d)" "failed 75 "

# --- 3. count 2, second GPU appears while waiting (round 2): waits, then takes both ---
reset_nv; csv 1 "${ONE_ELIG[@]}"; csv 2 "${ONE_BUSY[@]}"
( export REMOTE_GPU_WAIT=600 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch c2w )
got="$(jlog c2w)"
grep -q 'only 1 of 2 requested GPUs eligible' <<<"$got" && grep -qx 'CVD=1,0 GRANTED=2' <<<"$got" && ! grep -q degraded <<<"$got" \
  && ok count2-waits-then-both || bad "count2-waits-then-both: $got"

# --- 4. count 2, one eligible, MIN=1, WAIT=0: degraded, one id, GRANTED=1 -------------
reset_nv; csv 1 "${ONE_ELIG[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch c2d )
got="$(jlog c2d)"
grep -q 'GPU_COUNT degraded: granted 1 of 2 (REMOTE_GPU_COUNT_MIN=1)' <<<"$got" && ok degraded-line || bad "degraded-line: $got"
grep -qx 'CVD=1 GRANTED=1' <<<"$got" && ok degraded-cvd-granted || bad "degraded-cvd-granted: $got"
expect_eq degraded-status "$(jstate c2d)" "succeeded 0 1"
# at a nonzero deadline the degradation happens only after the waiting (sleep is a no-op: the loop spins 2 s)
reset_nv; csv 1 "${ONE_ELIG[@]}"
( export REMOTE_GPU_WAIT=2 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; launch c2dw )
got="$(jlog c2dw)"
grep -q 'only 1 of 2 requested GPUs eligible' <<<"$got" && grep -q 'GPU_COUNT degraded' <<<"$got" && grep -qx 'CVD=1 GRANTED=1' <<<"$got" \
  && ok degraded-after-wait || bad "degraded-after-wait: $got"

# --- 5. count 2 strict (MIN=2), one eligible: failed 75, command not run --------------
reset_nv; csv 1 "${ONE_ELIG[@]}"
( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=2; launch c2s )
expect_eq strict-status "$(jstate c2s)" "failed 75 "
[[ ! -e "$tmp/ran" ]] && ok strict-command-not-run || bad strict-command-not-run
grep -q 'ERROR: no eligible GPU within REMOTE_GPU_WAIT=0' "$LOGD/c2s.log" && ok strict-error-line || bad strict-error-line

# --- 6. none eligible: failed 75, command not run (count 1 and count 2) ---------------
for cnt in 1 2; do
  reset_nv; csv 1 "${NONE_ELIG[@]}"
  ( export REMOTE_GPU_WAIT=0 REMOTE_GPU_MIN_FREE_MIB=30000 REMOTE_GPU_COUNT=$cnt REMOTE_GPU_COUNT_MIN=1; launch none$cnt )
  expect_eq "none-eligible-status[count=$cnt]" "$(jstate none$cnt)" "failed 75 "
  [[ ! -e "$tmp/ran" ]] && ok "none-eligible-not-run[count=$cnt]" || bad "none-eligible-not-run[count=$cnt]"
done

# --- 7. refusals happen before any ssh ------------------------------------------------
rec="$tmp/ssh.rec"; : >"$rec"
ssh() { shift; printf '%s\n---\n' "$1" >>"$rec"; }
refuse() { # <name> <count> <min> <gpu_req>
  local rc=0
  ( export REMOTE_GPU_COUNT="$2" REMOTE_GPU_COUNT_MIN="$3"; run_command refuse "$4" true ) >/dev/null 2>&1 || rc=$?
  [[ $rc -ne 0 ]] && ok "refused[$1]" || bad "refused[$1] (rc=0)"
}
refuse pin-with-count2 2 1 0
refuse count3 3 1 auto
refuse count0 0 0 auto
refuse count-text two 1 auto
refuse min0 2 0 auto
refuse min-above-count 1 2 auto
refuse min3 2 3 auto
( export REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1 REMOTE_GPU_MIN_FREE_MIB_2=abc; run_command refuse auto true ) >/dev/null 2>&1 \
  && bad refused-min-free-2-text || ok refused-min-free-2-text
expect_eq refusals-no-ssh "$(wc -c <"$rec")" "0"
# a pinned single-GPU request is still accepted (its ssh validation is the old one)
rc=0; ( export REMOTE_GPU_COUNT=1 REMOTE_GPU_COUNT_MIN=1; run_command pinned 0 true ) >/dev/null 2>&1 || rc=$?
[[ -s "$rec" ]] && ok pin-count1-reaches-ssh || bad pin-count1-reaches-ssh

# --- 8. blinded log: gpu_count only when != 1 -----------------------------------------
: >"$rec"
( export REMOTE_GPU_COUNT=1 REMOTE_GPU_COUNT_MIN=1; run_command b1 auto true ) >/dev/null
! grep -q 'gpu_count' "$rec" && grep -q 'kind=run' "$rec" && ok blinded-detail-count1-unchanged || bad blinded-detail-count1-unchanged
: >"$rec"
( export REMOTE_GPU_COUNT=2 REMOTE_GPU_COUNT_MIN=1; run_command b2 auto true ) >/dev/null
grep -q 'gpu=auto.*gpu_count=2.*command=' "$rec" && ok blinded-detail-count2 || bad blinded-detail-count2

(( fails == 0 )) && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
