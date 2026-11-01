#!/usr/bin/env bash
# Test of the blinded-phase exec logging of scripts/remote against a fake host:
# `ssh` is replaced by a function that evaluates the remote command locally
# (bash -lc -> bash --noprofile --norc -c) with the host paths in a temp dir.
# Run: bash scripts/tests/remote_blinded_test.sh   (no network, no remote host)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

export HOME="$tmp/home"; mkdir -p "$HOME" "$tmp/bin" "$tmp/repo"
printf '#!/bin/bash\n[ "$1" = -lc ] && shift && set -- -c "$@"\nexec /bin/bash --noprofile --norc "$@"\n' >"$tmp/bin/bash"
chmod +x "$tmp/bin/bash"
export REMOTE_INCREMENT=T1
export REMOTE_REPO_DIR="$tmp/repo"
export REMOTE_STATE_ROOT="$tmp/state"
export REMOTE_SSH_RETRIES=0 REMOTE_SSH_RETRY_DELAY=0
# shellcheck disable=SC1091
source "$here/../remote"

# fake host: evaluate "bash -lc <quoted>" with the wrapper above
ssh() { shift; ( PATH="$tmp/bin:$PATH"; eval "$1" ); }

fails=0
ok() { echo "PASS $1"; }
bad() { echo "FAIL $1"; fails=$((fails + 1)); }
expect_eq() { if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1: got [$2] want [$3]"; fi; }

LOG="$REMOTE_STATE_ROOT/logs/exec-blinded.log"
MARK="$REMOTE_STATE_ROOT/blinded.marker"

# --- 1. no marker: command runs, nothing is logged, output unchanged ---------
out="$(exec_command 'echo "a b" && pwd')"
expect_eq no-marker-output "$out" "a b
$tmp/repo"
[[ ! -e "$LOG" ]] && ok no-marker-no-log || bad no-marker-no-log

# --- 2. blinded start / status ----------------------------------------------
blinded_command start --label "pilot 1" >/dev/null
[[ -e "$MARK" ]] && ok marker-created || bad marker-created
grep -q '^BLINDED_START=' "$MARK" && grep -q '^LABEL=.*pilot' "$MARK" && ok marker-content || bad marker-content
st="$(blinded_command status)"
grep -q 'blinded: active' <<<"$st" && ok status-active || bad status-active
(blinded_command start) >/dev/null 2>&1 && bad double-start-refused || ok double-start-refused

# --- 3. commands with special syntax: output identical, logged exactly --------
mkdir -p "$tmp/repo"
cmds=(
  'echo "it'"'"'s" '"'"'q u"o'"'"'te'
  'echo one && echo two; echo three'
  'echo "$HOME" '"'"'$HOME'"'"' ${HOME:-x}'
  $'echo line1\necho line2'
  'echo `echo tick` $(echo sub) | tr a-z A-Z'
  'printf "%s\n" "100%" \\ back'
)
for c in "${cmds[@]}"; do
  want="$(cd "$tmp/repo" && bash --noprofile --norc -c "$c" 2>&1)"
  got="$(exec_command "$c" 2>&1)"
  expect_eq "marker-output[$c]" "$got" "$want"
done
nlog="$(grep -c 'kind=exec' "$LOG")"
expect_eq log-line-count "$nlog" "${#cmds[@]}"
# each logged command decodes (shell-unquoting of the %q field) to the exact string
i=0
while IFS= read -r line; do
  field="${line##*$'\t'command=}"
  decoded="$(eval "printf %s $field")"
  expect_eq "logged-exact[$i]" "$decoded" "${cmds[$i]}"
  i=$((i + 1))
done < <(grep 'kind=exec' "$LOG")
head1="$(grep 'kind=exec' "$LOG" | head -1)"
[[ "$head1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z$'\t'increment=T1$'\t'who=[^[:space:]]+@[^[:space:]]+$'\t'kind=exec$'\t'attempt=1$'\t'command= ]] \
  && ok log-line-format || bad "log-line-format: $head1"

# --- 4. exit status through the wrapper (remote command text run directly) ----
for c in 'exit 7' 'false' 'true' 'echo x; exit 3' 'cd /nonexistent' 'exit 0'; do
  bare=0; ( cd "$tmp/repo" && bash --noprofile --norc -c "cd $REMOTE_REPO_DIR && $c" ) >/dev/null 2>&1 || bare=$?
  wrapped=0; bash --noprofile --norc -c "$(blinded_wrap "cd $REMOTE_REPO_DIR && $c" exec "command=x" 1)" >/dev/null 2>&1 || wrapped=$?
  expect_eq "exit-status[$c]" "$wrapped" "$bare"
done
# stdout/stderr separation untouched
o="$(bash --noprofile --norc -c "$(blinded_wrap 'echo OUT; echo ERR >&2' exec command=x 1)" 2>/dev/null)"
expect_eq stdout-only "$o" "OUT"
# without marker the same wrapped text is a pure no-op prefix
blinded_command stop >/dev/null
[[ ! -e "$MARK" ]] && ok marker-removed || bad marker-removed
n0="$(wc -l <"$LOG")"
o="$(exec_command 'echo after-stop')"
expect_eq after-stop-output "$o" "after-stop"
expect_eq after-stop-no-new-log "$(wc -l <"$LOG")" "$n0"
grep -q 'kind=blinded-start' "$LOG" && grep -q 'kind=blinded-stop' "$LOG" && ok start-stop-logged || bad start-stop-logged
(blinded_command stop) >/dev/null 2>&1 && bad stop-without-phase-refused || ok stop-without-phase-refused

# --- 5. retries are logged with their attempt number --------------------------
blinded_command start >/dev/null
REMOTE_SSH_RETRIES=2
before="$(grep -c 'kind=exec' "$LOG")"
exec_command 'exit 5' >/dev/null 2>&1 && bad failing-exec-fails || ok failing-exec-fails
att="$(grep 'kind=exec' "$LOG" | tail -3 | sed -E 's/.*attempt=([0-9]+).*/\1/' | paste -sd,)"
expect_eq retry-attempts "$att" "1,2,3"
REMOTE_SSH_RETRIES=0

# --- 6. a logging failure aborts the command (never an unlogged command) -------
rm -f "$tmp/side-effect"
mv "$REMOTE_STATE_ROOT/logs" "$REMOTE_STATE_ROOT/logs.bak"
: >"$REMOTE_STATE_ROOT/logs"   # a file where the log directory must be: mkdir -p fails
rc=0; exec_command "touch $tmp/side-effect" >/dev/null 2>&1 || rc=$?
[[ ! -e "$tmp/side-effect" && $rc -ne 0 ]] && ok log-failure-aborts-command || bad "log-failure-aborts-command (rc=$rc)"
rc=0; bash --noprofile --norc -c "$(blinded_wrap 'echo no' exec command=x 1)" >/dev/null 2>&1 || rc=$?
expect_eq log-failure-exit-97 "$rc" "97"
rm -f "$REMOTE_STATE_ROOT/logs"; mv "$REMOTE_STATE_ROOT/logs.bak" "$REMOTE_STATE_ROOT/logs"

# --- 7. run submissions are logged (ssh recorded, nothing executed) ------------
rec="$tmp/ssh.rec"; : >"$rec"
ssh() { shift; printf '%s\n---\n' "$1" >>"$rec"; cat >/dev/null 2>&1 </dev/null || true; }
rm -f "$LOG"
run_command myjob auto 'echo "x y" && ./a.out' >/dev/null
last="$(awk 'BEGIN{RS="\n---\n"} {l=$0} END{print l}' "$rec")"
grep -q 'blinded.marker' <<<"$last" && grep -q 'kind=run' <<<"$last" && grep -q 'job=myjob' <<<"$last" \
  && ok run-submission-guard-present || bad run-submission-guard-present
# the guard text itself, executed on the fake host, writes exactly one run line
ssh() { shift; ( PATH="$tmp/bin:$PATH"; eval "$1" ); }
bash --noprofile --norc -c "$(blinded_wrap 'true' run "job=myjob"$'\t'"gpu=auto" 1)"
expect_eq run-line-count "$(grep -c 'kind=run' "$LOG")" 1

# --- 8. the blinded log cannot be addressed as a job (launcher would truncate it) ----
echo keep-me >>"$LOG"
n_before="$(wc -l <"$LOG")"
: >"$rec"
ssh() { shift; printf '%s\n---\n' "$1" >>"$rec"; }
for fn in "run_command exec-blinded auto true" "status_command exec-blinded" "tail_command exec-blinded" \
          "wait_command exec-blinded" "cancel_command exec-blinded"; do
  rc=0; ( $fn ) >/dev/null 2>&1 || rc=$?
  expect_eq "reserved-job-name-refused[${fn%% *}]" "$rc" "1"
done
expect_eq reserved-name-no-ssh "$(wc -c <"$rec")" "0"
expect_eq reserved-name-log-intact "$(wc -l <"$LOG")" "$n_before"
rc=0; ( run_command exec-blinded2 auto true ) >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 ]] && ok other-job-names-unaffected || bad "other-job-names-unaffected (rc=$rc)"
! grep -q '(exit 97)' "$here/../remote" && grep -q 'helper (after its ssh retries) exits 1' "$here/../remote" \
  && ok usage-exit-status-text || bad usage-exit-status-text

# --- 9. remove-mirror refuses while the increment's blinded.marker exists ----------
ssh() { shift; ( PATH="$tmp/bin:$PATH"; eval "$1" ); }
mirror="$HOME/MacroFlow3D-T9"; mstate="$HOME/.macroflow3d-remote/macroflow3d-T9"
mkdir -p "$mirror" "$mstate/logs"
echo log >"$mstate/logs/exec-blinded.log"; : >"$mstate/blinded.marker"
rc=0; ( remove_mirror_command T9 ) >/dev/null 2>&1 || rc=$?
[[ $rc -ne 0 && -d "$mirror" && -e "$mstate/logs/exec-blinded.log" ]] \
  && ok remove-mirror-refused-during-phase || bad "remove-mirror-refused-during-phase (rc=$rc)"
rm -f "$mstate/blinded.marker"
rc=0; ( remove_mirror_command T9 ) >/dev/null 2>&1 || rc=$?
[[ $rc -eq 0 && ! -e "$mirror" && ! -e "$mstate" ]] \
  && ok remove-mirror-unchanged-without-phase || bad "remove-mirror-unchanged-without-phase (rc=$rc)"

(( fails == 0 )) && echo "ALL PASS" || { echo "$fails FAILED"; exit 1; }
