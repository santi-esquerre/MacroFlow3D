#!/usr/bin/env bash
# Validate the Lester equation (14) increment DAG harness.
#
# Semantics:
#   - Increments SF-NN are dependency-ordered (a DAG), not strictly sequential.
#     State lives in each spec's `State:` line and in the dashboard master
#     checklist (checked iff `done`); there is no NEXT pointer.
#   - READY = `pending` with every dependency `done` (or `Depends on: none`).
#   - Nonterminal = active|validating|awaiting_review|blocked; at most 2 may
#     exist at a time (one orchestrator session and one PR each).
#   - Any non-pending increment requires all its dependencies `done`.
#   - Dependency ids are validated for every state, including pending.
#   - Stale `- NEXT:`, `- Active runtime goal:` and `- Last completed
#     increment:` dashboard lines are rejected.
# Backticks below are literal Markdown delimiters in sed/grep patterns.
# shellcheck disable=SC2016
set -euo pipefail

if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    echo "ERROR: not inside a git repository" >&2
    exit 1
fi

dashboard="$repo_root/docs/plans/active/lester-eq14-streamfunction-solver-plan.md"
increment_dir="$repo_root/docs/plans/active/lester-eq14/increments"
max_nonterminal=2

failures=0
fail() {
    echo "ERROR: $*" >&2
    failures=$((failures + 1))
}

[[ -f "$dashboard" ]] || fail "missing dashboard: $dashboard"
[[ -d "$increment_dir" ]] || fail "missing increment directory: $increment_dir"

if (( failures > 0 )); then
    exit 1
fi

mapfile -t files < <(find "$increment_dir" -maxdepth 1 -type f -name 'SF-*.md' | sort)
expected_count=36
[[ ${#files[@]} -eq $expected_count ]] || \
    fail "expected $expected_count increment files, found ${#files[@]}"

declare -A states
declare -A goals
declare -A deps
nonterminal=()

required_headings=(
    "## Scientific or engineering intent"
    "## Preconditions"
    "## In scope"
    "## Out of scope"
    "## Files and symbols"
    "## Implementation specification"
    "## Expected numerical effect"
    "## Validation commands"
    "## Acceptance thresholds"
    "## Regression surface"
    "## Failure and rollback policy"
    "## Completion checklist"
    "## Advancement rule"
    "## Bitácora"
)

for index in "${!files[@]}"; do
    file="${files[$index]}"
    expected_id="$(printf 'SF-%02d' "$index")"
    basename_id="$(basename "$file" | cut -d- -f1-2)"
    [[ "$basename_id" == "$expected_id" ]] || \
        fail "expected $expected_id at position $index, found $basename_id"

    state="$(sed -n 's/^- State: `\([^`]*\)`$/\1/p' "$file" | head -n1)"
    goal="$(sed -n 's/^- Goal: `\([^`]*\)`$/\1/p' "$file" | head -n1)"
    depends="$(sed -n 's/^- Depends on: `\([^`]*\)`$/\1/p' "$file" | head -n1)"

    case "$state" in
        pending|active|validating|awaiting_review|blocked|done) ;;
        *) fail "$expected_id has invalid or missing State: '$state'" ;;
    esac
    [[ -n "$goal" ]] || fail "$expected_id has no exact Goal"
    [[ -n "$depends" ]] || fail "$expected_id has no Depends on field"
    if [[ -n "$goal" && -n "${goals[$goal]+set}" ]]; then
        fail "$expected_id duplicates Goal from ${goals[$goal]}"
    else
        goals["$goal"]="$expected_id"
    fi

    states["$expected_id"]="$state"
    deps["$expected_id"]="$depends"

    for heading in "${required_headings[@]}"; do
        grep -Fqx "$heading" "$file" || fail "$expected_id missing heading: $heading"
    done
    grep -Fqx '<!-- completion-checklist:start -->' "$file" || \
        fail "$expected_id missing completion checklist start marker"
    grep -Fqx '<!-- completion-checklist:end -->' "$file" || \
        fail "$expected_id missing completion checklist end marker"

    if [[ "$state" == "done" ]]; then
        if sed -n '/<!-- completion-checklist:start -->/,/<!-- completion-checklist:end -->/p' "$file" | \
            grep -Eq '^- \[ \]'; then
            fail "$expected_id is done but has unchecked completion items"
        fi
    fi

    case "$state" in
        active|validating|awaiting_review|blocked)
            nonterminal+=("$expected_id")
            ;;
    esac

    master_line="$(grep -E "^- \[[ x]\] \[$expected_id —" "$dashboard" || true)"
    [[ -n "$master_line" ]] || fail "dashboard missing checklist entry for $expected_id"
    master_target="$(printf '%s\n' "$master_line" | sed -n 's/.*](\([^)]*\)).*/\1/p')"
    [[ -n "$master_target" ]] || fail "dashboard entry for $expected_id has no link target"
    [[ -f "$(dirname "$dashboard")/$master_target" ]] || \
        fail "dashboard link for $expected_id does not resolve: $master_target"
    if [[ "$state" == "done" ]]; then
        [[ "$master_line" == "- [x]"* ]] || fail "$expected_id is done but dashboard is unchecked"
    else
        [[ "$master_line" == "- [ ]"* ]] || fail "$expected_id is unfinished but dashboard is checked"
    fi
done

(( ${#nonterminal[@]} <= max_nonterminal )) || \
    fail "more than $max_nonterminal nonterminal increments: ${nonterminal[*]}"

# Stale sequential-harness pointers must not survive in the dashboard.
for stale in '- NEXT:' '- Active runtime goal:' '- Last completed increment:'; do
    if grep -q -- "^$stale" "$dashboard"; then
        fail "dashboard still contains obsolete pointer line '$stale'"
    fi
done

ready=()
mapfile -t sorted_ids < <(printf '%s\n' "${!states[@]}" | sort)
for id in "${sorted_ids[@]}"; do
    state="${states[$id]}"
    depends="${deps[$id]}"
    all_done=1
    if [[ "$depends" != "none" ]]; then
        IFS=',' read -ra dep_ids <<< "$depends"
        for dep in "${dep_ids[@]}"; do
            dep="${dep// /}"
            if [[ -z "${states[$dep]+set}" ]]; then
                fail "$id references unknown dependency $dep"
                all_done=0
                continue
            fi
            if [[ "${states[$dep]}" != "done" ]]; then
                all_done=0
                if [[ "$state" != "pending" ]]; then
                    fail "$id is $state but dependency $dep is ${states[$dep]}"
                fi
            fi
        done
    fi
    if [[ "$state" == "pending" && $all_done -eq 1 ]]; then
        ready+=("$id")
    fi
done

if (( failures > 0 )); then
    echo "Lester increment harness: FAILED ($failures problem(s))" >&2
    exit 1
fi

nonterminal_sorted=""
if (( ${#nonterminal[@]} > 0 )); then
    nonterminal_sorted="$(printf '%s\n' "${nonterminal[@]}" | sort | paste -sd' ' -)"
fi
ready_str=""
if (( ${#ready[@]} > 0 )); then
    ready_str="${ready[*]}"
fi
echo "Lester increment harness: OK (${#files[@]} increments, ready=${ready_str:-none}, nonterminal=${nonterminal_sorted:-none})"
