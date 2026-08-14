#!/usr/bin/env bash
# Validate the sequential Lester equation (14) increment harness.
# Backticks below are literal Markdown delimiters in sed/grep patterns.
# shellcheck disable=SC2016
set -euo pipefail

if ! repo_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
    echo "ERROR: not inside a git repository" >&2
    exit 1
fi

dashboard="$repo_root/docs/plans/active/lester-eq14-streamfunction-solver-plan.md"
increment_dir="$repo_root/docs/plans/active/lester-eq14/increments"

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
expected_count=31
[[ ${#files[@]} -eq $expected_count ]] || \
    fail "expected $expected_count increment files, found ${#files[@]}"

declare -A states
declare -A paths
declare -A goals
active_count=0
first_not_done=""
seen_not_done=0

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
    paths["$expected_id"]="$file"

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
        (( seen_not_done == 0 )) || fail "$expected_id is done after an unfinished increment"
    else
        seen_not_done=1
        [[ -n "$first_not_done" ]] || first_not_done="$expected_id"
    fi

    case "$state" in
        active|validating|awaiting_review|blocked)
            active_count=$((active_count + 1))
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

(( active_count <= 1 )) || fail "more than one increment is active/nonterminal"

for id in "${!states[@]}"; do
    state="${states[$id]}"
    depends="$(sed -n 's/^- Depends on: `\([^`]*\)`$/\1/p' "${paths[$id]}" | head -n1)"
    if [[ "$state" != "pending" && "$depends" != "none" ]]; then
        IFS=',' read -ra dep_ids <<< "$depends"
        for dep in "${dep_ids[@]}"; do
            dep="${dep// /}"
            [[ -n "${states[$dep]+set}" ]] || {
                fail "$id references unknown dependency $dep"
                continue
            }
            [[ "${states[$dep]}" == "done" ]] || \
                fail "$id is $state but dependency $dep is ${states[$dep]}"
        done
    fi
done

if [[ -n "$first_not_done" ]]; then
    next_id="$(sed -n 's/^- NEXT: `\([^`]*\)`$/\1/p' "$dashboard" | head -n1)"
    [[ "$next_id" == "$first_not_done" ]] || \
        fail "dashboard NEXT is '$next_id'; expected '$first_not_done'"
    for id in "${!states[@]}"; do
        case "${states[$id]}" in
            active|validating|awaiting_review|blocked)
                [[ "$id" == "$first_not_done" ]] || \
                    fail "$id is ${states[$id]} but first unfinished is $first_not_done"
                ;;
        esac
    done
else
    grep -Fqx -- '- NEXT: `COMPLETE`' "$dashboard" || \
        fail "all increments are done but NEXT is not COMPLETE"
fi

if (( failures > 0 )); then
    echo "Lester increment harness: FAILED ($failures problem(s))" >&2
    exit 1
fi

echo "Lester increment harness: OK (${#files[@]} increments, next=${first_not_done:-COMPLETE})"
