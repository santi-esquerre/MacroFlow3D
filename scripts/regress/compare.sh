#!/usr/bin/env bash
# Regression byte-compare harness, consumer side: compare two scripts/regress/run.sh outputs case by case.
#
#   scripts/regress/compare.sh <ref_out> <head_out> [--ref-filter '<glob>=<sed-expr>']...
#
# Cases come from <head_out>/exit_codes.tsv.  Per case: exit codes equal; output tree file by file with cmp, strict file set, every
# `timings.json` ignored, `manifest.json` through json_strip_cmp.py dropping the keys timestamp and build.git_hash; the 20-run R set
# by the SF-40 bc_r_case rules (dir files + driver-log lines with timing fields masked); log lines of r_prod likewise.
# A head case absent in ref = DIFF; a case only in ref = NOT_COMPARED (reported, not a failure); an output without REGRESS_DONE = DIFF.
# --ref-filter: the sed expression is applied to every file whose path relative to the out dir matches the bash glob (`*` matches `/`),
# in BOTH trees, before comparing: the declaration of a pre-agreed expected difference (README.md).  Repeatable.
# Last line: `REGRESS IDENTICAL` or `REGRESS DIFF n=<k>` (k = differing items); exit 0 iff identical.
#
# Provenance: bc_res / bc_cmp / bc_json / bc_lines / bc_tree / bc_r_case and the comparison constants are adapted from
# docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts/run_sf41.sh at commit 3452a7d (SF-40 RR7 rules).
set -uo pipefail

SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PYTHON:-python3}
MANIFEST_DROPS=(timestamp build.git_hash)
TREE_JSON_DROPS=()
FILTER_GLOBS=()
FILTER_EXPRS=()

if [ $# -lt 2 ]; then
    echo "usage: compare.sh <ref_out> <head_out> [--ref-filter '<glob>=<sed-expr>']..." >&2
    exit 2
fi
REF_OUT=$1
HEAD_OUT=$2
shift 2
while [ $# -gt 0 ]; do
    if [ "$1" = "--ref-filter" ] && [ $# -ge 2 ] && [ "${2#*=}" != "$2" ]; then
        FILTER_GLOBS+=("${2%%=*}")
        FILTER_EXPRS+=("${2#*=}")
        shift 2
    else
        echo "compare.sh: bad argument '$1' (expected --ref-filter '<glob>=<sed-expr>')" >&2
        exit 2
    fi
done
for d in "$REF_OUT" "$HEAD_OUT"; do
    [ -d "$d" ] || { echo "compare.sh: '$d' is not a directory" >&2; exit 2; }
done
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

BAD=0       # differing items, all cases
CASE_BAD=0  # differing items, current case
CASE_N=0    # items of the current case
FH=""
FR=""

bc_res() { # bc_res <IDENTICAL|DIFF> <item...>
    CASE_N=$((CASE_N + 1))
    if [ "$1" = "IDENTICAL" ]; then
        return 0
    fi
    echo "  ${*:2}"
    CASE_BAD=$((CASE_BAD + 1))
    BAD=$((BAD + 1))
}

# fpair <rel>: FH / FR = the head / ref file of <rel>, passed through every matching --ref-filter (on copies; missing files stay as is)
fpair() {
    local rel=$1 i k=0
    FH=$HEAD_OUT/$rel
    FR=$REF_OUT/$rel
    for ((i = 0; i < ${#FILTER_GLOBS[@]}; i++)); do
        # shellcheck disable=SC2254
        case "$rel" in
        ${FILTER_GLOBS[$i]})
            k=$((k + 1))
            if [ -e "$FH" ]; then LC_ALL=C sed -e "${FILTER_EXPRS[$i]}" "$FH" > "$TMP/h$k" && FH=$TMP/h$k; fi
            if [ -e "$FR" ]; then LC_ALL=C sed -e "${FILTER_EXPRS[$i]}" "$FR" > "$TMP/r$k" && FR=$TMP/r$k; fi
            ;;
        esac
    done
}

bc_cmp() { # bc_cmp <item> <rel>
    fpair "$2"
    if [ ! -e "$FH" ] || [ ! -e "$FR" ]; then
        bc_res DIFF "MISSING $1 (head: $HEAD_OUT/$2 ref: $REF_OUT/$2)"
    elif cmp -s "$FH" "$FR"; then
        bc_res IDENTICAL "$1"
    else
        bc_res DIFF "DIFF $1"
    fi
}

bc_json() { # bc_json <item> <rel> <drop spec>...
    local item=$1 rel=$2 out rc d dargs=()
    shift 2
    fpair "$rel"
    if [ ! -e "$FH" ] || [ ! -e "$FR" ]; then
        bc_res DIFF "MISSING $item (head: $HEAD_OUT/$rel ref: $REF_OUT/$rel)"
        return 0
    fi
    for d in "$@"; do dargs+=(--drop "$d"); done
    out=$("$PY" "$SCRIPTS/json_strip_cmp.py" "$FH" "$FR" ${dargs[@]+"${dargs[@]}"})
    rc=$?
    if [ "$rc" -eq 0 ]; then
        bc_res IDENTICAL "$item"
    else
        bc_res DIFF "DIFF $item: $out"
    fi
}

bc_lines() { # bc_lines <item> <rel_log> <token>...   (lines_cmp.py: timing fields masked)
    local item=$1 rel=$2 out rc
    shift 2
    fpair "$rel"
    if [ ! -e "$FH" ] || [ ! -e "$FR" ]; then
        bc_res DIFF "MISSING $item (head: $HEAD_OUT/$rel ref: $REF_OUT/$rel)"
        return 0
    fi
    out=$("$PY" "$SCRIPTS/lines_cmp.py" "$FH" "$FR" "$@")
    rc=$?
    if [ "$rc" -eq 0 ]; then
        bc_res IDENTICAL "$item: $out"
    else
        bc_res DIFF "DIFF $item: $(echo "$out" | head -n 1)"
        echo "$out" | sed 's/^/      /' | head -n 8
    fi
}

tree_files() { # tree_files <dir>: files relative to <dir>, `timings.json` excluded, sorted
    (cd "$1" && find . -type f ! -name timings.json | sed 's#^\./##' | LC_ALL=C sort)
}

# bc_tree <item> <rel_dir>: every ref file present and identical, no extra head file.  manifest.json: MANIFEST_DROPS; other *.json:
# json_strip_cmp.py with TREE_JSON_DROPS when that is non-empty, else byte-exact.
bc_tree() {
    local item=$1 rel=$2 f extra
    local hd=$HEAD_OUT/$rel rd=$REF_OUT/$rel
    if [ ! -d "$rd" ] && [ ! -d "$hd" ]; then
        bc_res IDENTICAL "$item (no output tree in either)"
        return 0
    fi
    if [ ! -d "$rd" ] || [ ! -d "$hd" ]; then
        bc_res DIFF "MISSING $item (head dir: $hd ref dir: $rd)"
        return 0
    fi
    while IFS= read -r f; do
        case "$f" in
        manifest.json | */manifest.json) bc_json "$item:$f" "$rel/$f" "${MANIFEST_DROPS[@]}" ${TREE_JSON_DROPS[@]+"${TREE_JSON_DROPS[@]}"} ;;
        *.json)
            if [ "${#TREE_JSON_DROPS[@]}" -gt 0 ]; then
                bc_json "$item:$f" "$rel/$f" "${TREE_JSON_DROPS[@]}"
            else
                bc_cmp "$item:$f" "$rel/$f"
            fi
            ;;
        *) bc_cmp "$item:$f" "$rel/$f" ;;
        esac
    done < <(tree_files "$rd")
    extra=$(comm -13 <(tree_files "$rd") <(tree_files "$hd") | tr '\n' ' ')
    if [ -z "$extra" ]; then
        bc_res IDENTICAL "$item:fileset"
    else
        bc_res DIFF "DIFF $item:fileset (extra head files: $extra)"
    fi
}

tsv_rc() { # tsv_rc <tsv> <id> -> the recorded rc of <id> (last record), empty if none
    awk -F'\t' -v c="$2" '$1 == c { r = $2 } END { if (r != "") print r }' "$1"
}

bc_r_case() { # bc_r_case <id>   (SF-40 bc_r_case rules)
    local id=$1 kind name toks
    if [ "$id" = "r_prod" ]; then
        bc_lines "r:prod32:lines" "log/$id.log" CASE ORACLE ORACLE_SUMMARY EXTRA
        return 0
    fi
    kind=${id#r_}
    kind=${kind%%_*}
    name=${id#r_"${kind}"_}
    if [ "$kind" = "trace" ]; then
        TREE_JSON_DROPS=('**.timings_s' 'transport.seconds_*')
        bc_tree "r:$kind:$name" "r/$kind/$name"
        bc_lines "r:$kind:$name:lines" "log/$id.log" CASE PLANE BACKFLOW GROWTH RESIDUAL TRANSPORT_STATUS STATUS
    else
        TREE_JSON_DROPS=('**.timings_s')
        bc_tree "r:$kind:$name" "r/$kind/$name"
        toks="SETUP FIELD TRACK BACKFLOW TPLANE TTIME TSTATUS STATUS"
        [ "$kind" = "neutral" ] && toks="$toks PTRACK PPLANE PTIME PSTATUS PFAIL SEEDV"
        # shellcheck disable=SC2086
        bc_lines "r:$kind:$name:lines" "log/$id.log" $toks
    fi
    TREE_JSON_DROPS=()
}

# ---- main ---------------------------------------------------------------------------------------------------------
for d in "$REF_OUT" "$HEAD_OUT"; do
    if [ ! -e "$d/REGRESS_DONE" ] || [ ! -e "$d/exit_codes.tsv" ]; then
        CASE_BAD=0
        bc_res DIFF "INCOMPLETE $d: no REGRESS_DONE / exit_codes.tsv (run.sh did not finish)"
    fi
done
if [ "$BAD" -gt 0 ]; then
    echo "REGRESS DIFF n=$BAD"
    exit 1
fi
if [ -e "$REF_OUT/binaries.md5" ] && [ -e "$HEAD_OUT/binaries.md5" ] && cmp -s "$REF_OUT/binaries.md5" "$HEAD_OUT/binaries.md5"; then
    echo "NOTE: ref and head binaries.md5 are identical (same binaries; the comparison cannot detect a code change)"
fi

NCASE=0
while IFS=$'\t' read -r id hrc; do
    [ -n "$id" ] || continue
    NCASE=$((NCASE + 1))
    CASE_BAD=0
    CASE_N=0
    echo "== $id"
    rrc=$(tsv_rc "$REF_OUT/exit_codes.tsv" "$id")
    if [ -z "$rrc" ]; then
        bc_res DIFF "MISSING $id (case absent in ref)"
    else
        if [ "$rrc" = "$hrc" ]; then
            bc_res IDENTICAL "$id:rc"
        else
            bc_res DIFF "DIFF $id:rc ref=$rrc head=$hrc"
        fi
        case "$id" in
        pipe_*) bc_tree "$id" "pipe/${id#pipe_}" ;;
        r_*) bc_r_case "$id" ;;
        g_* | runner_b | runner_b_analysis) bc_tree "$id" "$id" ;;
        *) bc_res DIFF "UNKNOWN $id (no comparison rule)" ;;
        esac
    fi
    if [ "$CASE_BAD" -eq 0 ]; then
        echo "   $id IDENTICAL ($CASE_N items)"
    else
        echo "   $id DIFF ($CASE_BAD of $CASE_N items)"
    fi
done < <(awk -F'\t' '{ r[$1] = $2 } !($1 in s) { s[$1] = 1; o[++n] = $1 } END { for (i = 1; i <= n; i++) print o[i] "\t" r[o[i]] }' "$HEAD_OUT/exit_codes.tsv")

while IFS= read -r id; do
    [ -n "$id" ] || continue
    if [ -z "$(tsv_rc "$HEAD_OUT/exit_codes.tsv" "$id")" ]; then
        echo "NOT_COMPARED $id (case only in ref)"
    fi
done < <(cut -f1 "$REF_OUT/exit_codes.tsv" | LC_ALL=C sort -u)

echo "cases compared: $NCASE"
if [ "$BAD" -eq 0 ]; then
    echo "REGRESS IDENTICAL"
    exit 0
fi
echo "REGRESS DIFF n=$BAD"
exit 1
