#!/usr/bin/env bash
set -euo pipefail

root="${1:-artifacts/kh_reconstruction}"
summary="${root}/summary"
mkdir -p "$summary"

python3 - "$root" <<'PY'
import csv
import math
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
summary = root / "summary"

def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row and not row[0].startswith("#")]
    if not rows:
        return []
    header = rows[0]
    out = []
    for row in rows[1:]:
        if len(row) == len(header):
            out.append(dict(zip(header, row)))
    return out

def fval(row, key, default=math.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default

def values(rows, key):
    return [r[key] for r in rows if math.isfinite(r[key])]

def mean_or_nan(vals):
    return statistics.mean(vals) if vals else math.nan

def ci95(vals):
    if not vals:
        return math.nan
    if len(vals) == 1:
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))

runs = []
for seed_dir in sorted((root / "raw").glob("seed_*")):
    seed = int(seed_dir.name.split("_")[1])
    for backend in ("face", "kh"):
        run_dir = seed_dir / backend
        alpha_rows = read_csv(run_dir / "alpha_timeseries.csv")
        field_rows = read_csv(run_dir / "field_diagnostics.csv")
        runtime_rows = read_csv(run_dir / "runtime_diagnostics.csv")
        if not alpha_rows:
            continue
        final_alpha = alpha_rows[-1]
        field = field_rows[-1] if field_rows else {}
        runtime = runtime_rows[-1] if runtime_rows else {}
        runs.append({
            "seed": seed,
            "backend": backend,
            "alpha_L_final": fval(final_alpha, "alpha_x"),
            "alpha_T1_final": fval(final_alpha, "alpha_y"),
            "alpha_T2_final": fval(final_alpha, "alpha_z"),
            "helicity_norm_mean": fval(field, "helicity_norm_mean"),
            "div_abs_mean": fval(field, "div_abs_mean"),
            "runtime_seconds": fval(runtime, "transport_seconds"),
        })

with (summary / "ensemble_summary.csv").open("w", newline="") as f:
    fields = [
        "backend", "n_seeds",
        "alpha_L_mean", "alpha_L_ci95",
        "alpha_T1_mean", "alpha_T1_ci95",
        "alpha_T2_mean", "alpha_T2_ci95",
        "helicity_norm_mean", "div_abs_mean", "runtime_mean", "notes",
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for backend in ("face", "kh"):
        rows = [r for r in runs if r["backend"] == backend]
        alpha_l = values(rows, "alpha_L_final")
        alpha_t1 = values(rows, "alpha_T1_final")
        alpha_t2 = values(rows, "alpha_T2_final")
        helicity = values(rows, "helicity_norm_mean")
        div_abs = values(rows, "div_abs_mean")
        runtime = values(rows, "runtime_seconds")
        w.writerow({
            "backend": backend,
            "n_seeds": len(rows),
            "alpha_L_mean": mean_or_nan(alpha_l),
            "alpha_L_ci95": ci95(alpha_l),
            "alpha_T1_mean": mean_or_nan(alpha_t1),
            "alpha_T1_ci95": ci95(alpha_t1),
            "alpha_T2_mean": mean_or_nan(alpha_t2),
            "alpha_T2_ci95": ci95(alpha_t2),
            "helicity_norm_mean": mean_or_nan(helicity),
            "div_abs_mean": mean_or_nan(div_abs),
            "runtime_mean": mean_or_nan(runtime),
            "notes": "sampled diagnostics; see per-run CSVs",
        })

by_seed = {}
for r in runs:
    by_seed.setdefault(r["seed"], {})[r["backend"]] = r

with (summary / "alphaT_comparison_face_vs_kh.csv").open("w", newline="") as f:
    fields = ["seed", "alpha_T1_face", "alpha_T1_kh", "delta_alpha_T1_kh_minus_face",
              "alpha_T2_face", "alpha_T2_kh", "delta_alpha_T2_kh_minus_face"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for seed, pair in sorted(by_seed.items()):
        if "face" not in pair or "kh" not in pair:
            continue
        face, kh = pair["face"], pair["kh"]
        w.writerow({
            "seed": seed,
            "alpha_T1_face": face["alpha_T1_final"],
            "alpha_T1_kh": kh["alpha_T1_final"],
            "delta_alpha_T1_kh_minus_face": kh["alpha_T1_final"] - face["alpha_T1_final"],
            "alpha_T2_face": face["alpha_T2_final"],
            "alpha_T2_kh": kh["alpha_T2_final"],
            "delta_alpha_T2_kh_minus_face": kh["alpha_T2_final"] - face["alpha_T2_final"],
        })

with (summary / "helicity_comparison_face_vs_kh.csv").open("w", newline="") as f:
    fields = ["seed", "helicity_norm_face", "helicity_norm_kh", "delta_kh_minus_face"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for seed, pair in sorted(by_seed.items()):
        if "face" not in pair or "kh" not in pair:
            continue
        face, kh = pair["face"], pair["kh"]
        w.writerow({
            "seed": seed,
            "helicity_norm_face": face["helicity_norm_mean"],
            "helicity_norm_kh": kh["helicity_norm_mean"],
            "delta_kh_minus_face": kh["helicity_norm_mean"] - face["helicity_norm_mean"],
        })

with (summary / "runtime_comparison.csv").open("w", newline="") as f:
    fields = ["seed", "runtime_face_seconds", "runtime_kh_seconds", "kh_over_face"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for seed, pair in sorted(by_seed.items()):
        if "face" not in pair or "kh" not in pair:
            continue
        face, kh = pair["face"], pair["kh"]
        ratio = kh["runtime_seconds"] / face["runtime_seconds"] if face["runtime_seconds"] and math.isfinite(face["runtime_seconds"]) else math.nan
        w.writerow({
            "seed": seed,
            "runtime_face_seconds": face["runtime_seconds"],
            "runtime_kh_seconds": kh["runtime_seconds"],
            "kh_over_face": ratio,
        })

print(f"Wrote KH ensemble summaries under {summary}")
PY
