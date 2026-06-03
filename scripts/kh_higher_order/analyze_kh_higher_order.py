#!/usr/bin/env python3
"""Analyze KH higher-order ensemble outputs."""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import sys
from typing import Iterable


BACKENDS = [
    ("face", "FACE_TRILINEAR", "#1f77b4"),
    ("kh_linear", "KH_LINEAR", "#d62728"),
    ("kh_cubic", "KH_CUBIC_POTENTIAL_RECONSTRUCTION", "#2ca02c"),
    ("kh_logk_cubic", "KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION", "#9467bd"),
]
BACKEND_IDS = [b[0] for b in BACKENDS]
BACKEND_LABEL = {b[0]: b[1] for b in BACKENDS}
BACKEND_COLOR = {b[0]: b[2] for b in BACKENDS}
PAIR_SPECS = [
    ("paired_khlinear_vs_khcubic.csv", "kh_linear", "kh_cubic"),
    ("paired_khlinear_vs_khlogk.csv", "kh_linear", "kh_logk_cubic"),
    ("paired_face_vs_khcubic.csv", "face", "kh_cubic"),
    ("paired_face_vs_khlogk.csv", "face", "kh_logk_cubic"),
    ("paired_khcubic_vs_khlogk.csv", "kh_cubic", "kh_logk_cubic"),
]


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row and not row[0].startswith("#")]
    if not rows:
        return []
    header = rows[0]
    return [dict(zip(header, row)) for row in rows[1:] if len(row) == len(header)]


def fval(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else math.nan


def stdev(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def ci95(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return math.nan
    if len(vals) == 1:
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))


def quantile(values: list[float], p: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return math.nan
    if len(vals) == 1:
        return vals[0]
    pos = p * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def write_csv(path: pathlib.Path, rows: list[dict[str, object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def safe_metric(record: dict[str, object], key: str) -> float:
    value = record.get(key, math.nan)
    return float(value) if isinstance(value, (int, float)) else math.nan


def discover_runs(root: pathlib.Path):
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    series_by_key: dict[tuple[str, str], list[list[dict[str, float]]]] = {}

    raw_root = root / "raw"
    if not raw_root.exists():
        return records, failures, series_by_key

    for phase_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        phase = phase_dir.name
        for seed_dir in sorted(p for p in phase_dir.glob("seed_*") if p.is_dir()):
            try:
                seed = int(seed_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            for backend in BACKEND_IDS:
                run_dir = seed_dir / backend
                if not run_dir.exists():
                    failures.append(
                        {
                            "phase": phase,
                            "seed": seed,
                            "backend": backend,
                            "status": "missing_run_dir",
                            "missing_files": "run_dir",
                        }
                    )
                    continue
                alpha_path = run_dir / "alpha_timeseries.csv"
                if not alpha_path.exists():
                    alpha_path = run_dir / "analysis" / "macrodispersion.csv"
                field_path = run_dir / "field_diagnostics.csv"
                runtime_path = run_dir / "runtime_diagnostics.csv"
                transport_path = run_dir / "transport_diagnostics.csv"
                comparison_path = run_dir / "velocity_comparison.csv"
                missing = [
                    name
                    for name, path in [
                        ("alpha", alpha_path),
                        ("field", field_path),
                        ("runtime", runtime_path),
                        ("transport", transport_path),
                        ("comparison", comparison_path),
                    ]
                    if not path.exists()
                ]
                if missing:
                    failures.append(
                        {
                            "phase": phase,
                            "seed": seed,
                            "backend": backend,
                            "status": "missing_files",
                            "missing_files": ";".join(missing),
                        }
                    )
                    continue

                alpha_rows = read_csv(alpha_path)
                field_rows = read_csv(field_path)
                runtime_rows = read_csv(runtime_path)
                transport_rows = read_csv(transport_path)
                comparison_rows = read_csv(comparison_path)
                if not alpha_rows:
                    failures.append(
                        {
                            "phase": phase,
                            "seed": seed,
                            "backend": backend,
                            "status": "empty_alpha",
                            "missing_files": "alpha_rows",
                        }
                    )
                    continue
                alpha = alpha_rows[-1]
                field = field_rows[-1] if field_rows else {}
                runtime = runtime_rows[-1] if runtime_rows else {}
                transport = transport_rows[-1] if transport_rows else {}
                comparison = comparison_rows[-1] if comparison_rows else {}
                t1 = fval(alpha, "alpha_y")
                t2 = fval(alpha, "alpha_z")
                records.append(
                    {
                        "phase": phase,
                        "seed": seed,
                        "backend": backend,
                        "backend_label": BACKEND_LABEL[backend],
                        "alpha_L": fval(alpha, "alpha_x"),
                        "alpha_T1": t1,
                        "alpha_T2": t2,
                        "alpha_Tmean": 0.5 * (t1 + t2),
                        "speed_mean": fval(field, "speed_mean"),
                        "speed_max": fval(field, "speed_max"),
                        "div_abs_mean": fval(field, "div_abs_mean"),
                        "div_abs_max": fval(field, "div_abs_max"),
                        "curl_mag_mean": fval(field, "curl_mag_mean"),
                        "curl_mag_max": fval(field, "curl_mag_max"),
                        "helicity_mean": fval(field, "helicity_mean"),
                        "helicity_abs_mean": fval(field, "helicity_abs_mean"),
                        "helicity_norm_mean": fval(field, "helicity_norm_mean"),
                        "helicity_norm_std": fval(field, "helicity_norm_std"),
                        "helicity_norm_p50": fval(field, "helicity_norm_p50"),
                        "helicity_norm_p95": fval(field, "helicity_norm_p95"),
                        "k_interp_min": fval(field, "k_interp_min"),
                        "k_interp_max": fval(field, "k_interp_max"),
                        "k_interp_mean": fval(field, "k_interp_mean"),
                        "k_interp_nonpositive_count": fval(field, "k_interp_nonpositive_count"),
                        "k_interp_clamped_count": fval(field, "k_interp_clamped_count"),
                        "logk_interp_min": fval(field, "logk_interp_min"),
                        "logk_interp_max": fval(field, "logk_interp_max"),
                        "rel_l2_velocity_diff": fval(comparison, "rel_l2_diff"),
                        "diff_p95": fval(comparison, "diff_p95"),
                        "rel_diff_p95": fval(comparison, "rel_diff_p95"),
                        "rel_diff_max": fval(comparison, "rel_diff_max"),
                        "vector_correlation": fval(comparison, "vector_correlation"),
                        "runtime_seconds": fval(runtime, "transport_seconds"),
                        "active": fval(transport, "active"),
                        "problematic": fval(transport, "problematic"),
                        "final_time": fval(transport, "final_time"),
                        "var_x": fval(transport, "var_x"),
                        "var_y": fval(transport, "var_y"),
                        "var_z": fval(transport, "var_z"),
                    }
                )
                series_by_key.setdefault((phase, backend), []).append(
                    [
                        {
                            "t": fval(r, "t"),
                            "alpha_L": fval(r, "alpha_x"),
                            "alpha_T1": fval(r, "alpha_y"),
                            "alpha_T2": fval(r, "alpha_z"),
                            "alpha_Tmean": 0.5 * (fval(r, "alpha_y") + fval(r, "alpha_z")),
                        }
                        for r in alpha_rows
                    ]
                )
    return records, failures, series_by_key


def rows_for(records: list[dict[str, object]], phase: str, backend: str) -> list[dict[str, object]]:
    return [r for r in records if r["phase"] == phase and r["backend"] == backend]


def summarize_by_phase_backend(records: list[dict[str, object]]) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    phases = sorted({str(r["phase"]) for r in records})
    ensemble_rows: list[dict[str, object]] = []
    field_rows: list[dict[str, object]] = []
    transport_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    k_rows: list[dict[str, object]] = []

    for phase in phases:
        for backend in BACKEND_IDS:
            rows = rows_for(records, phase, backend)
            if not rows:
                continue
            ensemble_rows.append(
                {
                    "phase": phase,
                    "backend": backend,
                    "backend_label": BACKEND_LABEL[backend],
                    "n_seeds": len(rows),
                    "alpha_L_mean": mean(safe_metric(r, "alpha_L") for r in rows),
                    "alpha_L_ci95": ci95(safe_metric(r, "alpha_L") for r in rows),
                    "alpha_T1_mean": mean(safe_metric(r, "alpha_T1") for r in rows),
                    "alpha_T1_ci95": ci95(safe_metric(r, "alpha_T1") for r in rows),
                    "alpha_T2_mean": mean(safe_metric(r, "alpha_T2") for r in rows),
                    "alpha_T2_ci95": ci95(safe_metric(r, "alpha_T2") for r in rows),
                    "alpha_Tmean_mean": mean(safe_metric(r, "alpha_Tmean") for r in rows),
                    "alpha_Tmean_ci95": ci95(safe_metric(r, "alpha_Tmean") for r in rows),
                    "helicity_norm_mean": mean(safe_metric(r, "helicity_norm_mean") for r in rows),
                    "div_abs_mean": mean(safe_metric(r, "div_abs_mean") for r in rows),
                    "runtime_mean": mean(safe_metric(r, "runtime_seconds") for r in rows),
                    "problematic_total": sum(safe_metric(r, "problematic") for r in rows),
                }
            )
            field_rows.append(
                {
                    "phase": phase,
                    "backend": backend,
                    "backend_label": BACKEND_LABEL[backend],
                    "n_seeds": len(rows),
                    "speed_mean": mean(safe_metric(r, "speed_mean") for r in rows),
                    "speed_max": mean(safe_metric(r, "speed_max") for r in rows),
                    "div_abs_mean": mean(safe_metric(r, "div_abs_mean") for r in rows),
                    "div_abs_max": mean(safe_metric(r, "div_abs_max") for r in rows),
                    "curl_mag_mean": mean(safe_metric(r, "curl_mag_mean") for r in rows),
                    "helicity_mean": mean(safe_metric(r, "helicity_mean") for r in rows),
                    "helicity_abs_mean": mean(safe_metric(r, "helicity_abs_mean") for r in rows),
                    "helicity_norm_mean": mean(safe_metric(r, "helicity_norm_mean") for r in rows),
                    "helicity_norm_std_mean": mean(safe_metric(r, "helicity_norm_std") for r in rows),
                    "helicity_norm_p50_mean": mean(safe_metric(r, "helicity_norm_p50") for r in rows),
                    "helicity_norm_p95_mean": mean(safe_metric(r, "helicity_norm_p95") for r in rows),
                }
            )
            transport_rows.append(
                {
                    "phase": phase,
                    "backend": backend,
                    "backend_label": BACKEND_LABEL[backend],
                    "n_seeds": len(rows),
                    "alpha_L_mean": mean(safe_metric(r, "alpha_L") for r in rows),
                    "alpha_T1_mean": mean(safe_metric(r, "alpha_T1") for r in rows),
                    "alpha_T2_mean": mean(safe_metric(r, "alpha_T2") for r in rows),
                    "alpha_Tmean_mean": mean(safe_metric(r, "alpha_Tmean") for r in rows),
                    "active_mean": mean(safe_metric(r, "active") for r in rows),
                    "problematic_mean": mean(safe_metric(r, "problematic") for r in rows),
                    "var_x_mean": mean(safe_metric(r, "var_x") for r in rows),
                    "var_y_mean": mean(safe_metric(r, "var_y") for r in rows),
                    "var_z_mean": mean(safe_metric(r, "var_z") for r in rows),
                }
            )
            runtime_rows.append(
                {
                    "phase": phase,
                    "backend": backend,
                    "backend_label": BACKEND_LABEL[backend],
                    "n_seeds": len(rows),
                    "runtime_mean": mean(safe_metric(r, "runtime_seconds") for r in rows),
                    "runtime_ci95": ci95(safe_metric(r, "runtime_seconds") for r in rows),
                    "runtime_min": min(safe_metric(r, "runtime_seconds") for r in rows),
                    "runtime_max": max(safe_metric(r, "runtime_seconds") for r in rows),
                }
            )
            k_rows.append(
                {
                    "phase": phase,
                    "backend": backend,
                    "backend_label": BACKEND_LABEL[backend],
                    "n_seeds": len(rows),
                    "k_interp_min_mean": mean(safe_metric(r, "k_interp_min") for r in rows),
                    "k_interp_min_global": min(
                        (safe_metric(r, "k_interp_min") for r in rows if math.isfinite(safe_metric(r, "k_interp_min"))),
                        default=math.nan,
                    ),
                    "k_interp_max_mean": mean(safe_metric(r, "k_interp_max") for r in rows),
                    "k_interp_max_global": max(
                        (safe_metric(r, "k_interp_max") for r in rows if math.isfinite(safe_metric(r, "k_interp_max"))),
                        default=math.nan,
                    ),
                    "k_interp_mean_mean": mean(safe_metric(r, "k_interp_mean") for r in rows),
                    "k_interp_nonpositive_total": sum(
                        safe_metric(r, "k_interp_nonpositive_count") for r in rows
                    ),
                    "k_interp_clamped_total": sum(
                        safe_metric(r, "k_interp_clamped_count") for r in rows
                    ),
                    "logk_interp_min_global": min(
                        (safe_metric(r, "logk_interp_min") for r in rows if math.isfinite(safe_metric(r, "logk_interp_min"))),
                        default=math.nan,
                    ),
                    "logk_interp_max_global": max(
                        (safe_metric(r, "logk_interp_max") for r in rows if math.isfinite(safe_metric(r, "logk_interp_max"))),
                        default=math.nan,
                    ),
                }
            )
    return ensemble_rows, field_rows, transport_rows, runtime_rows, k_rows


def paired_rows(records: list[dict[str, object]], backend_a: str, backend_b: str) -> list[dict[str, object]]:
    lookup = {(str(r["phase"]), int(r["seed"]), str(r["backend"])): r for r in records}
    rows: list[dict[str, object]] = []
    keys = sorted({(str(r["phase"]), int(r["seed"])) for r in records})
    for phase, seed in keys:
        a = lookup.get((phase, seed, backend_a))
        b = lookup.get((phase, seed, backend_b))
        if not a or not b:
            continue
        rows.append(
            {
                "phase": phase,
                "seed": seed,
                "backend_a": BACKEND_LABEL[backend_a],
                "backend_b": BACKEND_LABEL[backend_b],
                "alpha_Tmean_a": safe_metric(a, "alpha_Tmean"),
                "alpha_Tmean_b": safe_metric(b, "alpha_Tmean"),
                "delta_alpha_Tmean": safe_metric(b, "alpha_Tmean") - safe_metric(a, "alpha_Tmean"),
                "alpha_L_a": safe_metric(a, "alpha_L"),
                "alpha_L_b": safe_metric(b, "alpha_L"),
                "delta_alpha_L": safe_metric(b, "alpha_L") - safe_metric(a, "alpha_L"),
                "helicity_norm_mean_a": safe_metric(a, "helicity_norm_mean"),
                "helicity_norm_mean_b": safe_metric(b, "helicity_norm_mean"),
                "delta_helicity_norm_mean": safe_metric(b, "helicity_norm_mean")
                - safe_metric(a, "helicity_norm_mean"),
                "div_abs_mean_a": safe_metric(a, "div_abs_mean"),
                "div_abs_mean_b": safe_metric(b, "div_abs_mean"),
                "delta_div_abs_mean": safe_metric(b, "div_abs_mean") - safe_metric(a, "div_abs_mean"),
                "runtime_seconds_a": safe_metric(a, "runtime_seconds"),
                "runtime_seconds_b": safe_metric(b, "runtime_seconds"),
                "delta_runtime_seconds": safe_metric(b, "runtime_seconds")
                - safe_metric(a, "runtime_seconds"),
                "problematic_a": safe_metric(a, "problematic"),
                "problematic_b": safe_metric(b, "problematic"),
                "delta_problematic": safe_metric(b, "problematic") - safe_metric(a, "problematic"),
                "rel_l2_velocity_diff_b": safe_metric(b, "rel_l2_velocity_diff"),
                "vector_correlation_b": safe_metric(b, "vector_correlation"),
            }
        )
    return rows


def aggregate_curves(series_by_key: dict[tuple[str, str], list[list[dict[str, float]]]], phase: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for backend in BACKEND_IDS:
        series = series_by_key.get((phase, backend), [])
        if not series:
            continue
        n = len(series)
        min_len = min(len(s) for s in series)
        for idx in range(min_len):
            row: dict[str, object] = {"phase": phase, "backend": backend, "t": series[0][idx]["t"], "n": n}
            for metric in ("alpha_L", "alpha_T1", "alpha_T2", "alpha_Tmean"):
                vals = [s[idx][metric] for s in series if idx < len(s)]
                row[f"{metric}_mean"] = mean(vals)
                row[f"{metric}_ci95"] = ci95(vals)
                row[f"{metric}_lower"] = row[f"{metric}_mean"] - row[f"{metric}_ci95"]
                row[f"{metric}_upper"] = row[f"{metric}_mean"] + row[f"{metric}_ci95"]
            rows.append(row)
    return rows


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        "<style>text{font-family:Arial,sans-serif;font-size:12px} "
        ".title{font-size:16px;font-weight:bold}.axis{stroke:#333;stroke-width:1} "
        ".grid{stroke:#ddd;stroke-width:1}</style>\n"
    )


def save_svg(path: pathlib.Path, body: str, width: int = 920, height: int = 560) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_header(width, height) + body + "</svg>\n")


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if not math.isfinite(v) or hi == lo:
        return 0.5 * (a + b)
    return a + (v - lo) * (b - a) / (hi - lo)


def line_plot_multi(path: pathlib.Path, title: str, rows: list[dict[str, object]], metric: str) -> None:
    if not rows:
        return
    width, height = 920, 560
    left, right, top, bottom = 70, 880, 50, 490
    xs = [float(r["t"]) for r in rows]
    vals = []
    for r in rows:
        vals.extend([float(r[f"{metric}_lower"]), float(r[f"{metric}_upper"])])
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(vals), max(vals)
    pad = 0.08 * (yhi - ylo if yhi > ylo else 1.0)
    ylo -= pad
    yhi += pad
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>']
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + frac * (bottom - top)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    legend_y = 72
    for backend, label, color in BACKENDS:
        b_rows = [r for r in rows if r["backend"] == backend]
        if not b_rows:
            continue
        upper = [
            (scale(float(r["t"]), xlo, xhi, left, right), scale(float(r[f"{metric}_upper"]), ylo, yhi, bottom, top))
            for r in b_rows
        ]
        lower = list(
            reversed(
                [
                    (
                        scale(float(r["t"]), xlo, xhi, left, right),
                        scale(float(r[f"{metric}_lower"]), ylo, yhi, bottom, top),
                    )
                    for r in b_rows
                ]
            )
        )
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        line = " ".join(
            f"{scale(float(r['t']), xlo, xhi, left, right):.1f},{scale(float(r[f'{metric}_mean']), ylo, yhi, bottom, top):.1f}"
            for r in b_rows
        )
        body.append(f'<polygon points="{poly}" fill="{color}" opacity="0.12"/>')
        body.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
        body.append(f'<line x1="640" y1="{legend_y}" x2="680" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="688" y="{legend_y+4}">{label}</text>')
        legend_y += 22
    body.append(f'<text x="{width/2}" y="540" text-anchor="middle">t</text>')
    body.append(
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})">{metric}</text>'
    )
    save_svg(path, "\n".join(body), width, height)


def boxplot_multi(path: pathlib.Path, title: str, groups: list[tuple[str, str, list[float]]]) -> None:
    vals = [v for _, _, g in groups for v in g if math.isfinite(v)]
    if not vals:
        return
    width, height = 920, 560
    left, right, top, bottom = 70, 880, 50, 470
    ylo, yhi = min(vals), max(vals)
    pad = 0.10 * (yhi - ylo if yhi > ylo else 1.0)
    ylo -= pad
    yhi += pad
    step = (right - left) / max(len(groups), 1)
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>']
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    for idx, (backend, label, g) in enumerate(groups):
        series = sorted(v for v in g if math.isfinite(v))
        if not series:
            continue
        x = left + step * (idx + 0.5)
        q1, med, q3 = quantile(series, 0.25), quantile(series, 0.50), quantile(series, 0.75)
        lo, hi = min(series), max(series)
        yq1, ymed, yq3 = [scale(v, ylo, yhi, bottom, top) for v in (q1, med, q3)]
        ylo_w, yhi_w = scale(lo, ylo, yhi, bottom, top), scale(hi, ylo, yhi, bottom, top)
        color = BACKEND_COLOR[backend]
        body.append(f'<line x1="{x:.1f}" y1="{ylo_w:.1f}" x2="{x:.1f}" y2="{yhi_w:.1f}" stroke="#333"/>')
        body.append(
            f'<rect x="{x-45:.1f}" y="{yq3:.1f}" width="90" height="{yq1-yq3:.1f}" fill="{color}" opacity="0.35" stroke="#333"/>'
        )
        body.append(f'<line x1="{x-45:.1f}" y1="{ymed:.1f}" x2="{x+45:.1f}" y2="{ymed:.1f}" stroke="#333" stroke-width="2"/>')
        body.append(f'<text x="{x:.1f}" y="500" text-anchor="middle">{label}</text>')
    save_svg(path, "\n".join(body), width, height)


def scatter_plot(path: pathlib.Path, title: str, points: list[tuple[float, float, str]], xlabel: str, ylabel: str) -> None:
    pts = [(x, y, b) for x, y, b in points if math.isfinite(x) and math.isfinite(y)]
    if not pts:
        return
    width, height = 860, 560
    left, right, top, bottom = 80, 820, 50, 490
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xpad = 0.08 * (xhi - xlo if xhi > xlo else 1.0)
    ypad = 0.08 * (yhi - ylo if yhi > ylo else 1.0)
    xlo -= xpad
    xhi += xpad
    ylo -= ypad
    yhi += ypad
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>']
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    for x, y, backend in pts:
        cx = scale(x, xlo, xhi, left, right)
        cy = scale(y, ylo, yhi, bottom, top)
        body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{BACKEND_COLOR.get(backend, "#444")}" opacity="0.72"/>')
    body.append(f'<text x="{width/2}" y="540" text-anchor="middle">{xlabel}</text>')
    body.append(
        f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})">{ylabel}</text>'
    )
    save_svg(path, "\n".join(body), width, height)


def order_plot(path: pathlib.Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    width, height = 920, 560
    left, right, top, bottom = 80, 880, 50, 490
    series = {
        "kh_linear": [(float(r["n"]), float(r["linear_rel_l2"])) for r in rows],
        "kh_cubic": [(float(r["n"]), float(r["cubic_rel_l2"])) for r in rows],
        "kh_logk_cubic": [(float(r["n"]), float(r["logk_rel_l2"])) for r in rows],
    }
    xs = [x for pts in series.values() for x, _ in pts]
    ys = [y for pts in series.values() for _, y in pts if y > 0]
    if not ys:
        return
    xlo, xhi = min(xs), max(xs)
    ly = [math.log10(y) for y in ys]
    ylo, yhi = min(ly), max(ly)
    pad = 0.08 * (yhi - ylo if yhi > ylo else 1.0)
    ylo -= pad
    yhi += pad
    body = ['<text x="460" y="28" text-anchor="middle" class="title">Manufactured Order Test</text>']
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    for backend, pts in series.items():
        coords = " ".join(
            f"{scale(x, xlo, xhi, left, right):.1f},{scale(math.log10(y), ylo, yhi, bottom, top):.1f}"
            for x, y in pts
        )
        body.append(f'<polyline points="{coords}" fill="none" stroke="{BACKEND_COLOR[backend]}" stroke-width="2"/>')
    body.append(f'<text x="{width/2}" y="540" text-anchor="middle">N</text>')
    body.append('<text x="20" y="280" text-anchor="middle" transform="rotate(-90 20 280)">log10 relative L2 error</text>')
    save_svg(path, "\n".join(body), width, height)


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("artifacts/kh_higher_order")
    summary_dir = root / "summary"
    plots_dir = root / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records, failures, series_by_key = discover_runs(root)
    if failures:
        write_csv(
            summary_dir / "failed_or_incomplete_seeds.csv",
            failures,
            ["phase", "seed", "backend", "status", "missing_files"],
        )
    else:
        write_csv(summary_dir / "failed_or_incomplete_seeds.csv", [], ["phase", "seed", "backend", "status", "missing_files"])

    ensemble_rows, field_rows, transport_rows, runtime_rows, k_rows = summarize_by_phase_backend(records)
    write_csv(
        summary_dir / "ensemble_summary.csv",
        ensemble_rows,
        [
            "phase",
            "backend",
            "backend_label",
            "n_seeds",
            "alpha_L_mean",
            "alpha_L_ci95",
            "alpha_T1_mean",
            "alpha_T1_ci95",
            "alpha_T2_mean",
            "alpha_T2_ci95",
            "alpha_Tmean_mean",
            "alpha_Tmean_ci95",
            "helicity_norm_mean",
            "div_abs_mean",
            "runtime_mean",
            "problematic_total",
        ],
    )
    write_csv(
        summary_dir / "field_diagnostics_summary.csv",
        field_rows,
        [
            "phase",
            "backend",
            "backend_label",
            "n_seeds",
            "speed_mean",
            "speed_max",
            "div_abs_mean",
            "div_abs_max",
            "curl_mag_mean",
            "helicity_mean",
            "helicity_abs_mean",
            "helicity_norm_mean",
            "helicity_norm_std_mean",
            "helicity_norm_p50_mean",
            "helicity_norm_p95_mean",
        ],
    )
    write_csv(
        summary_dir / "transport_diagnostics_summary.csv",
        transport_rows,
        [
            "phase",
            "backend",
            "backend_label",
            "n_seeds",
            "alpha_L_mean",
            "alpha_T1_mean",
            "alpha_T2_mean",
            "alpha_Tmean_mean",
            "active_mean",
            "problematic_mean",
            "var_x_mean",
            "var_y_mean",
            "var_z_mean",
        ],
    )
    write_csv(
        summary_dir / "k_interpolation_diagnostics.csv",
        k_rows,
        [
            "phase",
            "backend",
            "backend_label",
            "n_seeds",
            "k_interp_min_mean",
            "k_interp_min_global",
            "k_interp_max_mean",
            "k_interp_max_global",
            "k_interp_mean_mean",
            "k_interp_nonpositive_total",
            "k_interp_clamped_total",
            "logk_interp_min_global",
            "logk_interp_max_global",
        ],
    )
    write_csv(
        summary_dir / "runtime_summary.csv",
        runtime_rows,
        [
            "phase",
            "backend",
            "backend_label",
            "n_seeds",
            "runtime_mean",
            "runtime_ci95",
            "runtime_min",
            "runtime_max",
        ],
    )

    for filename, a, b in PAIR_SPECS:
        rows = paired_rows(records, a, b)
        write_csv(
            summary_dir / filename,
            rows,
            [
                "phase",
                "seed",
                "backend_a",
                "backend_b",
                "alpha_Tmean_a",
                "alpha_Tmean_b",
                "delta_alpha_Tmean",
                "alpha_L_a",
                "alpha_L_b",
                "delta_alpha_L",
                "helicity_norm_mean_a",
                "helicity_norm_mean_b",
                "delta_helicity_norm_mean",
                "div_abs_mean_a",
                "div_abs_mean_b",
                "delta_div_abs_mean",
                "runtime_seconds_a",
                "runtime_seconds_b",
                "delta_runtime_seconds",
                "problematic_a",
                "problematic_b",
                "delta_problematic",
                "rel_l2_velocity_diff_b",
                "vector_correlation_b",
            ],
        )

    phases = sorted({str(r["phase"]) for r in records})
    focus_phase = "gaussian_smooth" if "gaussian_smooth" in phases else (phases[0] if phases else "")
    curve_rows = aggregate_curves(series_by_key, focus_phase) if focus_phase else []
    if curve_rows:
        write_csv(
            summary_dir / "alpha_timeseries_mean_ci.csv",
            curve_rows,
            [
                "phase",
                "backend",
                "t",
                "n",
                "alpha_L_mean",
                "alpha_L_ci95",
                "alpha_L_lower",
                "alpha_L_upper",
                "alpha_T1_mean",
                "alpha_T1_ci95",
                "alpha_T1_lower",
                "alpha_T1_upper",
                "alpha_T2_mean",
                "alpha_T2_ci95",
                "alpha_T2_lower",
                "alpha_T2_upper",
                "alpha_Tmean_mean",
                "alpha_Tmean_ci95",
                "alpha_Tmean_lower",
                "alpha_Tmean_upper",
            ],
        )
        line_plot_multi(plots_dir / "alpha_L_mean_ci.svg", f"alpha_L(t) — {focus_phase}", curve_rows, "alpha_L")
        line_plot_multi(plots_dir / "alpha_T1_mean_ci.svg", f"alpha_T1(t) — {focus_phase}", curve_rows, "alpha_T1")
        line_plot_multi(plots_dir / "alpha_T2_mean_ci.svg", f"alpha_T2(t) — {focus_phase}", curve_rows, "alpha_T2")
        line_plot_multi(plots_dir / "alpha_T_mean_ci.svg", f"alpha_T_mean(t) — {focus_phase}", curve_rows, "alpha_Tmean")

    focus_records = [r for r in records if r["phase"] == focus_phase]
    if focus_records:
        by_backend = {b: [r for r in focus_records if r["backend"] == b] for b in BACKEND_IDS}
        boxplot_multi(
            plots_dir / "alpha_T_mean_final_boxplot.svg",
            f"Final alpha_T_mean by backend — {focus_phase}",
            [(b, BACKEND_LABEL[b], [safe_metric(r, "alpha_Tmean") for r in by_backend[b]]) for b in BACKEND_IDS],
        )
        boxplot_multi(
            plots_dir / "helicity_norm_boxplot.svg",
            f"Mean normalized helicity — {focus_phase}",
            [(b, BACKEND_LABEL[b], [safe_metric(r, "helicity_norm_mean") for r in by_backend[b]]) for b in BACKEND_IDS],
        )
        boxplot_multi(
            plots_dir / "divergence_boxplot.svg",
            f"Mean absolute divergence — {focus_phase}",
            [(b, BACKEND_LABEL[b], [safe_metric(r, "div_abs_mean") for r in by_backend[b]]) for b in BACKEND_IDS],
        )
        boxplot_multi(
            plots_dir / "runtime_comparison.svg",
            f"Runtime by backend — {focus_phase}",
            [(b, BACKEND_LABEL[b], [safe_metric(r, "runtime_seconds") for r in by_backend[b]]) for b in BACKEND_IDS],
        )
        scatter_plot(
            plots_dir / "helicity_vs_alphaT.svg",
            f"Helicity vs alpha_T_mean — {focus_phase}",
            [(safe_metric(r, "helicity_norm_mean"), safe_metric(r, "alpha_Tmean"), str(r["backend"])) for r in focus_records],
            "mean normalized helicity",
            "final alpha_T_mean",
        )
        scatter_plot(
            plots_dir / "divergence_vs_alphaT.svg",
            f"Divergence vs alpha_T_mean — {focus_phase}",
            [(safe_metric(r, "div_abs_mean"), safe_metric(r, "alpha_Tmean"), str(r["backend"])) for r in focus_records],
            "mean |div q|",
            "final alpha_T_mean",
        )
        face_lookup = {(int(r["seed"]), str(r["backend"])): r for r in focus_records}
        delta_div_points = []
        vel_diff_points = []
        for backend in ("kh_linear", "kh_cubic", "kh_logk_cubic"):
            for r in by_backend[backend]:
                face = face_lookup.get((int(r["seed"]), "face"))
                if not face:
                    continue
                delta_alpha = safe_metric(r, "alpha_Tmean") - safe_metric(face, "alpha_Tmean")
                delta_div = safe_metric(r, "div_abs_mean") - safe_metric(face, "div_abs_mean")
                delta_div_points.append((delta_div, delta_alpha, backend))
                vel_diff_points.append((safe_metric(r, "rel_l2_velocity_diff"), delta_alpha, backend))
        scatter_plot(
            plots_dir / "delta_divergence_vs_delta_alphaT.svg",
            f"Delta divergence vs delta alpha_T_mean — {focus_phase}",
            delta_div_points,
            "delta mean |div q| vs FACE",
            "delta final alpha_T_mean vs FACE",
        )
        scatter_plot(
            plots_dir / "velocity_diff_vs_delta_alphaT.svg",
            f"Velocity diff vs delta alpha_T_mean — {focus_phase}",
            vel_diff_points,
            "relative velocity L2 difference vs FACE",
            "delta final alpha_T_mean vs FACE",
        )

    order_rows = read_csv(summary_dir / "manufactured_order_tests.csv")
    order_plot(plots_dir / "manufactured_order_error.svg", order_rows)

    print(f"Wrote summaries to {summary_dir}")
    print(f"Wrote plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
