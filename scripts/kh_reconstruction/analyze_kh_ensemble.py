#!/usr/bin/env python3
"""Analyze KH potential reconstruction ensemble outputs.

This script intentionally uses only the Python standard library so the
experiment can be reproduced on a bare WSL/remote environment.
"""

from __future__ import annotations

import csv
import math
import pathlib
import statistics
import sys
from typing import Iterable


BACKENDS = ("face", "kh")
BACKEND_LABEL = {"face": "FACE_TRILINEAR", "kh": "KH_POTENTIAL_RECONSTRUCTION"}


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
    if len(vals) <= 1:
        return 0.0 if vals else math.nan
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


def paired_stats(name: str, face: list[float], kh: list[float]) -> dict[str, float | str | int]:
    pairs = [(f, k) for f, k in zip(face, kh) if math.isfinite(f) and math.isfinite(k)]
    diffs = [k - f for f, k in pairs]
    return {
        "metric": name,
        "n": len(pairs),
        "face_mean": mean(f for f, _ in pairs),
        "kh_mean": mean(k for _, k in pairs),
        "delta_kh_minus_face_mean": mean(diffs),
        "delta_kh_minus_face_ci95": ci95(diffs),
        "delta_kh_minus_face_sd": stdev(diffs),
    }


def load_final_records(root: pathlib.Path) -> list[dict[str, float | str | int]]:
    records: list[dict[str, float | str | int]] = []
    raw = root / "raw"
    for seed in range(100):
        for backend in BACKENDS:
            run = raw / f"seed_{seed:03d}" / backend
            alpha_rows = read_csv(run / "alpha_timeseries.csv")
            field_rows = read_csv(run / "field_diagnostics.csv")
            runtime_rows = read_csv(run / "runtime_diagnostics.csv")
            transport_rows = read_csv(run / "transport_diagnostics.csv")
            comparison_rows = read_csv(run / "velocity_comparison.csv")
            if not alpha_rows:
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
                    "seed": seed,
                    "backend": backend,
                    "alpha_L": fval(alpha, "alpha_x"),
                    "alpha_T1": t1,
                    "alpha_T2": t2,
                    "alpha_Tmean": 0.5 * (t1 + t2),
                    "helicity_norm_mean": fval(field, "helicity_norm_mean"),
                    "div_abs_mean": fval(field, "div_abs_mean"),
                    "runtime_seconds": fval(runtime, "transport_seconds"),
                    "active": fval(transport, "active"),
                    "problematic": fval(transport, "problematic"),
                    "rel_l2_velocity_diff": fval(comparison, "rel_l2_diff"),
                }
            )
    return records


def load_alpha_series(root: pathlib.Path) -> dict[str, list[list[dict[str, float]]]]:
    out: dict[str, list[list[dict[str, float]]]] = {b: [] for b in BACKENDS}
    raw = root / "raw"
    for seed in range(100):
        for backend in BACKENDS:
            rows = read_csv(raw / f"seed_{seed:03d}" / backend / "alpha_timeseries.csv")
            if not rows:
                continue
            out[backend].append(
                [
                    {
                        "t": fval(r, "t"),
                        "alpha_L": fval(r, "alpha_x"),
                        "alpha_T1": fval(r, "alpha_y"),
                        "alpha_T2": fval(r, "alpha_z"),
                    }
                    for r in rows
                ]
            )
    return out


def write_csv(path: pathlib.Path, rows: list[dict[str, object]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<style>text{font-family:Arial,sans-serif;font-size:12px} '
        '.title{font-size:16px;font-weight:bold}.axis{stroke:#333;stroke-width:1} '
        '.grid{stroke:#ddd;stroke-width:1}.face{stroke:#1f77b4;fill:none;stroke-width:2} '
        '.kh{stroke:#d62728;fill:none;stroke-width:2}.band-face{fill:#1f77b4;opacity:.14} '
        '.band-kh{fill:#d62728;opacity:.14}</style>\n'
    )


def save_svg(path: pathlib.Path, body: str, width: int = 900, height: int = 540) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_header(width, height) + body + "</svg>\n")


def scale(v: float, lo: float, hi: float, a: float, b: float) -> float:
    if not math.isfinite(v) or hi == lo:
        return 0.5 * (a + b)
    return a + (v - lo) * (b - a) / (hi - lo)


def line_plot(path: pathlib.Path, title: str, rows: list[dict[str, float]], metric: str) -> None:
    width, height = 900, 540
    left, right, top, bottom = 70, 870, 50, 480
    xs = [r["t"] for r in rows if r["backend"] == "face"]
    vals = []
    for r in rows:
        vals.extend([r[f"{metric}_mean"] - r[f"{metric}_ci95"], r[f"{metric}_mean"] + r[f"{metric}_ci95"]])
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(vals), max(vals)
    pad = 0.08 * (yhi - ylo if yhi > ylo else 1.0)
    ylo -= pad
    yhi += pad

    def pts(backend: str, suffix: str) -> list[tuple[float, float]]:
        return [
            (
                scale(r["t"], xlo, xhi, left, right),
                scale(r[f"{metric}_{suffix}"], ylo, yhi, bottom, top),
            )
            for r in rows
            if r["backend"] == backend
        ]

    body = [f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>']
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + frac * (bottom - top)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')

    for backend, cls, band_cls, label_x in [("face", "face", "band-face", 690), ("kh", "kh", "band-kh", 690)]:
        upper = pts(backend, "upper")
        lower = list(reversed(pts(backend, "lower")))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in upper + lower)
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts(backend, "mean"))
        body.append(f'<polygon points="{poly}" class="{band_cls}"/>')
        body.append(f'<polyline points="{line}" class="{cls}"/>')
    body.append('<line x1="690" y1="70" x2="730" y2="70" class="face"/>')
    body.append('<text x="738" y="74">FACE_TRILINEAR</text>')
    body.append('<line x1="690" y1="94" x2="730" y2="94" class="kh"/>')
    body.append('<text x="738" y="98">KH_POTENTIAL_RECONSTRUCTION</text>')
    body.append(f'<text x="{width/2}" y="520" text-anchor="middle">t</text>')
    body.append(
        f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})">{metric}</text>'
    )
    save_svg(path, "\n".join(body), width, height)


def boxplot(path: pathlib.Path, title: str, groups: list[tuple[str, list[float]]]) -> None:
    width, height = 900, 520
    left, right, top, bottom = 70, 860, 50, 450
    vals = [v for _, g in groups for v in g if math.isfinite(v)]
    ylo, yhi = min(vals), max(vals)
    pad = 0.10 * (yhi - ylo if yhi > ylo else 1.0)
    ylo -= pad
    yhi += pad
    step = (right - left) / len(groups)
    body = [f'<text x="{width/2}" y="28" text-anchor="middle" class="title">{title}</text>']
    body.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    for idx, (label, g) in enumerate(groups):
        vals = sorted(v for v in g if math.isfinite(v))
        x = left + step * (idx + 0.5)
        q1, med, q3 = quantile(vals, 0.25), quantile(vals, 0.50), quantile(vals, 0.75)
        lo, hi = min(vals), max(vals)
        yq1, ymed, yq3 = [scale(v, ylo, yhi, bottom, top) for v in (q1, med, q3)]
        ylo_w, yhi_w = scale(lo, ylo, yhi, bottom, top), scale(hi, ylo, yhi, bottom, top)
        fill = "#9ecae1" if "FACE" in label else "#f4a6a6"
        body.append(f'<line x1="{x:.1f}" y1="{ylo_w:.1f}" x2="{x:.1f}" y2="{yhi_w:.1f}" stroke="#333"/>')
        body.append(f'<rect x="{x-45:.1f}" y="{yq3:.1f}" width="90" height="{yq1-yq3:.1f}" fill="{fill}" stroke="#333"/>')
        body.append(f'<line x1="{x-45:.1f}" y1="{ymed:.1f}" x2="{x+45:.1f}" y2="{ymed:.1f}" stroke="#333" stroke-width="2"/>')
        body.append(f'<text x="{x:.1f}" y="475" text-anchor="middle">{label}</text>')
    save_svg(path, "\n".join(body), width, height)


def scatter(path: pathlib.Path, title: str, points: list[tuple[float, float, str]], xlabel: str, ylabel: str) -> None:
    width, height = 820, 540
    left, right, top, bottom = 80, 780, 50, 470
    xs = [p[0] for p in points if math.isfinite(p[0]) and math.isfinite(p[1])]
    ys = [p[1] for p in points if math.isfinite(p[0]) and math.isfinite(p[1])]
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
    for x, y, group in points:
        color = "#1f77b4" if group == "face" else "#d62728"
        cx = scale(x, xlo, xhi, left, right)
        cy = scale(y, ylo, yhi, bottom, top)
        body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="{color}" opacity="0.72"/>')
    body.append(f'<text x="{width/2}" y="520" text-anchor="middle">{xlabel}</text>')
    body.append(
        f'<text x="20" y="{height/2}" text-anchor="middle" transform="rotate(-90 20 {height/2})">{ylabel}</text>'
    )
    save_svg(path, "\n".join(body), width, height)


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("artifacts/kh_reconstruction")
    summary_dir = root / "summary"
    plots_dir = root / "plots"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    records = load_final_records(root)
    by_backend = {b: [r for r in records if r["backend"] == b] for b in BACKENDS}

    stat_rows: list[dict[str, object]] = []
    for metric in [
        "alpha_L",
        "alpha_T1",
        "alpha_T2",
        "alpha_Tmean",
        "helicity_norm_mean",
        "div_abs_mean",
        "runtime_seconds",
    ]:
        stat_rows.append(
            paired_stats(
                metric,
                [float(r[metric]) for r in by_backend["face"]],
                [float(r[metric]) for r in by_backend["kh"]],
            )
        )
    for backend in BACKENDS:
        stat_rows.append(
            {
                "metric": f"{backend}_alpha_T1_minus_T2",
                "n": len(by_backend[backend]),
                "face_mean": math.nan,
                "kh_mean": math.nan,
                "delta_kh_minus_face_mean": mean(
                    float(r["alpha_T1"]) - float(r["alpha_T2"]) for r in by_backend[backend]
                ),
                "delta_kh_minus_face_ci95": ci95(
                    float(r["alpha_T1"]) - float(r["alpha_T2"]) for r in by_backend[backend]
                ),
                "delta_kh_minus_face_sd": stdev(
                    float(r["alpha_T1"]) - float(r["alpha_T2"]) for r in by_backend[backend]
                ),
            }
        )
    write_csv(
        summary_dir / "kh_statistical_analysis.csv",
        stat_rows,
        [
            "metric",
            "n",
            "face_mean",
            "kh_mean",
            "delta_kh_minus_face_mean",
            "delta_kh_minus_face_ci95",
            "delta_kh_minus_face_sd",
        ],
    )

    series = load_alpha_series(root)
    curve_rows: list[dict[str, object]] = []
    for backend in BACKENDS:
        n = len(series[backend])
        if n == 0:
            continue
        for idx in range(len(series[backend][0])):
            t = series[backend][0][idx]["t"]
            row: dict[str, object] = {"backend": backend, "t": t, "n": n}
            for metric in ("alpha_L", "alpha_T1", "alpha_T2"):
                vals = [s[idx][metric] for s in series[backend] if idx < len(s)]
                row[f"{metric}_mean"] = mean(vals)
                row[f"{metric}_ci95"] = ci95(vals)
                row[f"{metric}_lower"] = row[f"{metric}_mean"] - row[f"{metric}_ci95"]
                row[f"{metric}_upper"] = row[f"{metric}_mean"] + row[f"{metric}_ci95"]
            curve_rows.append(row)
    write_csv(
        summary_dir / "alpha_timeseries_mean_ci.csv",
        curve_rows,
        [
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
        ],
    )

    line_plot(plots_dir / "alpha_L_mean_ci.svg", "alpha_L(t): FACE vs KH", curve_rows, "alpha_L")
    line_plot(plots_dir / "alpha_T1_mean_ci.svg", "alpha_T1(t): FACE vs KH", curve_rows, "alpha_T1")
    line_plot(plots_dir / "alpha_T2_mean_ci.svg", "alpha_T2(t): FACE vs KH", curve_rows, "alpha_T2")
    boxplot(
        plots_dir / "alphaT_final_boxplots.svg",
        "Final transverse alpha by seed",
        [
            ("T1 FACE", [float(r["alpha_T1"]) for r in by_backend["face"]]),
            ("T1 KH", [float(r["alpha_T1"]) for r in by_backend["kh"]]),
            ("T2 FACE", [float(r["alpha_T2"]) for r in by_backend["face"]]),
            ("T2 KH", [float(r["alpha_T2"]) for r in by_backend["kh"]]),
        ],
    )
    boxplot(
        plots_dir / "helicity_norm_boxplot.svg",
        "Mean normalized helicity by seed",
        [
            ("FACE", [float(r["helicity_norm_mean"]) for r in by_backend["face"]]),
            ("KH", [float(r["helicity_norm_mean"]) for r in by_backend["kh"]]),
        ],
    )
    scatter(
        plots_dir / "helicity_vs_alphaT.svg",
        "Helicity vs final mean transverse alpha",
        [
            (float(r["helicity_norm_mean"]), float(r["alpha_Tmean"]), str(r["backend"]))
            for r in records
        ],
        "mean normalized helicity",
        "final (alpha_T1 + alpha_T2)/2",
    )

    paired = {int(r["seed"]): r for r in by_backend["face"]}
    scatter_points = []
    for kh in by_backend["kh"]:
        seed = int(kh["seed"])
        face = paired.get(seed)
        if not face:
            continue
        delta_alpha_t = float(kh["alpha_Tmean"]) - float(face["alpha_Tmean"])
        scatter_points.append((float(kh["rel_l2_velocity_diff"]), delta_alpha_t, "kh"))
    scatter(
        plots_dir / "velocity_diff_vs_delta_alphaT.svg",
        "Velocity difference vs KH-FACE transverse alpha change",
        scatter_points,
        "relative velocity L2 difference",
        "delta final mean transverse alpha",
    )

    key_lines = [
        "# KH ensemble key results",
        "",
        f"n_seeds: {len(by_backend['face'])}",
        "",
    ]
    for row in stat_rows:
        key_lines.append(
            f"- {row['metric']}: face_mean={row['face_mean']}, kh_mean={row['kh_mean']}, "
            f"delta_mean={row['delta_kh_minus_face_mean']} +/- {row['delta_kh_minus_face_ci95']} (95% CI)"
        )
    (summary_dir / "kh_key_results.md").write_text("\n".join(key_lines) + "\n")

    print(f"Wrote analysis CSVs to {summary_dir}")
    print(f"Wrote SVG plots to {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
