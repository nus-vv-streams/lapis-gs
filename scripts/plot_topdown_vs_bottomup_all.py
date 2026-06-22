#!/usr/bin/env python3
"""
Plot top-down vs bottom-up across every dataset on a single figure.

Reads one ``metrics_<dataset>.csv`` per dataset (written by
plot_opacity_comparison.py) from a directory, averages metrics across the
scenes within each dataset, and produces a single PNG with:

    rows    = datasets (auto-discovered from filenames in --csv_dir)
    columns = PSNR | SSIM | LPIPS
    curves  = 2 per subplot (top-down vs bottom-up; pass --topdown_method and
              --bottomup_method to choose which methods are compared)

X-axis is model size (MB), per-LOD-averaged across the dataset's scenes.

Example
-------
    python scripts/plot_topdown_vs_bottomup_all.py \\
        --csv_dir ./plots \\
        --out ./plots/comparison_all_datasets.png
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

METRICS = ("PSNR", "SSIM", "LPIPS")
METRIC_BETTER = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓"}

# Canonical row order if these datasets are present (others append alphabetically).
PREFERRED_DATASET_ORDER = ("nerf_synthetic", "360", "tandt", "db")


def parse_float(x):
    if x is None or x == "" or x == "None":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def aggregate(rows, methods):
    """Return curves[method] = list of dicts {k, size_mb, PSNR, SSIM, LPIPS},
    averaged across all scenes present in `rows`, sorted by size_mb."""
    buckets: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["method"] not in methods:
            continue
        try:
            k = int(r["k"])
        except (TypeError, ValueError, KeyError):
            continue
        sz = parse_float(r.get("ply_mb"))
        if sz is not None:
            buckets[(r["method"], k)]["size"].append(sz)
        for metric in METRICS:
            v = parse_float(r.get(metric))
            if v is not None:
                buckets[(r["method"], k)][metric].append(v)

    curves = {m: [] for m in methods}
    for (m, k), vals in buckets.items():
        if not vals["size"]:
            continue
        entry = {"k": k, "size_mb": mean(vals["size"])}
        for metric in METRICS:
            entry[metric] = mean(vals[metric]) if vals[metric] else None
        curves[m].append(entry)
    for m in curves:
        curves[m].sort(key=lambda e: e["size_mb"])
    return curves


def discover_csvs(csv_dir: Path) -> list[tuple[str, Path]]:
    """Find metrics_<dataset>.csv files, return [(dataset, path)] sorted by
    PREFERRED_DATASET_ORDER then alphabetical for the rest."""
    pattern = re.compile(r"^metrics_(.+)\.csv$")
    pairs = []
    for f in csv_dir.iterdir():
        m = pattern.match(f.name)
        if m and f.is_file():
            pairs.append((m.group(1), f))
    # Sort: preferred first (in given order), then unknowns alphabetically.
    preferred_index = {d: i for i, d in enumerate(PREFERRED_DATASET_ORDER)}
    pairs.sort(key=lambda p: (preferred_index.get(p[0], len(PREFERRED_DATASET_ORDER)),
                              p[0]))
    return pairs


def plot_grid(datasets, topdown, bottomup, out_path, suptitle):
    nrows = len(datasets)
    ncols = len(METRICS)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.0 * nrows),
                             squeeze=False)

    methods = [topdown, bottomup]

    for i, (dataset, csv_path) in enumerate(datasets):
        rows = read_rows(csv_path)
        scenes_in_csv = sorted({r["scene"] for r in rows})
        curves = aggregate(rows, methods)

        for j, metric in enumerate(METRICS):
            ax = axes[i][j]
            plotted_any = False
            for m in methods:
                pts = [e for e in curves[m] if e[metric] is not None]
                if not pts:
                    continue
                xs = [e["size_mb"] for e in pts]
                ys = [e[metric] for e in pts]
                line, = ax.plot(xs, ys, marker="o", linewidth=2, label=m)
                for e, x, y in zip(pts, xs, ys):
                    ax.annotate(f"L{e['k']}", xy=(x, y),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=7, color=line.get_color(), alpha=0.7)
                plotted_any = True

            ax.set_xlabel("model size (MB)")
            ax.grid(True, linestyle="--", alpha=0.5)
            if i == 0:
                ax.set_title(f"{metric} ({METRIC_BETTER[metric]} = better)")
            if j == 0:
                # First column shows the dataset name above the metric label.
                n_scenes = len(scenes_in_csv)
                scene_blurb = f"\n(avg of {n_scenes} scene{'s' if n_scenes != 1 else ''})"
                ax.set_ylabel(f"{dataset}{scene_blurb}\n\n{metric}",
                              fontsize=10, fontweight="bold")
            else:
                ax.set_ylabel(metric)
            if plotted_any:
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f"no data for\n{topdown} / {bottomup}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--csv_dir", type=Path, default=Path("./plots"),
                   help="Directory containing metrics_<dataset>.csv files (default: ./plots).")
    p.add_argument("--out", type=Path,
                   default=Path("./plots/comparison_all_datasets.png"))
    p.add_argument("--topdown_method", default="topdown_opacity",
                   help="Method name for the top-down curve (default: topdown_opacity).")
    p.add_argument("--bottomup_method", default="bottomup_opacity",
                   help="Method name for the bottom-up curve (default: bottomup_opacity).")
    p.add_argument("--suptitle",
                   default="Top-down vs bottom-up LapisGS — per-dataset comparison")
    args = p.parse_args()

    if not args.csv_dir.is_dir():
        raise SystemExit(f"CSV directory not found: {args.csv_dir}")

    datasets = discover_csvs(args.csv_dir)
    if not datasets:
        raise SystemExit(f"No metrics_<dataset>.csv files found in {args.csv_dir}")

    print(f"Discovered datasets: {[d for d, _ in datasets]}")
    plot_grid(datasets, args.topdown_method, args.bottomup_method,
              args.out, args.suptitle)


if __name__ == "__main__":
    main()
