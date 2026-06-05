#!/usr/bin/env python3
"""
Plot PSNR / SSIM / LPIPS curves across multiple LapisGS variants per scene.

Reads each LOD layer's ``results.json`` (produced by ``metrics.py``). Auto-
detects the two naming conventions the codebase uses for a layer directory at
resolution R = 2^(n_layers-1-k):

    top-down:   {model_base}/{dataset}/{scene}/{method}/L{k}_res{R}/
    bottom-up:  {model_base}/{dataset}/{scene}/{method}/{scene}_res{R}/

so a single invocation can mix top-down and bottom-up methods.

Produces one figure per scene with three subplots (one per metric) and one
line per method. Also writes a tidy CSV with the raw numbers.

Example
-------
    python scripts/plot_opacity_comparison.py \\
        --model_base /home/e/e0686126/gs/model \\
        --dataset db \\
        --scenes playroom drjohnson \\
        --methods topdown_opacity topdown_no_opacity \\
                  bottomup_opacity bottomup_no_opacity \\
        --out_dir ./plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

METRICS = ("PSNR", "SSIM", "LPIPS")
DEFAULT_METHODS = (
    "topdown_opacity",
    "topdown_no_opacity",
    "bottomup_opacity",
    "bottomup_no_opacity",
)


def candidate_layer_dirs(method_dir: Path, scene: str, k: int, res: int) -> list[Path]:
    """Both naming conventions used in this repo."""
    return [
        method_dir / f"L{k}_res{res}",       # top-down (train_full_pipeline_topdown.py)
        method_dir / f"{scene}_res{res}",    # bottom-up (train_full_pipeline.py)
    ]


def find_layer_dir(method_dir: Path, scene: str, k: int, res: int) -> Path | None:
    """Return the first candidate dir that contains results.json, else None."""
    for cand in candidate_layer_dirs(method_dir, scene, k, res):
        if (cand / "results.json").is_file():
            return cand
    return None


def read_metrics(layer_dir: Path) -> dict | None:
    """Return {PSNR,SSIM,LPIPS} for the latest 'ours_*' in results.json, or None."""
    results_path = layer_dir / "results.json"
    if not results_path.is_file():
        return None
    with open(results_path) as f:
        data = json.load(f)
    if not data:
        return None
    # results.json: {"<model_path>": {"ours_<iter>": {"SSIM":..,"PSNR":..,"LPIPS":..}}}
    outer = next(iter(data.values()))
    if not outer:
        return None
    # Pick the highest iteration if multiple exist.
    def _iter_num(key: str) -> int:
        tail = key.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0
    inner_key = max(outer.keys(), key=_iter_num)
    inner = outer[inner_key]
    return {m: inner.get(m) for m in METRICS}


def collect(model_base: Path, dataset: str, scenes: list[str],
            methods: list[str], n_layers: int):
    """Return (flat_rows, curves[scene][method][metric] -> list of length n_layers)."""
    flat_rows: list[dict] = []
    curves: dict[str, dict[str, dict[str, list]]] = {}
    for scene in scenes:
        curves[scene] = {
            m: {metric: [None] * n_layers for metric in METRICS}
            for m in methods
        }
        for method in methods:
            method_dir = model_base / dataset / scene / method
            for k in range(n_layers):
                res = 2 ** (n_layers - 1 - k)
                ldir = find_layer_dir(method_dir, scene, k, res)
                row = {"dataset": dataset, "scene": scene, "method": method,
                       "k": k, "res": res,
                       "layer_dir": str(ldir) if ldir else ""}
                if ldir is None:
                    cands = [str(c) for c in candidate_layer_dirs(method_dir, scene, k, res)]
                    print(f"[warn] no results.json found for {method}/L{k} (tried: {cands})")
                    row.update({m: None for m in METRICS})
                else:
                    vals = read_metrics(ldir)
                    if vals is None:
                        print(f"[warn] empty results.json: {ldir}")
                        row.update({m: None for m in METRICS})
                    else:
                        row.update(vals)
                        for metric in METRICS:
                            curves[scene][method][metric][k] = vals[metric]
                flat_rows.append(row)
    return flat_rows, curves


def plot_per_scene(curves, methods: list[str], n_layers: int,
                   out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    xs = list(range(n_layers))
    xticks = [f"L{k}\nres{2 ** (n_layers - 1 - k)}" for k in xs]
    # LPIPS is "lower is better"; annotate via title.
    metric_dir = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓"}

    for scene, by_method in curves.items():
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
        if len(METRICS) == 1:
            axes = [axes]
        for ax, metric in zip(axes, METRICS):
            plotted_any = False
            for method in methods:
                ys = by_method[method][metric]
                if all(y is None for y in ys):
                    continue
                ax.plot(xs, ys, marker="o", linewidth=2, label=method)
                plotted_any = True
            ax.set_xticks(xs)
            ax.set_xticklabels(xticks)
            ax.set_xlabel("LOD")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} ({metric_dir[metric]} = better)")
            ax.grid(True, linestyle="--", alpha=0.5)
            if plotted_any:
                ax.legend()
        fig.suptitle(f"{dataset}/{scene} — dynamic-opacity comparison", fontsize=12)
        fig.tight_layout()
        fp = out_dir / f"compare_{dataset}_{scene}.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fp}")


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_base", required=True, type=Path)
    p.add_argument("--dataset", required=True)
    p.add_argument("--scenes", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                   help=f"Method dirnames to compare (default: {' '.join(DEFAULT_METHODS)})")
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--out_dir", type=Path, default=Path("./plots"))
    args = p.parse_args()

    rows, curves = collect(args.model_base, args.dataset, args.scenes,
                           args.methods, args.n_layers)
    write_csv(rows, args.out_dir / f"metrics_{args.dataset}.csv")
    plot_per_scene(curves, args.methods, args.n_layers, args.out_dir, args.dataset)


if __name__ == "__main__":
    main()
