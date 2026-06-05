#!/usr/bin/env python3
"""
Plot PSNR / SSIM / LPIPS vs model size across multiple LapisGS variants per scene.

For each (scene, method, LOD) combination, reads:
  * the metric values from results.json (written by metrics.py)
  * the model size in MB from the saved point_cloud.ply on disk
  * the splat count from the PLY header (recorded in the CSV)

Auto-detects the two naming conventions the codebase uses for a layer
directory at resolution R = 2^(n_layers-1-k):

    top-down:   {model_base}/{dataset}/{scene}/{method}/L{k}_res{R}/
    bottom-up:  {model_base}/{dataset}/{scene}/{method}/{scene}_res{R}/

so a single invocation can mix top-down and bottom-up methods.

Produces one figure per scene with three subplots (one per metric). X axis is
model size (MB), y axis is the metric, one curve per method.

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
    """Return {PSNR,SSIM,LPIPS} for the latest 'ours_*' in results.json, or None.

    metrics.py writes ``full_dict[scene_dir]`` to results.json, so the file
    layout is ``{"ours_<iter>": {"SSIM": .., "PSNR": .., "LPIPS": ..}}`` —
    method/iteration on the outer level, metrics directly underneath.
    """
    results_path = layer_dir / "results.json"
    if not results_path.is_file():
        return None
    with open(results_path) as f:
        data = json.load(f)
    if not data:
        return None

    def _iter_num(key: str) -> int:
        tail = key.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    iter_key = max(data.keys(), key=_iter_num)
    inner = data[iter_key]
    if not isinstance(inner, dict):
        return None
    return {m: inner.get(m) for m in METRICS}


def _read_splat_count(ply_path: Path) -> int | None:
    """Read 'element vertex N' from a PLY header without loading the binary body."""
    try:
        with open(ply_path, "rb") as f:
            for _ in range(200):  # header guard
                line = f.readline()
                if not line:
                    break
                s = line.decode("ascii", errors="replace").strip()
                if s.startswith("element vertex "):
                    return int(s.split()[-1])
                if s == "end_header":
                    break
    except OSError:
        pass
    return None


def find_point_cloud(layer_dir: Path) -> Path | None:
    """Locate the saved point_cloud.ply for the latest training iteration."""
    pc_root = layer_dir / "point_cloud"
    if not pc_root.is_dir():
        return None
    iters = []
    for d in pc_root.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            tail = d.name.split("_", 1)[1]
            if tail.isdigit():
                iters.append((int(tail), d))
    if not iters:
        return None
    iters.sort()
    cand = iters[-1][1] / "point_cloud.ply"
    return cand if cand.is_file() else None


def collect(model_base: Path, dataset: str, scenes: list[str],
            methods: list[str], n_layers: int):
    """Return (flat_rows, curves).

    curves[scene][method] is a list of dicts (one per LOD that produced data),
    each with keys {k, res, size_mb, n_splats, PSNR, SSIM, LPIPS}.
    """
    flat_rows: list[dict] = []
    curves: dict[str, dict[str, list[dict]]] = {}
    for scene in scenes:
        curves[scene] = {m: [] for m in methods}
        for method in methods:
            method_dir = model_base / dataset / scene / method
            for k in range(n_layers):
                res = 2 ** (n_layers - 1 - k)
                ldir = find_layer_dir(method_dir, scene, k, res)
                row = {"dataset": dataset, "scene": scene, "method": method,
                       "k": k, "res": res,
                       "layer_dir": str(ldir) if ldir else "",
                       "ply_path": "", "ply_bytes": None, "ply_mb": None,
                       "n_splats": None}
                if ldir is None:
                    cands = [str(c) for c in candidate_layer_dirs(method_dir, scene, k, res)]
                    print(f"[warn] no results.json found for {method}/L{k} (tried: {cands})")
                    row.update({m: None for m in METRICS})
                    flat_rows.append(row)
                    continue

                vals = read_metrics(ldir)
                ply = find_point_cloud(ldir)
                if ply is not None:
                    sz = ply.stat().st_size
                    row["ply_path"] = str(ply)
                    row["ply_bytes"] = sz
                    row["ply_mb"] = sz / (1024 * 1024)
                    row["n_splats"] = _read_splat_count(ply)

                if vals is None:
                    print(f"[warn] empty results.json: {ldir}")
                    row.update({m: None for m in METRICS})
                else:
                    row.update(vals)
                    # Need both metric AND model size to plot this point.
                    if row["ply_mb"] is None:
                        print(f"[warn] no PLY found under {ldir}, point will be omitted from curves")
                    else:
                        curves[scene][method].append({
                            "k": k, "res": res,
                            "size_mb": row["ply_mb"],
                            "n_splats": row["n_splats"],
                            **{m: vals[m] for m in METRICS},
                        })
                flat_rows.append(row)
            # Sort the per-method points by model size (smallest first).
            curves[scene][method].sort(key=lambda p: p["size_mb"])
    return flat_rows, curves


def plot_per_scene(curves, methods: list[str], out_dir: Path, dataset: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # LPIPS is "lower is better"; annotate via title.
    metric_dir = {"PSNR": "↑", "SSIM": "↑", "LPIPS": "↓"}

    for scene, by_method in curves.items():
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
        if len(METRICS) == 1:
            axes = [axes]
        for ax, metric in zip(axes, METRICS):
            plotted_any = False
            for method in methods:
                points = [p for p in by_method[method]
                          if p[metric] is not None and p["size_mb"] is not None]
                if not points:
                    continue
                xs = [p["size_mb"] for p in points]
                ys = [p[metric] for p in points]
                line, = ax.plot(xs, ys, marker="o", linewidth=2, label=method)
                # Annotate each marker with its LOD index for orientation.
                for p, x, y in zip(points, xs, ys):
                    ax.annotate(f"L{p['k']}", xy=(x, y),
                                xytext=(4, 4), textcoords="offset points",
                                fontsize=8, color=line.get_color(), alpha=0.7)
                plotted_any = True
            ax.set_xlabel("model size (MB)")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} ({metric_dir[metric]} = better)")
            ax.grid(True, linestyle="--", alpha=0.5)
            if plotted_any:
                ax.legend()
        fig.suptitle(f"{dataset}/{scene} — quality vs model size", fontsize=12)
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
    plot_per_scene(curves, args.methods, args.out_dir, args.dataset)


if __name__ == "__main__":
    main()
