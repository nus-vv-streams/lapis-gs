#!/usr/bin/env python3
"""
Write a tiny ``model_size.json`` sidecar capturing the saved point_cloud.ply's
file size and splat count. Run before deleting the PLY so the plot script can
still reconstruct the model-size axis after cleanup.

Output (at ``{model_path}/model_size.json``):
    {
        "ply_path": "<absolute path>",
        "ply_bytes": <int>,
        "n_splats": <int>
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _read_splat_count(ply_path: Path) -> int | None:
    """Read 'element vertex N' from a PLY header without loading the body."""
    try:
        with open(ply_path, "rb") as f:
            for _ in range(200):
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model_path", required=True, type=Path,
                   help="Layer directory containing point_cloud/iteration_*/point_cloud.ply")
    args = p.parse_args()

    layer_dir: Path = args.model_path
    ply = find_point_cloud(layer_dir)
    if ply is None:
        print(f"[record_model_size] no point_cloud.ply found under {layer_dir}", file=sys.stderr)
        return 1

    data = {
        "ply_path": str(ply),
        "ply_bytes": ply.stat().st_size,
        "n_splats": _read_splat_count(ply),
    }
    out = layer_dir / "model_size.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[record_model_size] wrote {out}: {data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
