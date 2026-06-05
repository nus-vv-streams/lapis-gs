#
# Top-down LOD construction, step 3 (scoring + partition).
#
# Loads a PRUNED 3DGS model (output of prune_finetune.py, with exactly
# n_layers*layer_size Gaussians), scores every Gaussian by L3GS importance
# (sum of blending opacity over all training views, volume-weighted), sorts the
# model so the most important Gaussians come first, and partitions it into
# n_layers per-layer bucket PLYs of exactly layer_size each.
#
# If the input model's splat count does NOT equal n_layers*layer_size, this
# script warns and falls back to truncating/padding the bands accordingly.
#
# Outputs under --out_dir:
#   sorted_full.ply        the full model reordered by descending importance
#   imp_score.npz          sorted importance scores (1-D, length = #Gaussians)
#   sort_index.npy         original row index in sorted order
#   layer_1.ply ... layer_N.ply   importance buckets (layer_1 = most important)
#
# Run on a CUDA machine with the `dgr_l3gs` rasterizer installed (see
# gaussian_renderer.count_render for the build command).
#

import os
import numpy as np
import torch
from argparse import ArgumentParser

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from prune import prune_list, calculate_v_imp_score, extract_band


def score_sort_partition(dataset, pipe, iteration, out_dir, n_layers, layer_size, v_pow):
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        print(f"[score] computing importance over {len(scene.getTrainCameras())} train cameras "
              f"for {gaussians.get_xyz.shape[0]} Gaussians ...")
        _gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
        v_list = calculate_v_imp_score(gaussians, imp_list, v_pow)

        sort_index = gaussians.sort_gaussians(v_list)
        v_sorted = v_list[sort_index]

        # Save the importance-sorted full model and the score/index sidecars.
        gaussians.save_ply(os.path.join(out_dir, "sorted_full.ply"))
        np.savez(os.path.join(out_dir, "imp_score.npz"), v_sorted.detach().cpu().numpy())
        np.save(os.path.join(out_dir, "sort_index.npy"), sort_index.detach().cpu().numpy())

        M = gaussians.get_xyz.shape[0]

        # Two partition modes:
        #   * layer_size set (>0): L3GS mode. Expect M == n_layers*layer_size
        #     (prune_finetune.py guarantees this). Slice into N bands of exactly
        #     layer_size.
        #   * layer_size unset (None/<=0): equal-split mode. Don't prune; split
        #     all M splats into N near-equal bands (sizes sum exactly to M).
        if layer_size is None or layer_size <= 0:
            base, rem = divmod(M, n_layers)
            sizes = [base + 1] * rem + [base] * (n_layers - rem)
            print(f"[score] equal-split mode: M={M}, n_layers={n_layers} -> band sizes {sizes}")
            boundaries = []
            cursor = 0
            for s in sizes:
                boundaries.append((cursor, cursor + s))
                cursor += s
        else:
            expected = n_layers * layer_size
            if M != expected:
                print(f"[score] WARNING: model has {M} Gaussians, expected n_layers*layer_size="
                      f"{expected}. This script normally runs on the output of prune_finetune.py.")
            boundaries = [((k - 1) * layer_size, min(k * layer_size, M))
                          for k in range(1, n_layers + 1)]

        for k, (start, end) in enumerate(boundaries, start=1):
            if start >= M:
                print(f"[score] layer {k}: no Gaussians left (start={start} >= M={M}); skipping.")
                continue
            band = extract_band(gaussians, start, end)
            band_path = os.path.join(out_dir, f"layer_{k}.ply")
            band.save_ply(band_path)
            print(f"[score] layer {k}: rows [{start}:{end}) -> {band_path} ({end - start} splats)")

        print(f"[score] done. sorted_full + {n_layers} buckets written to {out_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Top-down: score, sort, and partition a trained 3DGS model.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="Which saved iteration of the full model to load.")
    parser.add_argument("--out_dir", required=True, type=str, help="Directory for sorted model + buckets.")
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--layer_size", default=None, type=int,
                        help="Gaussians per layer bucket (d). If unset (or <=0), "
                             "split the full model equally into n_layers bands.")
    parser.add_argument("--v_pow", default=0.1, type=float, help="Volume-weighting exponent in the importance score.")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Scoring " + args.model_path)

    safe_state(args.quiet)
    score_sort_partition(model.extract(args), pipeline.extract(args),
                         args.iteration, args.out_dir, args.n_layers, args.layer_size, args.v_pow)
