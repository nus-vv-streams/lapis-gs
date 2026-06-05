#
# Top-down layered training pipeline (LapisGS regime, L3GS-style construction).
#
# Bottom-up LapisGS (train_full_pipeline.py) trains coarsest-first and *adds*
# fresh Gaussians per finer level. This top-down variant instead:
#   1. trains ONE full-detail model at res1 (normal 3DGS, densification on);
#   2. scores every Gaussian by L3GS importance, sorts, and partitions the top
#      n_layers*layer_size into buckets (layer_1 = most important);
#   3. builds layers coarse->fine: layer k uses bucket k as its trainable splats
#      on top of a frozen cumulative base, fine-tuned at resolution 2^(n_layers-k).
#
# The per-layer regime is LapisGS's (frozen base with dynamic ancestor opacity)
# but with densification disabled, so layer k has exactly k*layer_size splats and
# the coarse prefix stays nested for streaming.
#

import os
import sys
from argparse import ArgumentParser


def run(cmd):
    print("\n+ " + cmd, flush=True)
    rc = os.system(cmd)
    if rc != 0:
        raise SystemExit(f"Command failed (exit {rc}): {cmd}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Top-down layered training pipeline.")
    parser.add_argument('--model_base', type=str, required=True, help="Path to the model root directory")
    parser.add_argument('--dataset_base', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--scene', type=str, required=True, help="Name of the scene")
    parser.add_argument('--method', type=str, default="lapis_topdown", help="Output method subfolder name")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of LOD layers (data folders must exist for res 2^(n_layers-k))")
    parser.add_argument('--layer_size', type=int, default=45000, help="Gaussians per layer (d)")
    parser.add_argument('--full_iterations', type=int, default=30000, help="Iterations for the full res1 pretrain")
    parser.add_argument('--layer_iterations', type=int, default=30000, help="Iterations for each layer fine-tune")
    parser.add_argument('--v_pow', type=float, default=0.1, help="Volume-weighting exponent in importance score")
    parser.add_argument('--lambda_dssim', type=float, default=0.2)
    parser.add_argument('--no_dynamic_opacity', action='store_true', help="Fully freeze ancestor layers (L3GS regime) instead of LapisGS dynamic-opacity ancestors")
    parser.add_argument('-w', '--white_background', action='store_true', help="Pass -w to train/score (NeRF-synthetic scenes)")
    args = parser.parse_args(sys.argv[1:])

    train_bin = "train.py"
    score_bin = "score_sort_partition.py"
    N = args.n_layers
    wbg = " -w" if args.white_background else ""

    method_dir = os.path.join(args.model_base, args.dataset_name, args.scene, args.method)
    os.makedirs(method_dir, exist_ok=True)

    def source_for_res(res):
        return os.path.join(args.dataset_base, args.dataset_name, args.scene, f"{args.scene}_res{res}")

    # ---- Step 1: full-detail pretrain at res1 (densification ON) ----
    full_dir = os.path.join(method_dir, f"{args.scene}_full_res1")
    os.makedirs(full_dir, exist_ok=True)
    run(f"python {train_bin} -s {source_for_res(1)} -m {full_dir} --data_device cuda "
        f"--lambda_dssim {args.lambda_dssim} --iterations {args.full_iterations}{wbg}")

    # ---- Step 2: score + sort + partition into buckets ----
    buckets_dir = os.path.join(method_dir, "buckets")
    run(f"python {score_bin} -m {full_dir} --out_dir {buckets_dir} "
        f"--n_layers {N} --layer_size {args.layer_size} --iteration {args.full_iterations} "
        f"--v_pow {args.v_pow}{wbg}")

    # ---- Step 3: build layers coarse -> fine ----
    prev_dir = None
    for k in range(1, N + 1):
        res = 2 ** (N - k)  # k=1 -> coarsest (largest downsample); k=N -> res1
        model_dir = os.path.join(method_dir, f"L{k}_res{res}")
        os.makedirs(model_dir, exist_ok=True)
        band = os.path.join(buckets_dir, f"layer_{k}.ply")

        cmd = (f"python {train_bin} -s {source_for_res(res)} -m {model_dir} --data_device cuda "
               f"--lambda_dssim {args.lambda_dssim} --iterations {args.layer_iterations} "
               f"--init_gs_path {band} --no_densify{wbg}")
        if k > 1:
            foundation = os.path.join(prev_dir, "point_cloud",
                                      f"iteration_{args.layer_iterations}", "point_cloud.ply")
            cmd += f" --foundation_gs_path {foundation}"
            if not args.no_dynamic_opacity:
                cmd += " --dynamic_opacity"
        run(cmd)
        prev_dir = model_dir

    print("\nTop-down layered training complete.")
    print(f"Layer checkpoints under: {method_dir}")
