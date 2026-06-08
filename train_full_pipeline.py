import os
import sys
from argparse import ArgumentParser


if __name__ == "__main__":

    parser = ArgumentParser(description="Bottom-up LapisGS training pipeline")
    parser.add_argument('--model_base', type=str, required=True, help="Path to the model root directory")
    parser.add_argument('--dataset_base', type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--scene', type=str, required=True, help="Name of the scene")
    # --method is the OUTPUT FOLDER name (any string). For backward compatibility, if
    # it's literally "lapis" or "freeze" and --regime is not given, --regime is taken
    # from --method. Otherwise --regime must be set explicitly.
    parser.add_argument('--method', type=str, required=True,
                        help="Output method subfolder name (any string).")
    parser.add_argument('--regime', type=str, default=None, choices=["lapis", "freeze"],
                        help="Training regime: 'lapis' (dynamic ancestor opacity) or 'freeze' "
                             "(fully frozen ancestors). Defaults to --method when --method is "
                             "'lapis' or 'freeze'; otherwise required.")
    parser.add_argument('--lambda_dssim', type=float, default=0.2)
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--no_eval', action='store_true',
                        help="Disable train/test split (--eval is on by default).")
    parser.add_argument('-w', '--white_background', action='store_true',
                        help="Pass -w to train.py (NeRF-synthetic scenes).")
    args = parser.parse_args(sys.argv[1:])

    # Resolve regime.
    if args.regime is None:
        if args.method in ("lapis", "freeze"):
            args.regime = args.method
        else:
            parser.error("--regime must be set when --method is not 'lapis' or 'freeze'.")

    model_base = args.model_base
    dataset_base = args.dataset_base
    dataset_name = args.dataset_name
    scene = args.scene
    method = args.method            # folder name
    regime = args.regime            # behavior switch

    resolution_scales = [8, 4, 2, 1]
    train_bin = "train.py"

    # Common passthrough flags.
    wbg = " -w" if args.white_background else ""
    eval_arg = "" if args.no_eval else " --eval"
    common = (f"--data_device cuda --lambda_dssim {args.lambda_dssim} "
              f"--iterations {args.iterations}{wbg}{eval_arg}")

    for idx, resolution in enumerate(resolution_scales):
        print(f"Training {method} (regime={regime}) for {scene} at resolution {resolution}")

        model_dir = os.path.join(model_base, dataset_name, scene, method, f"{scene}_res{resolution}")
        os.makedirs(model_dir, exist_ok=True)
        source_dir = os.path.join(dataset_base, dataset_name, scene, f"{scene}_res{resolution}")

        cmd = f"python {train_bin} -s {source_dir} -m {model_dir} {common}"
        if resolution != 8:
            prev_res = resolution_scales[idx - 1]
            foundation = (f"{model_base}/{dataset_name}/{scene}/{method}/{scene}_res{prev_res}"
                          f"/point_cloud/iteration_{args.iterations}/point_cloud.ply")
            cmd += f" --foundation_gs_path {foundation}"
            if regime == "lapis":
                cmd += " --dynamic_opacity"
            # regime == "freeze": no extra flag (full ancestor freeze is the default
            # when foundation_gs_path is set without --dynamic_opacity).

        print(f"+ {cmd}")
        rc = os.system(cmd)
        if rc != 0:
            raise SystemExit(f"Command failed (exit {rc}): {cmd}")
