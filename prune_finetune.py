#
# Top-down LOD construction, step 2: prune + recover.
#
# Faithful port of L3GS's prune_finetune_fixnum.py (the released code, not the
# iterative algorithm in the paper text). Loads a pretrained full 3DGS model,
# fine-tunes it, performs a one-shot importance-based prune to a target splat
# count at a configured iteration, then fine-tunes the survivors to recover
# quality. The resulting model has exactly `--target_num` splats and is the
# input to score_sort_partition.py.
#
# Default schedule (matches L3GS):
#   iterations           35_000
#   prune_iterations     [30_001]
#   target_num           180_000  (4 layers x 45_000)
#   prune_type           v_important_score
#   v_pow                0.1
#   ExponentialLR        gamma=0.95, step every 400 iters
#
# Run on a CUDA machine with `dgr_l3gs` installed (see gaussian_renderer.count_render).
#

import os
import sys
import uuid
import numpy as np
import torch
from random import randint
from argparse import ArgumentParser, Namespace
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state
from arguments import ModelParams, OptimizationParams, PipelineParams
from prune import prune_list, calculate_v_imp_score


def _prepare_output(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") or str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))


def training(dataset, opt, pipe, args):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    # Override Scene's SfM init with the pretrained full model PLY.
    if not args.start_pointcloud:
        raise ValueError("--start_pointcloud is required (the pretrained full 3DGS PLY).")
    gaussians.load_ply(args.start_pointcloud)
    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0],), device="cuda")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # L3GS recovery-phase LR decay: ExponentialLR stepping every 400 iters.
    gaussians.scheduler = ExponentialLR(gaussians.optimizer, gamma=0.95)

    viewpoint_stack = None
    ema_loss = 0.0
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Prune+finetune")

    for iteration in range(1, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        if iteration % 400 == 0:
            gaussians.scheduler.step()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 200 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}", "N": gaussians.get_xyz.shape[0]})
                progress_bar.update(200)

            # ---- one-shot prune to target_num at args.prune_iterations[i] ----
            if iteration in args.prune_iterations:
                current_num = gaussians.get_xyz.shape[0]
                print(f"\n[prune] before: {current_num} Gaussians")
                prune_percent = 1.0 - (args.target_num / current_num)
                if prune_percent <= 0:
                    raise RuntimeError(
                        f"target_num={args.target_num} >= current_num={current_num}. "
                        f"Either lower n_layers*layer_size, raise --densify_grad_threshold / "
                        f"--densify_until_iter in step 1 (to grow more splats), or omit "
                        f"--layer_size entirely to skip pruning and equal-split the full model."
                    )
                else:
                    i = args.prune_iterations.index(iteration)
                    _gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
                    if args.prune_type == "important_score":
                        score = imp_list
                    elif args.prune_type == "v_important_score":
                        score = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
                    elif args.prune_type == "max_v_important_score":
                        score = imp_list * torch.max(gaussians.get_scaling, dim=1)[0]
                    elif args.prune_type == "count":
                        score = _gaussian_list
                    elif args.prune_type == "opacity":
                        score = gaussians.get_opacity.detach()
                    else:
                        raise ValueError(f"Unsupported prune_type: {args.prune_type}")
                    decayed_percent = (args.prune_decay ** i) * prune_percent
                    gaussians.prune_gaussians(decayed_percent, score)
                    print(f"[prune] after:  {gaussians.get_xyz.shape[0]} Gaussians "
                          f"(target {args.target_num}, prune_percent={decayed_percent:.4f})")

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                out_dir = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}")
                os.makedirs(out_dir, exist_ok=True)
                gaussians.save_ply(os.path.join(out_dir, "point_cloud.ply"))

    progress_bar.close()

    # Also write imp_score.npz for the final pruned model (sidecar for analysis).
    _gaussian_list, imp_list = prune_list(gaussians, scene, pipe, background)
    v_list = calculate_v_imp_score(gaussians, imp_list, args.v_pow)
    np.savez(os.path.join(args.model_path, "imp_score.npz"), v_list.detach().cpu().numpy())


if __name__ == "__main__":
    parser = ArgumentParser(description="L3GS-style prune + finetune to a fixed Gaussian count.")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--start_pointcloud", type=str, required=True,
                        help="Pretrained full 3DGS PLY to load.")
    parser.add_argument("--target_num", type=int, default=180_000,
                        help="Target Gaussian count after prune.")
    parser.add_argument("--prune_iterations", nargs="+", type=int, default=[30_001],
                        help="Iterations at which to prune to target_num.")
    parser.add_argument("--prune_type", type=str, default="v_important_score",
                        choices=["important_score", "v_important_score",
                                 "max_v_important_score", "count", "opacity"])
    parser.add_argument("--prune_decay", type=float, default=1.0)
    parser.add_argument("--v_pow", type=float, default=0.1)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[35_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Prune+finetune to " + args.model_path)
    safe_state(args.quiet)
    _prepare_output(args)
    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    print("\nPrune+finetune complete.")
