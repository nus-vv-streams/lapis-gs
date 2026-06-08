#
# Top-down LOD construction helpers for LapisGS.
#
# These mirror L3GS / LightGaussian "global significance" pruning, ported so it
# can run against LapisGS's own GaussianModel / Scene / renderer. The importance
# of a Gaussian is the sum of its blending opacity over every pixel it touches,
# accumulated across all training views, then weighted by a (volume)^v_pow term.
#

import torch
from torch import nn

from gaussian_renderer import count_render
from scene import GaussianModel


def calculate_v_imp_score(gaussians, imp_list, v_pow):
    """Volume-weighted importance score (LightGaussian GSG variant).

    :param gaussians: GaussianModel (exposes get_scaling).
    :param imp_list: per-Gaussian accumulated opacity importance (1-D tensor).
    :param v_pow: power applied to the (volume / 90th-percentile-volume) ratio.
    :return: v_list, the adjusted per-Gaussian importance used for ranking.
    """
    volume = torch.prod(gaussians.get_scaling, dim=1)
    index = int(len(volume) * 0.9)
    sorted_volume, _ = torch.sort(volume, descending=True)
    kth_percent_largest = sorted_volume[index]
    v_list = torch.pow(volume / kth_percent_largest, v_pow)
    v_list = v_list * imp_list
    return v_list


def prune_list(gaussians, scene, pipe, background):
    """Accumulate per-Gaussian hit-count and opacity-importance over all train cameras.

    Returns (gaussian_list, imp_list), each a 1-D tensor of length #Gaussians.
    """
    viewpoint_stack = scene.getTrainCameras().copy()
    gaussian_list, imp_list = None, None

    viewpoint_cam = viewpoint_stack.pop()
    render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
    gaussian_list = render_pkg["gaussians_count"].detach()
    imp_list = render_pkg["important_score"].detach()

    for _ in range(len(viewpoint_stack)):
        viewpoint_cam = viewpoint_stack.pop()
        render_pkg = count_render(viewpoint_cam, gaussians, pipe, background)
        gaussian_list += render_pkg["gaussians_count"].detach()
        imp_list += render_pkg["important_score"].detach()

    return gaussian_list, imp_list


def extract_band(source: GaussianModel, start: int, end: int) -> GaussianModel:
    """Return a new GaussianModel holding rows [start:end) of `source`.

    Used to carve an importance-sorted full model into per-layer buckets. The
    returned model is independent (cloned tensors) and saveable via save_ply.
    """
    band = GaussianModel(source.max_sh_degree)
    band._xyz = nn.Parameter(source._xyz[start:end].detach().clone().requires_grad_(True))
    band._features_dc = nn.Parameter(source._features_dc[start:end].detach().clone().requires_grad_(True))
    band._features_rest = nn.Parameter(source._features_rest[start:end].detach().clone().requires_grad_(True))
    band._opacity = nn.Parameter(source._opacity[start:end].detach().clone().requires_grad_(True))
    band._scaling = nn.Parameter(source._scaling[start:end].detach().clone().requires_grad_(True))
    band._rotation = nn.Parameter(source._rotation[start:end].detach().clone().requires_grad_(True))
    band.active_sh_degree = source.active_sh_degree
    band.max_radii2D = torch.zeros((max(end - start, 0),), device=source._xyz.device)
    return band
