#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    # Importance-scoring rasterizer (adds an `f_count` mode that returns a
    # per-Gaussian significance score) used for top-down LOD construction.
    # Packaged as "dgr_l3gs" rather than "diff_gaussian_rasterization" so it
    # coexists in the same env with LapisGS's standard rasterizer, which has an
    # incompatible GaussianRasterizationSettings API. Import it as `dgr_l3gs`
    # (see gaussian_renderer.count_render).
    name="dgr_l3gs",
    packages=['dgr_l3gs'],
    ext_modules=[
        CUDAExtension(
            name="dgr_l3gs._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
