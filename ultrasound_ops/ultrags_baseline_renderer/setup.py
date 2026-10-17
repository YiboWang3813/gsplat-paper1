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

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6")

nvcc_args = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-Xcompiler",
    "-fPIC",
    "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
    "-I" + os.path.join(THIS_DIR, "third_party/glm/"),
]
if os.environ.get("ULTRAGS_BASELINE_CUDA_SM89", "0") == "1":
    nvcc_args.extend(["-gencode=arch=compute_89,code=sm_89"])

setup(
    name="ultrags_baseline_renderer",
    packages=find_packages(),
    version="0.1.0",
    description="UltraGS baseline ultrasound rasterization extension.",
    ext_modules=[
        CUDAExtension(
            name="ultrags_baseline_renderer._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_args,
            },
        )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
