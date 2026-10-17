import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

setup(
    name="slice_adapted_3dgs_renderer",
    version="0.1.0",
    description="Naive CUDA additive splat backend for slice_adapted_3dgs.",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="slice_adapted_3dgs_renderer._C",
            sources=[
                os.path.join("src", "bindings.cpp"),
                os.path.join("src", "render.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xcompiler",
                    "-fPIC",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

