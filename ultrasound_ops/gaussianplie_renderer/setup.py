import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_INCLUDE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))

sources = [
    os.path.join("src", "bindings.cpp"),
    os.path.join("src", "mapper", "core_tile.cu"),
    os.path.join("src", "mapper", "core_pixel.cu"),
    os.path.join("src", "shader", "forward.cu"),
    os.path.join("src", "shader", "backward.cu"),
    os.path.join("src", "shader", "plane_cond2d.cu"),
    os.path.join("src", "shader", "us_focus.cu"),
    os.path.join("src", "shader", "wrapper_shader.cu"),
]


setup(
    name="gaussianplie_renderer",
    version="0.1.0",
    description="GaussianPile renderer API with GaussianPile reference math and CUDA-backed projected-slice path.",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="gaussianplie_renderer._C",
            sources=sources,
            include_dirs=[
                os.path.join(THIS_DIR, "include"),
                COMMON_INCLUDE_DIR,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xcompiler",
                    "-fPIC",
                    "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
