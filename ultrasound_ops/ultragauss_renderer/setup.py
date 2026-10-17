import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_INCLUDE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))

sources = [
    os.path.join("src", "bindings.cpp"),
    os.path.join("src", "mapper", "wrapper_mapper.cu"),
    os.path.join("src", "mapper", "core.cu"),
    os.path.join("src", "shader", "wrapper_shader.cu"),
    os.path.join("src", "shader", "forward.cu"),
    os.path.join("src", "shader", "backward.cu"),
]

setup(
    name="ultragauss_renderer",
    version="0.1.0",
    description="UltraGauss ultrasound renderer extension (mapper + shader).",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="ultragauss_renderer._C",
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
