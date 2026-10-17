import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 1. 路径配置
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_INCLUDE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))

# 2. 源文件列表 
sources = [
    os.path.join("src", "bindings.cpp"),
    os.path.join("src", "wrapper.cu"),
    os.path.join("src", "forward.cu"), 
    os.path.join("src", "backward.cu"), 
]

setup(
    name="s2s_shader",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="s2s_shader._C",
            sources=sources,
            include_dirs=[
                os.path.join(THIS_DIR, "include"), # 本项目的头文件
                COMMON_INCLUDE_DIR,                # <--- 关键：上一级的 common 目录
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xcompiler", "-fPIC",
                    "-DTHRUST_IGNORE_CUB_VERSION_CHECK"
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)