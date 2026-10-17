import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 获取当前脚本所在目录的绝对路径
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# 这里的源文件对应你刚才的 grid_builder 逻辑
sources = [
    os.path.join("src", "bindings.cpp"),
    os.path.join("src", "grid_builder.cu"),
    os.path.join("src", "grid_core.cu"),
]

setup(
    name="grid_mapping",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="grid_mapping._C",
            sources=sources,
            include_dirs=[os.path.join(THIS_DIR, "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo", # 生成行信息，方便 Nsight Compute 调优
                    "-Xcompiler", "-fPIC",
                    # 处理不同环境下的 Thrust/CUB 版本冲突
                    "-DTHRUST_IGNORE_CUB_VERSION_CHECK" 
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)