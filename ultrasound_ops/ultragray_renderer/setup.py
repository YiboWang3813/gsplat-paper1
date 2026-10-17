import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_INCLUDE_DIR = os.path.abspath(os.path.join(THIS_DIR, "../common"))

# Default to the A100 target used for UltraGray acceptance runs. Additional SMs
# can be requested explicitly because some older PyTorch versions do not parse
# newer architectures such as 8.9 in TORCH_CUDA_ARCH_LIST.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

nvcc_extra_args = [
    "-O3",
    "--use_fast_math",
    "-lineinfo",
    "-Xcompiler",
    "-fPIC",
    "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
]
extra_sms = []
if os.environ.get("ULTRAGRAY_CUDA_SM89", "0") == "1":
    extra_sms.append("89")
extra_sms.extend(
    sm.strip().replace(".", "")
    for sm in os.environ.get("ULTRAGRAY_CUDA_SMS", "").replace(";", ",").split(",")
    if sm.strip()
)
for sm in dict.fromkeys(extra_sms):
    nvcc_extra_args.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")

sources = [
    os.path.join("src", "bindings.cpp"),
    os.path.join("src", "mapper", "core.cu"),
    os.path.join("src", "mapper", "wrapper_mapper.cu"),
    os.path.join("src", "shader", "forward.cu"),
    os.path.join("src", "shader", "backward.cu"),
    os.path.join("src", "shader", "wrapper_shader.cu"),
]

setup(
    name="ultragray_renderer",
    version="0.1.0",
    description="Unified UltraG-Ray renderer operator (mapper + shader).",
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="ultragray_renderer._C",
            sources=sources,
            include_dirs=[
                os.path.join(THIS_DIR, "include"),
                COMMON_INCLUDE_DIR,
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": nvcc_extra_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
