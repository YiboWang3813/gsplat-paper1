"""Build the ultrasound extension in a locked cache keyed by sources and ABI."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch


def build_spec(backend="ours") -> tuple[Path, list[Path], dict]:
    from torch.utils.cpp_extension import CUDA_HOME

    if not CUDA_HOME or not os.environ.get("CC") or not os.environ.get("CXX"):
        raise RuntimeError(
            "Select CUDA_HOME, CC and CXX explicitly (use paper1/bin/paper1)."
        )
    base = Path(__file__).resolve().parents[1] / "cuda/csrc/ultrasound"
    if backend not in (
        "ours",
        "ultragauss",
        "ultragray",
        "gaussianpile",
        "slice3dgs",
        "ultrags",
    ):
        raise ValueError(f"Unknown native ultrasound backend: {backend}")
    root = base if backend == "ours" else base / "backends" / backend
    sources = sorted((root / "src").rglob("*.cu")) + sorted(
        (root / "src").glob("*.cpp")
    )
    if backend == "ultrags":
        sources = [
            root / name
            for name in (
                "ext.cpp",
                "rasterize_points.cu",
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
            )
        ]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if backend == "ours" and path.is_relative_to(base / "backends"):
            continue
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    identity = {
        "source_sha256": digest.hexdigest(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "cuda_home": str(Path(CUDA_HOME).resolve()),
        "nvcc": subprocess.check_output(
            [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
        ),
        "cxx": subprocess.check_output([os.environ["CXX"], "--version"], text=True),
        "cc": subprocess.check_output([os.environ["CC"], "--version"], text=True),
        "architecture": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "cflags": ["-O3", "-std=c++17"],
        "cuda_cflags": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
    }
    if backend != "ours":
        identity["backend"] = backend
        common = hashlib.sha256()
        for path in sorted((base / "common").glob("*.cuh")):
            common.update(path.read_bytes())
        identity["common_sha256"] = common.hexdigest()
    if not sources or not identity["architecture"]:
        raise RuntimeError(
            "Missing ultrasound sources or explicit TORCH_CUDA_ARCH_LIST."
        )
    return root, sources, identity


@functools.lru_cache(maxsize=12)
def load_backend(verbose: bool = False, backend="ours"):
    import fcntl

    from torch.utils.cpp_extension import load

    root, sources, identity = build_spec(backend)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = (
        Path(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                "/home/yibo/shared/cache/torch_extensions/paper1",
            )
        )
        / key
    )
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        module = load(
            name="gsplat_ultrasound_" + key,
            sources=[str(p) for p in sources],
            extra_cflags=identity["cflags"],
            extra_cuda_cflags=identity["cuda_cflags"],
            extra_include_paths=[
                str(root / "include"),
                str(root),
                str(root / "third_party/glm"),
                str(
                    Path(__file__).resolve().parents[1] / "cuda/csrc/ultrasound/common"
                ),
            ],
            build_directory=str(directory),
            verbose=verbose,
        )
        (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        return module


class Backend:
    def __init__(self, name="ours"):
        self.name = name

    def __getattr__(self, name):
        function = getattr(load_backend(backend=self.name), name)

        def call(*args, **kwargs):
            reference = next(
                (value for value in args if isinstance(value, torch.Tensor)), None
            )
            if reference is None or not reference.is_cuda:
                raise ValueError("Ultrasound kernel calls require CUDA tensor inputs.")
            with torch.cuda.device(reference.device):
                return function(*args, **kwargs)

        return call


backend = Backend()
