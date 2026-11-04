# Paper1 gsplat fork

## Hong Kong native refactor (2026-11-04)

- Current policy is /home/yibo/AGENTS.md and ../AGENTS.md; latest instructions
  supersede the historical relocation notes below.
- Work on refactor/native-ultrasound. Native ultrasound Python API lives in
  gsplat/ultrasound, kernels in gsplat/cuda/csrc/ultrasound, density strategy
  in gsplat/strategy/ultrasound.py. No native imports of standalone renderers.
- Historical ultrasound_ops packages are migration references for remaining
  backends; their transfer does not certify native adaptation.
- Builds use a content-addressed cache under shared/cache with a real file lock.
- Use ../bin/paper1 build and an available RTX4090 locally, not eehpc Slurm.
- Never replace paper2's shared gsplat installation or shared Torch/CUDA stack.

Development branch: `paper1/main`; upstream base: `v1.4.0`.

- `ultrasound_ops/` owns the preserved custom CUDA sources and common headers.
- `gsplat.ultrasound` exposes lazy, explicit backend discovery/loading.
- Preserve physical units, quaternion/pose conventions and finite-slice/global-path semantics.
- Historical compiled binaries live in `../cache/legacy_builds/`, not in Git.
- Build a new ABI separately; never delete or overwrite the frozen reference binary.
- Current workspace rules are `../docs/03_engineering/workspace_rules.md`.
- Run CUDA compilation and GPU verification through Slurm.
