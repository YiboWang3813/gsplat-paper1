# Paper1 gsplat fork

- Read /home/yibo/AGENTS.md and ../AGENTS.md. Current branch: refactor/native-ultrasound; upstream base: v1.4.0.
- Native ultrasound API: gsplat/ultrasound; kernels: gsplat/cuda/csrc/ultrasound; density control: gsplat/strategy/ultrasound.py. Retired standalone ultrasound_ops sources have been removed.
- Preserve physical units, pose/quaternion conventions, finite-slice/global-path semantics and training budgets.
- Use ../bin/paper1 and the shared runtime. Builds use compatible source/Torch/Toolkit/architecture cache keys and resource locks; never change paper2's source routing or shared packages.
- This server does not use Slurm. Verify an actually idle RTX4090 and hold the physical-device lock before GPU execution. Existing CPU checks do not authorize training.
- Runs and generated binaries stay outside source Git, in ../outputs and shared/cache as appropriate. Docs and graph/table projects belong in ../docs.
- Do not push rewritten history or modify other users' jobs without explicit authorization.
