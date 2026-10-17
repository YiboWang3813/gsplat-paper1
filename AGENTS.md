# Paper1 gsplat fork

Development branch: `paper1/main`; upstream base: `v1.4.0`.

- `ultrasound_ops/` owns the preserved custom CUDA sources and common headers.
- `gsplat.ultrasound` exposes lazy, explicit backend discovery/loading.
- Preserve physical units, quaternion/pose conventions and finite-slice/global-path semantics.
- Historical compiled binaries live in `../cache/legacy_builds/`, not in Git.
- Build a new ABI separately; never delete or overwrite the frozen reference binary.
- Current workspace rules are `../docs/03_engineering/workspace_rules.md`.
- Run CUDA compilation and GPU verification through Slurm.
