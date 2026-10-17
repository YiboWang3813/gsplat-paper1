#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"

find . -name "*.so" -delete
rm -rf build *.egg-info
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation -v

"${PYTHON_BIN}" - <<'PY'
import ultrags_baseline_renderer
from ultrags_baseline_renderer import _C

required = ("rasterize_gaussians", "rasterize_gaussians_backward", "mark_visible")
missing = [name for name in required if not hasattr(_C, name)]
if missing:
    raise RuntimeError(f"Missing exported symbols: {missing}")
print("ultrags_baseline_renderer smoke check passed.")
PY
