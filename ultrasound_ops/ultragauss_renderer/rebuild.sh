#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip uninstall -y ultragauss_renderer || true
"${PYTHON_BIN}" setup.py clean --all || true
"${PYTHON_BIN}" -m pip install -v -e .

"${PYTHON_BIN}" - <<'PY'
import ultragauss_renderer
from ultragauss_renderer import _C

required = ("MapperForwardCUDA", "ShaderForwardCUDA", "ShaderBackwardCUDA")
missing = [name for name in required if not hasattr(_C, name)]
if missing:
    raise RuntimeError(f"Missing exported symbols: {missing}")
print("ultragauss_renderer smoke check passed.")
PY
