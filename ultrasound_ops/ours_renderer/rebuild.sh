#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

find . -name "*.so" -delete
rm -rf build *.egg-info
"${PYTHON_BIN}" -m pip install -e . --no-build-isolation -v

"${PYTHON_BIN}" - <<'PY'
import ours_renderer
from ours_renderer import _C

required = (
    "ProjectedSliceForwardCUDA",
    "ProjectedSliceBackwardCUDA",
)
missing = [name for name in required if not hasattr(_C, name)]
if missing:
    raise RuntimeError(f"Missing exported symbols: {missing}")
print("ours_renderer smoke check passed.")
PY
