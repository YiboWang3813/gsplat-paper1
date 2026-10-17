#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN=${PYTHON_BIN:-python}

find . -name "*.so" -delete
rm -rf build *.egg-info
$PYTHON_BIN -m pip install -e . --no-build-isolation -v
