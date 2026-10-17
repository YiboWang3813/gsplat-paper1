#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
python -m pip install -e . --no-build-isolation -v

