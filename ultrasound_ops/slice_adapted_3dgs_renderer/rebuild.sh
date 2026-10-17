#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
rm -rf build slice_adapted_3dgs_renderer.egg-info
python setup.py build_ext --inplace

