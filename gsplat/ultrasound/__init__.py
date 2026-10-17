"""Preserved ultrasound CUDA backends owned by the Paper1 gsplat fork.

These backends retain their validated physics and historical import names.
They are distinct from gsplat's perspective RGB rasterization API.
"""
from pathlib import Path
import importlib
import sys

_BACKENDS = {
    'ours': ('ours_renderer', 'ours_renderer'),
    'ultragauss': ('ultragauss_renderer', 'ultragauss_renderer'),
    'ultragray': ('ultragray_renderer', 'ultragray_renderer'),
    'gaussianpile': ('gaussianplie_renderer', 'gaussianplie_renderer'),
    'slice_adapted_3dgs': ('slice_adapted_3dgs_renderer', 'slice_adapted_3dgs_renderer'),
    'ultrags_baseline': ('ultrags_baseline_renderer', 'ultrags_baseline_renderer'),
    'ultrags_renderer': ('ultrags_renderer', 'ultrags_renderer'),
    # The historical UltraGS method uses this mapper/shader pair, not the
    # separate ultrags_renderer package. Keep their identities explicit.
    'ellipsoid_mapper': ('old/ellipsoid_mapper', 'ellipsoid_mapper'),
    'ultrags_shader': ('old/ultrags_shader', 'ultrags_shader'),
    'grid_mapping': ('grid_mapping', 'grid_mapping'),
    's2s_shader': ('s2s_shader', 's2s_shader'),
}


def backend_source(name):
    """Return the source directory; reject unknown backend names."""
    if name not in _BACKENDS:
        raise ValueError('Unknown ultrasound backend {!r}; choose {}'.format(name, sorted(_BACKENDS)))
    return Path(__file__).resolve().parents[2]/'ultrasound_ops'/_BACKENDS[name][0]


def load_backend(name):
    """Load a backend compiled for the caller's Python/PyTorch/CUDA ABI.

    No rebuilding or dependency installation occurs implicitly.
    """
    source = backend_source(name)
    package = _BACKENDS[name][1]
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(package)
    loaded = Path(module.__file__).resolve()
    if source.resolve() not in loaded.parents:
        raise RuntimeError('Backend was shadowed by another installation: '+str(loaded))
    return module


__all__ = ['backend_source', 'load_backend']
