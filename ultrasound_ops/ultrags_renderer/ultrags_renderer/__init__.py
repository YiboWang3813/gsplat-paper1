from .api import UltragsRenderer
from .api_batched import UltragsBatchedRenderer
from .check import should_validate_inputs, validate_batched_render_inputs, validate_render_inputs
from .types import BatchSliceGeometry, SliceGeometry, UltragsRenderConfig

OPTIMIZED_BUILD_TAG = "ultrags_renderer_batched_v1"

__all__ = [
    "UltragsRenderer",
    "UltragsBatchedRenderer",
    "UltragsRenderConfig",
    "SliceGeometry",
    "BatchSliceGeometry",
    "should_validate_inputs",
    "validate_render_inputs",
    "validate_batched_render_inputs",
    "OPTIMIZED_BUILD_TAG",
]
