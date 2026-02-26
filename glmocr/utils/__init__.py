"""Utility functions module."""

from .image_utils import smart_resize, load_image_to_base64, crop_image_region
from .lock_utils import (
    acquire_conversion_lock,
    release_conversion_lock,
    wait_for_conversion_completion,
)
from .logging import (
    get_logger,
    get_profiler,
    configure_logging,
    set_log_level,
)
from .result_postprocess_utils import (
    find_consecutive_repeat,
    clean_repeated_content,
    clean_formula_number,
)


def __getattr__(name):
    # Lazy imports for layout-only symbols that require opencv-python.
    _viz_names = {"draw_layout_boxes", "save_layout_visualization", "get_colormap"}
    if name in _viz_names:
        from . import visualization_utils

        return getattr(visualization_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "smart_resize",
    "load_image_to_base64",
    "crop_image_region",
    "acquire_conversion_lock",
    "release_conversion_lock",
    "wait_for_conversion_completion",
    "get_logger",
    "get_profiler",
    "configure_logging",
    "set_log_level",
    "draw_layout_boxes",
    "save_layout_visualization",
    "get_colormap",
    "find_consecutive_repeat",
    "clean_repeated_content",
    "clean_formula_number",
]
