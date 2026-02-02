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
from .visualization_utils import (
    draw_layout_boxes,
    save_layout_visualization,
    get_colormap,
)
from .result_postprocess_utils import (
    find_consecutive_repeat,
    clean_repeated_content,
    clean_formula_number,
)

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
