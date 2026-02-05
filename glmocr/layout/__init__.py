"""Layout detection module."""

from .base import BaseLayoutDetector

from typing import Optional


_layout_import_error: Optional[BaseException] = None
_layout_import_error_is_dependency: bool = False

try:
    from .layout_detector import PPDocLayoutDetector
except (ModuleNotFoundError, ImportError) as e:  # pragma: no cover
    PPDocLayoutDetector = None  # type: ignore
    _layout_import_error = e
    _layout_import_error_is_dependency = True
except Exception as e:  # pragma: no cover
    # Dependencies may already be installed; importing the detector can still fail
    # due to version incompatibilities or other runtime errors.
    PPDocLayoutDetector = None  # type: ignore
    _layout_import_error = e
    _layout_import_error_is_dependency = False


def _raise_layout_import_error() -> None:
    if _layout_import_error_is_dependency:
        message = (
            "Layout detector dependencies are missing or incompatible. "
            "Try: pip install 'glmocr[layout]'. "
            f"Original error: {type(_layout_import_error).__name__}: {_layout_import_error}"
        )
    else:
        message = (
            "Layout detector failed to import (dependencies may already be installed). "
            "See the original error for the real cause. "
            f"Original error: {type(_layout_import_error).__name__}: {_layout_import_error}"
        )

    raise ImportError(message) from _layout_import_error


__all__ = ["BaseLayoutDetector", "PPDocLayoutDetector"]
