"""GLM-OCR Python SDK."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.1.1"
__author__ = "ZHIPUAI"

__all__ = [
    "dataloader",
    "layout",
    "postprocess",
    "utils",
    "Pipeline",
    "PipelineResult",
    "GlmOcrConfig",
    "load_config",
    "MaaSClient",
    "MissingApiKeyError",
    "GlmOcr",
    "parse",
]


_LAZY_SUBMODULES = {"dataloader", "layout", "postprocess", "utils"}
_LAZY_ATTRS = {
    "Pipeline": ("pipeline", "Pipeline"),
    "PipelineResult": ("parser_result", "PipelineResult"),
    "GlmOcrConfig": ("config", "GlmOcrConfig"),
    "load_config": ("config", "load_config"),
    "MaaSClient": ("maas_client", "MaaSClient"),
    "MissingApiKeyError": ("maas_client", "MissingApiKeyError"),
    "GlmOcr": ("api", "GlmOcr"),
    "parse": ("api", "parse"),
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, attr_name)


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))


if TYPE_CHECKING:  # pragma: no cover
    from . import dataloader, layout, postprocess, utils
    from .api import GlmOcr, parse
    from .config import GlmOcrConfig, load_config
    from .maas_client import MaaSClient, MissingApiKeyError
    from .parser_result import PipelineResult
    from .pipeline import Pipeline
