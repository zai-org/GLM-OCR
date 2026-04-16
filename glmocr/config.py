"""Configuration models and loaders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Environment variable prefix for all GLM-OCR settings.
ENV_PREFIX = "GLMOCR_"


def _find_dotenv(start: Optional[Path] = None) -> Optional[Path]:
    """Walk up from *start* (default: cwd) looking for a ``.env`` file.

    Returns the first ``.env`` found, or ``None``.
    """
    cur = (start or Path.cwd()).resolve()
    for directory in (cur, *cur.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


# Mapping: env-var name (without prefix) → nested config dict path.
# Only the most commonly needed knobs are listed here so that an agent can
# configure the SDK entirely through environment variables / .env files.
_ENV_MAP: Dict[str, str] = {
    # mode
    "MODE": "pipeline.maas.enabled",  # "maas" | "selfhosted"
    # MaaS settings
    "API_KEY": "pipeline.maas.api_key",
    "API_URL": "pipeline.maas.api_url",
    "MODEL": "pipeline.maas.model",
    "TIMEOUT": "pipeline.maas.request_timeout",
    # Self-hosted OCR API settings
    "OCR_API_URL": "pipeline.ocr_api.api_url",
    "OCR_API_KEY": "pipeline.ocr_api.api_key",
    "OCR_API_HOST": "pipeline.ocr_api.api_host",
    "OCR_API_PORT": "pipeline.ocr_api.api_port",
    "OCR_MODEL": "pipeline.ocr_api.model",
    # Allow overriding which GPU(s) the layout model uses
    "LAYOUT_CUDA_VISIBLE_DEVICES": "pipeline.layout.cuda_visible_devices",
    # Explicit device for layout model: "cpu", "cuda", "cuda:0", etc.
    "LAYOUT_DEVICE": "pipeline.layout.device",
    # Logging
    "LOG_LEVEL": "logging.level",
}

PRIMARY_API_KEY_ENV = "ZHIPU_API_KEY"
LEGACY_API_KEY_ENV = "GLMOCR_API_KEY"


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class ServerConfig(_BaseConfig):
    host: str = "0.0.0.0"
    port: int = 5002
    debug: bool = False


class LoggingConfig(_BaseConfig):
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: Optional[str] = None


class OCRApiConfig(_BaseConfig):
    api_host: str = "localhost"
    api_port: int = 5002

    # For MaaS / HTTPS / non-default endpoints
    api_scheme: Optional[str] = None
    api_path: str = "/v1/chat/completions"
    api_url: Optional[str] = None
    api_key: Optional[str] = None

    # Model name included in API requests (required by Ollama/MLX).
    model: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    verify_ssl: bool = False

    # API mode: "openai" (default) or "ollama_generate"
    # Use "ollama_generate" for Ollama's native /api/generate endpoint
    api_mode: str = "openai"

    connect_timeout: int = 30
    request_timeout: int = 120

    # Retry behavior (for transient upstream failures like 429/5xx)
    retry_max_attempts: int = 2  # total attempts = 1 + retry_max_attempts
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_max_seconds: float = 8.0
    retry_jitter_ratio: float = 0.2
    retry_status_codes: List[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )

    # HTTP connection pool size. Should be >= pipeline max_workers to avoid
    # "Connection pool is full" when layout mode runs concurrent requests. Default 128.
    connection_pool_size: Optional[int] = 128


class MaaSApiConfig(_BaseConfig):
    """Configuration for Zhipu MaaS GLM-OCR API.

    When using MaaS mode, the SDK acts as a thin wrapper that forwards requests
    directly to the Zhipu cloud API without local processing.
    """

    # Enable MaaS mode (passthrough to Zhipu cloud API)
    # Default: True — MaaS is the default mode after `pip install glmocr` (no GPU needed)
    enabled: bool = True

    # API endpoint (default: Zhipu GLM-OCR layout_parsing API)
    api_url: str = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"

    # Model name
    model: str = "glm-ocr"

    # API key (required for MaaS mode)
    api_key: Optional[str] = None

    # SSL verification
    verify_ssl: bool = True

    # Timeouts (seconds)
    connect_timeout: int = 30
    request_timeout: int = 300

    # Retry settings
    retry_max_attempts: int = 2
    retry_backoff_base_seconds: float = 0.5
    retry_backoff_max_seconds: float = 8.0
    retry_jitter_ratio: float = 0.2
    retry_status_codes: List[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )

    # Connection pool size
    connection_pool_size: int = 16


class PageLoaderConfig(_BaseConfig):
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 0.00001
    top_k: int = 1
    repetition_penalty: float = 1.1

    t_patch_size: int = 2
    patch_expand_factor: int = 1
    image_expect_length: int = 6144
    image_format: str = "JPEG"
    min_pixels: int = 112 * 112
    max_pixels: int = 14 * 14 * 4 * 1280

    task_prompt_mapping: Optional[Dict[str, str]] = None

    pdf_dpi: int = 200
    pdf_max_pages: Optional[int] = None
    pdf_verbose: bool = False


class ResultFormatterConfig(_BaseConfig):
    filter_nested: bool = True
    min_overlap_ratio: float = 0.8
    output_format: str = "both"  # json | markdown | both
    enable_merge_formula_numbers: bool = True
    enable_merge_text_blocks: bool = True
    enable_format_bullet_points: bool = True
    label_visualization_mapping: Dict[str, Any] = Field(
        default_factory=lambda: {
            "image": ["chart", "image"],
            "table": ["table"],
            "formula": ["display_formula", "inline_formula"],
            "text": [
                "abstract",
                "algorithm",
                "content",
                "doc_title",
                "figure_title",
                "paragraph_title",
                "reference_content",
                "text",
                "vertical_text",
                "vision_footnote",
                "seal",
                "formula_number",
            ],
        }
    )


class LayoutConfig(_BaseConfig):
    model_dir: Optional[str] = None
    threshold: float = 0.3
    threshold_by_class: Optional[Dict[Union[int, str], float]] = None
    batch_size: int = 8
    workers: int = 1
    cuda_visible_devices: str = "0"
    # Explicit device placement for the layout model.
    # - null (default): auto-select using cuda_visible_devices if CUDA is
    #   available, otherwise CPU.  This preserves backward compatibility.
    # - "cpu": force CPU even when CUDA is available.
    # - "cuda": use the default CUDA device.
    # - "cuda:N": use a specific CUDA device (overrides cuda_visible_devices).
    device: Optional[str] = None
    img_size: Optional[int] = None
    layout_nms: bool = True
    layout_unclip_ratio: Optional[Any] = None
    layout_merge_bboxes_mode: Union[str, Dict[int, str]] = "large"
    label_task_mapping: Optional[Dict[str, Any]] = None
    use_polygon: bool = False
    id2label: Optional[Dict[Union[int, str], str]] = None

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: Optional[str]) -> Optional[str]:
        """Validate the layout device string.

        Allowed values:
        - None / null (auto-select based on CUDA availability)
        - "cpu"
        - "cuda"
        - "cuda:<int>" (e.g., "cuda:0", "cuda:1")
        """
        if value is None:
            return value
        v = value.strip()
        if v == "":
            return None
        if v == "cpu" or v == "cuda":
            return v
        if v.startswith("cuda:"):
            index_part = v[5:]
            if index_part.isdigit():
                return v
        raise ValueError(
            "Invalid layout device value. Expected one of: None, 'cpu', 'cuda', "
            "or 'cuda:<int>' (e.g., 'cuda:0')."
        )


class PipelineConfig(_BaseConfig):
    # MaaS mode configuration (Zhipu cloud API passthrough)
    maas: MaaSApiConfig = Field(default_factory=MaaSApiConfig)

    page_loader: PageLoaderConfig = Field(default_factory=PageLoaderConfig)
    ocr_api: OCRApiConfig = Field(default_factory=OCRApiConfig)
    result_formatter: ResultFormatterConfig = Field(
        default_factory=ResultFormatterConfig
    )
    layout: LayoutConfig = Field(default_factory=LayoutConfig)

    # Parallel recognition workers (VLM/API concurrent requests)
    max_workers: int = 16

    # Queue sizes for async pipeline.
    page_maxsize: int = 100
    region_maxsize: Optional[int] = 800


def _set_nested(data: Dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a value in a nested dict using a dotted key path."""
    keys = dotted_path.split(".")
    d = data
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _coerce_env_value(dotted_path: str, raw: str) -> Any:
    """Coerce a raw environment-variable string to the expected Python type."""
    # Boolean fields
    if dotted_path == "pipeline.maas.enabled":
        return raw.strip().lower() in ("maas", "true", "1", "yes")
    # Integer fields
    if dotted_path.endswith((".api_port", ".request_timeout", ".connect_timeout")):
        return int(raw)
    return raw


def _collect_env_overrides(
    env_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Read SDK env values from ``.env`` + real environment variables.

    Args:
        env_file: Explicit path to a ``.env`` file.  When provided, this file
            is used instead of the auto-discovered one.  Raises
            ``FileNotFoundError`` if the path does not exist.

    Priority: real ``os.environ`` > ``.env`` file.  This means a user can
    always override a ``.env`` value by exporting the variable in the shell.

    API key special case:
    - Primary: ``ZHIPU_API_KEY``
    - Legacy fallback: ``GLMOCR_API_KEY``
    """
    # 1. Load .env file (does NOT mutate os.environ)
    if env_file is not None:
        dotenv_path = Path(env_file)
        if not dotenv_path.is_file():
            raise FileNotFoundError(f".env file not found: {dotenv_path}")
    else:
        dotenv_path = _find_dotenv()
    dotenv_vars: Dict[str, Optional[str]] = (
        dotenv_values(dotenv_path) if dotenv_path else {}
    )

    # 2. Merge: real env > .env
    merged: Dict[str, str] = {}
    for env_suffix in _ENV_MAP:
        if env_suffix == "API_KEY":
            # Prefer unified env key for SDK skill, fallback to legacy key.
            val = os.environ.get(PRIMARY_API_KEY_ENV)
            if val is None:
                val = os.environ.get(LEGACY_API_KEY_ENV)
            if val is None:
                val = dotenv_vars.get(PRIMARY_API_KEY_ENV)  # type: ignore[assignment]
            if val is None:
                val = dotenv_vars.get(LEGACY_API_KEY_ENV)  # type: ignore[assignment]
            if val is not None:
                merged[env_suffix] = val
            continue

        full_key = f"{ENV_PREFIX}{env_suffix}"
        # Real env takes precedence
        val = os.environ.get(full_key)
        if val is None:
            val = dotenv_vars.get(full_key)  # type: ignore[assignment]
        if val is not None:
            merged[env_suffix] = val

    # 3. Build nested config dict
    overrides: Dict[str, Any] = {}
    for env_suffix, raw in merged.items():
        dotted_path = _ENV_MAP[env_suffix]
        _set_nested(overrides, dotted_path, _coerce_env_value(dotted_path, raw))
    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


class GlmOcrConfig(_BaseConfig):
    """Top-level config model."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def default_path(cls) -> str:
        return str(Path(__file__).with_name("config.yaml"))

    @classmethod
    def from_yaml(cls, path: Optional[Union[str, Path]] = None) -> "GlmOcrConfig":
        path = Path(path or cls.default_path())
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)

    @classmethod
    def from_env(
        cls,
        config_path: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "GlmOcrConfig":
        """Build config with layered priority (highest → lowest):

        1. CLI ``--set`` overrides (``_dotted`` dict)
        2. Keyword overrides (``api_key``, ``mode``, …)
        3. ``GLMOCR_*`` environment variables / ``.env`` file
        4. YAML config file
        5. Built-in defaults

        This is the **agent-friendly** entry-point.  An agent (or any
        programmatic caller) can configure the SDK entirely through keyword
        arguments or environment variables without touching a YAML file.
        Primary API key env var is ``ZHIPU_API_KEY`` (``GLMOCR_API_KEY`` is
        still supported as a legacy fallback).

        Accepted keyword overrides (a useful subset – the full YAML structure
        is also accepted via nested dicts):

        * ``api_key``        – MaaS / OCR API key
        * ``api_url``        – MaaS API endpoint URL
        * ``model``          – model name
        * ``mode``           – ``"maas"`` or ``"selfhosted"``
        * ``timeout``        – request timeout in seconds
        * ``log_level``      – logging level (DEBUG / INFO / …)
        * ``env_file``       – explicit path to a ``.env`` file

        Any other keyword is silently ignored so that callers can safely
        forward ``**kwargs`` without worrying about typos crashing the SDK.

        Examples::

            # Pure env-var driven (e.g. in a .env file)
            #   ZHIPU_API_KEY=xxx
            #   GLMOCR_MODE=maas
            cfg = GlmOcrConfig.from_env()

            # Explicit overrides (highest priority)
            cfg = GlmOcrConfig.from_env(api_key="sk-xxx", mode="maas")

            # With a custom YAML base
            cfg = GlmOcrConfig.from_env(config_path="my.yaml", api_key="sk")
        """
        # --- Priority (applied in order, later wins): ---
        # 1. YAML baseline (lowest)
        yaml_path = Path(config_path or cls.default_path())
        if yaml_path.exists():
            data: Dict[str, Any] = (
                yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            )
        else:
            # If no YAML and no explicit path requested, start from scratch.
            if config_path is not None:
                raise FileNotFoundError(f"Config file not found: {yaml_path}")
            data = {}

        # 2. Environment variable overrides (.env + GLMOCR_*)
        env_file = overrides.pop("env_file", None)
        env_data = _collect_env_overrides(env_file=env_file)
        if env_data:
            _deep_merge(data, env_data)

        # 3. Keyword overrides (Python API convenience names)
        _KW_MAP = {
            "api_key": "pipeline.maas.api_key",
            "api_url": "pipeline.maas.api_url",
            "mode": "pipeline.maas.enabled",
            "timeout": "pipeline.maas.request_timeout",
            "log_level": "logging.level",
            # Self-hosted OCR API
            "ocr_api_host": "pipeline.ocr_api.api_host",
            "ocr_api_port": "pipeline.ocr_api.api_port",
            # Layout GPU binding
            "cuda_visible_devices": "pipeline.layout.cuda_visible_devices",
            "layout_device": "pipeline.layout.device",
        }

        # `model` is shared by both MaaS and self-hosted modes.
        # Keep MaaS behavior while also forwarding it to OCR API so that
        # `GlmOcr(mode="selfhosted", model="...")` works as expected.
        if "model" in overrides and overrides["model"] is not None:
            model_value = str(overrides["model"])
            _set_nested(data, "pipeline.maas.model", model_value)
            _set_nested(data, "pipeline.ocr_api.model", model_value)

        for kw, dotted in _KW_MAP.items():
            if kw in overrides and overrides[kw] is not None:
                raw = overrides[kw]
                _set_nested(data, dotted, _coerce_env_value(dotted, str(raw)))

        # 4. CLI --set overrides (highest priority)
        for dotted, value in overrides.get("_dotted", {}).items():
            _set_nested(data, dotted, value)

        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def load_config(
    path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> GlmOcrConfig:
    """Load config with priority: CLI --set > keyword > env-vars > YAML > defaults.

    This is a drop-in replacement for the old ``load_config(path)``.
    When called without arguments it behaves exactly as before (YAML only).
    When keyword overrides or ``GLMOCR_*`` env-vars are present they take
    precedence.  CLI ``--set`` overrides (passed via ``_dotted``) have the
    highest priority.
    """
    return GlmOcrConfig.from_env(config_path=path, **overrides)
