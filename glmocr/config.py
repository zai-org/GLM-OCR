"""Configuration models and loaders.  """

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field

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
    # Layout
    "ENABLE_LAYOUT": "pipeline.enable_layout",
    # Logging
    "LOG_LEVEL": "logging.level",
}


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

    # Model name included in API requests.
    model: Optional[str] = None
    headers: Dict[str, str] = Field(default_factory=dict)
    verify_ssl: bool = False

    connect_timeout: int = 300
    request_timeout: int = 300

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
    enabled: bool = False

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
    max_tokens: int = 16384
    temperature: float = 0.01
    top_p: float = 0.00001
    top_k: int = 1
    repetition_penalty: float = 1.1

    t_patch_size: int = 2
    patch_expand_factor: int = 1
    image_expect_length: int = 6144
    image_format: str = "JPEG"
    min_pixels: int = 112 * 112
    max_pixels: int = 14 * 14 * 4 * 1280

    default_prompt: str = (
        "Recognize the text in the image and output in Markdown format. "
        "Preserve the original layout (headings/paragraphs/tables/formulas). "
        "Do not fabricate content that does not exist in the image."
    )
    task_prompt_mapping: Optional[Dict[str, str]] = None

    pdf_dpi: int = 200
    pdf_max_pages: Optional[int] = None
    pdf_verbose: bool = False


class ResultFormatterConfig(_BaseConfig):
    filter_nested: bool = True
    min_overlap_ratio: float = 0.8
    output_format: str = "both"  # json | markdown | both
    label_visualization_mapping: Dict[str, Any] = Field(default_factory=dict)


class LayoutConfig(_BaseConfig):
    model_dir: Optional[str] = None
    threshold: float = 0.4
    threshold_by_class: Optional[Dict[Union[int, str], float]] = None
    batch_size: int = 8
    workers: int = 1
    cuda_visible_devices: str = "0"
    img_size: Optional[int] = None
    layout_nms: bool = True
    layout_unclip_ratio: Optional[Any] = None
    layout_merge_bboxes_mode: Union[str, Dict[int, str]] = "large"
    label_task_mapping: Optional[Dict[str, Any]] = None


class PipelineConfig(_BaseConfig):
    enable_layout: bool = False

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
    region_maxsize: Optional[int] = None


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
    if dotted_path in ("pipeline.maas.enabled", "pipeline.enable_layout"):
        # Special handling for MODE: "maas" → True, anything else → False
        if dotted_path == "pipeline.maas.enabled":
            return raw.strip().lower() in ("maas", "true", "1", "yes")
        return raw.strip().lower() in ("true", "1", "yes")
    # Integer fields
    if dotted_path.endswith((".api_port", ".request_timeout", ".connect_timeout")):
        return int(raw)
    return raw


def _collect_env_overrides() -> Dict[str, Any]:
    """Read GLMOCR_* values from ``.env`` file + real environment variables.

    Priority: real ``os.environ`` > ``.env`` file.  This means a user can
    always override a ``.env`` value by exporting the variable in the shell.
    """
    # 1. Load .env file (does NOT mutate os.environ)
    dotenv_path = _find_dotenv()
    dotenv_vars: Dict[str, Optional[str]] = (
        dotenv_values(dotenv_path) if dotenv_path else {}
    )

    # 2. Merge: real env > .env
    merged: Dict[str, str] = {}
    for env_suffix in _ENV_MAP:
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
        """Build config with priority: *overrides* > env-vars > YAML > defaults.

        This is the **agent-friendly** entry-point.  An agent (or any
        programmatic caller) can configure the SDK entirely through keyword
        arguments or ``GLMOCR_*`` environment variables without touching a
        YAML file.

        Accepted keyword overrides (a useful subset – the full YAML structure
        is also accepted via nested dicts):

        * ``api_key``        – MaaS / OCR API key
        * ``api_url``        – MaaS API endpoint URL
        * ``model``          – model name
        * ``mode``           – ``"maas"`` or ``"selfhosted"``
        * ``timeout``        – request timeout in seconds
        * ``enable_layout``  – whether to run layout detection
        * ``log_level``      – logging level (DEBUG / INFO / …)

        Any other keyword is silently ignored so that callers can safely
        forward ``**kwargs`` without worrying about typos crashing the SDK.

        Examples::

            # Pure env-var driven (e.g. in a .env file)
            #   GLMOCR_API_KEY=xxx
            #   GLMOCR_MODE=maas
            cfg = GlmOcrConfig.from_env()

            # Explicit overrides (highest priority)
            cfg = GlmOcrConfig.from_env(api_key="sk-xxx", mode="maas")

            # With a custom YAML base
            cfg = GlmOcrConfig.from_env(config_path="my.yaml", api_key="sk")
        """
        # 1. YAML baseline
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

        # 2. Environment variable overrides
        env_data = _collect_env_overrides()
        if env_data:
            _deep_merge(data, env_data)

        # 3. Keyword overrides (flat convenience names → nested paths)
        _KW_MAP = {
            "api_key": "pipeline.maas.api_key",
            "api_url": "pipeline.maas.api_url",
            "model": "pipeline.maas.model",
            "mode": "pipeline.maas.enabled",
            "timeout": "pipeline.maas.request_timeout",
            "enable_layout": "pipeline.enable_layout",
            "log_level": "logging.level",
        }
        for kw, dotted in _KW_MAP.items():
            if kw in overrides and overrides[kw] is not None:
                raw = overrides[kw]
                _set_nested(data, dotted, _coerce_env_value(dotted, str(raw)))

        return cls.model_validate(data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def load_config(
    path: Optional[Union[str, Path]] = None,
    **overrides: Any,
) -> GlmOcrConfig:
    """Load config with priority: *overrides* > env-vars > YAML > defaults.

    This is a drop-in replacement for the old ``load_config(path)``.
    When called without arguments it behaves exactly as before (YAML only).
    When keyword overrides or ``GLMOCR_*`` env-vars are present they take
    precedence.
    """
    return GlmOcrConfig.from_env(config_path=path, **overrides)
