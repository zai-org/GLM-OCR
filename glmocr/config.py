"""Configuration models and loaders.  """

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union, List

import yaml
from pydantic import BaseModel, ConfigDict, Field


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

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def load_config(path: Optional[Union[str, Path]] = None) -> GlmOcrConfig:
    """Load config from YAML (returns a new instance each time)."""

    return GlmOcrConfig.from_yaml(path)
