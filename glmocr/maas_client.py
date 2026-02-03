"""MaaS API Client for GLM-OCR.

This client provides a simple passthrough to the Zhipu MaaS API
(https://open.bigmodel.cn/api/paas/v4/layout_parsing).

When using MaaS mode, the SDK acts as a thin wrapper that:
1. Forwards the request to the MaaS API
2. Returns the response directly without additional processing

The MaaS service already runs the complete OCR pipeline internally,
so no local layout detection or parallel processing is needed.
"""

from __future__ import annotations

import base64
import os
import random
import time
import traceback
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter

from glmocr.utils.logging import get_logger, get_profiler

if TYPE_CHECKING:
    from glmocr.config import MaaSApiConfig

logger = get_logger(__name__)
profiler = get_profiler(__name__)


# Default MaaS API endpoint
DEFAULT_MAAS_URL = "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
DEFAULT_MAAS_MODEL = "glm-ocr"
MAX_MAAS_IMAGE_BYTES = 10 * 1024 * 1024


def _sniff_mime_from_bytes(data: bytes) -> str:
    # PDF
    if data[:5] == b"%PDF-":
        return "application/pdf"
    # PNG
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    # JPEG
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    return "application/octet-stream"


def _as_data_uri(mime: str, b64: str) -> str:
    return f"data:{mime};base64,{b64}"


class MaaSClient:
    """
    Client for Zhipu MaaS GLM-OCR API.

    This is a simple passthrough client that forwards requests to the MaaS API
    without doing any business logic processing. The MaaS service handles
    layout detection, parallel OCR, and result formatting internally.

    Usage:
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="your-api-key")
        client = MaaSClient(config)

        # Parse a local image file
        result = client.parse("document.png")

        # Parse a URL
        result = client.parse("https://example.com/document.png")

        # Parse with options
        result = client.parse(
            "document.pdf",
            return_crop_images=True,
            need_layout_visualization=True,
            start_page_id=1,
            end_page_id=5,
        )
    """

    def __init__(self, config: "MaaSApiConfig"):
        """Initialize the MaaS client.

        Args:
            config: MaaSApiConfig instance with API settings.
        """
        self.config = config

        # API endpoint
        self.api_url = config.api_url or DEFAULT_MAAS_URL
        self.model = config.model or DEFAULT_MAAS_MODEL

        # Authentication
        self.api_key = config.api_key or os.getenv("GLMOCR_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required for MaaS mode. "
                "Set it in config.yaml or GLMOCR_API_KEY environment variable."
            )

        # SSL verification
        self.verify_ssl = config.verify_ssl

        # Timeouts
        self.connect_timeout = config.connect_timeout
        self.request_timeout = config.request_timeout

        # Retry configuration
        self.retry_max_attempts = config.retry_max_attempts
        self.retry_backoff_base_seconds = config.retry_backoff_base_seconds
        self.retry_backoff_max_seconds = config.retry_backoff_max_seconds
        self.retry_jitter_ratio = config.retry_jitter_ratio
        self.retry_status_codes = set(config.retry_status_codes)

        # HTTP session
        self._session: Optional[requests.Session] = None
        self._pool_maxsize = config.connection_pool_size or 16

    def _make_session(self) -> requests.Session:
        """Create a Session with connection pooling."""
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=1,
            pool_maxsize=self._pool_maxsize,
            max_retries=0,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def start(self):
        """Initialize the HTTP session."""
        if self._session is None:
            self._session = self._make_session()
        logger.debug("MaaS client initialized for %s", self.api_url)

    def stop(self):
        """Close the HTTP session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def _sleep_backoff(
        self, attempt_index: int, retry_after_seconds: Optional[float] = None
    ) -> None:
        """Sleep with exponential backoff."""
        if retry_after_seconds is not None and retry_after_seconds > 0:
            sleep_s = min(
                float(retry_after_seconds), float(self.retry_backoff_max_seconds)
            )
        else:
            base = float(self.retry_backoff_base_seconds)
            sleep_s = min(
                base * (2**attempt_index), float(self.retry_backoff_max_seconds)
            )

        jitter = sleep_s * float(self.retry_jitter_ratio)
        if jitter > 0:
            sleep_s = max(0.0, sleep_s + random.uniform(-jitter, jitter))

        time.sleep(sleep_s)

    @staticmethod
    def _parse_retry_after_seconds(response: requests.Response) -> Optional[float]:
        """Parse Retry-After header if present."""
        ra = response.headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except Exception:
            return None

    def _prepare_file(self, source: Union[str, Path, bytes]) -> str:
        """Prepare file content for API request.

        Args:
            source: File path, URL, base64 string, data URI, or raw bytes.

        Returns:
            URL string or base64-encoded data.
        """
        # If it's bytes, encode to base64
        if isinstance(source, bytes):
            b64 = base64.b64encode(source).decode("utf-8")
            return _as_data_uri(_sniff_mime_from_bytes(source), b64)

        source_str = str(source)

        # If it's a URL, return as-is
        if source_str.startswith(("http://", "https://")):
            return source_str

        # If it's a data URI, extract the base64 part
        if source_str.startswith("data:"):
            # MaaS endpoint accepts data URIs directly.
            return source_str

        # Check if it looks like base64 (not a file path)
        # Base64 strings are typically long and don't contain path separators
        if self._looks_like_base64(source_str):
            # MaaS requires data URI even for base64. Try to infer mime.
            candidate = "".join(source_str.split())
            # Strip optional "<|base64|>" prefix.
            if candidate.startswith("<|base64|>"):
                candidate = candidate[len("<|base64|>") :]
            pad = (-len(candidate)) % 4
            if pad:
                candidate = candidate + ("=" * pad)
            try:
                decoded = base64.b64decode(candidate)
                mime = _sniff_mime_from_bytes(decoded)
            except Exception:
                mime = "application/octet-stream"
            return _as_data_uri(mime, "".join(source_str.split()))

        # If it's a file path, read and encode
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_bytes = path.read_bytes()

        # Keep PDFs as-is, but wrap as data URI for MaaS.
        if path.suffix.lower() == ".pdf" or file_bytes[:5] == b"%PDF-":
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            return _as_data_uri("application/pdf", b64)

        # MaaS only accepts JPG/PNG images. If the file is not actually JPEG/PNG
        # (e.g. WEBP renamed to .jpeg), re-encode to a supported format.
        try:
            from PIL import Image

            img = Image.open(BytesIO(file_bytes))
            fmt = (img.format or "").upper()
            if fmt in ("JPEG", "JPG", "PNG"):
                b64 = base64.b64encode(file_bytes).decode("utf-8")
                mime = _sniff_mime_from_bytes(file_bytes)
                return _as_data_uri(mime, b64)

            # Convert to PNG first (lossless for screenshots/text).
            img_converted = (
                img.convert("RGBA")
                if img.mode in ("RGBA", "LA")
                else img.convert("RGB")
            )
            buf = BytesIO()
            if img_converted.mode == "RGBA":
                img_converted.save(buf, format="PNG")
            else:
                # Try PNG; if too large, fallback to JPEG.
                img_converted.save(buf, format="PNG")
                if buf.tell() > MAX_MAAS_IMAGE_BYTES:
                    buf = BytesIO()
                    img_converted.save(buf, format="JPEG", quality=92, optimize=True)
            converted_bytes = buf.getvalue()
            b64 = base64.b64encode(converted_bytes).decode("utf-8")
            mime = _sniff_mime_from_bytes(converted_bytes)
            return _as_data_uri(mime, b64)
        except Exception:
            # If PIL cannot decode, fall back to raw bytes.
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            mime = _sniff_mime_from_bytes(file_bytes)
            return _as_data_uri(mime, b64)

    @staticmethod
    def _looks_like_base64(s: str) -> bool:
        """Heuristically detect base64 payloads.

        We intentionally avoid using path-separator heuristics because valid
        base64 often contains '/'. Instead we try a strict base64 decode.
        """
        if not isinstance(s, str):
            return False

        candidate = "".join(s.split())
        if len(candidate) < 128:
            return False

        # Reject obvious filenames/paths early.
        if candidate.startswith(("http://", "https://", "file://", "data:")):
            return False
        if "\\" in candidate:
            return False

        # If it has a short extension-like suffix, assume it's a filename.
        if "." in candidate and len(candidate.rsplit(".", 1)[-1]) <= 5:
            return False

        # Pad to a multiple of 4 for base64 decoding.
        pad = (-len(candidate)) % 4
        if pad:
            candidate = candidate + ("=" * pad)

        try:
            # validate=True enforces correct alphabet.
            base64.b64decode(candidate, validate=True)
            return True
        except Exception:
            return False

    def parse(
        self,
        source: Union[str, Path, bytes, List[Union[str, Path, bytes]]],
        return_crop_images: bool = False,
        need_layout_visualization: bool = False,
        start_page_id: Optional[int] = None,
        end_page_id: Optional[int] = None,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Parse a document using the MaaS API.

        This method sends the document directly to the Zhipu MaaS API and returns
        the response. No local processing is performed.

        Args:
            source: File path, URL, bytes, or list of them.
                   For multiple files, each will be processed separately.
            return_crop_images: Whether to return cropped region images.
            need_layout_visualization: Whether to return layout visualization.
            start_page_id: Start page for PDF (1-indexed).
            end_page_id: End page for PDF (1-indexed).
            request_id: Custom request ID.
            user_id: User ID for abuse monitoring (6-128 chars).
            **kwargs: Additional parameters to pass to the API.

        Returns:
            API response dict containing:
            - id: Task ID
            - created: Unix timestamp
            - model: Model name
            - md_results: Markdown formatted results
            - layout_details: Detailed layout information
            - layout_visualization: Visualization URLs (if requested)
            - data_info: Document metadata
            - usage: Token usage statistics
            - request_id: Request identifier
        """
        if self._session is None:
            self.start()

        # Handle list of sources
        if isinstance(source, list):
            results = []
            for src in source:
                results.append(
                    self.parse(
                        src,
                        return_crop_images=return_crop_images,
                        need_layout_visualization=need_layout_visualization,
                        start_page_id=start_page_id,
                        end_page_id=end_page_id,
                        request_id=request_id,
                        user_id=user_id,
                        **kwargs,
                    )
                )
            return {"results": results}

        # Prepare the request payload
        file_content = self._prepare_file(source)

        payload: Dict[str, Any] = {
            "model": self.model,
            "file": file_content,
        }

        if return_crop_images:
            payload["return_crop_images"] = True
        if need_layout_visualization:
            payload["need_layout_visualization"] = True
        if start_page_id is not None:
            payload["start_page_id"] = start_page_id
        if end_page_id is not None:
            payload["end_page_id"] = end_page_id
        if request_id:
            payload["request_id"] = request_id
        if user_id:
            payload["user_id"] = user_id

        # Add any extra parameters
        payload.update(kwargs)

        return self._send_request(payload)

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to the MaaS API with retry logic.

        Args:
            payload: Request payload.

        Returns:
            API response dict.

        Raises:
            requests.exceptions.RequestException: On network errors.
            ValueError: On API errors.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        total_attempts = int(self.retry_max_attempts) + 1
        last_error: Optional[str] = None

        for attempt in range(total_attempts):
            try:
                with profiler.measure("maas_http_request"):
                    response = self._session.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=(self.connect_timeout, self.request_timeout),
                        verify=self.verify_ssl,
                    )

                if response.status_code == 200:
                    return response.json()

                status = int(response.status_code)
                body_preview = (response.text or "")[:500]

                # Retry on specific status codes
                if status in self.retry_status_codes and attempt < total_attempts - 1:
                    retry_after = self._parse_retry_after_seconds(response)
                    logger.warning(
                        "MaaS API returned status %s (attempt %d/%d). Retrying... response: %s",
                        status,
                        attempt + 1,
                        total_attempts,
                        body_preview,
                    )
                    self._sleep_backoff(
                        attempt_index=attempt, retry_after_seconds=retry_after
                    )
                    continue

                # Non-retryable error
                logger.error(
                    "MaaS API request failed with status %s: %s", status, body_preview
                )
                raise ValueError(
                    f"MaaS API request failed with status {status}: {body_preview}"
                )

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                if attempt < total_attempts - 1:
                    logger.warning(
                        "MaaS API request error (attempt %d/%d): %s. Retrying...",
                        attempt + 1,
                        total_attempts,
                        last_error,
                    )
                    self._sleep_backoff(attempt_index=attempt)
                    continue
                logger.error("MaaS API request failed: %s", last_error)
                logger.debug(traceback.format_exc())
                raise

            except Exception as e:
                logger.error("Unexpected error during MaaS API request: %s", e)
                logger.debug(traceback.format_exc())
                raise

        raise ValueError(f"MaaS API request failed after {total_attempts} attempts")

    def parse_url(
        self,
        url: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Parse a document from URL.

        Convenience method that passes the URL directly to the API.

        Args:
            url: Document URL.
            **kwargs: Additional parameters (see parse()).

        Returns:
            API response dict.
        """
        return self.parse(url, **kwargs)

    def parse_base64(
        self,
        data: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Parse a document from base64-encoded data.

        Args:
            data: Base64-encoded document data.
            **kwargs: Additional parameters (see parse()).

        Returns:
            API response dict.
        """
        # MaaS currently expects data URIs. If caller passes a raw base64 string,
        # wrap it as a data URI (mime is inferred when possible).
        file_value = data
        if isinstance(file_value, str) and not file_value.startswith(
            (
                "http://",
                "https://",
                "data:",
            )
        ):
            file_value = self._prepare_file(file_value)

        payload: Dict[str, Any] = {"model": self.model, "file": file_value}
        payload.update(kwargs)
        return self._send_request(payload)
