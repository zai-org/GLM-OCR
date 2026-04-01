"""GLM-OCR Python API

Python API for calling the document parsing pipeline from your code.

Two modes are supported:
1. MaaS Mode (maas.enabled=true): Forwards requests to Zhipu's cloud API.
   No GPU required; the cloud handles all processing.
2. Self-hosted Mode (maas.enabled=false): Uses local vLLM/SGLang service.
   Requires GPU; SDK handles layout detection, parallel OCR, etc.

Supported input types: file paths (``str``), ``pathlib.Path``, raw ``bytes``
(image or PDF content), and URLs (file://, http://, data:).

Agent-friendly usage::

    # Only needs ZHIPU_API_KEY in environment (or pass api_key directly)
    from glmocr import GlmOcr

    parser = GlmOcr(api_key="sk-xxx", mode="maas")
    result = parser.parse("document.png")
    result = parser.parse(open("doc.pdf", "rb").read())   # bytes
    print(result.to_dict())
"""

import os
import re
from typing import Any, Dict, Generator, List, Literal, Optional, Union, overload
from pathlib import Path

from glmocr.config import load_config
from glmocr.parser_result import PipelineResult
from glmocr.utils.image_asset_utils import export_image_assets
from glmocr.utils.logging import get_logger, ensure_logging_configured

logger = get_logger(__name__)

# Backward compatibility: ParseResult is PipelineResult
ParseResult = PipelineResult


class GlmOcr:
    """Main GLM-OCR entrypoint.

    Provides a Python API for document parsing. Automatically detects whether
    to use MaaS mode or self-hosted mode based on config.

    Configuration priority:  constructor args > env vars > YAML > defaults.

    Examples::

        # --- Agent-friendly: zero YAML ---
        import glmocr
        parser = glmocr.GlmOcr(api_key="sk-xxx")          # MaaS auto-enabled
        parser = glmocr.GlmOcr(mode="maas")                # uses ZHIPU_API_KEY env

        # --- Classic: YAML-based ---
        parser = glmocr.GlmOcr(config_path="config.yaml")

        # --- Parse (paths, bytes, or mixed) ---
        result = parser.parse("image.png")
        result = parser.parse(open("doc.pdf", "rb").read())
        results = parser.parse(["img.png", pdf_bytes])
        for r in results:
            print(r.markdown_result)
            print(r.to_dict())           # structured, JSON-serialisable
            r.save(output_dir="./output")

        parser.close()   # or use `with GlmOcr(...) as parser:`
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        mode: Optional[str] = None,
        timeout: Optional[int] = None,
        log_level: Optional[str] = None,
        env_file: Optional[str] = None,
        # Extra knobs for self-hosted mode & GPU binding
        ocr_api_host: Optional[str] = None,
        ocr_api_port: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        layout_device: Optional[str] = None,
        detect_printed_page_numbers: Optional[bool] = None,
        **kwargs: Any,
    ):
        """Initialize GlmOcr.

        All keyword arguments are optional.  When provided they override any
        value coming from the YAML file or environment variables
        (primary API key: ``ZHIPU_API_KEY``).

        Args:
            config_path: YAML config file path (optional).
            api_key:  API key for MaaS / self-hosted OCR API.
            api_url:  MaaS API endpoint URL.
            model:    Model name.
            mode:     ``"maas"`` (cloud) or ``"selfhosted"`` (local vLLM/SGLang).
                      If *api_key* is provided without an explicit *mode*,
                      mode defaults to ``"maas"``.
            timeout:  Request timeout in seconds.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            env_file: Path to a ``.env`` file to load API key and other settings from.
            layout_device: Device for the layout model: ``"cpu"``, ``"cuda"``,
                or ``"cuda:N"``.  Defaults to auto-detection via
                ``cuda_visible_devices``.
        """
        # If an API key is available (constructor arg or env var), default to MaaS.
        # This ensures `GlmOcr()` with ZHIPU_API_KEY in env auto-selects MaaS
        # even when the user has an old YAML with maas.enabled=false.
        _has_api_key = api_key is not None or bool(
            os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLMOCR_API_KEY")
        )
        if _has_api_key and mode is None:
            mode = "maas"

        # Build config: overrides > env vars > YAML > defaults
        self.config_model = load_config(
            config_path,
            api_key=api_key,
            api_url=api_url,
            model=model,
            mode=mode,
            timeout=timeout,
            log_level=log_level,
            env_file=env_file,
            ocr_api_host=ocr_api_host,
            ocr_api_port=ocr_api_port,
            cuda_visible_devices=cuda_visible_devices,
            layout_device=layout_device,
            detect_printed_page_numbers=detect_printed_page_numbers,
            **kwargs,
        )
        # Apply logging config for API/SDK usage.
        ensure_logging_configured(
            level=self.config_model.logging.level,
            format_string=self.config_model.logging.format,
        )

        # Check if MaaS mode is enabled
        self._use_maas = self.config_model.pipeline.maas.enabled
        self._pipeline = None
        self._maas_client = None

        if self._use_maas:
            # MaaS mode: use MaaSClient for direct API passthrough
            from glmocr.maas_client import MaaSClient

            self._maas_client = MaaSClient(self.config_model.pipeline.maas)
            self._maas_client.start()
            logger.info("GLM-OCR initialized in MaaS mode (cloud API passthrough)")
        else:
            # Self-hosted mode: use full Pipeline
            from glmocr.pipeline import Pipeline

            self._pipeline = Pipeline(config=self.config_model.pipeline)
            self._pipeline.start()
            logger.info("GLM-OCR initialized in self-hosted mode")

    # ------------------------------------------------------------------
    # Input normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_url(image: Union[str, Path]) -> str:
        """Convert a path/URL to a ``file://`` URL."""
        if isinstance(image, Path):
            return f"file://{image.absolute()}"
        if isinstance(image, str):
            if image.startswith(("http://", "https://", "data:", "file://")):
                return image
            return f"file://{Path(image).absolute()}"
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _maas_source(image: Union[str, bytes, Path]):
        """Return ``(source, display_name)`` suitable for the MaaS client."""
        if isinstance(image, bytes):
            return image, "<bytes>"
        if isinstance(image, Path):
            return str(image), str(image)
        if isinstance(image, str) and image.startswith("file://"):
            p = image[7:]
            return p, p
        return image, str(image)

    # ------------------------------------------------------------------
    # parse() and overloads
    # ------------------------------------------------------------------

    @overload
    def parse(
        self,
        images: Union[str, bytes, Path],
        *,
        stream: Literal[False] = ...,
        save_layout_visualization: bool = ...,
        **kwargs: Any,
    ) -> PipelineResult: ...

    @overload
    def parse(
        self,
        images: List[Union[str, bytes, Path]],
        *,
        stream: Literal[False] = ...,
        save_layout_visualization: bool = ...,
        **kwargs: Any,
    ) -> List[PipelineResult]: ...

    @overload
    def parse(
        self,
        images: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
        *,
        stream: Literal[True],
        save_layout_visualization: bool = ...,
        **kwargs: Any,
    ) -> Generator[PipelineResult, None, None]: ...

    def parse(
        self,
        images: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
        *,
        stream: bool = False,
        save_layout_visualization: bool = True,
        preserve_order: bool = True,
        **kwargs: Any,
    ) -> Union[
        PipelineResult, List[PipelineResult], Generator[PipelineResult, None, None]
    ]:
        """Parse images or documents.

        Accepts file paths, URLs, ``pathlib.Path`` objects, or raw ``bytes``
        (image or PDF content).  Format is auto-detected from magic bytes.

        Args:
            images: A single input or a list.  Each element can be:

                * ``str``   – local path or URL (file://, http://, data:)
                * ``bytes`` – raw image / PDF bytes
                * ``Path``  – ``pathlib.Path`` to a file

            stream: If ``True``, yields one :class:`PipelineResult` at a time.
            save_layout_visualization: Whether to save layout visualization artifacts.
            preserve_order: Whether to keep output order consistent with input order.
            **kwargs: Additional parameters for MaaS mode (return_crop_images,
                     need_layout_visualization, start_page_id, end_page_id, etc.)

        Returns:
            - ``stream=False``, single input → ``PipelineResult``
            - ``stream=False``, list input  → ``List[PipelineResult]``
            - ``stream=True``               → ``Generator[PipelineResult, ...]``

        Examples::

            result = parser.parse("image.png")
            result = parser.parse(Path("image.png"))
            result = parser.parse(open("image.png", "rb").read())
            results = parser.parse(["img1.png", pdf_bytes])
        """
        _single = isinstance(images, (str, bytes, Path))
        if _single:
            images = [images]

        if stream:
            return self._parse_stream(
                images,
                save_layout_visualization,
                preserve_order=preserve_order,
                **kwargs,
            )

        if self._use_maas:
            result_list = self._parse_maas(images, save_layout_visualization, **kwargs)
        else:
            result_list = self._parse_selfhosted(
                images,
                save_layout_visualization,
                preserve_order=preserve_order,
            )

        return result_list[0] if _single else result_list

    def _parse_stream(
        self,
        images: List[Union[str, bytes, Path]],
        save_layout_visualization: bool = True,
        preserve_order: bool = True,
        **kwargs: Any,
    ) -> Generator[PipelineResult, None, None]:
        """Internal: yield one PipelineResult per input. Used by parse(stream=True)."""
        if self._use_maas:
            if save_layout_visualization:
                kwargs.setdefault("need_layout_visualization", True)
            for image in images:
                source, display = self._maas_source(image)
                try:
                    response = self._maas_client.parse(source, **kwargs)
                    result = self._maas_response_to_pipeline_result(response, display)
                    yield result
                except Exception as e:
                    logger.error("MaaS API error for %s: %s", display, e)
                    result = PipelineResult(
                        json_result=[],
                        markdown_result="",
                        original_images=[display],
                    )
                    result._error = str(e)
                    yield result
            return
        for result in self._stream_parse_selfhosted(
            images,
            save_layout_visualization=save_layout_visualization,
            preserve_order=preserve_order,
        ):
            yield result

    def _parse_maas(
        self,
        images: List[Union[str, bytes, Path]],
        save_layout_visualization: bool = True,
        **kwargs: Any,
    ) -> List[PipelineResult]:
        """Parse using MaaS API (passthrough mode)."""
        results = []

        if save_layout_visualization:
            kwargs.setdefault("need_layout_visualization", True)

        for image in images:
            source, display = self._maas_source(image)
            try:
                response = self._maas_client.parse(source, **kwargs)
                result = self._maas_response_to_pipeline_result(response, display)
                results.append(result)
            except Exception as e:
                logger.error("MaaS API error for %s: %s", display, e)
                result = PipelineResult(
                    json_result=[],
                    markdown_result="",
                    original_images=[display],
                )
                result._error = str(e)
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # MaaS bbox coordinate conversion
    # ------------------------------------------------------------------
    # The MaaS API returns bbox_2d in **absolute pixel coordinates** of
    # its own internal rendering (e.g. 2040×2640 for a letter-sized PDF
    # page).  The rest of the SDK (self-hosted pipeline,
    # resolve_image_regions) uses **normalised 0-1000 coordinates**.
    #
    # To keep everything consistent we convert here, right after receiving
    # the MaaS response, so that json_result and markdown_result always
    # contain normalised coords regardless of the backend.

    @staticmethod
    def _normalise_bbox(
        bbox: Optional[List[int]],
        page_w: int,
        page_h: int,
    ) -> Optional[List[int]]:
        """Convert absolute-pixel bbox to normalised 0-1000 coords."""
        if not bbox or len(bbox) != 4 or page_w <= 0 or page_h <= 0:
            return bbox
        x1, y1, x2, y2 = bbox
        return [
            round(x1 * 1000 / page_w),
            round(y1 * 1000 / page_h),
            round(x2 * 1000 / page_w),
            round(y2 * 1000 / page_h),
        ]

    # Regex for Markdown image refs: ![](page=0,bbox=[431, 1762, 1061, 2189])
    _MD_BBOX_RE = re.compile(r"(!\[\]\(page=(\d+),bbox=\[([\d,\s]+)\])\)")

    @classmethod
    def _normalise_markdown_bboxes(
        cls,
        markdown: str,
        pages_info: List[Dict[str, int]],
    ) -> str:
        """Replace absolute-pixel bbox values in Markdown image refs with
        normalised 0-1000 values so that the result formatter resolves
        the correct region.
        """
        if not pages_info or not markdown:
            return markdown

        def _replace(m: re.Match) -> str:
            page_idx = int(m.group(2))
            if page_idx >= len(pages_info):
                return m.group(0)  # can't normalise, keep original

            page_w = pages_info[page_idx].get("width", 0)
            page_h = pages_info[page_idx].get("height", 0)
            if page_w <= 0 or page_h <= 0:
                return m.group(0)

            raw_coords = [int(c.strip()) for c in m.group(3).split(",")]
            if len(raw_coords) != 4:
                return m.group(0)

            norm = cls._normalise_bbox(raw_coords, page_w, page_h)
            return f"![](page={page_idx},bbox={norm})"

        return cls._MD_BBOX_RE.sub(_replace, markdown)

    def _maas_response_to_pipeline_result(
        self, response: Dict[str, Any], source: str
    ) -> PipelineResult:
        """Convert MaaS API response to PipelineResult."""
        # Extract layout_details (list of pages, each page is a list of regions)
        layout_details = response.get("layout_details", [])

        # Per-page pixel dimensions from MaaS (used for bbox normalisation).
        data_info = response.get("data_info", {})
        pages_info: List[Dict[str, int]] = data_info.get("pages", [])

        # Convert to SDK format: [[{index, label, content, bbox_2d}, ...], ...]
        json_result = []
        for page_idx, page_regions in enumerate(layout_details):
            # Resolve page dimensions for normalisation.
            if page_idx < len(pages_info):
                page_w = pages_info[page_idx].get("width", 0)
                page_h = pages_info[page_idx].get("height", 0)
            else:
                page_w = page_h = 0

            page_result = []
            for region in page_regions:
                bbox = region.get("bbox_2d")
                if page_w > 0 and page_h > 0 and bbox:
                    bbox = self._normalise_bbox(bbox, page_w, page_h)
                page_result.append(
                    {
                        "index": region.get("index", 0),
                        "label": region.get("label", "text"),
                        "native_label": region.get("label", "text"),
                        "content": region.get("content", ""),
                        "bbox_2d": bbox,
                        "layout_index": region.get("index", 0),
                        "layout_score": float(region.get("score") or 0.0),
                    }
                )
            json_result.append(page_result)

        # Get markdown result and normalise the bbox refs inside it.
        markdown_result = response.get("md_results", "")
        markdown_result = self._normalise_markdown_bboxes(
            markdown_result,
            pages_info,
        )

        json_result, markdown_result, image_files = export_image_assets(
            json_result,
            markdown_result,
            source,
            enable_image_asset_export=self.config_model.pipeline.result_formatter.enable_image_asset_export,
            markdown_image_preference=self.config_model.pipeline.result_formatter.markdown_image_preference,
            image_match_iou_threshold=self.config_model.pipeline.result_formatter.image_match_iou_threshold,
            image_match_containment_threshold=self.config_model.pipeline.result_formatter.image_match_containment_threshold,
            rendered_image_dpi=self.config_model.pipeline.result_formatter.rendered_image_dpi,
        )

        page_metadata = None
        page_number_candidates = None
        document_page_numbering = None
        if self.config_model.pipeline.result_formatter.detect_printed_page_numbers:
            from glmocr.postprocess import ResultFormatter

            formatter = ResultFormatter(self.config_model.pipeline.result_formatter)
            (
                page_number_candidates,
                document_page_numbering,
                page_metadata,
            ) = formatter.extract_printed_page_data(json_result)

        from glmocr.postprocess import ResultFormatter

        ResultFormatter._strip_layout_metadata(json_result)

        # Create PipelineResult
        result = PipelineResult(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=[source],
            image_files=image_files or None,
            page_metadata=page_metadata,
            page_number_candidates=page_number_candidates,
            document_page_numbering=document_page_numbering,
        )

        # Store additional MaaS response data
        result._maas_response = response
        result._layout_visualization = response.get("layout_visualization", [])
        result._data_info = response.get("data_info", {})
        result._usage = response.get("usage", {})

        return result

    def _build_selfhosted_request(
        self,
        images: List[Union[str, bytes, Path]],
    ) -> Dict[str, Any]:
        """Build request from mixed inputs (paths, URLs, or raw bytes)."""
        messages: List[Dict[str, Any]] = [{"role": "user", "content": []}]
        for image in images:
            if isinstance(image, bytes):
                messages[0]["content"].append({"type": "image_bytes", "data": image})
            else:
                url = self._to_url(image)
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
        return {"messages": messages}

    def _parse_selfhosted(
        self,
        images: List[Union[str, bytes, Path]],
        save_layout_visualization: bool = True,
        preserve_order: bool = True,
    ) -> List[PipelineResult]:
        """Parse using self-hosted vLLM/SGLang pipeline."""
        request_data = self._build_selfhosted_request(images)
        results = list(
            self._pipeline.process(
                request_data,
                save_layout_visualization=save_layout_visualization,
                preserve_order=preserve_order,
            )
        )
        return results

    def _stream_parse_selfhosted(
        self,
        images: List[Union[str, bytes, Path]],
        save_layout_visualization: bool = True,
        preserve_order: bool = True,
    ) -> Generator[PipelineResult, None, None]:
        """Streaming variant of self-hosted parse()."""
        request_data = self._build_selfhosted_request(images)
        for result in self._pipeline.process(
            request_data,
            save_layout_visualization=save_layout_visualization,
            preserve_order=preserve_order,
        ):
            yield result

    def parse_maas(
        self,
        source: Union[str, Path, bytes],
        return_crop_images: bool = False,
        need_layout_visualization: bool = False,
        start_page_id: Optional[int] = None,
        end_page_id: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Direct MaaS API call (raw response).

        This method provides direct access to the MaaS API response without
        converting to PipelineResult. Useful when you need the full API response.

        Only available when maas.enabled=true in config.

        Args:
            source: File path, URL, or bytes.
            return_crop_images: Whether to return cropped images.
            need_layout_visualization: Whether to return layout visualization.
            start_page_id: Start page for PDF (1-indexed).
            end_page_id: End page for PDF (1-indexed).
            **kwargs: Additional API parameters.

        Returns:
            Raw MaaS API response dict.

        Raises:
            RuntimeError: If not in MaaS mode.
        """
        if not self._use_maas:
            raise RuntimeError(
                "parse_maas() is only available when maas.enabled=true in config"
            )

        return self._maas_client.parse(
            source,
            return_crop_images=return_crop_images,
            need_layout_visualization=need_layout_visualization,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            **kwargs,
        )

    def get_queue_stats(self) -> Optional[Dict[str, int]]:
        """Return current pipeline queue sizes, or ``None`` if unavailable."""
        if self._pipeline is not None:
            return self._pipeline.get_queue_stats()
        return None

    def close(self):
        """Close the parser and release resources."""
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        if self._maas_client:
            self._maas_client.stop()
            self._maas_client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor."""
        try:
            self.close()
        except Exception:
            pass


# Convenience function
@overload
def parse(
    images: Union[str, bytes, Path],
    config_path: Optional[str] = ...,
    save_layout_visualization: bool = ...,
) -> PipelineResult: ...


@overload
def parse(
    images: List[Union[str, bytes, Path]],
    config_path: Optional[str] = ...,
    save_layout_visualization: bool = ...,
) -> List[PipelineResult]: ...


@overload
def parse(
    images: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
    config_path: Optional[str] = ...,
    save_layout_visualization: bool = ...,
    *,
    stream: Literal[True],
    **kwargs: Any,
) -> Generator[PipelineResult, None, None]: ...


def parse(
    images: Union[str, bytes, Path, List[Union[str, bytes, Path]]],
    config_path: Optional[str] = None,
    save_layout_visualization: bool = True,
    *,
    stream: bool = False,
    preserve_order: bool = True,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    timeout: Optional[int] = None,
    log_level: Optional[str] = None,
    env_file: Optional[str] = None,
    **kwargs: Any,
) -> Union[PipelineResult, List[PipelineResult], Generator[PipelineResult, None, None]]:
    """Convenience function: parse images or documents in one call.

    Creates a :class:`GlmOcr` instance, runs parsing, and cleans up.

    Examples::

        import glmocr

        # Minimal – only needs ZHIPU_API_KEY env var
        results = glmocr.parse("image.png")

        # Explicit API key
        results = glmocr.parse("image.png", api_key="sk-xxx")

        # Self-hosted mode
        results = glmocr.parse("image.png", mode="selfhosted")

        for r in glmocr.parse(["a.pdf", "b.pdf"], stream=True):
            r.save(output_dir="./output")

    Args:
        images: Single input or list.  Each element can be ``str`` (path/URL),
            ``bytes`` (raw image/PDF), or ``pathlib.Path``.
        config_path: Config file path.
        save_layout_visualization: Whether to save layout visualization.
        stream: If ``True``, returns a generator.
        preserve_order: Whether to keep output order consistent with input order.
        api_key:  API key.
        api_url:  MaaS API endpoint URL.
        model:    Model name.
        mode:     ``"maas"`` or ``"selfhosted"``.
        timeout:  Request timeout in seconds.
        log_level: Logging level.
    """
    with GlmOcr(
        config_path=config_path,
        api_key=api_key,
        api_url=api_url,
        model=model,
        mode=mode,
        timeout=timeout,
        log_level=log_level,
        env_file=env_file,
    ) as parser:
        return parser.parse(
            images,
            stream=stream,
            save_layout_visualization=save_layout_visualization,
            preserve_order=preserve_order,
            **kwargs,
        )
