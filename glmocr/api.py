"""GLM-OCR Python API

Python API for calling the document parsing pipeline from your code.

Two modes are supported:
1. MaaS Mode (maas.enabled=true): Forwards requests to Zhipu's cloud API.
   No GPU required; the cloud handles all processing.
2. Self-hosted Mode (maas.enabled=false): Uses local vLLM/SGLang service.
   Requires GPU; SDK handles layout detection, parallel OCR, etc.

Agent-friendly usage::

    # Only needs ZHIPU_API_KEY in environment (or pass api_key directly)
    from glmocr import GlmOcr

    parser = GlmOcr(api_key="sk-xxx", mode="maas")
    results = parser.parse("document.png")
    print(results[0].to_dict())
"""

import json
import os
import re
import shutil
import tempfile
from typing import Any, Dict, Generator, List, Literal, Optional, Union, overload
from pathlib import Path

from glmocr.config import load_config
from glmocr.parser_result import PipelineResult
from glmocr.utils.logging import get_logger, ensure_logging_configured

logger = get_logger(__name__)

# Backward compatibility: ParseResult is PipelineResult
ParseResult = PipelineResult

# Default extraction prompt used by GLM-OCR information extraction mode.
_DEFAULT_EXTRACTION_PROMPT = "请按下列JSON格式输出图中信息:"


def _json_schema_to_template(schema: Dict[str, Any]) -> Any:
    """Convert a JSON Schema dict to an empty-value template for GLM-OCR.

    Handles ``$defs``/``definitions``, ``$ref``, ``allOf``/``anyOf``/``oneOf``,
    nested objects, and arrays.  All leaf values become ``""``.
    """
    defs = schema.get("$defs", schema.get("definitions", {}))

    def _resolve(s: Any) -> Any:
        if not isinstance(s, dict):
            return ""
        if "$ref" in s:
            ref_name = s["$ref"].rsplit("/", 1)[-1]
            if ref_name in defs:
                return _convert(defs[ref_name])
            return ""
        return _convert(s)

    def _convert(s: dict) -> Any:
        for key in ("allOf", "anyOf", "oneOf"):
            if key in s and s[key]:
                return _resolve(s[key][0])
        schema_type = s.get("type", "string")
        if schema_type == "object":
            props = s.get("properties", {})
            return {k: _resolve(v) for k, v in props.items()}
        if schema_type == "array":
            return [_resolve(s.get("items", {}))]
        return ""

    return _resolve(schema)


def _resolve_schema_template(schema: Any) -> Dict[str, Any]:
    """Normalise *schema* into the empty-value JSON template GLM-OCR expects.

    Accepted inputs:

    * **dict without ``"type"``/``"properties"``** – treated as a ready-made
      template (the format shown in the GLM-OCR docs).
    * **JSON Schema dict** (has ``"type": "object"`` + ``"properties"``) –
      converted automatically.  This is what Zod produces via
      ``zodToJsonSchema()``.
    * **Pydantic model class** – calls ``model_json_schema()`` then converts.
    """
    # Pydantic v2 model class
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        return _json_schema_to_template(schema.model_json_schema())

    if not isinstance(schema, dict):
        raise TypeError(
            f"schema must be a dict or Pydantic model class, got {type(schema)}"
        )

    # JSON Schema dict
    if schema.get("type") == "object" and "properties" in schema:
        return _json_schema_to_template(schema)

    # Already a raw template dict
    return schema


def _parse_json_from_text(text: str) -> Any:
    """Best-effort extraction of a JSON object from *text*.

    Tries direct ``json.loads`` first, then falls back to extracting from
    Markdown fenced code blocks.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try Markdown ```json ... ``` blocks
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse extraction response as JSON: {text[:500]}")


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

        # --- Parse ---
        results = parser.parse("image.png")
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
        enable_layout: Optional[bool] = None,
        log_level: Optional[str] = None,
        env_file: Optional[str] = None,
        # Extra knobs for self-hosted mode & GPU binding
        ocr_api_host: Optional[str] = None,
        ocr_api_port: Optional[int] = None,
        cuda_visible_devices: Optional[str] = None,
        layout_device: Optional[str] = None,
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
            enable_layout: Whether to run layout detection.
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
            enable_layout=enable_layout,
            log_level=log_level,
            env_file=env_file,
            ocr_api_host=ocr_api_host,
            ocr_api_port=ocr_api_port,
            cuda_visible_devices=cuda_visible_devices,
            layout_device=layout_device,
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
            self.enable_layout = True  # MaaS always includes layout
            logger.info("GLM-OCR initialized in MaaS mode (cloud API passthrough)")
        else:
            # Self-hosted mode: use full Pipeline
            from glmocr.pipeline import Pipeline

            self._pipeline = Pipeline(config=self.config_model.pipeline)
            self.enable_layout = self._pipeline.enable_layout
            self._pipeline.start()
            logger.info("GLM-OCR initialized in self-hosted mode")

    # Type alias for accepted input sources
    InputSource = Union[str, bytes, Path]

    @overload
    def parse(
        self,
        images: "GlmOcr.InputSource",
        *,
        stream: Literal[False] = ...,
        save_layout_visualization: bool = ...,
        **kwargs: Any,
    ) -> PipelineResult: ...

    @overload
    def parse(
        self,
        images: List["GlmOcr.InputSource"],
        *,
        stream: Literal[False] = ...,
        save_layout_visualization: bool = ...,
        **kwargs: Any,
    ) -> List[PipelineResult]: ...

    @overload
    def parse(
        self,
        images: Union["GlmOcr.InputSource", List["GlmOcr.InputSource"]],
        *,
        stream: Literal[True],
        save_layout_visualization: bool = ...,
        **kwargs: Any,
    ) -> Generator[PipelineResult, None, None]: ...

    def parse(
        self,
        images: Union["GlmOcr.InputSource", List["GlmOcr.InputSource"]],
        *,
        stream: bool = False,
        save_layout_visualization: bool = True,
        **kwargs: Any,
    ) -> Union[
        PipelineResult, List[PipelineResult], Generator[PipelineResult, None, None]
    ]:
        """Predict / parse images or documents.

        Supports local paths, ``Path`` objects, URLs (file://, http://, https://,
        data:// — including presigned URLs), and raw ``bytes``.
        Supports image files (jpg, png, bmp, gif, webp) and PDF files.

        Args:
            images: A single input or list of inputs. Each input can be:

                - ``str``: local file path, or URL (http/https/file/data).
                  Presigned URLs (e.g. S3) are supported.
                - ``bytes``: raw file content (image or PDF).
                  Useful for multipart/form-data uploads.
                - ``Path``: a ``pathlib.Path`` to a local file.

            stream: If ``True``, yields one :class:`PipelineResult` at a time (avoids
                holding all results in memory). If ``False``, returns a single result
                or a list, depending on *images*.
            save_layout_visualization: Whether to save layout visualization artifacts.
            **kwargs: Additional parameters for MaaS mode (return_crop_images,
                     need_layout_visualization, start_page_id, end_page_id, etc.)

        Returns:
            - When ``stream=False`` (default): a single ``PipelineResult`` if *images*
              is a single input, or a ``List[PipelineResult]`` if *images* is a list.
            - When ``stream=True``: a generator that yields one ``PipelineResult``
              per input.

        Example:
            # Single file — returns one PipelineResult
            result = parser.parse("image.png")

            # Path object
            result = parser.parse(Path("document.pdf"))

            # Presigned URL
            result = parser.parse("https://bucket.s3.amazonaws.com/doc.pdf?X-Amz-...")

            # Raw bytes (e.g. from a multipart/form-data upload)
            result = parser.parse(uploaded_file.read())

            # Mixed list
            results = parser.parse([b"...pdf bytes...", "https://presigned/img.png"])

            # Stream to avoid large in-memory results
            for r in parser.parse(["a.pdf", "b.pdf"], stream=True):
                r.save(output_dir="./output")
        """
        _single = isinstance(images, (str, bytes, Path))
        if _single:
            images = [images]

        if stream:
            return self._parse_stream(images, save_layout_visualization, **kwargs)

        if self._use_maas:
            result_list = self._parse_maas(images, save_layout_visualization, **kwargs)
        else:
            result_list = self._parse_selfhosted(images, save_layout_visualization)

        return result_list[0] if _single else result_list

    @staticmethod
    def _guess_suffix(data: bytes) -> str:
        """Guess file suffix from magic bytes."""
        if data[:5] == b"%PDF-":
            return ".pdf"
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        if data[:3] == b"GIF":
            return ".gif"
        if data[:2] == b"BM":
            return ".bmp"
        return ".bin"

    def _resolve_inputs(
        self, images: List[Union[str, bytes, Path]]
    ) -> tuple:
        """Convert bytes/Path inputs to file path strings.

        Returns:
            (resolved_paths, temp_dir) — *temp_dir* is ``None`` when no temp
            files were created; otherwise the caller must clean it up.
        """
        resolved: List[str] = []
        temp_dir: Optional[str] = None

        for idx, img in enumerate(images):
            if isinstance(img, bytes):
                if temp_dir is None:
                    temp_dir = tempfile.mkdtemp(prefix="glmocr_upload_")
                suffix = self._guess_suffix(img)
                path = os.path.join(temp_dir, f"input_{idx}{suffix}")
                with open(path, "wb") as f:
                    f.write(img)
                resolved.append(path)
            elif isinstance(img, Path):
                resolved.append(str(img.absolute()))
            else:
                resolved.append(str(img))

        return resolved, temp_dir

    def _parse_stream(
        self,
        images: List[str],
        save_layout_visualization: bool = True,
        **kwargs: Any,
    ) -> Generator[PipelineResult, None, None]:
        """Internal: yield one PipelineResult per input. Used by parse(stream=True)."""
        if self._use_maas:
            if save_layout_visualization:
                kwargs.setdefault("need_layout_visualization", True)
            for image in images:
                img = image
                if img.startswith("file://"):
                    img = img[7:]
                try:
                    response = self._maas_client.parse(img, **kwargs)
                    result = self._maas_response_to_pipeline_result(response, img)
                    yield result
                except Exception as e:
                    logger.error("MaaS API error for %s: %s", img, e)
                    result = PipelineResult(
                        json_result=[],
                        markdown_result="",
                        original_images=[img],
                    )
                    result._error = str(e)
                    yield result
            return
        for result in self._stream_parse_selfhosted(
            images,
            save_layout_visualization=save_layout_visualization,
        ):
            yield result

    def _parse_maas(
        self,
        images: List[str],
        save_layout_visualization: bool = True,
        **kwargs,
    ) -> List[PipelineResult]:
        """Parse using MaaS API (passthrough mode)."""
        results = []

        # Map save_layout_visualization to MaaS API parameter
        if save_layout_visualization:
            kwargs.setdefault("need_layout_visualization", True)

        for image in images:
            # Resolve file:// URLs to actual paths
            if image.startswith("file://"):
                image = image[7:]

            try:
                response = self._maas_client.parse(image, **kwargs)
                result = self._maas_response_to_pipeline_result(response, image)
                results.append(result)
            except Exception as e:
                logger.error("MaaS API error for %s: %s", image, e)
                # Return an error result
                result = PipelineResult(
                    json_result=[],
                    markdown_result="",
                    original_images=[image],
                )
                result._error = str(e)
                results.append(result)

        return results

    # ------------------------------------------------------------------
    # MaaS bbox coordinate conversion
    # ------------------------------------------------------------------
    # The MaaS API returns bbox_2d in **absolute pixel coordinates** of
    # its own internal rendering (e.g. 2040×2640 for a letter-sized PDF
    # page).  The rest of the SDK (self-hosted pipeline, crop_image_region,
    # crop_and_replace_images) uses **normalised 0-1000 coordinates**.
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
        normalised 0-1000 values so that ``crop_and_replace_images`` crops
        from the correct region.
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
                        "content": region.get("content", ""),
                        "bbox_2d": bbox,
                    }
                )
            json_result.append(page_result)

        # Get markdown result and normalise the bbox refs inside it.
        markdown_result = response.get("md_results", "")
        markdown_result = self._normalise_markdown_bboxes(
            markdown_result,
            pages_info,
        )

        # Create PipelineResult
        result = PipelineResult(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=[source],
        )

        # Store additional MaaS response data
        result._maas_response = response
        result._layout_visualization = response.get("layout_visualization", [])
        result._data_info = response.get("data_info", {})
        result._usage = response.get("usage", {})

        return result

    def _parse_selfhosted(
        self,
        images: List[Union[str, bytes, Path]],
        save_layout_visualization: bool = True,
    ) -> List[PipelineResult]:
        """Parse using self-hosted vLLM/SGLang pipeline."""
        resolved, temp_dir = self._resolve_inputs(images)
        try:
            messages = [{"role": "user", "content": []}]
            for image in resolved:
                if image.startswith(("http://", "https://", "data:", "file://")):
                    url = image
                else:
                    url = f"file://{Path(image).absolute()}"
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
            request_data = {"messages": messages}

            layout_vis_dir = None
            if self._pipeline.enable_layout and save_layout_visualization:
                layout_vis_dir = tempfile.mkdtemp(prefix="layout_vis_")

            results = list(
                self._pipeline.process(
                    request_data,
                    save_layout_visualization=save_layout_visualization,
                    layout_vis_output_dir=layout_vis_dir,
                )
            )
            return results
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _stream_parse_selfhosted(
        self,
        images: List[Union[str, bytes, Path]],
        save_layout_visualization: bool = True,
    ) -> Generator[PipelineResult, None, None]:
        """Streaming variant of self-hosted parse().

        Wraps ``Pipeline.process(...)`` and yields results as soon as they
        become available from the async pipeline.
        """
        resolved, temp_dir = self._resolve_inputs(images)
        try:
            messages = [{"role": "user", "content": []}]
            for image in resolved:
                if image.startswith(("http://", "https://", "data:", "file://")):
                    url = image
                else:
                    url = f"file://{Path(image).absolute()}"
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": url}}
                )
            request_data = {"messages": messages}

            layout_vis_dir = None
            if self._pipeline.enable_layout and save_layout_visualization:
                layout_vis_dir = tempfile.mkdtemp(prefix="layout_vis_")

            for result in self._pipeline.process(
                request_data,
                save_layout_visualization=save_layout_visualization,
                layout_vis_output_dir=layout_vis_dir,
            ):
                yield result
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def extract(
        self,
        images: Union["GlmOcr.InputSource", List["GlmOcr.InputSource"]],
        *,
        schema: Union[Dict[str, Any], type],
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Extract structured data from documents according to *schema*.

        Uses GLM-OCR's information extraction mode: the model receives the
        document image together with a JSON template and returns a populated
        version of that template.

        Args:
            images: One or more document images (paths, URLs, bytes, or Path
                objects).
            schema: Describes the desired output structure.  Accepts:

                - A **dict with empty values** (GLM-OCR native template)::

                      {"invoice_no": "", "total": "", "items": [{"desc": "", "qty": ""}]}

                - A **JSON Schema dict** (what Zod's ``zodToJsonSchema()``
                  produces)::

                      {"type": "object", "properties": {"invoice_no": {"type": "string"}, ...}}

                - A **Pydantic model class**::

                      class Invoice(BaseModel):
                          invoice_no: str
                          total: str

            prompt: Custom prompt prefix.  Defaults to the standard Chinese
                extraction prompt used by GLM-OCR.
            **kwargs: Extra parameters forwarded to the MaaS / self-hosted API.

        Returns:
            A single ``dict`` when *images* is a single input, or a
            ``list[dict]`` when *images* is a list.

        Raises:
            ValueError: If the model response cannot be parsed as JSON.
            RuntimeError: If used in self-hosted mode (not yet supported).

        Example::

            # --- Raw template (GLM-OCR native) ---
            data = parser.extract("id_card.png", schema={
                "id_number": "",
                "name": "",
                "date_of_birth": "",
            })

            # --- JSON Schema (from Zod via zodToJsonSchema) ---
            data = parser.extract("invoice.pdf", schema={
                "type": "object",
                "properties": {
                    "invoice_no": {"type": "string"},
                    "total": {"type": "number"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "amount": {"type": "number"},
                            },
                        },
                    },
                },
            })

            # --- Pydantic model ---
            from pydantic import BaseModel

            class IdCard(BaseModel):
                id_number: str
                name: str
                date_of_birth: str

            data = parser.extract("id_card.png", schema=IdCard)
        """
        template = _resolve_schema_template(schema)
        prefix = prompt or _DEFAULT_EXTRACTION_PROMPT
        full_prompt = f"{prefix}\n{json.dumps(template, ensure_ascii=False, indent=4)}"

        _single = isinstance(images, (str, bytes, Path))
        if _single:
            images = [images]

        if not self._use_maas:
            raise RuntimeError(
                "extract() currently requires MaaS mode. "
                "Initialize with mode='maas' or set maas.enabled=true."
            )

        results: List[Dict[str, Any]] = []
        for image in images:
            img = image
            if isinstance(img, str) and img.startswith("file://"):
                img = img[7:]

            response = self._maas_client.parse(img, prompt=full_prompt, **kwargs)
            md_results = response.get("md_results", "")
            extracted = _parse_json_from_text(md_results)
            results.append(extracted)

        return results[0] if _single else results

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
    images: str,
    config_path: Optional[str] = ...,
    save_layout_visualization: bool = ...,
) -> PipelineResult: ...


@overload
def parse(
    images: List[str],
    config_path: Optional[str] = ...,
    save_layout_visualization: bool = ...,
) -> List[PipelineResult]: ...


@overload
def parse(
    images: Union[str, List[str]],
    config_path: Optional[str] = ...,
    save_layout_visualization: bool = ...,
    *,
    stream: Literal[True],
    **kwargs: Any,
) -> Generator[PipelineResult, None, None]: ...


def parse(
    images: Union[str, List[str]],
    config_path: Optional[str] = None,
    save_layout_visualization: bool = True,
    *,
    stream: bool = False,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    timeout: Optional[int] = None,
    enable_layout: Optional[bool] = None,
    log_level: Optional[str] = None,
    env_file: Optional[str] = None,
    **kwargs: Any,
) -> Union[PipelineResult, List[PipelineResult], Generator[PipelineResult, None, None]]:
    """Convenience function: parse images or documents in one call.

    Creates a :class:`GlmOcr` instance, runs parsing, and cleans up.
    All keyword arguments are forwarded to the ``GlmOcr`` constructor.

    Examples::

        import glmocr

        # Minimal – only needs ZHIPU_API_KEY env var
        results = glmocr.parse("image.png")

        # Explicit API key
        results = glmocr.parse("image.png", api_key="sk-xxx")

        # Self-hosted mode
        results = glmocr.parse("image.png", mode="selfhosted")

        # Stream to avoid large in-memory results
        for r in glmocr.parse(["a.pdf", "b.pdf"], stream=True):
            r.save(output_dir="./output")

    The return type mirrors the input type and stream:
    - ``str``, stream=False → ``PipelineResult``
    - ``List[str]``, stream=False → ``List[PipelineResult]``
    - ``stream=True`` → ``Generator[PipelineResult, None, None]``

    Args:
        images: Image path or URL (single ``str`` or ``List[str]``).
        config_path: Config file path.
        save_layout_visualization: Whether to save layout visualization.
        stream: If ``True``, returns a generator that yields one result at a time.
        api_key:  API key.
        api_url:  MaaS API endpoint URL.
        model:    Model name.
        mode:     ``"maas"`` or ``"selfhosted"``.
        timeout:  Request timeout in seconds.
        enable_layout: Whether to run layout detection.
        log_level: Logging level.

    Returns:
        A single ``PipelineResult``, a list, or a generator, depending on input and stream.

    Example:
        result = parse("image.png")
        result.save(output_dir="./output")

        results = parse(["img1.png", "doc.pdf"])
        for r in results:
            r.save(output_dir="./output")

        for r in parse(["a.pdf", "b.pdf"], stream=True):
            r.save(output_dir="./output")
    """
    with GlmOcr(
        config_path=config_path,
        api_key=api_key,
        api_url=api_url,
        model=model,
        mode=mode,
        timeout=timeout,
        enable_layout=enable_layout,
        log_level=log_level,
        env_file=env_file,
    ) as parser:
        return parser.parse(
            images,
            stream=stream,
            save_layout_visualization=save_layout_visualization,
            **kwargs,
        )


def extract(
    images: Union[str, List[str]],
    *,
    schema: Union[Dict[str, Any], type],
    prompt: Optional[str] = None,
    config_path: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    mode: Optional[str] = None,
    timeout: Optional[int] = None,
    log_level: Optional[str] = None,
    env_file: Optional[str] = None,
    **kwargs: Any,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Convenience function: extract structured data in one call.

    Creates a :class:`GlmOcr` instance, runs extraction, and cleans up.

    Examples::

        import glmocr

        data = glmocr.extract(
            "id_card.png",
            schema={"id_number": "", "name": "", "date_of_birth": ""},
            api_key="sk-xxx",
        )

    Args:
        images: Image path or URL (single ``str`` or ``list[str]``).
        schema: Extraction schema (template dict, JSON Schema, or Pydantic model).
        prompt: Custom extraction prompt prefix.
        config_path: Config file path.
        api_key:  API key.
        api_url:  MaaS API endpoint URL.
        model:    Model name.
        mode:     ``"maas"`` or ``"selfhosted"``.
        timeout:  Request timeout in seconds.
        log_level: Logging level.
        env_file: Path to ``.env`` file.

    Returns:
        A single ``dict`` or a ``list[dict]``, depending on input.
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
        return parser.extract(images, schema=schema, prompt=prompt, **kwargs)
