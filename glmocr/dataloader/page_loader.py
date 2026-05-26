"""Page loader - unified document/image loading and preprocessing.

PageLoader is responsible for:
1. Loading various input formats (images, PDFs, base64, URLs)
2. Converting inputs into a list of PIL Images
3. Building OCR API request payloads

Supported inputs:
- Local file paths (images, PDFs)
- file:// URLs
- data:image/... base64 URLs
- data:application/pdf;base64,... data URIs
"""

from __future__ import annotations

import os
import base64
import time
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple, Union

from PIL import Image


from glmocr.utils.image_utils import (
    load_image_to_base64,
    pdf_to_images_pil,
    pdf_to_images_pil_iter,
)
from glmocr.utils.logging import get_logger, get_profiler

Image.MAX_IMAGE_PIXELS = None
if TYPE_CHECKING:
    from glmocr.config import PageLoaderConfig

logger = get_logger(__name__)
profiler = get_profiler(__name__)


class PageLoader:
    """Page loader.

    Unifies image and PDF inputs and provides two output modes:
    1. load_pages(): returns a list of PIL Images (for layout detection or custom logic)
    2. build_request(): builds OCR API request payloads

    Example:
        from glmocr.config import PageLoaderConfig

        loader = PageLoader(PageLoaderConfig())

        # Load as PIL images
        pages = loader.load_pages(["doc.pdf", "image.png"])

        # Build an API request
        request_data = loader.build_request(request_data)
    """

    def __init__(self, config: "PageLoaderConfig"):
        """Initialize.

        Args:
            config: PageLoaderConfig instance.
        """
        self.config = config

        # Image processing parameters
        self.t_patch_size = config.t_patch_size
        self.patch_expand_factor = config.patch_expand_factor
        self.image_expect_length = config.image_expect_length
        self.image_format = config.image_format
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels

        # API request parameters
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.top_k = config.top_k
        self.repetition_penalty = config.repetition_penalty

        # Task prompt mapping
        self.task_prompt_mapping = config.task_prompt_mapping

        # PDF-to-image parameters
        self.pdf_dpi = config.pdf_dpi
        self.pdf_max_pages = config.pdf_max_pages
        self.pdf_verbose = config.pdf_verbose

    # =========================================================================
    # Page loading
    # =========================================================================

    def load_pages(
        self, sources: Union[str, bytes, List[Union[str, bytes]]]
    ) -> List[Image.Image]:
        """Load sources into a list of PIL Images.

        Supports image files, PDFs, and raw bytes (PDFs are expanded into
        multiple pages).

        Args:
            sources: Single path/URL/bytes or a list.

        Returns:
            List[PIL.Image.Image]
        """
        if isinstance(sources, (str, bytes)):
            sources = [sources]

        all_pages = []
        for source in sources:
            pages = self._load_source(source)
            all_pages.extend(pages)

        return all_pages

    def load_pages_with_unit_indices(
        self, sources: Union[str, bytes, List[Union[str, bytes]]]
    ) -> Tuple[List[Image.Image], List[int]]:
        """Load sources into pages and return unit index per page.

        Each input is one "unit". For a PDF, all its pages share the same
        unit index. Used by streaming mode to yield one result per input unit.

        Args:
            sources: Single path/URL/bytes or a list.

        Returns:
            (all_pages, unit_indices) where unit_indices[i] is the unit index
            of page i (i.e. which input it came from).
        """
        if isinstance(sources, (str, bytes)):
            sources = [sources]

        all_pages: List[Image.Image] = []
        unit_indices: List[int] = []
        for unit_idx, source in enumerate(sources):
            pages = self._load_source(source)
            all_pages.extend(pages)
            unit_indices.extend([unit_idx] * len(pages))
        return all_pages, unit_indices

    def iter_pages_with_unit_indices(
        self, sources: Union[str, bytes, List[Union[str, bytes]]]
    ):
        """Stream pages one at a time with unit index per page.

        Yields (page, unit_idx) so the pipeline can enqueue each page as soon
        as it is rendered (e.g. PDF: render one page → yield → next page).

        Args:
            sources: Single path/URL/bytes or a list.

        Yields:
            (PIL.Image, unit_idx) for each page.
        """
        if isinstance(sources, (str, bytes)):
            sources = [sources]
        for unit_idx, source in enumerate(sources):
            try:
                for page in self._iter_source(source):
                    yield page, unit_idx
            except Exception as e:
                logger.warning("Skipping source (unit %d): %s", unit_idx, e)

    def _iter_source(self, source: Union[str, bytes]):
        """Yield pages from a single source one at a time."""
        if isinstance(source, bytes):
            if source[:5] == b"%PDF-":
                yield from self._iter_pdf_bytes(source)
            else:
                yield Image.open(BytesIO(source))
            return

        # data:application/pdf;base64,... data URI
        if source.startswith("data:application/pdf"):
            _, b64_data = source.split(",", 1)
            yield from self._iter_pdf_bytes(base64.b64decode(b64_data))
            return

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            yield from self._iter_pdf(file_path)
        else:
            yield self._load_image(source)

    def _compute_end_page(self) -> Optional[int]:
        """Parse pdf_max_pages into 0-based inclusive end page index, or None for last page."""
        if self.pdf_max_pages is None:
            return None
        try:
            mp = int(self.pdf_max_pages)
            if mp > 0:
                return mp - 1  # 0-based inclusive
        except Exception:
            pass
        return None

    def _iter_pdf(self, file_path: str):
        """Yield PDF pages one at a time (streaming)."""
        end_page = self._compute_end_page()
        for image in pdf_to_images_pil_iter(
            file_path,
            dpi=self.pdf_dpi,
            max_width_or_height=3500,
            start_page_id=0,
            end_page_id=end_page,
        ):
            yield image

    def _iter_pdf_bytes(self, data: bytes):
        """Yield PDF pages from raw bytes one at a time."""
        end_page = self._compute_end_page()
        for image in pdf_to_images_pil_iter(
            data,
            dpi=self.pdf_dpi,
            max_width_or_height=3500,
            start_page_id=0,
            end_page_id=end_page,
        ):
            yield image

    def _load_pdf_bytes(self, data: bytes) -> List[Image.Image]:
        """Load all pages from PDF bytes."""
        t0 = time.perf_counter()
        end_page = self._compute_end_page()
        pages = pdf_to_images_pil(
            data,
            dpi=self.pdf_dpi,
            max_width_or_height=3500,
            start_page_id=0,
            end_page_id=end_page,
        )
        profiler.log(
            "pdf_to_images_pil(<bytes>)",
            (time.perf_counter() - t0) * 1000,
        )
        return pages

    def _load_source(self, source: Union[str, bytes]) -> List[Image.Image]:
        """Load a single source and return a list of pages.

        PDFs return all pages; images return a single-page list.
        """
        if isinstance(source, bytes):
            if source[:5] == b"%PDF-":
                return self._load_pdf_bytes(source)
            return [Image.open(BytesIO(source))]

        # data:application/pdf;base64,... data URI
        if source.startswith("data:application/pdf"):
            _, b64_data = source.split(",", 1)
            return self._load_pdf_bytes(base64.b64decode(b64_data))

        if source.startswith("file://"):
            file_path = source[7:]
        else:
            file_path = source

        # Detect PDF
        if os.path.isfile(file_path) and file_path.lower().endswith(".pdf"):
            return self._load_pdf(file_path)

        # Otherwise load as a single image page
        return [self._load_image(source)]

    def _load_image(self, source: str) -> Image.Image:
        """Load a single image."""
        try:
            # data:image/... URL
            if source.startswith("data:image"):
                header, base64_data = source.split(",", 1)
                image_data = base64.b64decode(base64_data)
                return Image.open(BytesIO(image_data))

            elif source.startswith("file://"):
                return Image.open(source[7:])

            # Local file
            elif os.path.isfile(source):
                return Image.open(source)

            else:
                raise ValueError(f"Invalid image source: {source}")

        except Exception as e:
            raise RuntimeError(f"Error loading image '{source}': {e}")

    def _load_pdf(self, file_path: str) -> List[Image.Image]:
        """Load all pages from a PDF file."""
        t0 = time.perf_counter()
        end_page = self._compute_end_page()
        pages = pdf_to_images_pil(
            file_path,
            dpi=self.pdf_dpi,
            max_width_or_height=3500,
            start_page_id=0,
            end_page_id=end_page,
        )
        profiler.log(
            f"pdf_to_images_pil({os.path.basename(file_path)})",
            (time.perf_counter() - t0) * 1000,
        )
        return pages

    # =========================================================================
    # API request building
    # =========================================================================

    def build_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build an OCR API request from request_data.

        Args:
            request_data: Raw request data containing messages.

        Returns:
            Updated request data.
        """
        # Set default parameters
        if "max_tokens" not in request_data:
            request_data["max_tokens"] = self.max_tokens
        if "top_p" not in request_data:
            request_data["top_p"] = self.top_p
        if "temperature" not in request_data:
            request_data["temperature"] = self.temperature
        if "top_k" not in request_data:
            request_data["top_k"] = self.top_k
        if "repetition_penalty" not in request_data:
            request_data["repetition_penalty"] = self.repetition_penalty

        # Process messages
        messages = request_data["messages"]
        processed_messages = []

        for msg in messages:
            if msg["role"] in ("system", "assistant", "tool"):
                processed_messages.append(msg)
            elif msg["role"] in ("user", "observation"):
                processed_messages.append(self._process_msg_standard(msg))
            else:
                raise ValueError(f"{msg['role']} is not a valid role for a message.")

        request_data["messages"] = processed_messages
        return request_data

    def build_request_from_image(
        self, image: Image.Image, task_type: str = "text"
    ) -> Dict[str, Any]:
        """Build an API request from a PIL Image.

        Args:
            image: PIL Image.
            task_type: Task type (text/table/formula, etc.).

        Returns:
            Full API request payload.
        """
        prompt_text = ""
        if self.task_prompt_mapping:
            prompt_text = self.task_prompt_mapping.get(task_type, "")

        encoded_image = load_image_to_base64(
            image,
            t_patch_size=self.t_patch_size,
            max_pixels=self.max_pixels,
            image_format=self.image_format,
            patch_expand_factor=self.patch_expand_factor,
            min_pixels=self.min_pixels,
        )

        content: list = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{self.image_format.lower()};base64,{encoded_image}"
                },
            },
        ]
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})

        return {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }

    def _process_msg_standard(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Standard mode: encode images inside a message."""
        msg_content: List[Dict] = msg["content"]
        processed_content = []

        for content in msg_content:
            if content["type"] == "text":
                processed_content.append(content)
            elif content["type"] == "image_url":
                image_url = content["image_url"]["url"]
                with profiler.measure("load_image_to_base64"):
                    encoded_image = load_image_to_base64(
                        image_url,
                        t_patch_size=self.t_patch_size,
                        max_pixels=self.max_pixels,
                        image_format=self.image_format,
                        patch_expand_factor=self.patch_expand_factor,
                        min_pixels=self.min_pixels,
                    )
                processed_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{self.image_format.lower()};base64,{encoded_image}"
                        },
                    }
                )
            else:
                raise ValueError(f"{content['type']} is not a valid type.")

        return {"role": msg["role"], "content": processed_content}
