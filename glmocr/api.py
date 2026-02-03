"""GLM-OCR Python API

Python API for calling the document parsing pipeline from your code.

Two modes are supported:
1. MaaS Mode (maas.enabled=true): Forwards requests to Zhipu's cloud API.
   No GPU required; the cloud handles all processing.
2. Self-hosted Mode (maas.enabled=false): Uses local vLLM/SGLang service.
   Requires GPU; SDK handles layout detection, parallel OCR, etc.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from glmocr.config import load_config
from glmocr.parser_result import PipelineResult
from glmocr.utils.logging import get_logger, ensure_logging_configured

logger = get_logger(__name__)

# Backward compatibility: ParseResult is PipelineResult
ParseResult = PipelineResult


class GlmOcr:
    """Main GLM-OCR entrypoint.

    Provides a Python API for document parsing. Automatically detects whether
    to use MaaS mode or self-hosted mode based on config.

    Example:
        from glmocr.api import GlmOcr

        # Initialize (pipeline is created on instantiation)
        parser = GlmOcr(config_path="config.yaml")

        # Predict (returns list of PipelineResult, one per input unit)
        results = parser.predict("image1.png")
        for result in results:
            print(result.json_result)
            result.save(output_dir="./output")

        # Cleanup
        parser.close()
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize GlmOcr and create the pipeline.

        Args:
            config_path: Config file path. If None, loads the default config.
        """
        # Load config (instance-based; no global singleton)
        self.config_model = load_config(config_path)
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

    def parse(
        self,
        images: Union[str, List[str]],
        save_layout_visualization: bool = True,
        **kwargs,
    ) -> List[PipelineResult]:
        """Predict / parse images or documents.

        Supports local paths and URLs (file://, http://, https://, data:).
        Supports image files (jpg, png, bmp, gif, webp) and PDF files.

        Args:
            images: Image path/URL (single or list).
            save_layout_visualization: Whether to save layout visualization artifacts.
            **kwargs: Additional parameters for MaaS mode (return_crop_images,
                     need_layout_visualization, start_page_id, end_page_id, etc.)

        Returns:
            List[PipelineResult]: One result per input (one image or one PDF). Use
            result.save() to persist each.

        Example:
            results = parser.parse("image.png")
            results = parser.parse(["img1.png", "doc.pdf"])
            for r in results:
                r.save(output_dir="./output")
        """
        if isinstance(images, str):
            images = [images]

        if self._use_maas:
            return self._parse_maas(images, save_layout_visualization, **kwargs)
        else:
            return self._parse_selfhosted(images, save_layout_visualization)

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

    def _maas_response_to_pipeline_result(
        self, response: Dict[str, Any], source: str
    ) -> PipelineResult:
        """Convert MaaS API response to PipelineResult."""
        # Extract layout_details (list of pages, each page is a list of regions)
        layout_details = response.get("layout_details", [])

        # Convert to SDK format: [[{index, label, content, bbox_2d}, ...], ...]
        json_result = []
        for page_regions in layout_details:
            page_result = []
            for region in page_regions:
                page_result.append(
                    {
                        "index": region.get("index", 0),
                        "label": region.get("label", "text"),
                        "content": region.get("content", ""),
                        "bbox_2d": region.get("bbox_2d"),
                    }
                )
            json_result.append(page_result)

        # Get markdown result
        markdown_result = response.get("md_results", "")

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
        images: List[str],
        save_layout_visualization: bool = True,
    ) -> List[PipelineResult]:
        """Parse using self-hosted vLLM/SGLang pipeline."""
        import tempfile

        messages = [{"role": "user", "content": []}]
        for image in images:
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
def parse(
    images: Union[str, List[str]],
    config_path: Optional[str] = None,
    save_layout_visualization: bool = True,
) -> List[PipelineResult]:
    """Convenience function: predict / parse images or documents.

    Args:
        images: Image path or URL (single or list).
        config_path: Config file path.
        save_layout_visualization: Whether to save layout visualization.

    Returns:
        List[PipelineResult]: One result per input unit.

    Example:
        results = predict("image.png")
        for r in results:
            r.save(output_dir="./output")
    """
    with GlmOcr(config_path=config_path) as parser:
        return parser.parse(images, save_layout_visualization=save_layout_visualization)
