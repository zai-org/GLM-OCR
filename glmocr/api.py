"""GLM-OCR Python API

Python API for calling the document parsing pipeline from your code.
"""

from typing import List, Optional, Union
from pathlib import Path

from glmocr.pipeline import Pipeline
from glmocr.config import load_config
from glmocr.parser_result import PipelineResult
from glmocr.utils.logging import get_logger, ensure_logging_configured

logger = get_logger(__name__)

# Backward compatibility: ParseResult is PipelineResult
ParseResult = PipelineResult


class GlmOcr:
    """Main GLM-OCR entrypoint.

    Provides a Python API for document parsing.

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
        # CLI/server may already have configured logging explicitly; this will not override that.
        ensure_logging_configured(
            level=self.config_model.logging.level,
            format_string=self.config_model.logging.format,
        )
        self._pipeline = None

        # Create pipeline with typed config
        self._pipeline = Pipeline(config=self.config_model.pipeline)
        self.enable_layout = self._pipeline.enable_layout
        self._pipeline.start()

    def parse(
        self,
        images: Union[str, List[str]],
        save_layout_visualization: bool = True,
    ) -> List[PipelineResult]:
        """Predict / parse images or documents (async pipeline; one result per input unit).

        Supports local paths and URLs (file://, http://, https://, data:).
        Supports image files (jpg, png, bmp, gif, webp) and PDF files.

        Args:
            images: Image path/URL (single or list).
            save_layout_visualization: Whether to save layout visualization artifacts.

        Returns:
            List[PipelineResult]: One result per input (one image or one PDF). Use
            result.save() to persist each.

        Example:
            results = parser.predict("image.png")
            results = parser.predict(["img1.png", "doc.pdf"])
            for r in results:
                r.save(output_dir="./output")
        """
        import tempfile

        if isinstance(images, str):
            images = [images]

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

    def close(self):
        """Close the parser and release resources."""
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

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
