"""Pipeline result with layout visualization support."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from glmocr.utils.logging import get_logger

from .base import BaseParserResult

logger = get_logger(__name__)


class PipelineResult(BaseParserResult):
    """Pipeline result for one input unit (one image or one PDF).

    Supports saving JSON, Markdown, and optional layout visualization.
    """

    def __init__(
        self,
        json_result: Union[str, dict, list],
        markdown_result: Optional[str],
        original_images: List[str],
        image_files: Optional[dict] = None,
        raw_json_result: Optional[list] = None,
        layout_vis_images: Optional[Dict[int, Any]] = None,
        page_metadata: Optional[List[Dict[str, Any]]] = None,
        page_number_candidates: Optional[List[Dict[str, Any]]] = None,
        document_page_numbering: Optional[Dict[str, Any]] = None,
    ):
        """Initialize.

        Args:
            json_result: JSON result (string, dict, or list).
            markdown_result: Markdown result.
            original_images: Original image paths for this unit.
            image_files: Mapping of ``filename`` → PIL Image for image-type
                regions; saved directly to ``imgs/`` during :meth:`save`.
            raw_json_result: Raw model output before post-processing (optional).
            layout_vis_images: Mapping of ``page_idx`` → PIL Image for layout
                visualization; saved to ``layout_vis/`` during :meth:`save`.
            page_metadata: Derived per-page printed page metadata.
            page_number_candidates: Raw printed page-number candidates.
            document_page_numbering: Document-level numbering inference.
        """
        super().__init__(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=original_images,
            image_files=image_files,
            raw_json_result=raw_json_result,
            page_metadata=page_metadata,
            page_number_candidates=page_number_candidates,
            document_page_numbering=document_page_numbering,
        )
        self.layout_vis_images = layout_vis_images

    def save(
        self,
        output_dir: Union[str, Path] = "./output",
        save_layout_visualization: bool = True,
    ) -> None:
        """Save JSON, Markdown, and optionally layout visualization."""
        self._save_json_and_markdown(output_dir)

        if not save_layout_visualization or not self.layout_vis_images:
            return

        if self.original_images:
            stem = self._sanitize_name(Path(self.original_images[0]).stem)
            target_dir = Path(output_dir).absolute() / stem / "layout_vis"
        else:
            target_dir = Path(output_dir).absolute() / "result" / "layout_vis"

        target_dir.mkdir(parents=True, exist_ok=True)

        stem_name = (
            self._sanitize_name(Path(self.original_images[0]).stem)
            if self.original_images
            else "result"
        )
        single = len(self.layout_vis_images) == 1
        for page_idx, vis_img in self.layout_vis_images.items():
            name = f"{stem_name}.jpg" if single else f"{stem_name}_page{page_idx}.jpg"
            try:
                vis_img.save(target_dir / name, quality=95)
            except Exception as e:
                logger.warning("Failed to save layout vis %s: %s", name, e)

        self.layout_vis_images = None
        logger.debug("Layout visualization saved to %s", target_dir)
