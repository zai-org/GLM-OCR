"""Base parser result.

Defines common fields and JSON/Markdown save logic.
"""

from __future__ import annotations

import json
import re
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


class BaseParserResult(ABC):
    """Base parser result.

    Common interface: json_result, markdown_result, original_images; abstract save().
    """

    def __init__(
        self,
        json_result: Union[str, dict, list],
        markdown_result: Optional[str] = None,
        original_images: Optional[List[str]] = None,
        image_files: Optional[Dict[str, Any]] = None,
        raw_json_result: Optional[list] = None,
        page_metadata: Optional[List[Dict[str, Any]]] = None,
        page_number_candidates: Optional[List[Dict[str, Any]]] = None,
        document_page_numbering: Optional[Dict[str, Any]] = None,
    ):
        """Initialize.

        Args:
            json_result: JSON result (string, dict, or list).
            markdown_result: Markdown result (optional).
            original_images: Original image paths.
            image_files: Mapping of ``filename`` → PIL Image for image-type
                regions, to be saved under ``imgs/`` during :meth:`save`.
            raw_json_result: Raw model output before post-processing;
                saved as ``{name}_model.json`` alongside the final result.
            page_metadata: Derived per-page printed page metadata.
            page_number_candidates: Raw printed page-number candidates.
            document_page_numbering: Document-level numbering inference.
        """
        if isinstance(json_result, str):
            try:
                self.json_result: Union[str, dict, list] = json.loads(json_result)
            except json.JSONDecodeError:
                self.json_result = json_result
        else:
            self.json_result = json_result

        self.markdown_result = markdown_result
        self.original_images = [
            str(Path(p).absolute()) for p in (original_images or [])
        ]
        self.image_files = image_files
        self.raw_json_result = raw_json_result
        self.page_metadata = page_metadata
        self.page_number_candidates = page_number_candidates
        self.document_page_numbering = document_page_numbering

    @abstractmethod
    def save(
        self,
        output_dir: Union[str, Path] = "./output",
        save_layout_visualization: bool = True,
    ) -> None:
        """Save result to disk. Subclasses implement layout vis etc."""
        pass

    def _save_json_and_markdown(self, output_dir: Union[str, Path]) -> None:
        """Save JSON and Markdown to output_dir (by first image name or 'result')."""
        output_dir = Path(output_dir).absolute()
        if self.original_images:
            image_path = Path(self.original_images[0])
            base_name = self._sanitize_name(image_path.stem)
            output_path = output_dir / base_name
        else:
            output_path = output_dir / "result"

        output_path.mkdir(parents=True, exist_ok=True)
        base_name = output_path.name

        # JSON
        json_file = output_path / f"{base_name}.json"
        try:
            json_data = self.json_result
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError:
                    pass

            has_printed_page_data = (
                bool(self.page_metadata)
                or bool(self.page_number_candidates)
                or self.document_page_numbering is not None
            )

            if has_printed_page_data:
                json_data = {
                    "json_result": json_data,
                    "page_metadata": (
                        self.page_metadata if self.page_metadata is not None else []
                    ),
                    "page_number_candidates": (
                        self.page_number_candidates
                        if self.page_number_candidates is not None
                        else []
                    ),
                    "document_page_numbering": self.document_page_numbering,
                }

            with open(json_file, "w", encoding="utf-8") as f:
                if isinstance(json_data, (dict, list)):
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(json_data))
        except Exception as e:
            logger.warning("Failed to save JSON: %s", e)
            traceback.print_exc()

        # Raw model output (before post-processing)
        if self.raw_json_result is not None:
            raw_file = output_path / f"{base_name}_model.json"
            try:
                with open(raw_file, "w", encoding="utf-8") as f:
                    json.dump(self.raw_json_result, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning("Failed to save raw JSON: %s", e)

        # Markdown
        if self.markdown_result and self.markdown_result.strip():
            md_file = output_path / f"{base_name}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(self.markdown_result)

        # Image files produced by the result formatter
        if self.image_files:
            imgs_dir = output_path / "imgs"
            imgs_dir.mkdir(parents=True, exist_ok=True)
            for filename, img in self.image_files.items():
                try:
                    img.save(imgs_dir / filename, quality=95)
                except Exception as e:
                    logger.warning("Failed to save image %s: %s", filename, e)
            self.image_files = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the result.

        Useful for agents and programmatic consumers that need a structured
        representation without touching the file system.
        """
        d: dict = {
            "json_result": self.json_result,
            "markdown_result": self.markdown_result or "",
            "original_images": self.original_images,
        }
        if self.page_metadata is not None:
            d["page_metadata"] = self.page_metadata
        if self.page_number_candidates is not None:
            d["page_number_candidates"] = self.page_number_candidates
        if self.document_page_numbering is not None:
            d["document_page_numbering"] = self.document_page_numbering
        # Include optional metadata set by MaaS mode.
        for attr in ("_usage", "_data_info", "_error"):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr.lstrip("_")] = val
        return d

    @staticmethod
    def _sanitize_name(value: str) -> str:
        """Sanitize a string for use as a directory/file name.

        Strips characters that are illegal on Windows (<>:"/\\|?*) and
        control characters (0x00-0x1F).  Also removes trailing spaces
        and dots which Windows silently drops, causing path mismatches.
        """
        value = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", value)
        value = value.rstrip(" .")
        return value or "result"

    def to_json(self, **kwargs: Any) -> str:
        """Serialise the result to a JSON string.

        Keyword arguments are forwarded to :func:`json.dumps`.
        """
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(images={len(self.original_images)})"
