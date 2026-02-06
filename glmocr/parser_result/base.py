"""Base parser result.

Defines common fields and JSON/Markdown save logic.
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

from glmocr.utils.logging import get_logger
from glmocr.utils.markdown_utils import crop_and_replace_images

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
    ):
        """Initialize.

        Args:
            json_result: JSON result (string, dict, or list).
            markdown_result: Markdown result (optional).
            original_images: Original image paths.
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

    @abstractmethod
    def save(
        self,
        output_dir: Union[str, Path] = "./results",
        save_layout_visualization: bool = True,
    ) -> None:
        """Save result to disk. Subclasses implement layout vis etc."""
        pass

    def _save_json_and_markdown(self, output_dir: Union[str, Path]) -> None:
        """Save JSON and Markdown to output_dir (by first image name or 'result')."""
        output_dir = Path(output_dir).absolute()
        if self.original_images:
            image_path = Path(self.original_images[0])
            output_path = output_dir / image_path.stem
        else:
            output_path = output_dir / "result"

        output_path.mkdir(parents=True, exist_ok=True)
        base_name = output_path.name

        # JSON
        json_file = output_path / f"{base_name}.json"
        try:
            if isinstance(self.json_result, (dict, list)):
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(self.json_result, f, ensure_ascii=False, indent=2)
            elif isinstance(self.json_result, str):
                try:
                    data = json.loads(self.json_result)
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    with open(json_file, "w", encoding="utf-8") as f:
                        f.write(self.json_result)
            else:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(self.json_result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save JSON: %s", e)
            traceback.print_exc()

        # Markdown (with image crop/replace if original_images)
        if self.markdown_result and self.markdown_result.strip():
            md_text = self.markdown_result
            if self.original_images:
                try:
                    imgs_dir = output_path / "imgs"
                    md_text, _ = crop_and_replace_images(
                        md_text,
                        self.original_images,
                        imgs_dir,
                        image_prefix="cropped",
                    )
                except Exception as e:
                    logger.warning("Failed to process image regions: %s", e)
            md_file = output_path / f"{base_name}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_text)

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
        # Include optional metadata set by MaaS mode.
        for attr in ("_usage", "_data_info", "_error"):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr.lstrip("_")] = val
        return d

    def to_json(self, **kwargs: Any) -> str:
        """Serialise the result to a JSON string.

        Keyword arguments are forwarded to :func:`json.dumps`.
        """
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(images={len(self.original_images)})"
