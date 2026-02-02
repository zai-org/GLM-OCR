"""Result formatter - unified OCR post-processing.

ResultFormatter is responsible for:
1. Formatting OCR outputs
2. Filtering nested regions
3. Producing JSON and Markdown outputs

Applies to:
- OCR-only mode: format single-page results
- Layout mode: merge per-region results and format
"""

from __future__ import annotations

import re
import json
from copy import deepcopy
from wordfreq import zipf_frequency
from typing import TYPE_CHECKING, List, Dict, Tuple, Any

from glmocr.postprocess.base_post_processor import BasePostProcessor
from glmocr.utils.logging import get_logger, get_profiler
from glmocr.utils.result_postprocess_utils import (
    clean_repeated_content,
    clean_formula_number,
)

if TYPE_CHECKING:
    from glmocr.config import ResultFormatterConfig

logger = get_logger(__name__)
profiler = get_profiler(__name__)


class ResultFormatter(BasePostProcessor):
    """Result formatter.

    Formats OCR recognition outputs into JSON and Markdown.

    Example:
        from glmocr.config import ResultFormatterConfig

        formatter = ResultFormatter(ResultFormatterConfig())

        # Layout mode: process grouped results
        json_str, md_str = formatter.process(grouped_results)

        # OCR-only mode: format a single output
        json_str, md_str = formatter.format_ocr_result(content)
    """

    def __init__(self, config: "ResultFormatterConfig"):
        """Initialize.

        Args:
            config: ResultFormatterConfig instance.
        """
        super().__init__(config)

        # Label mapping (for layout mode)
        self.label_visualization_mapping = config.label_visualization_mapping

        # Output format
        self.output_format = config.output_format

    # =========================================================================
    # OCR-only mode
    # =========================================================================

    def format_ocr_result(self, content: str, page_idx: int = 0) -> Tuple[str, str]:
        """Format an OCR-only result.

        Args:
            content: Raw OCR output.
            page_idx: Page index.

        Returns:
            (json_str, markdown_str)
        """
        # Clean content
        content = self._clean_content(content)

        # Build JSON result
        json_result = [
            [
                {
                    "index": 0,
                    "label": "text",
                    "content": content,
                    "bbox_2d": None,
                }
            ]
        ]

        json_str = json.dumps(json_result, ensure_ascii=False)
        markdown_str = content

        return json_str, markdown_str

    def format_multi_page_results(self, contents: List[str]) -> Tuple[str, str]:
        """Format multi-page OCR-only results.

        Args:
            contents: OCR output per page.

        Returns:
            (json_str, markdown_str)
        """
        json_results = []
        markdown_parts = []

        for page_idx, content in enumerate(contents):
            content = self._clean_content(content)
            json_results.append(
                [
                    {
                        "index": 0,
                        "label": "text",
                        "content": content,
                        "bbox_2d": None,
                    }
                ]
            )
            markdown_parts.append(content)

        json_str = json.dumps(json_results, ensure_ascii=False)
        markdown_str = "\n\n---\n\n".join(markdown_parts)

        return json_str, markdown_str

    # =========================================================================
    # Layout mode
    # =========================================================================

    def process(self, grouped_results: List[List[Dict]]) -> Tuple[str, str]:
        """Process grouped results in layout mode.

        Args:
            grouped_results: Region recognition results grouped by page.

        Returns:
            (json_str, markdown_str)
        """
        json_final_results = []

        with profiler.measure("format_regions"):
            for page_idx, results in enumerate(grouped_results):
                # Sort
                sorted_results = sorted(results, key=lambda x: x.get("index", 0))

                # Process each region
                json_page_results = []
                valid_idx = 0

                for item in sorted_results:
                    result = deepcopy(item)
                    result["native_label"] = result.get("label", "text")

                    # Map labels
                    result["label"] = self._map_label(result["label"])

                    # Format content
                    result["content"] = self._format_content(
                        result["content"],
                        result["label"],
                        result["native_label"],
                    )

                    # Skip empty content (after formatting)
                    content = result.get("content")
                    if isinstance(content, str) and content.strip() == "":
                        continue

                    # Update index
                    result["index"] = valid_idx
                    result.pop("task_type", None)
                    result.pop("score", None)
                    valid_idx += 1

                    json_page_results.append(result)

                # Merge formula with formula_number
                json_page_results = self._merge_formula_numbers(json_page_results)

                # Merge hyphenated text blocks
                json_page_results = self._merge_text_blocks(json_page_results)

                # Format bullet points
                json_page_results = self._format_bullet_points(json_page_results)

                json_final_results.append(json_page_results)

        # Generate markdown results
        with profiler.measure("generate_markdown"):
            markdown_final_results = []
            for page_idx, json_page_results in enumerate(json_final_results):
                markdown_page_results = []
                for result in json_page_results:
                    content = result["content"]
                    if result["label"] == "image":
                        markdown_page_results.append(
                            f"![](page={page_idx},bbox={result.get('bbox_2d', [])})"
                        )
                    elif content:
                        markdown_page_results.append(content)
                markdown_final_results.append("\n\n".join(markdown_page_results))

        with profiler.measure("serialize_json"):
            json_str = json.dumps(json_final_results, ensure_ascii=False)
        markdown_str = "\n\n".join(markdown_final_results)

        return json_str, markdown_str

    # =========================================================================
    # Content handling
    # =========================================================================

    def _clean_content(self, content: str) -> str:
        """Clean OCR output content."""
        if content is None:
            return ""

        # Remove leading/trailing literal \t
        content = re.sub(r"^(\\t)+", "", content).lstrip()
        content = re.sub(r"(\\t)+$", "", content).rstrip()

        # Remove repeated punctuation
        content = re.sub(r"(\.)\1{2,}", r"\1\1\1", content)
        content = re.sub(r"(·)\1{2,}", r"\1\1\1", content)
        content = re.sub(r"(_)\1{2,}", r"\1\1\1", content)
        content = re.sub(r"(\\_)\1{2,}", r"\1\1\1", content)

        # Remove repeated substrings (for long content)
        if len(content) >= 2048:
            content = clean_repeated_content(content)

        return content.strip()

    def _format_content(self, content: Any, label: str, native_label: str) -> str:
        """Format a region's content."""
        if content is None:
            return content

        content = self._clean_content(str(content))

        # Title formatting
        if native_label == "doc_title":
            # Remove existing # symbols at the beginning
            content = re.sub(r"^#+\s*", "", content)
            content = "# " + content
        elif native_label == "paragraph_title":
            # Remove existing - or # symbols at the beginning
            if content.startswith("- ") or content.startswith("* "):
                content = content[2:].lstrip()
            content = re.sub(r"^#+\s*", "", content)
            content = "## " + content.lstrip()

        # Formula formatting
        if label == "formula":
            if content.startswith("$$") and content.endswith("$$"):
                content = content[2:-2].strip()
                content = "$$\n" + content + "\n$$"
            elif content.startswith("\\[") and content.endswith("\\]"):
                content = content[2:-2].strip()
                content = "$$\n" + content + "\n$$"
            elif content.startswith("\\(") and content.endswith("\\)"):
                content = content[2:-2].strip()
                content = "$$\n" + content + "\n$$"
            else:
                content = "$$\n" + content + "\n$$"

        # Text formatting
        if label == "text":
            # Bullet points
            if (
                content.startswith("·")
                or content.startswith("•")
                or content.startswith("* ")
            ):
                content = "- " + content[1:].lstrip()

            # Allow multiple digits for numbers, single letter for alphabetic
            match = re.match(r"^(\(|\（)(\d+|[A-Za-z])(\)|\）)(.*)$", content)
            if match:
                _, symbol, _, rest = match.groups()
                content = f"({symbol}) {rest.lstrip()}"

            # Allow multiple digits for numbers, single letter for alphabetic
            match = re.match(r"^(\d+|[A-Za-z])(\.|\)|\）)(.*)$", content)
            if match:
                symbol, sep, rest = match.groups()
                sep = ")" if sep == "）" else sep
                content = f"{symbol}{sep} {rest.lstrip()}"

            # Replace single newlines with double newlines
            content = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content)

        return content

    def _map_label(self, label: str) -> str:
        """Map labels to standardized types."""
        if label in self.label_visualization_mapping.get("image", []):
            return "image"
        if label in self.label_visualization_mapping.get("text", []):
            return "text"
        if label in self.label_visualization_mapping.get("table", []):
            return "table"
        if label in self.label_visualization_mapping.get("formula", []):
            return "formula"
        return label

    # =========================================================================
    # Text block processing
    # =========================================================================

    def _merge_text_blocks(self, json_page_results: List[Dict]) -> List[Dict]:
        """Merge hyphenated text blocks.

        Merges text blocks separated by hyphens if the combined word is valid.
        """
        if not json_page_results:
            return json_page_results

        merged_results = []
        skip_indices = set()

        for i, block in enumerate(json_page_results):
            if i in skip_indices:
                continue

            if block.get("label") != "text":
                merged_results.append(block)
                continue

            content = block.get("content", "")
            if not isinstance(content, str):
                merged_results.append(block)
                continue

            content_stripped = content.rstrip()
            if not content_stripped:
                merged_results.append(block)
                continue

            # Check if ends with hyphen
            if not content_stripped.endswith("-"):
                merged_results.append(block)
                continue

            # Look for next text block starting with lowercase
            merged = False
            for j in range(i + 1, len(json_page_results)):
                if json_page_results[j].get("label") == "text":
                    next_content = json_page_results[j].get("content", "")
                    if isinstance(next_content, str):
                        next_stripped = next_content.lstrip()
                        if next_stripped and next_stripped[0].islower():
                            words_before = content_stripped[:-1].split()
                            next_words = next_stripped.split()

                            if words_before and next_words:
                                word_fragment_before = words_before[-1]
                                word_fragment_after = next_words[0]
                                merged_word = word_fragment_before + word_fragment_after

                                # Validate merged word
                                zipf_score = zipf_frequency(merged_word.lower(), "en")
                                if zipf_score >= 2.5:
                                    merged_content = (
                                        content_stripped[:-1] + next_content.lstrip()
                                    )
                                    merged_block = deepcopy(block)
                                    merged_block["content"] = merged_content

                                    merged_results.append(merged_block)
                                    skip_indices.add(j)
                                    merged = True
                            break

            if not merged:
                merged_results.append(block)

        # Reassign indices
        for idx, block in enumerate(merged_results):
            block["index"] = idx

        return merged_results

    def _format_bullet_points(
        self, json_page_results: List[Dict], left_align_threshold: float = 10.0
    ) -> List[Dict]:
        """Detect and add missing bullet points to list items.

        If a text block is between two bullet points and left-aligned with them,
        add a bullet point to it as well.
        """
        if len(json_page_results) < 3:
            return json_page_results

        for i in range(1, len(json_page_results) - 1):
            current_block = json_page_results[i]
            prev_block = json_page_results[i - 1]
            next_block = json_page_results[i + 1]

            # Only process text blocks
            if current_block.get("native_label") != "text":
                continue

            if (
                prev_block.get("native_label") != "text"
                or next_block.get("native_label") != "text"
            ):
                continue

            current_content = current_block.get("content", "")
            if current_content.startswith("- "):
                continue

            prev_content = prev_block.get("content", "")
            next_content = next_block.get("content", "")

            # Both prev and next must be bullet points
            if not (prev_content.startswith("- ") and next_content.startswith("- ")):
                continue

            # Check left alignment
            current_bbox = current_block.get("bbox_2d", [])
            prev_bbox = prev_block.get("bbox_2d", [])
            next_bbox = next_block.get("bbox_2d", [])

            if not (current_bbox and prev_bbox and next_bbox):
                continue

            current_left = current_bbox[0]
            prev_left = prev_bbox[0]
            next_left = next_bbox[0]

            if (
                abs(current_left - prev_left) <= left_align_threshold
                and abs(current_left - next_left) <= left_align_threshold
            ):
                current_block["content"] = "- " + current_content

        return json_page_results

    def _merge_formula_numbers(self, json_page_results: List[Dict]) -> List[Dict]:
        """Merge formula_number into adjacent formula block using \\tag{}.

        Handles two cases:
        1. formula followed by formula_number: formula -> formula_number
        2. formula_number followed by formula: formula_number -> formula

        Example:
            formula: "$$E = mc^2$$"
            formula_number: "(1)"
            merged: "$$E = mc^2 \\tag{1}$$"
        """
        if not json_page_results:
            return json_page_results

        merged_results = []
        skip_indices = set()

        for i, block in enumerate(json_page_results):
            if i in skip_indices:
                continue

            native_label = block.get("native_label", "")

            # Case 1: formula_number followed by formula
            if native_label == "formula_number":
                # Look for the next block being formula
                if i + 1 < len(json_page_results):
                    next_block = json_page_results[i + 1]
                    if next_block.get("label") == "formula":
                        # Extract formula number content
                        number_content = block.get("content", "").strip()
                        number_clean = clean_formula_number(number_content)

                        # Get formula content and try to add tag
                        formula_content = next_block.get("content", "")
                        merged_block = deepcopy(next_block)

                        # Add tag only if formula ends with \n$$
                        if formula_content.endswith("\n$$"):
                            merged_block["content"] = (
                                formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                            )

                        merged_results.append(merged_block)
                        skip_indices.add(
                            i + 1
                        )  # Skip the formula block (already merged)
                        continue  # Skip formula_number block

                # No formula follows, skip formula_number anyway
                continue

            # Case 2: formula followed by formula_number
            if block.get("label") == "formula":
                # Look for the next block being formula_number
                if i + 1 < len(json_page_results):
                    next_block = json_page_results[i + 1]
                    if next_block.get("native_label") == "formula_number":
                        # Extract formula number content
                        number_content = next_block.get("content", "").strip()
                        number_clean = clean_formula_number(number_content)

                        # Get formula content and try to add tag
                        formula_content = block.get("content", "")
                        merged_block = deepcopy(block)

                        # Add tag only if formula ends with \n$$
                        if formula_content.endswith("\n$$"):
                            merged_block["content"] = (
                                formula_content[:-3] + f" \\tag{{{number_clean}}}\n$$"
                            )

                        merged_results.append(merged_block)
                        skip_indices.add(i + 1)  # Skip the formula_number block
                        continue

                # No formula_number follows, keep formula as-is
                merged_results.append(block)
                continue

            # Other blocks, keep as-is
            merged_results.append(block)

        # Reassign indices
        for idx, block in enumerate(merged_results):
            block["index"] = idx

        return merged_results
