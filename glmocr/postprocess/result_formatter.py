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

import collections
import re
import json
from copy import deepcopy
from typing import TYPE_CHECKING, List, Dict, Tuple, Any, Optional

try:  # Optional dependency for better English word validation quality.
    from wordfreq import zipf_frequency
except Exception:  # pragma: no cover
    zipf_frequency = None

from glmocr.postprocess.base_post_processor import BasePostProcessor
from glmocr.utils.logging import get_logger, get_profiler
from glmocr.utils.result_postprocess_utils import (
    clean_repeated_content,
    clean_formula_number,
    normalize_inline_formula,
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
        json_str, md_str, image_files = formatter.process(grouped_results)

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
        self.enable_merge_formula_numbers = config.enable_merge_formula_numbers
        self.enable_merge_text_blocks = config.enable_merge_text_blocks
        self.enable_format_bullet_points = config.enable_format_bullet_points
        self.detect_printed_page_numbers = config.detect_printed_page_numbers
        self.page_metadata: Optional[List[Dict[str, Any]]] = None
        self.page_number_candidates: Optional[List[Dict[str, Any]]] = None
        self.document_page_numbering: Optional[Dict[str, Any]] = None

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

    def process(
        self,
        grouped_results: List[List[Dict]],
        cropped_images: Dict[tuple, Any] | None = None,
        image_prefix: str = "cropped",
    ) -> Tuple[str, str, Dict[str, Any]]:
        """Process grouped results in layout mode.

        Args:
            grouped_results: Region recognition results grouped by page.
            cropped_images: Pre-cropped PIL images keyed by
                ``(local_page_idx, *bbox)``; when provided, image regions
                are resolved to final file paths directly in the markdown
                and JSON output.
            image_prefix: Filename prefix for saved images.

        Returns:
            (json_str, markdown_str, image_files) where *image_files* maps
            ``filename`` → PIL Image for the caller to persist.
        """
        self.page_metadata = None
        self.page_number_candidates = None
        self.document_page_numbering = None

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
                    result["layout_index"] = result.get(
                        "layout_index", result.get("index", 0)
                    )
                    result["layout_score"] = float(
                        result.get("layout_score", result.get("score") or 0.0)
                    )
                    result["native_label"] = result.get("label", "text")

                    # Map labels
                    result["label"] = self._map_label(result["label"])

                    # Format content
                    result["content"] = self._format_content(
                        result["content"],
                        result["label"],
                        result["native_label"],
                    )

                    # Skip empty or failed content (after formatting)
                    if result["label"] != "image":
                        content = result.get("content")
                        if content is None or (
                            isinstance(content, str) and content.strip() == ""
                        ):
                            continue

                    # Update index
                    result["index"] = valid_idx
                    result.pop("task_type", None)
                    result.pop("score", None)
                    valid_idx += 1

                    json_page_results.append(result)

                # Merge formula with formula_number
                if self.enable_merge_formula_numbers:
                    json_page_results = self._merge_formula_numbers(json_page_results)

                # Merge hyphenated text blocks
                if self.enable_merge_text_blocks:
                    json_page_results = self._merge_text_blocks(json_page_results)

                # Format bullet points
                if self.enable_format_bullet_points:
                    json_page_results = self._format_bullet_points(json_page_results)

                json_final_results.append(json_page_results)

        if self.detect_printed_page_numbers:
            (
                self.page_number_candidates,
                self.document_page_numbering,
                self.page_metadata,
            ) = self.extract_printed_page_data(json_final_results)

        self._strip_layout_metadata(json_final_results)

        # Generate markdown results and resolve image regions
        image_files: Dict[str, Any] = {}
        image_counter = 0
        with profiler.measure("generate_markdown"):
            markdown_final_results = []
            for page_idx, json_page_results in enumerate(json_final_results):
                markdown_page_results = []
                for result in json_page_results:
                    content = result["content"]
                    if result["label"] == "image":
                        bbox = result.get("bbox_2d", [])
                        key = (page_idx, *bbox) if bbox else None
                        img = (
                            cropped_images.get(key) if cropped_images and key else None
                        )
                        if img is not None:
                            filename = (
                                f"{image_prefix}_page{page_idx}_idx{image_counter}.jpg"
                            )
                            rel_path = f"imgs/{filename}"
                            image_files[filename] = img
                            result["image_path"] = rel_path
                            markdown_page_results.append(
                                f"![Image {page_idx}-{image_counter}]({rel_path})"
                            )
                            image_counter += 1
                    elif content:
                        markdown_page_results.append(content)
                markdown_final_results.append("\n\n".join(markdown_page_results))

        with profiler.measure("serialize_json"):
            json_str = json.dumps(json_final_results, ensure_ascii=False)
        markdown_str = "\n\n".join(markdown_final_results)

        return json_str, markdown_str, image_files

    def extract_printed_page_data(
        self,
        pages: List[List[Dict[str, Any]]],
    ) -> Tuple[
        List[Dict[str, Any]],
        Optional[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """Extract number candidates and derived printed page metadata."""
        candidates = self._extract_page_number_candidates(pages)
        document_page_numbering = self._infer_document_page_numbering(candidates)
        page_metadata = self._build_printed_page_metadata(candidates)
        return candidates, document_page_numbering, page_metadata

    def _extract_page_number_candidates(
        self,
        pages: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Extract raw `number` candidates for printed page inference."""
        candidates: List[Dict[str, Any]] = []
        for page_index, page_blocks in enumerate(pages):
            for block in page_blocks:
                candidate = self._build_page_number_candidate(page_index, block)
                if candidate is not None:
                    candidates.append(candidate)
        return candidates

    def _build_page_number_candidate(
        self,
        page_index: int,
        block: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Build a normalized page-number candidate from one layout block."""
        if block.get("native_label") != "number":
            return None

        bbox = block.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None

        label = self._normalize_printed_page_label(block.get("content"))
        if label is None:
            return None

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0 or width > 140 or height > 120:
            return None
        if not self._is_margin_candidate(x1, y1, x2, y2):
            return None

        return {
            "page_index": page_index,
            "label": "number",
            "content": label,
            "layout_index": block.get("layout_index", block.get("index", 0)),
            "bbox_2d": bbox,
            "layout_score": float(block.get("layout_score") or 0.0),
            "numeric_like": label.isdigit(),
            "roman_like": self._is_roman_like(label),
        }

    @staticmethod
    def _is_margin_candidate(x1: int, y1: int, x2: int, y2: int) -> bool:
        """Return whether a candidate lies in a plausible page-margin folio area."""
        in_margin_band = y1 <= 120 or y2 >= 880
        in_outer_margin = x1 <= 180 or x2 >= 820
        return in_margin_band and in_outer_margin

    @staticmethod
    def _is_roman_like(content: str) -> bool:
        """Check whether a label looks like a Roman numeral folio."""
        return bool(re.fullmatch(r"(?i)[ivxlcdm]+", content))

    def _infer_document_page_numbering(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Infer document-level numbering from number-only candidates."""
        if not candidates:
            return None

        best_candidates = self._best_candidates_by_page(candidates)
        page_count = len(best_candidates)
        numeric_candidates = [c for c in best_candidates if c["numeric_like"]]
        roman_candidates = [c for c in best_candidates if c["roman_like"]]

        if numeric_candidates:
            offsets = collections.Counter(
                int(c["content"]) - int(c["page_index"]) for c in numeric_candidates
            )
            page_offset, support = offsets.most_common(1)[0]
            return {
                "strategy": "visual_sequence",
                "confidence": round(support / max(1, page_count), 3),
                "sequence_type": "arabic",
                "page_offset": page_offset,
                "candidate_pages": page_count,
            }

        if roman_candidates:
            return {
                "strategy": "visual_sequence",
                "confidence": round(len(roman_candidates) / max(1, page_count), 3),
                "sequence_type": "roman",
                "page_offset": None,
                "candidate_pages": len(roman_candidates),
            }

        return None

    def _build_printed_page_metadata(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build per-page printed page metadata from selected candidates."""
        if not candidates:
            return []

        metadata: List[Dict[str, Any]] = []
        for candidate in self._best_candidates_by_page(candidates):
            metadata.append(
                {
                    "page_index": candidate["page_index"],
                    "printed_page_label": candidate["content"],
                    "printed_page_block_index": candidate["layout_index"],
                    "printed_page_bbox_2d": candidate["bbox_2d"],
                    "printed_page_confidence": candidate["layout_score"],
                }
            )
        return metadata

    def _best_candidates_by_page(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select the strongest candidate per page."""
        by_page: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for candidate in candidates:
            by_page[int(candidate["page_index"])].append(candidate)
        return [
            min(by_page[page_index], key=self._candidate_sort_key)
            for page_index in sorted(by_page)
        ]

    @staticmethod
    def _candidate_sort_key(block: Dict[str, Any]) -> tuple[int, int, int, int]:
        """Prefer blocks nearest to outer top/bottom page margins."""
        bbox = block.get("bbox_2d") or [0, 0, 1000, 1000]
        x1, y1, x2, y2 = bbox
        top_distance = y1
        bottom_distance = 1000 - y2
        edge_distance = min(top_distance, bottom_distance)
        side_distance = min(x1, 1000 - x2)
        return (
            edge_distance,
            side_distance,
            -int(block.get("layout_score", 0) * 1000),
            int(block.get("layout_index", block.get("index", 0))),
        )

    @staticmethod
    def _normalize_printed_page_label(content: Any) -> Optional[str]:
        """Normalize OCR text from a printed page-number candidate."""
        if not isinstance(content, str):
            return None
        label = content.strip()
        if not label or len(label) > 12:
            return None
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-./]*", label):
            return None
        if not (re.search(r"\d", label) or ResultFormatter._is_roman_like(label)):
            return None
        return label

    @staticmethod
    def _strip_layout_metadata(pages: List[List[Dict[str, Any]]]) -> None:
        """Remove broad layout-only metadata from final JSON blocks."""
        for page in pages:
            for block in page:
                block.pop("layout_index", None)
                block.pop("layout_score", None)

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

        content = normalize_inline_formula(content)

        return content.strip()

    def _format_content(self, content: Any, label: str, native_label: str) -> str:
        """Format a region's content."""
        if content is None:
            return content

        if label == "table":
            if content.startswith("<table") and content.endswith("</table>"):
                content = content.strip()
            else:
                content = self._clean_content(str(content))
        elif label == "formula":
            if content.startswith("$$") and content.endswith("$$"):
                content = content.strip()
            else:
                content = self._clean_content(str(content))
        else:
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
            if (
                content.startswith("$$")
                or content.startswith("\\[")
                or content.startswith("\\(")
            ):
                content = content[2:].strip()
            if (
                content.endswith("$$")
                or content.endswith("\\]")
                or content.endswith("\\)")
            ):
                content = content[:-2].strip()
            content = "$$\n" + content + "\n$$"

        # Text formatting
        if label == "text":
            # Code blocks
            if content.startswith("```") and (not content.endswith("```")):
                content = content + "\n```"

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

    def _is_likely_valid_merged_word(self, merged_word: str) -> bool:
        """Check whether a hyphen-merged token looks like a valid word.

        Uses `wordfreq` when available, and falls back to a lightweight
        regex heuristic when `wordfreq` is not installed.
        """
        token = merged_word.strip().lower()
        if not token:
            return False

        if zipf_frequency is not None:
            try:
                return zipf_frequency(token, "en") >= 2.5
            except Exception:
                pass

        # Fallback heuristic (dependency-free):
        # - alphabetic-ish token
        # - length in a reasonable range
        # - avoid obviously malformed merges
        if not re.fullmatch(r"[a-z][a-z'\-]{2,30}", token):
            return False
        if "--" in token or "''" in token:
            return False
        return True

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
                                if self._is_likely_valid_merged_word(merged_word):
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
