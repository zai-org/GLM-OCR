"""Result post-processing utilities."""

import re
from typing import Optional
from collections import Counter


def find_consecutive_repeat(
    s: str, min_unit_len: int = 10, min_repeats: int = 10
) -> Optional[str]:
    """Find and remove consecutive repeated patterns.

    Args:
        s: Input string to check for repeats.
        min_unit_len: Minimum length of the repeating unit.
        min_repeats: Minimum number of repetitions to detect.

    Returns:
        String with repeats removed, or None if no repeats found.
    """
    n = len(s)
    if n < min_unit_len * min_repeats:
        return None

    # Dynamically calculate max_unit_len
    max_unit_len = n // min_repeats
    if max_unit_len < min_unit_len:
        return None

    # Use DOTALL mode to match newlines
    pattern = re.compile(
        r"(.{"
        + str(min_unit_len)
        + ","
        + str(max_unit_len)
        + r"}?)\1{"
        + str(min_repeats - 1)
        + ",}",
        re.DOTALL,
    )
    match = pattern.search(s)
    if match:
        return s[: match.start()] + match.group(1)
    return None


def clean_repeated_content(
    content: str,
    min_len: int = 10,
    min_repeats: int = 10,
    line_threshold: int = 10,
) -> str:
    """Remove repeated content (both consecutive and line-level).

    Args:
        content: Input content string.
        min_len: Minimum length of repeating pattern.
        min_repeats: Minimum number of repetitions.
        line_threshold: Threshold for line-level repeat detection.

    Returns:
        Content with repeated parts removed.
    """
    stripped_content = content.strip()
    if not stripped_content:
        return content

    # 1. Consecutive repeat detection (supports multi-line patterns)
    if len(stripped_content) > min_len * min_repeats:
        result = find_consecutive_repeat(
            stripped_content, min_unit_len=min_len, min_repeats=min_repeats
        )
        if result is not None:
            return result

    # 2. Line-level repeat detection
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines >= line_threshold and lines:
        common, count = Counter(lines).most_common(1)[0]
        if count >= line_threshold and (count / total_lines) >= 0.8:
            for i, line in enumerate(lines):
                if line == common:
                    consecutive = sum(
                        1
                        for j in range(i, min(i + 3, len(lines)))
                        if lines[j] == common
                    )
                    if consecutive >= 3:
                        original_lines = content.split("\n")
                        non_empty_count = 0
                        for idx, orig_line in enumerate(original_lines):
                            if orig_line.strip():
                                non_empty_count += 1
                                if non_empty_count == i + 1:
                                    return "\n".join(original_lines[: idx + 1])
                        break
    return content


def clean_formula_number(number_content: str) -> str:
    """Clean up formula number by removing parentheses.

    Args:
        number_content: Raw formula number string, e.g., "(1)", "（2.1）", "3"

    Returns:
        Cleaned number string, e.g., "1", "2.1", "3"
    """
    number_clean = number_content.strip()
    if number_clean.startswith("(") and number_clean.endswith(")"):
        number_clean = number_clean[1:-1]
    elif number_clean.startswith("（") and number_clean.endswith("）"):
        number_clean = number_clean[1:-1]
    return number_clean
