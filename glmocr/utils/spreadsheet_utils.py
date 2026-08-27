"""Spreadsheet (CSV / TSV) export utilities.

Extracts tables from OCR results so downstream spreadsheet and data-analysis
workflows can consume them directly without post-processing (#225).
"""

from __future__ import annotations

import csv
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Union


class _HTMLTableParser(HTMLParser):
    """Minimal HTML table parser producing rows of cell strings."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[List[List[str]]] = []
        self._table_depth = 0
        self._rows: List[List[str]] = []
        self._row: List[str] = []
        self._cell: List[str] = []
        self._in_cell = False

    # -- helpers ---------------------------------------------------------

    def _flush_cell(self) -> None:
        text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
        self._row.append(text)
        self._cell = []

    def _flush_row(self) -> None:
        if self._row:
            self._rows.append(self._row)
        self._row = []

    def _flush_table(self) -> None:
        self._flush_row()
        if self._rows:
            self.tables.append(self._rows)
        self._rows = []

    # -- HTMLParser hooks --------------------------------------------------

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif self._table_depth and tag == "tr":
            self._row = []
        elif self._table_depth and tag in ("td", "th"):
            self._cell = []
            self._in_cell = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._table_depth:
            self._table_depth -= 1
            if self._table_depth == 0:
                self._flush_table()
        elif self._table_depth and tag in ("td", "th"):
            self._flush_cell()
            self._in_cell = False
        elif self._table_depth and tag == "tr":
            self._flush_row()

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)


def _parse_html_tables(html: str) -> List[List[List[str]]]:
    parser = _HTMLTableParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        return []
    return parser.tables


_MD_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
_MD_SEP_RE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")


def _parse_markdown_table(text: str) -> Union[List[List[str]], None]:
    """Parse a GitHub-style pipe table; returns rows or None."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    if not _MD_SEP_RE.match(lines[1]) or "---" not in lines[1]:
        return None
    rows: List[List[str]] = []
    for i, line in enumerate(lines):
        m = _MD_ROW_RE.match(line)
        if not m:
            return None
        if i == 1:
            continue  # separator row
        cells = [c.strip() for c in m.group(1).split("|")]
        rows.append(cells)
    return rows or None


def extract_tables(json_result: Any) -> List[Dict[str, Any]]:
    """Extract all tables from a formatted OCR JSON result.

    Returns a list of ``{"page": int, "index": int, "rows": List[List[str]]}``
    in document order.
    """
    if isinstance(json_result, str):
        return []
    pages = json_result if isinstance(json_result, list) else [json_result]
    tables: List[Dict[str, Any]] = []
    for page_idx, page in enumerate(pages):
        if isinstance(page, dict):
            page = page.get("result", page.get("blocks", [page]))
        if not isinstance(page, list):
            continue
        t_idx = 0
        for block in page:
            if not isinstance(block, dict):
                continue
            label = block.get("label") or block.get("type")
            content = block.get("content")
            if not label or label != "table" or not content:
                continue
            content = str(content).strip()
            parsed: Union[List[List[List[str]]], None] = None
            if content.lower().startswith("<table"):
                parsed = _parse_html_tables(content)
            else:
                single = _parse_markdown_table(content)
                parsed = [single] if single else None
            if not parsed:
                continue
            for rows in parsed:
                tables.append(
                    {"page": page_idx, "index": t_idx, "rows": rows}
                )
                t_idx += 1
    return tables


def write_spreadsheet(
    rows: List[List[str]],
    path: Union[str, Path],
    delimiter: str = ",",
) -> None:
    """Write a 2-D list of strings as CSV (default) or TSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, delimiter=delimiter).writerows(rows)


def save_tables(
    tables: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    base_name: str,
) -> List[Path]:
    """Write each extracted table as ``<name>.csv`` and ``<name>.tsv``.

    Returns the list of files actually written.
    """
    if not tables:
        return []
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    multi = len(tables) > 1
    for fmt, delim in (("csv", ","), ("tsv", "\t")):
        if multi:
            for t in tables:
                p = out_dir / f"{base_name}_p{t['page']}_t{t['index']}.{fmt}"
                write_spreadsheet(t["rows"], p, delimiter=delim)
                written.append(p)
        else:
            p = out_dir / f"{base_name}.{fmt}"
            write_spreadsheet(tables[0]["rows"], p, delimiter=delim)
            written.append(p)
    return written
