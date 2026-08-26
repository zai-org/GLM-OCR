"""Tests for CSV/TSV table export (#225).

Run:
    python -m pytest glmocr/tests/test_spreadsheet_export.py -v
or standalone:
    python glmocr/tests/test_spreadsheet_export.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from glmocr.utils.spreadsheet_utils import (  # noqa: E402
    extract_tables,
    save_tables,
    write_spreadsheet,
)

HTML_TABLE = (
    "<table><tr><th>Name</th><th>Qty</th></tr>"
    "<tr><td>Apples</td><td>12</td></tr>"
    "<tr><td>Pears, green</td><td>7</td></tr></table>"
)

MD_TABLE = "| A | B |\n| --- | --- |\n| 1 | 2 |\n"


def _result(content):
    return [{"label": "table", "content": content}]


def test_extract_html_table():
    tables = extract_tables([_result(HTML_TABLE)])
    assert len(tables) == 1
    rows = tables[0]["rows"]
    assert rows[0] == ["Name", "Qty"]
    assert rows[1] == ["Apples", "12"]


def test_extract_html_comma_cell():
    """Commas inside cells must survive CSV quoting."""
    tables = extract_tables([_result(HTML_TABLE)])
    rows = tables[0]["rows"]
    assert any("Pears, green" in cell for row in rows for cell in row)


def test_extract_markdown_table():
    tables = extract_tables([_result(MD_TABLE)])
    assert len(tables) == 1
    assert tables[0]["rows"][0] == ["A", "B"]
    assert tables[0]["rows"][1] == ["1", "2"]


def test_multi_page_indexing():
    res = [_result(MD_TABLE), _result(HTML_TABLE)]
    tables = extract_tables(res)
    assert [t["page"] for t in tables] == [0, 1]


def test_no_tables_returns_empty():
    assert extract_tables([{"label": "text", "content": "hello"}]) == []


def test_write_csv_roundtrip():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.csv")
    write_spreadsheet([["a", "b,c"], ["d", "e"]], p)
    with open(p, encoding="utf-8-sig") as f:
        body = f.read()
    assert '"b,c"' in body


def test_write_tsv_delimiter():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "t.tsv")
    write_spreadsheet([["a", "b"], ["c", "d"]], p, delimiter="\t")
    with open(p, encoding="utf-8-sig") as f:
        lines = f.read().strip().splitlines()
    assert lines == ["a\tb", "c\td"]


def test_save_tables_single_and_multi():
    single = save_tables(extract_tables([_result(MD_TABLE)]), tempfile.mkdtemp(), "doc")
    multi = save_tables(
        extract_tables([_result(MD_TABLE), _result(HTML_TABLE)]),
        tempfile.mkdtemp(),
        "doc",
    )
    # one table -> one .csv + one .tsv; two tables -> four files
    assert len(single) == 2
    assert len(multi) == 4
    assert all(p.suffix in (".csv", ".tsv") for p in single + multi)


def test_save_tables_empty_writes_nothing():
    assert save_tables([], tempfile.mkdtemp(), "doc") == []


def test_json_string_result_ignored_safely():
    assert extract_tables(json.dumps(_result(MD_TABLE))) == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
