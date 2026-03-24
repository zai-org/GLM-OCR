"""Detect example — returns table bounding boxes without running table OCR.

Run:
    python examples/detect/run.py

Drop example.png into examples/detect/ before running.
"""

from __future__ import annotations

import json
from pathlib import Path

from glmocr.api import GlmOcr

HERE = Path(__file__).resolve().parent
PNG = HERE / "example.png"
CONFIG = HERE / "config.yaml"


def main() -> None:
    if not PNG.exists():
        raise FileNotFoundError(f"Place your image at: {PNG}")

    with GlmOcr(config_path=str(CONFIG), mode="selfhosted") as parser:
        result = parser.parse(str(PNG))

    regions = result.json_result[0] if result.json_result else []
    detect_regions = [r for r in regions if r.get("task_type") == "detect" or (r.get("label") == "table" and r.get("content") == "")]
    print(f"\n--- {len(detect_regions)} detect region(s) ---")
    for r in detect_regions:
        bbox = r.get("bbox_2d")
        print(f"  label={r.get('label')}  bbox_2d={bbox}  content={r.get('content')!r}")

    print("\nFull json_result:")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
