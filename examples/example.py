"""Generate example outputs.

This script parses all files under examples/source/ and writes results into
examples/result/ (one folder per input file).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from glmocr.api import GlmOcr


def main() -> int:
    here = Path(__file__).resolve().parent
    source_dir = here / "source"
    output_dir = here / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise RuntimeError(f"Missing examples source dir: {source_dir}")

    inputs = sorted(
        [
            *source_dir.glob("*.png"),
            *source_dir.glob("*.jpg"),
            *source_dir.glob("*.jpeg"),
            *source_dir.glob("*.pdf"),
        ]
    )
    if not inputs:
        raise RuntimeError(f"No input files found under: {source_dir}")

    print(f"Found {len(inputs)} inputs under {source_dir}")
    print(f"Writing results to {output_dir}")

    poppler_ok = any(
        shutil.which(cmd) is not None for cmd in ("pdfinfo", "pdftoppm", "pdftocairo")
    )
    if not poppler_ok and any(p.suffix.lower() == ".pdf" for p in inputs):
        print(
            "Poppler not found (pdfinfo/pdftoppm/pdftocairo). "
            "PDF inputs will be skipped. On macOS: brew install poppler"
        )

    total = len(inputs)
    processed = 0
    skipped = 0
    failed = 0

    with GlmOcr() as parser:
        for idx, p in enumerate(inputs, start=1):
            print(f"\n[{idx}/{total}] Parsing: {p.name}")
            if p.suffix.lower() == ".pdf" and not poppler_ok:
                print(f"[{idx}/{total}] Skipping PDF (missing poppler)")
                skipped += 1
                continue

            try:
                result = parser.parse(str(p))
                result.save(output_dir=output_dir)
                processed += 1
                print(f"[{idx}/{total}] Done: {p.name}")
            except Exception as e:
                failed += 1
                print(f"[{idx}/{total}] Failed: {p.name}: {e}")
                continue

    print(f"\nAll done. processed={processed}, skipped={skipped}, failed={failed}, total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
