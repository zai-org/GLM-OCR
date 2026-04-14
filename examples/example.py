"""Generate example outputs.

This script parses all files under examples/source/ and writes results into
examples/result/ (one folder per input file).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from glmocr.api import GlmOcr

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _get_pdf_pages(pdf_path: Path) -> int:
    """Best-effort page count for local PDFs via pdfinfo."""
    try:
        output = subprocess.check_output(
            ["pdfinfo", str(pdf_path)],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in output.splitlines():
            if line.lower().startswith("pages:"):
                pages = int(line.split(":", 1)[1].strip())
                return max(1, pages)
    except Exception:
        pass
    return 1


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
    work_units = [
        _get_pdf_pages(p) if p.suffix.lower() == ".pdf" and poppler_ok else 1
        for p in inputs
    ]
    total_units = sum(work_units)
    processed = 0
    skipped = 0
    failed = 0
    progress = (
        tqdm(total=total_units, desc="Parsing requests", unit="page")
        if tqdm is not None and total_units > 1
        else None
    )

    with GlmOcr() as parser:
        for idx, p in enumerate(inputs, start=1):
            unit_count = work_units[idx - 1]
            print(f"\n[{idx}/{total}] Parsing: {p.name} ({unit_count} page(s))")
            if p.suffix.lower() == ".pdf" and not poppler_ok:
                print(f"[{idx}/{total}] Skipping PDF (missing poppler)")
                skipped += 1
                if progress is not None:
                    progress.update(unit_count)
                continue

            try:
                result = parser.parse(str(p))
                result.save(output_dir=output_dir)
                processed += 1
                print(f"[{idx}/{total}] Done: {p.name}")
            except Exception as e:
                failed += 1
                print(f"[{idx}/{total}] Failed: {p.name}: {e}")
            finally:
                if progress is not None:
                    progress.update(unit_count)

    if progress is not None:
        progress.close()

    print(f"\nAll done. processed={processed}, skipped={skipped}, failed={failed}, total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
