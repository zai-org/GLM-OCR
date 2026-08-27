"""Regression tests for RGBA-to-JPEG saving (#215).

JPEG does not support an alpha channel; extracted images from transparent
PNG sources must be flattened to RGB before ``Image.save`` is called,
otherwise the save fails and the image silently disappears from output.

Run:
    python -m pytest glmocr/tests/test_flatten_to_rgb.py -v
or standalone:
    python glmocr/tests/test_flatten_to_rgb.py
"""

import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PIL import Image

from glmocr.parser_result.base import flatten_to_rgb


def _assert_jpeg_writable(img: Image.Image) -> None:
    out = flatten_to_rgb(img)
    assert out.mode == "RGB", f"expected RGB, got {out.mode}"
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    reloaded = Image.open(buf)
    assert reloaded.format == "JPEG"


def test_rgba_image():
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 128))
    _assert_jpeg_writable(img)


def test_la_image():
    img = Image.new("LA", (10, 10), (0, 200))
    _assert_jpeg_writable(img)


def test_palette_image():
    img = Image.new("P", (10, 10))
    img.putpalette([255, 255, 255] * 256)
    _assert_jpeg_writable(img)


def test_palette_with_transparency():
    img = Image.new("P", (10, 10))
    img.putpalette([255, 255, 255] * 256)
    img.info["transparency"] = 0
    _assert_jpeg_writable(img)


def test_l_image():
    _assert_jpeg_writable(Image.new("L", (10, 10), 128))


def test_rgb_passthrough_no_copy_needed():
    img = Image.new("RGB", (10, 10))
    assert flatten_to_rgb(img) is img


def test_transparency_flattened_onto_white():
    """A fully transparent pixel must become white, not black."""
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    out = flatten_to_rgb(img)
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_end_to_end_save_to_disk_as_jpg(tmp_path=None):
    """Simulates the pipeline: RGBA source saved under a .jpg filename."""
    img = Image.new("RGBA", (16, 16), (10, 20, 30, 255))
    target = (tmp_path or tempfile.mkdtemp())
    path = os.path.join(target, "extracted_page0_idx0.jpg")
    flatten_to_rgb(img).save(path, quality=95)
    with Image.open(path) as reloaded:
        assert reloaded.format == "JPEG"
        assert reloaded.mode == "RGB"


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
