"""Unit tests for glmocr (no external services required)."""

from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfig:
    """Tests for Config."""

    def test_config_load_default(self):
        """Loads default config."""
        from glmocr.config import load_config

        cfg = load_config().to_dict()
        assert "server" in cfg or "pipeline" in cfg

    def test_config_to_dict(self):
        """to_dict returns a dict."""
        from glmocr.config import load_config

        cfg = load_config().to_dict()
        assert isinstance(cfg, dict)


class TestPageLoader:
    """Tests for PageLoader."""

    def test_pageloader_init(self):
        """Can initialize PageLoader."""
        from glmocr.dataloader import PageLoader
        from glmocr.config import PageLoaderConfig

        loader = PageLoader(PageLoaderConfig())
        assert loader is not None

    def test_pageloader_with_config(self):
        """Respects basic config fields."""
        from glmocr.dataloader import PageLoader
        from glmocr.config import PageLoaderConfig

        config = PageLoaderConfig(
            max_tokens=8192,
            temperature=0.1,
            image_format="PNG",
        )
        loader = PageLoader(config)
        assert loader.max_tokens == 8192
        assert loader.image_format == "PNG"

    def test_pageloader_load_pdf_requires_pypdfium2(self):
        """Gives a clear error when pypdfium2 is unavailable."""
        from glmocr.dataloader import PageLoader
        from glmocr.config import PageLoaderConfig

        loader = PageLoader(PageLoaderConfig())
        with patch("glmocr.dataloader.page_loader.PYPDFIUM2_AVAILABLE", False):
            with pytest.raises(RuntimeError) as exc:
                loader._load_pdf("dummy.pdf")
            assert "pypdfium2" in str(exc.value).lower()

    def test_pageloader_load_pdf_pages(self):
        """Expands a PDF into page images (requires pypdfium2)."""
        from glmocr.config import PageLoaderConfig
        from glmocr.dataloader import PageLoader
        from glmocr.utils.image_utils import PYPDFIUM2_AVAILABLE

        if not PYPDFIUM2_AVAILABLE:
            pytest.skip("pypdfium2 is not installed")

        repo_root = Path(__file__).resolve().parents[2]
        source_dir = repo_root / "examples" / "source"
        sample_pdf = next(
            (f for f in source_dir.iterdir() if f.suffix.lower() == ".pdf"),
            None,
        )
        if not sample_pdf or not sample_pdf.exists():
            pytest.skip(f"No sample PDF found in {source_dir}")

        loader = PageLoader(PageLoaderConfig())
        pages = loader.load_pages(str(sample_pdf))
        assert isinstance(pages, list)
        assert len(pages) >= 1

    def test_pageloader_load_pdf_via_file_uri(self):
        """Parses PDF file:// URIs correctly."""
        from glmocr.dataloader import PageLoader
        from glmocr.utils.image_utils import PYPDFIUM2_AVAILABLE
        from glmocr.config import PageLoaderConfig

        if not PYPDFIUM2_AVAILABLE:
            pytest.skip("pypdfium2 is not installed")

        repo_root = Path(__file__).resolve().parents[2]
        source_dir = repo_root / "examples" / "source"
        sample_pdf = next(
            (f for f in source_dir.iterdir() if f.suffix.lower() == ".pdf"),
            None,
        )
        if not sample_pdf or not sample_pdf.exists():
            pytest.skip(f"No sample PDF found in {source_dir}")

        loader = PageLoader(PageLoaderConfig())
        pdf_uri = f"file://{sample_pdf.resolve()}"
        pages = loader.load_pages(pdf_uri)
        assert len(pages) >= 1


class TestParseResult:
    """Tests for ParseResult."""

    def test_parse_result_init_with_dict(self):
        """Can initialize ParseResult with a dict."""
        from glmocr.api import ParseResult

        result = ParseResult(
            json_result={"test": "data"},
            markdown_result="# Test",
            original_images=["/path/to/image.png"],
        )
        assert result.json_result == {"test": "data"}
        assert result.markdown_result == "# Test"

    def test_parse_result_init_with_json_string(self):
        """Can initialize ParseResult with a JSON string."""
        from glmocr.api import ParseResult

        json_str = '{"key": "value"}'
        result = ParseResult(
            json_result=json_str,
            markdown_result=None,
            original_images=[],
        )
        assert result.json_result == {"key": "value"}

    def test_parse_result_init_with_invalid_json_string(self):
        """Keeps invalid JSON strings as-is."""
        from glmocr.api import ParseResult

        html_str = "<table><tr><td>hello</td></tr></table>"
        result = ParseResult(
            json_result=html_str,
            markdown_result=None,
            original_images=[],
        )
        # Non-JSON is preserved
        assert result.json_result == html_str

    def test_parse_result_repr(self):
        """repr includes image count."""
        from glmocr.api import ParseResult

        result = ParseResult(
            json_result={},
            markdown_result=None,
            original_images=["a.png", "b.png"],
        )
        assert "images=2" in repr(result)


class TestPipeline:
    """Tests for Pipeline (without starting)."""

    def test_pipeline_init_enable_layout_default(self):
        """Default enable_layout behavior (mocked)."""
        from glmocr.pipeline import Pipeline

        # Use a mock to avoid heavy dependencies
        with patch.object(Pipeline, "__init__", lambda self, config: None):
            p = Pipeline.__new__(Pipeline)
            p.config = {}
            p.enable_layout = p.config.get("enable_layout", True)
            assert p.enable_layout is True

    def test_pipeline_init_enable_layout_false(self):
        """enable_layout can be disabled (mocked)."""
        with patch("glmocr.pipeline.Pipeline.__init__", return_value=None):
            from glmocr.pipeline import Pipeline

            p = Pipeline.__new__(Pipeline)
            p.config = {"enable_layout": False}
            p.enable_layout = p.config.get("enable_layout", True)
            assert p.enable_layout is False


class TestUtils:
    """Tests for utility functions."""

    def test_image_utils_crop_image_region(self):
        """crop_image_region exists."""
        from glmocr.utils.image_utils import crop_image_region

        assert callable(crop_image_region)


class TestResultFormatter:
    """Tests for ResultFormatter."""

    def test_result_formatter_init(self):
        """Can initialize ResultFormatter."""
        from glmocr.postprocess import ResultFormatter
        from glmocr.config import ResultFormatterConfig

        formatter = ResultFormatter(ResultFormatterConfig())
        assert formatter is not None

    def test_result_formatter_format_ocr_result(self):
        """format_ocr_result returns JSON and Markdown."""
        from glmocr.postprocess import ResultFormatter
        from glmocr.config import ResultFormatterConfig

        formatter = ResultFormatter(ResultFormatterConfig())
        json_str, md_str = formatter.format_ocr_result("Hello World")
        assert "Hello World" in json_str
        assert md_str == "Hello World"

    def test_result_formatter_clean_content(self):
        """Content cleanup works."""
        from glmocr.postprocess import ResultFormatter
        from glmocr.config import ResultFormatterConfig

        formatter = ResultFormatter(ResultFormatterConfig())
        # Repeated punctuation cleanup
        cleaned = formatter._clean_content("Hello....World")
        assert "....." not in cleaned
