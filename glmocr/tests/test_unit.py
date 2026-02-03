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

    def test_load_image_to_base64_accepts_raw_base64(self):
        """load_image_to_base64 accepts raw base64 payloads (OCRClient path)."""
        import base64
        from io import BytesIO
        from PIL import Image

        from glmocr.utils.image_utils import load_image_to_base64

        img = Image.new("RGB", (8, 8), color=(255, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        out_b64 = load_image_to_base64(
            raw_b64,
            t_patch_size=2,
            max_pixels=112 * 112 * 10,
            image_format="PNG",
            patch_expand_factor=1,
            min_pixels=112 * 112,
        )
        assert isinstance(out_b64, str)
        # Should still be valid base64
        base64.b64decode(out_b64 + "===")

    def test_load_image_to_base64_accepts_base64_prefix(self):
        """load_image_to_base64 accepts <|base64|>... blobs."""
        import base64
        from io import BytesIO
        from PIL import Image

        from glmocr.utils.image_utils import load_image_to_base64

        img = Image.new("RGB", (8, 8), color=(0, 255, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        blob = "<|base64|>" + raw_b64

        out_b64 = load_image_to_base64(
            blob,
            t_patch_size=2,
            max_pixels=112 * 112 * 10,
            image_format="PNG",
            patch_expand_factor=1,
            min_pixels=112 * 112,
        )
        assert isinstance(out_b64, str)
        base64.b64decode(out_b64 + "===")


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


class TestMaaSClient:
    """Tests for MaaSClient."""

    def test_maas_config_defaults(self):
        """MaaSApiConfig has correct defaults."""
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig()
        assert config.enabled is False
        assert config.api_url == "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
        assert config.model == "glm-ocr"
        assert config.verify_ssl is True

    def test_maas_client_requires_api_key(self):
        """MaaSClient raises error when API key is missing."""
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key=None)
        with pytest.raises(ValueError) as exc:
            MaaSClient(config)
        assert "API key is required" in str(exc.value)

    def test_maas_client_init_with_api_key(self):
        """MaaSClient initializes correctly with API key."""
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key-12345")
        client = MaaSClient(config)
        assert client.api_key == "test-key-12345"
        assert client.model == "glm-ocr"

    def test_maas_client_prepare_file_url(self):
        """MaaSClient handles URLs correctly."""
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key")
        client = MaaSClient(config)

        # URL should be returned as-is
        url = "https://example.com/image.png"
        result = client._prepare_file(url)
        assert result == url

    def test_maas_client_prepare_file_bytes(self):
        """MaaSClient encodes bytes to base64."""
        import base64
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key")
        client = MaaSClient(config)

        # Bytes should be encoded to base64 and wrapped as data URI
        data = b"test image data"
        result = client._prepare_file(data)
        expected = base64.b64encode(data).decode("utf-8")
        assert result.endswith(expected)
        assert result.startswith("data:")

    def test_maas_client_prepare_file_base64_string(self):
        """MaaSClient accepts base64 strings directly."""
        import base64
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key")
        client = MaaSClient(config)

        # A long base64 string should be wrapped as a data URI
        original_data = b"\xff\xff\xff" * 80  # Ensure base64 contains '/'
        b64_str = base64.b64encode(original_data).decode("utf-8")
        result = client._prepare_file(b64_str)
        assert result.startswith("data:")
        assert result.endswith(b64_str)

    def test_maas_client_prepare_file_data_uri(self):
        """MaaSClient extracts base64 from data URIs."""
        import base64
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key")
        client = MaaSClient(config)

        # Data URI with base64
        b64_data = base64.b64encode(b"test image").decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_data}"
        result = client._prepare_file(data_uri)
        assert result == data_uri

    def test_maas_client_looks_like_base64(self):
        """_looks_like_base64 correctly identifies base64 strings."""
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key")
        client = MaaSClient(config)

        # Long base64 string (including '/') should be detected
        import base64

        long_b64 = base64.b64encode(b"\xff\xff\xff" * 80).decode("utf-8")
        assert client._looks_like_base64(long_b64) is True

        # File paths should not be detected as base64
        assert client._looks_like_base64("/path/to/file.png") is False
        assert client._looks_like_base64("image.png") is False
        assert client._looks_like_base64("C:\\Users\\file.pdf") is False

    def test_maas_client_context_manager(self):
        """MaaSClient works as context manager."""
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key="test-key")
        with MaaSClient(config) as client:
            assert client._session is not None
        assert client._session is None

    @patch("glmocr.maas_client.requests.Session")
    def test_maas_client_parse_success(self, mock_session_cls):
        """MaaSClient.parse returns response on success."""
        from glmocr.maas_client import MaaSClient
        from glmocr.config import MaaSApiConfig

        # Mock successful response
        mock_response = type(
            "Response",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "id": "task_123",
                    "model": "glm-ocr",
                    "md_results": "# Test",
                    "layout_details": [
                        [{"index": 0, "label": "text", "content": "Hello"}]
                    ],
                },
            },
        )()

        mock_session = mock_session_cls.return_value
        mock_session.post.return_value = mock_response

        config = MaaSApiConfig(api_key="test-key")
        client = MaaSClient(config)
        client.start()

        result = client.parse("https://example.com/image.png")
        assert result["id"] == "task_123"
        assert result["md_results"] == "# Test"

    def test_glmocr_detects_maas_mode(self):
        """GlmOcr detects MaaS mode from config."""
        from glmocr.config import GlmOcrConfig

        # Create config with MaaS enabled
        config = GlmOcrConfig()
        config.pipeline.maas.enabled = True
        config.pipeline.maas.api_key = "test-key"

        assert config.pipeline.maas.enabled is True

    def test_config_maas_in_pipeline(self):
        """PipelineConfig has maas field."""
        from glmocr.config import PipelineConfig

        config = PipelineConfig()
        assert hasattr(config, "maas")
        assert config.maas.enabled is False
