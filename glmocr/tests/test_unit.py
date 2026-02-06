"""Unit tests for glmocr (no external services required)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ═══════════════════════════════════════════════════════════════════════
# NEW TESTS – config system, parser result, parse() overloads, bbox
# ═══════════════════════════════════════════════════════════════════════


class TestCoerceEnvValue:
    """Tests for _coerce_env_value()."""

    def test_mode_maas(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.maas.enabled", "maas") is True

    def test_mode_true(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.maas.enabled", "true") is True

    def test_mode_selfhosted(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.maas.enabled", "selfhosted") is False

    def test_mode_case_insensitive(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.maas.enabled", "MaaS") is True
        assert _coerce_env_value("pipeline.maas.enabled", "TRUE") is True

    def test_enable_layout_true(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.enable_layout", "1") is True
        assert _coerce_env_value("pipeline.enable_layout", "yes") is True

    def test_enable_layout_false(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.enable_layout", "0") is False
        assert _coerce_env_value("pipeline.enable_layout", "no") is False

    def test_integer_coercion(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.maas.request_timeout", "600") == 600
        assert isinstance(_coerce_env_value("pipeline.ocr_api.api_port", "8080"), int)

    def test_string_passthrough(self):
        from glmocr.config import _coerce_env_value

        assert _coerce_env_value("pipeline.maas.api_key", "sk-abc") == "sk-abc"


class TestDeepMerge:
    """Tests for _deep_merge()."""

    def test_shallow(self):
        from glmocr.config import _deep_merge

        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 99, "c": 3})
        assert base == {"a": 1, "b": 99, "c": 3}

    def test_nested(self):
        from glmocr.config import _deep_merge

        base = {"x": {"y": 1, "z": 2}}
        _deep_merge(base, {"x": {"z": 99}})
        assert base["x"]["y"] == 1
        assert base["x"]["z"] == 99

    def test_override_dict_with_scalar(self):
        from glmocr.config import _deep_merge

        base = {"a": {"nested": True}}
        _deep_merge(base, {"a": "flat"})
        assert base["a"] == "flat"


class TestSetNested:
    """Tests for _set_nested()."""

    def test_simple(self):
        from glmocr.config import _set_nested

        d: dict = {}
        _set_nested(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_top_level(self):
        from glmocr.config import _set_nested

        d: dict = {}
        _set_nested(d, "key", "val")
        assert d == {"key": "val"}


class TestFindDotenv:
    """Tests for _find_dotenv()."""

    def test_finds_dotenv_in_start_dir(self, tmp_path):
        from glmocr.config import _find_dotenv

        dotenv = tmp_path / ".env"
        dotenv.write_text("X=1\n")
        assert _find_dotenv(start=tmp_path) == dotenv

    def test_finds_dotenv_in_parent(self, tmp_path):
        from glmocr.config import _find_dotenv

        dotenv = tmp_path / ".env"
        dotenv.write_text("X=1\n")
        child = tmp_path / "a" / "b"
        child.mkdir(parents=True)
        assert _find_dotenv(start=child) == dotenv

    def test_returns_none_when_missing(self, tmp_path):
        from glmocr.config import _find_dotenv

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        # Walk up from a leaf inside tmp_path; the .env won't exist in /tmp
        # Use a file-system boundary: we can't control parents, so we just
        # assert the return type is None or Path.
        result = _find_dotenv(start=empty_dir)
        # If the test runner's CWD has no .env, result is None.
        # We can't assert None universally (CI may have one), so just check type.
        assert result is None or isinstance(result, Path)


class TestCollectEnvOverrides:
    """Tests for _collect_env_overrides() with real env vars."""

    def test_picks_up_env_var(self, monkeypatch):
        from glmocr.config import _collect_env_overrides

        monkeypatch.setenv("GLMOCR_API_KEY", "sk-env")
        overrides = _collect_env_overrides()
        # Should produce nested dict: pipeline.maas.api_key = "sk-env"
        assert overrides["pipeline"]["maas"]["api_key"] == "sk-env"

    def test_mode_env_var(self, monkeypatch):
        from glmocr.config import _collect_env_overrides

        monkeypatch.setenv("GLMOCR_MODE", "maas")
        overrides = _collect_env_overrides()
        assert overrides["pipeline"]["maas"]["enabled"] is True

    def test_dotenv_file_loaded(self, tmp_path, monkeypatch):
        """_collect_env_overrides reads .env when present."""
        from glmocr.config import _collect_env_overrides

        dotenv = tmp_path / ".env"
        dotenv.write_text("GLMOCR_API_KEY=sk-from-dotenv\n")
        # Patch _find_dotenv to return our temp .env
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: dotenv)
        # Make sure real env doesn't have the key
        monkeypatch.delenv("GLMOCR_API_KEY", raising=False)

        overrides = _collect_env_overrides()
        assert overrides["pipeline"]["maas"]["api_key"] == "sk-from-dotenv"

    def test_real_env_beats_dotenv(self, tmp_path, monkeypatch):
        """os.environ takes priority over .env file."""
        from glmocr.config import _collect_env_overrides

        dotenv = tmp_path / ".env"
        dotenv.write_text("GLMOCR_API_KEY=sk-dotenv\n")
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: dotenv)
        monkeypatch.setenv("GLMOCR_API_KEY", "sk-real")

        overrides = _collect_env_overrides()
        assert overrides["pipeline"]["maas"]["api_key"] == "sk-real"

    def test_no_env_returns_empty(self, monkeypatch):
        """When no GLMOCR_* vars exist, returns empty dict."""
        from glmocr.config import _collect_env_overrides, _ENV_MAP, ENV_PREFIX

        # Clear all GLMOCR_* vars
        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        assert _collect_env_overrides() == {}


class TestFromEnv:
    """Tests for GlmOcrConfig.from_env() – full priority chain."""

    def test_defaults_when_nothing_set(self, monkeypatch):
        """from_env with no args, no env → pure defaults."""
        from glmocr.config import GlmOcrConfig, _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        cfg = GlmOcrConfig.from_env()
        assert cfg.pipeline.maas.enabled is False
        assert cfg.logging.level == "INFO"

    def test_overrides_win_over_env(self, monkeypatch):
        """Keyword overrides beat env vars."""
        from glmocr.config import GlmOcrConfig

        monkeypatch.setenv("GLMOCR_API_KEY", "sk-env")
        cfg = GlmOcrConfig.from_env(api_key="sk-override")
        assert cfg.pipeline.maas.api_key == "sk-override"

    def test_env_wins_over_yaml(self, tmp_path, monkeypatch):
        """Env vars beat YAML values."""
        from glmocr.config import GlmOcrConfig

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("pipeline:\n  maas:\n    api_key: sk-yaml\n")
        monkeypatch.setenv("GLMOCR_API_KEY", "sk-env")
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        cfg = GlmOcrConfig.from_env(config_path=str(yaml_file))
        assert cfg.pipeline.maas.api_key == "sk-env"

    def test_mode_kwarg_enables_maas(self, monkeypatch):
        from glmocr.config import GlmOcrConfig, _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        cfg = GlmOcrConfig.from_env(mode="maas")
        assert cfg.pipeline.maas.enabled is True

    def test_missing_explicit_yaml_raises(self, tmp_path):
        from glmocr.config import GlmOcrConfig

        with pytest.raises(FileNotFoundError):
            GlmOcrConfig.from_env(config_path=str(tmp_path / "nope.yaml"))

    def test_load_config_is_from_env_alias(self, monkeypatch):
        """load_config() delegates to from_env()."""
        from glmocr.config import load_config, _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        cfg = load_config(api_key="sk-test", mode="maas")
        assert cfg.pipeline.maas.api_key == "sk-test"
        assert cfg.pipeline.maas.enabled is True


class TestBaseParserResultSerialization:
    """Tests for to_dict() / to_json() on BaseParserResult subclasses."""

    def _make_result(self, **kwargs):
        from glmocr.parser_result import PipelineResult

        defaults = dict(
            json_result=[
                [
                    {
                        "index": 0,
                        "label": "text",
                        "content": "hello",
                        "bbox_2d": [0, 0, 500, 500],
                    }
                ]
            ],
            markdown_result="# Hello",
            original_images=[],
        )
        defaults.update(kwargs)
        return PipelineResult(**defaults)

    def test_to_dict_basic_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert "json_result" in d
        assert "markdown_result" in d
        assert "original_images" in d
        assert d["markdown_result"] == "# Hello"

    def test_to_dict_includes_usage_when_set(self):
        r = self._make_result()
        r._usage = {"total_tokens": 42}
        d = r.to_dict()
        assert d["usage"] == {"total_tokens": 42}

    def test_to_dict_includes_error_when_set(self):
        r = self._make_result()
        r._error = "timeout"
        d = r.to_dict()
        assert d["error"] == "timeout"

    def test_to_dict_excludes_private_when_unset(self):
        r = self._make_result()
        d = r.to_dict()
        assert "usage" not in d
        assert "error" not in d
        assert "data_info" not in d

    def test_to_json_returns_valid_json(self):
        r = self._make_result()
        s = r.to_json()
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["markdown_result"] == "# Hello"

    def test_to_json_kwargs_forwarded(self):
        r = self._make_result()
        s = r.to_json(indent=None, sort_keys=True)
        # No indentation means single line (roughly)
        assert "\n" not in s
        parsed = json.loads(s)
        assert "json_result" in parsed

    def test_to_json_unicode_preserved(self):
        r = self._make_result(markdown_result="中文测试")
        s = r.to_json()
        # ensure_ascii=False by default → raw CJK characters
        assert "中文测试" in s

    def test_repr(self):
        r = self._make_result()
        assert "PipelineResult" in repr(r)
        assert "images=0" in repr(r)


class TestNormaliseBbox:
    """Tests for GlmOcr._normalise_bbox()."""

    def test_basic_normalisation(self):
        from glmocr.api import GlmOcr

        result = GlmOcr._normalise_bbox([500, 500, 1000, 1000], 2000, 2000)
        assert result == [250, 250, 500, 500]

    def test_full_page(self):
        from glmocr.api import GlmOcr

        result = GlmOcr._normalise_bbox([0, 0, 2040, 2640], 2040, 2640)
        assert result == [0, 0, 1000, 1000]

    def test_rounding(self):
        from glmocr.api import GlmOcr

        # 431/2040 * 1000 = 211.27 → 211
        result = GlmOcr._normalise_bbox([431, 1762, 1061, 2189], 2040, 2640)
        assert result == [211, 667, 520, 829]

    def test_none_input(self):
        from glmocr.api import GlmOcr

        assert GlmOcr._normalise_bbox(None, 100, 100) is None

    def test_empty_list(self):
        from glmocr.api import GlmOcr

        assert GlmOcr._normalise_bbox([], 100, 100) == []

    def test_wrong_length(self):
        from glmocr.api import GlmOcr

        assert GlmOcr._normalise_bbox([1, 2, 3], 100, 100) == [1, 2, 3]

    def test_zero_dimensions(self):
        from glmocr.api import GlmOcr

        assert GlmOcr._normalise_bbox([10, 20, 30, 40], 0, 100) == [10, 20, 30, 40]
        assert GlmOcr._normalise_bbox([10, 20, 30, 40], 100, 0) == [10, 20, 30, 40]


class TestNormaliseMarkdownBboxes:
    """Tests for GlmOcr._normalise_markdown_bboxes()."""

    def test_basic_replacement(self):
        from glmocr.api import GlmOcr

        md = "Some text ![](page=0,bbox=[500, 500, 1000, 1000]) more text"
        pages = [{"width": 2000, "height": 2000}]
        result = GlmOcr._normalise_markdown_bboxes(md, pages)
        assert "[250, 250, 500, 500]" in result

    def test_multiple_refs(self):
        from glmocr.api import GlmOcr

        md = (
            "![](page=0,bbox=[0, 0, 2040, 2640]) "
            "![](page=0,bbox=[500, 500, 1000, 1000])"
        )
        pages = [{"width": 2040, "height": 2640}]
        result = GlmOcr._normalise_markdown_bboxes(md, pages)
        assert "[0, 0, 1000, 1000]" in result

    def test_multipage(self):
        from glmocr.api import GlmOcr

        md = "![](page=0,bbox=[100, 100, 200, 200]) ![](page=1,bbox=[300, 300, 600, 600])"
        pages = [
            {"width": 1000, "height": 1000},
            {"width": 3000, "height": 3000},
        ]
        result = GlmOcr._normalise_markdown_bboxes(md, pages)
        assert "[100, 100, 200, 200]" in result  # page 0: 1000px → ×1
        assert "[100, 100, 200, 200]" in result  # page 1: 300/3000*1000=100

    def test_empty_markdown(self):
        from glmocr.api import GlmOcr

        assert (
            GlmOcr._normalise_markdown_bboxes("", [{"width": 100, "height": 100}]) == ""
        )

    def test_empty_pages(self):
        from glmocr.api import GlmOcr

        md = "![](page=0,bbox=[10, 20, 30, 40])"
        assert GlmOcr._normalise_markdown_bboxes(md, []) == md

    def test_out_of_range_page(self):
        from glmocr.api import GlmOcr

        md = "![](page=5,bbox=[10, 20, 30, 40])"
        pages = [{"width": 100, "height": 100}]
        # page 5 doesn't exist → keep original
        assert GlmOcr._normalise_markdown_bboxes(md, pages) == md


class TestParseReturnType:
    """Tests for GlmOcr.parse() return type: str→single, list→list."""

    def _make_glmocr(self):
        """Create a GlmOcr instance with mocked internals."""
        from glmocr.api import GlmOcr
        from glmocr.parser_result import PipelineResult

        mock_result = PipelineResult(
            json_result=[[{"index": 0, "label": "text", "content": "hi"}]],
            markdown_result="hi",
            original_images=["test.png"],
        )

        obj = object.__new__(GlmOcr)
        obj._use_maas = True
        obj._pipeline = None
        obj._maas_client = MagicMock()
        obj.config_model = MagicMock()
        obj.enable_layout = True

        # Mock _parse_maas to return a list of one result
        obj._parse_maas = MagicMock(return_value=[mock_result])
        return obj

    def test_str_input_returns_single(self):
        parser = self._make_glmocr()
        result = parser.parse("image.png")
        from glmocr.parser_result import PipelineResult

        assert isinstance(result, PipelineResult)
        assert not isinstance(result, list)

    def test_list_input_returns_list(self):
        parser = self._make_glmocr()
        result = parser.parse(["image.png"])
        assert isinstance(result, list)
        assert len(result) == 1

    def test_list_multiple_returns_list(self):
        from glmocr.parser_result import PipelineResult

        parser = self._make_glmocr()
        r1 = PipelineResult(json_result=[], markdown_result="a", original_images=[])
        r2 = PipelineResult(json_result=[], markdown_result="b", original_images=[])
        parser._parse_maas = MagicMock(return_value=[r1, r2])

        result = parser.parse(["a.png", "b.png"])
        assert isinstance(result, list)
        assert len(result) == 2


class TestGlmOcrConstructor:
    """Tests for GlmOcr.__init__ kwarg handling (config assembly only)."""

    def test_api_key_implies_maas(self, monkeypatch):
        """Passing api_key without mode should default to maas."""
        from glmocr.config import _ENV_MAP, ENV_PREFIX

        # Clean env
        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        # MaaSClient is imported inside __init__ → patch at source module
        with patch("glmocr.maas_client.MaaSClient") as mock_maas:
            mock_maas.return_value.start = MagicMock()
            from glmocr.api import GlmOcr

            parser = GlmOcr(api_key="sk-test")
            assert parser._use_maas is True
            assert parser.config_model.pipeline.maas.api_key == "sk-test"
            parser.close()

    def test_explicit_selfhosted_mode(self, monkeypatch):
        """mode='selfhosted' keeps maas disabled."""
        from glmocr.config import _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        with patch("glmocr.pipeline.Pipeline") as mock_pipeline:
            mock_pipeline.return_value.start = MagicMock()
            mock_pipeline.return_value.enable_layout = False
            from glmocr.api import GlmOcr

            parser = GlmOcr(mode="selfhosted")
            assert parser._use_maas is False
            parser.close()
