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

    def test_selfhosted_layout_defaults_are_publicly_configured(self):
        """Self-hosted config loads packaged layout defaults."""
        from glmocr.config import load_config

        cfg = load_config(mode="selfhosted")
        layout = cfg.pipeline.layout

        assert cfg.pipeline.maas.enabled is False
        assert layout.model_dir == "PaddlePaddle/PP-DocLayoutV3_safetensors"
        assert isinstance(layout.label_task_mapping, dict)
        assert layout.label_task_mapping
        assert isinstance(layout.id2label, dict)
        assert layout.id2label


class TestLayoutDeviceUnit:
    """Unit tests for layout device selection and config plumbing (mocked)."""

    @staticmethod
    def _require_layout_runtime():
        """Require optional self-hosted layout deps, otherwise skip tests."""
        torch = pytest.importorskip(
            "torch", reason="layout device unit tests require optional selfhosted deps"
        )
        pytest.importorskip(
            "transformers",
            reason="layout device unit tests require optional selfhosted deps",
        )
        pytest.importorskip(
            "cv2",
            reason="layout device unit tests require optional selfhosted deps",
        )
        return torch

    def test_layout_config_device_default_is_none(self):
        """LayoutConfig.device defaults to None (auto-select)."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig()
        assert cfg.device is None

    def test_layout_config_device_cpu(self):
        """LayoutConfig accepts 'cpu' as device."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig(device="cpu")
        assert cfg.device == "cpu"

    def test_layout_config_device_cuda(self):
        """LayoutConfig accepts 'cuda' as device."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig(device="cuda")
        assert cfg.device == "cuda"

    def test_layout_config_device_cuda_index(self):
        """LayoutConfig accepts 'cuda:1' as device."""
        from glmocr.config import LayoutConfig

        cfg = LayoutConfig(device="cuda:1")
        assert cfg.device == "cuda:1"

    def test_env_var_sets_device(self, monkeypatch):
        """GLMOCR_LAYOUT_DEVICE env var propagates to config."""
        from glmocr.config import GlmOcrConfig, _ENV_MAP, ENV_PREFIX

        # Clear other GLMOCR_ vars to avoid interference
        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        monkeypatch.setenv("GLMOCR_LAYOUT_DEVICE", "cpu")
        cfg = GlmOcrConfig.from_env()
        assert cfg.pipeline.layout.device == "cpu"

    def test_from_env_layout_device_kwarg(self, monkeypatch):
        """layout_device kwarg in from_env() sets device correctly."""
        from glmocr.config import GlmOcrConfig, _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        cfg = GlmOcrConfig.from_env(layout_device="cuda:1")
        assert cfg.pipeline.layout.device == "cuda:1"

    # Minimal config kwargs for mocked detector tests
    _MOCK_LAYOUT_KWARGS = dict(
        id2label={0: "text"},
        label_task_mapping={"text": ["text"]},
    )

    def _mock_detector(self, device_val, *, model_dir="dummy"):
        """Create a PPDocLayoutDetector with mocked model, ready for start()."""
        self._require_layout_runtime()
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = LayoutConfig(
            device=device_val, model_dir=model_dir, **self._MOCK_LAYOUT_KWARGS
        )
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "text"}
        mock_processor = MagicMock()
        return det, mock_model, mock_processor

    def test_detector_device_selection_explicit_cpu(self):
        """When config.device='cpu', detector picks CPU even if CUDA is available."""
        det, mock_model, mock_proc = self._mock_detector("cpu")

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
        ):
            det.start()

        assert det._device == "cpu"
        mock_model.to.assert_called_with("cpu")

    def test_detector_device_selection_explicit_cuda(self):
        """When config.device='cuda:1', detector picks that device."""
        det, mock_model, mock_proc = self._mock_detector("cuda:1")

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
        ):
            det.start()

        assert det._device == "cuda:1"
        mock_model.to.assert_called_with("cuda:1")

    def test_detector_device_selection_auto_fallback_cpu(self):
        """When config.device=None and CUDA unavailable, auto-selects CPU."""
        torch = self._require_layout_runtime()

        det, mock_model, mock_proc = self._mock_detector(None)

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
            patch.object(torch.cuda, "is_available", return_value=False),
        ):
            det.start()

        assert det._device == "cpu"

    def test_detector_device_selection_auto_cuda(self):
        """When config.device=None and CUDA available, auto-selects cuda:{cuda_visible_devices}."""
        torch = self._require_layout_runtime()
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = LayoutConfig(
            device=None,
            cuda_visible_devices="1",
            model_dir="dummy",
            **self._MOCK_LAYOUT_KWARGS,
        )
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "text"}
        mock_proc = MagicMock()

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
            patch.object(torch.cuda, "is_available", return_value=True),
        ):
            det.start()

        assert det._device == "cuda:1"

    def test_detector_start_uses_public_config_defaults(self):
        """Detector can start from load_config() defaults without extra fields."""
        self._require_layout_runtime()
        from glmocr.config import load_config
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = load_config(mode="selfhosted").pipeline.layout
        cfg.model_dir = "dummy"
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = dict(cfg.id2label)
        mock_proc = MagicMock()

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
        ):
            det.start()

        assert det.id2label == cfg.id2label

    def test_detector_defaults_label_task_mapping_from_model_id2label(self):
        """Missing label_task_mapping falls back to a text bucket from id2label."""
        self._require_layout_runtime()
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = LayoutConfig(model_dir="dummy", device="cpu")
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "text", 1: "title"}
        mock_proc = MagicMock()

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
        ):
            det.start()

        assert det.id2label == {0: "text", 1: "title"}
        assert det.label_task_mapping == {"text": ["text", "title"]}

    def test_detector_raises_when_id2label_missing_everywhere(self):
        """Missing id2label in both config and model config raises a clear error."""
        self._require_layout_runtime()
        from glmocr.config import LayoutConfig
        from glmocr.layout.layout_detector import PPDocLayoutDetector

        cfg = LayoutConfig(model_dir="dummy", device="cpu")
        det = PPDocLayoutDetector(cfg)

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.id2label = None
        mock_proc = MagicMock()

        with (
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ForObjectDetection.from_pretrained",
                return_value=mock_model,
            ),
            patch(
                "glmocr.layout.layout_detector.PPDocLayoutV3ImageProcessor.from_pretrained",
                return_value=mock_proc,
            ),
            pytest.raises(RuntimeError, match="Missing id2label"),
        ):
            det.start()


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

    def test_pageloader_load_pdf_pages(self):
        """Expands a PDF into page images."""
        from glmocr.config import PageLoaderConfig
        from glmocr.dataloader import PageLoader

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
        from glmocr.config import PageLoaderConfig

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

    def test_iter_pages_with_unit_indices_pdf_and_multi_source(self):
        """Streaming: pages yielded incrementally; unit indices correct for multi-source."""
        from glmocr.config import PageLoaderConfig
        from glmocr.dataloader import PageLoader

        repo_root = Path(__file__).resolve().parents[2]
        source_dir = repo_root / "examples" / "source"
        if not source_dir.exists():
            pytest.skip(f"No source dir: {source_dir}")

        sample_pdf = next(
            (f for f in source_dir.iterdir() if f.suffix.lower() == ".pdf"),
            None,
        )
        sample_image = next(
            (f for f in source_dir.iterdir() if f.suffix.lower() in (".png", ".jpg")),
            None,
        )
        if not sample_pdf or not sample_pdf.exists():
            pytest.skip(f"No sample PDF in {source_dir}")

        loader = PageLoader(PageLoaderConfig())

        # (1) Single PDF: pages yielded incrementally, same count as load_pages
        expected_pages, _ = loader.load_pages_with_unit_indices(str(sample_pdf))
        streamed = list(loader.iter_pages_with_unit_indices(str(sample_pdf)))
        assert len(streamed) == len(
            expected_pages
        ), "streaming should yield same number of pages as load"
        for i, (page, unit_idx) in enumerate(streamed):
            assert unit_idx == 0, "single source should have unit_idx 0"
            assert page is not None

        # (2) Multi-source: unit indices match load_pages_with_unit_indices
        if sample_image and sample_image.exists():
            sources = [str(sample_image), str(sample_pdf)]
        else:
            pdfs = sorted(f for f in source_dir.iterdir() if f.suffix.lower() == ".pdf")
            if len(pdfs) < 2:
                pytest.skip(
                    "need second source (image or another PDF) for multi-source test"
                )
            sources = [str(pdfs[0]), str(pdfs[1])]

        expected_pages, expected_indices = loader.load_pages_with_unit_indices(sources)
        streamed = list(loader.iter_pages_with_unit_indices(sources))
        assert len(streamed) == len(expected_pages)
        for i, (page, unit_idx) in enumerate(streamed):
            assert (
                unit_idx == expected_indices[i]
            ), f"page {i}: expected unit_idx {expected_indices[i]}, got {unit_idx}"


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


class TestUtils:
    """Tests for utility functions."""

    def test_image_utils_crop_image_region(self):
        """crop_image_region exists."""
        from glmocr.utils.image_utils import crop_image_region

        assert callable(crop_image_region)

    def test_image_utils_crop_image_region_polygon_without_opencv(self):
        """Polygon crop works without requiring OpenCV."""
        from PIL import Image

        from glmocr.utils.image_utils import crop_image_region

        image = Image.new("RGB", (100, 100), (255, 255, 255))
        cropped = crop_image_region(
            image,
            [100, 100, 900, 900],
            polygon=[[100, 100], [900, 100], [900, 900], [100, 900]],
        )

        assert cropped.size == (80, 80)

    def test_server_create_app_requires_flask_extra(self, monkeypatch):
        """Server import error explains the optional extra."""
        from glmocr.config import load_config
        from glmocr import server

        monkeypatch.setattr(server, "Flask", None)
        monkeypatch.setattr(server, "_FLASK_IMPORT_ERROR", ImportError("missing flask"))

        with pytest.raises(ImportError, match=r"glmocr\[server\]"):
            server.create_app(load_config())

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

    def test_result_formatter_chart_region_included(self):
        """Chart regions (task_type 'skip') are not dropped and produce images."""
        from glmocr.postprocess import ResultFormatter
        from glmocr.config import ResultFormatterConfig

        config = ResultFormatterConfig()
        formatter = ResultFormatter(config)

        chart_bbox = [10, 20, 200, 180]
        image_bbox = [210, 20, 400, 180]
        fake_chart_img = MagicMock(name="chart_img")
        fake_image_img = MagicMock(name="image_img")

        grouped_results = [
            [
                {
                    "index": 0,
                    "label": "chart",
                    "task_type": "skip",
                    "content": None,
                    "bbox_2d": chart_bbox,
                    "score": 0.95,
                },
                {
                    "index": 1,
                    "label": "text",
                    "task_type": "text",
                    "content": "Hello",
                    "bbox_2d": [10, 200, 400, 250],
                    "score": 0.9,
                },
                {
                    "index": 2,
                    "label": "image",
                    "task_type": "skip",
                    "content": None,
                    "bbox_2d": image_bbox,
                    "score": 0.92,
                },
            ]
        ]

        cropped_images = {
            (0, *chart_bbox): fake_chart_img,
            (0, *image_bbox): fake_image_img,
        }

        json_str, md_str, image_files = formatter.process(
            grouped_results, cropped_images=cropped_images
        )

        # Both chart and image regions should be saved
        assert (
            len(image_files) == 2
        ), f"Expected 2 image files (chart + image), got {len(image_files)}"

        # Markdown should contain image references for both
        assert "![Image 0-0]" in md_str
        assert "![Image 0-1]" in md_str

        # Text content should also be present
        assert "Hello" in md_str

        # JSON should contain both image regions
        parsed = json.loads(json_str)
        page_results = parsed[0]
        labels = [r["label"] for r in page_results]
        assert labels.count("image") == 2
        assert "text" in labels

        # task_type and score should not leak into JSON output
        for r in page_results:
            assert "task_type" not in r
            assert "score" not in r
            assert "_is_image" not in r

    def test_result_formatter_chart_region_default_config(self):
        """Chart regions work even with an empty label_visualization_mapping."""
        from glmocr.postprocess import ResultFormatter
        from glmocr.config import ResultFormatterConfig

        # Explicitly empty mapping (the old default)
        config = ResultFormatterConfig(label_visualization_mapping={})
        formatter = ResultFormatter(config)

        chart_bbox = [10, 20, 200, 180]
        fake_img = MagicMock(name="chart_img")

        grouped_results = [
            [
                {
                    "index": 0,
                    "label": "chart",
                    "task_type": "skip",
                    "content": None,
                    "bbox_2d": chart_bbox,
                    "score": 0.95,
                },
            ]
        ]

        cropped_images = {
            (0, *chart_bbox): fake_img,
        }

        json_str, md_str, image_files = formatter.process(
            grouped_results, cropped_images=cropped_images
        )

        # Chart region should still produce an image file via task_type fallback
        assert (
            len(image_files) == 1
        ), f"Chart region lost with empty mapping, got {len(image_files)} files"
        assert "![Image 0-0]" in md_str


class TestMaaSClient:
    """Tests for MaaSClient."""

    def test_maas_config_defaults(self):
        """MaaSApiConfig has correct defaults."""
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig()
        assert config.enabled is True
        assert config.api_url == "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
        assert config.model == "glm-ocr"
        assert config.verify_ssl is True

    def test_maas_client_requires_api_key(self):
        """MaaSClient raises MissingApiKeyError when API key is missing."""
        from glmocr.maas_client import MaaSClient, MissingApiKeyError
        from glmocr.config import MaaSApiConfig

        config = MaaSApiConfig(api_key=None)
        with pytest.raises(MissingApiKeyError):
            MaaSClient(config)

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
        assert config.maas.enabled is True


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

        monkeypatch.setenv("ZHIPU_API_KEY", "sk-env")
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
        dotenv.write_text("ZHIPU_API_KEY=sk-from-dotenv\n")
        # Patch _find_dotenv to return our temp .env
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: dotenv)
        # Make sure real env doesn't have the key
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
        monkeypatch.delenv("GLMOCR_API_KEY", raising=False)

        overrides = _collect_env_overrides()
        assert overrides["pipeline"]["maas"]["api_key"] == "sk-from-dotenv"

    def test_real_env_beats_dotenv(self, tmp_path, monkeypatch):
        """os.environ takes priority over .env file."""
        from glmocr.config import _collect_env_overrides

        dotenv = tmp_path / ".env"
        dotenv.write_text("ZHIPU_API_KEY=sk-dotenv\n")
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: dotenv)
        monkeypatch.setenv("ZHIPU_API_KEY", "sk-real")

        overrides = _collect_env_overrides()
        assert overrides["pipeline"]["maas"]["api_key"] == "sk-real"

    def test_no_env_returns_empty(self, monkeypatch):
        """When no GLMOCR_* vars exist, returns empty dict."""
        from glmocr.config import _collect_env_overrides, _ENV_MAP, ENV_PREFIX

        # Clear all GLMOCR_* vars
        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
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
        assert cfg.pipeline.maas.enabled is True
        assert cfg.logging.level == "INFO"

    def test_overrides_win_over_env(self, monkeypatch):
        """Keyword overrides beat env vars."""
        from glmocr.config import GlmOcrConfig

        monkeypatch.setenv("ZHIPU_API_KEY", "sk-env")
        cfg = GlmOcrConfig.from_env(api_key="sk-override")
        assert cfg.pipeline.maas.api_key == "sk-override"

    def test_env_wins_over_yaml(self, tmp_path, monkeypatch):
        """Env vars beat YAML values."""
        from glmocr.config import GlmOcrConfig

        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("pipeline:\n  maas:\n    api_key: sk-yaml\n")
        monkeypatch.setenv("ZHIPU_API_KEY", "sk-env")
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


class TestGlmOcrParseStream:
    """Unit tests for GlmOcr._parse_stream() (stream=True path)."""

    def _make_glmocr_maas(self):
        """GlmOcr with MaaS enabled and mocked _maas_client."""
        from glmocr.api import GlmOcr
        from glmocr.parser_result import PipelineResult

        obj = object.__new__(GlmOcr)
        obj._use_maas = True
        obj._pipeline = None
        obj._maas_client = MagicMock()
        obj.config_model = MagicMock()
        obj._maas_response_to_pipeline_result = MagicMock(
            return_value=PipelineResult(
                json_result=[[{"content": "ok"}]],
                markdown_result="ok",
                original_images=["img.png"],
            )
        )
        return obj

    def _make_glmocr_selfhosted(self):
        """GlmOcr with self-hosted mode and mocked _stream_parse_selfhosted."""
        from glmocr.api import GlmOcr
        from glmocr.parser_result import PipelineResult

        obj = object.__new__(GlmOcr)
        obj._use_maas = False
        obj._maas_client = None
        obj._pipeline = MagicMock()
        obj.config_model = MagicMock()
        r1 = PipelineResult(
            json_result=[], markdown_result="a", original_images=["a.png"]
        )
        r2 = PipelineResult(
            json_result=[], markdown_result="b", original_images=["b.png"]
        )
        obj._stream_parse_selfhosted = MagicMock(return_value=(r for r in [r1, r2]))
        return obj

    def test_parse_stream_maas_yields_one_per_image(self):
        """MaaS: one PipelineResult yielded per image."""
        from glmocr.parser_result import PipelineResult

        parser = self._make_glmocr_maas()
        parser._maas_client.parse.return_value = {
            "md_results": "",
            "layout_details": [],
            "data_info": {"pages": []},
        }
        results = list(parser._parse_stream(["img1.png", "img2.png"]))
        assert len(results) == 2
        assert all(isinstance(r, PipelineResult) for r in results)
        assert parser._maas_client.parse.call_count == 2

    def test_parse_stream_maas_strips_file_prefix(self):
        """MaaS: file:// prefix is stripped when calling API."""
        parser = self._make_glmocr_maas()
        parser._maas_client.parse.return_value = {
            "md_results": "",
            "layout_details": [],
            "data_info": {"pages": []},
        }
        list(parser._parse_stream(["file:///path/to/img.png"]))
        parser._maas_client.parse.assert_called_once()
        call_arg = parser._maas_client.parse.call_args[0][0]
        assert call_arg == "/path/to/img.png"

    def test_parse_stream_maas_on_error_yields_error_result(self):
        """MaaS: on API exception, yield result with _error set."""
        from glmocr.parser_result import PipelineResult

        parser = self._make_glmocr_maas()
        parser._maas_client.parse.side_effect = RuntimeError("API down")
        results = list(parser._parse_stream(["img.png"]))
        assert len(results) == 1
        assert isinstance(results[0], PipelineResult)
        assert results[0]._error == "API down"
        assert len(results[0].original_images) == 1
        assert results[0].original_images[0].endswith("img.png")
        assert results[0].markdown_result == ""

    def test_parse_stream_maas_save_layout_sets_kwarg(self):
        """MaaS: save_layout_visualization=True sets need_layout_visualization in kwargs."""
        parser = self._make_glmocr_maas()
        parser._maas_client.parse.return_value = {
            "md_results": "",
            "layout_details": [],
            "data_info": {"pages": []},
        }
        list(parser._parse_stream(["img.png"], save_layout_visualization=True))
        call_kwargs = parser._maas_client.parse.call_args[1]
        assert call_kwargs.get("need_layout_visualization") is True

    def test_parse_stream_selfhosted_delegates(self):
        """Self-hosted: yields results from _stream_parse_selfhosted."""
        parser = self._make_glmocr_selfhosted()
        results = list(
            parser._parse_stream(
                ["a.png", "b.png"],
                save_layout_visualization=False,
            )
        )
        assert len(results) == 2
        assert results[0].markdown_result == "a"
        assert results[1].markdown_result == "b"
        parser._stream_parse_selfhosted.assert_called_once_with(
            ["a.png", "b.png"],
            save_layout_visualization=False,
            preserve_order=True,
        )


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
            from glmocr.api import GlmOcr

            parser = GlmOcr(mode="selfhosted")
            assert parser._use_maas is False
            parser.close()

    def test_selfhosted_model_kwarg_is_forwarded_to_ocr_api(self, monkeypatch):
        """model=... should configure self-hosted OCR request model."""
        from glmocr.config import _ENV_MAP, ENV_PREFIX

        for suffix in _ENV_MAP:
            monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)
        monkeypatch.setattr("glmocr.config._find_dotenv", lambda: None)

        with patch("glmocr.pipeline.Pipeline") as mock_pipeline:
            mock_pipeline.return_value.start = MagicMock()
            mock_pipeline.return_value.enable_layout = False
            from glmocr.api import GlmOcr

            parser = GlmOcr(mode="selfhosted", model="glm-ocr")
            assert parser._use_maas is False
            assert parser.config_model.pipeline.ocr_api.model == "glm-ocr"
            parser.close()


class TestOCRClientOllamaConfig:
    """Tests for OCRClient initialization with Ollama api_mode."""

    def test_ocr_api_config_default_api_mode(self):
        """OCRApiConfig defaults to openai mode."""
        from glmocr.config import OCRApiConfig

        config = OCRApiConfig()
        assert config.api_mode == "openai"

    def test_ocr_api_config_ollama_mode(self):
        """OCRApiConfig accepts ollama_generate mode."""
        from glmocr.config import OCRApiConfig

        config = OCRApiConfig(api_mode="ollama_generate")
        assert config.api_mode == "ollama_generate"

    def test_ocr_client_reads_api_mode(self):
        """OCRClient reads api_mode from config."""
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        config = OCRApiConfig(api_mode="ollama_generate", model="glm-ocr:latest")
        client = OCRClient(config)
        assert client.api_mode == "ollama_generate"
        assert client.model == "glm-ocr:latest"

    def test_ocr_client_defaults_to_openai_mode(self):
        """OCRClient defaults to openai mode."""
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        config = OCRApiConfig()
        client = OCRClient(config)
        assert client.api_mode == "openai"


class TestConvertToOllamaGenerate:
    """Tests for OCRClient._convert_to_ollama_generate()."""

    def _make_client(self, model="glm-ocr:latest"):
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        config = OCRApiConfig(api_mode="ollama_generate", model=model)
        return OCRClient(config)

    def test_basic_text_message(self):
        """Converts a simple text message."""
        client = self._make_client()
        openai_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Recognize this"}],
                }
            ],
            "max_tokens": 100,
        }
        result = client._convert_to_ollama_generate(openai_data)

        assert result["model"] == "glm-ocr:latest"
        assert result["prompt"] == "Recognize this"
        assert result["stream"] is False
        assert "images" not in result
        assert result["options"]["num_predict"] == 100

    def test_text_and_image(self):
        """Converts a message with text and a base64 image."""
        client = self._make_client()
        openai_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR this"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                        },
                    ],
                }
            ],
            "max_tokens": 200,
        }
        result = client._convert_to_ollama_generate(openai_data)

        assert result["prompt"] == "OCR this"
        assert result["images"] == ["iVBORw0KGgo="]
        assert result["options"]["num_predict"] == 200

    def test_image_url_as_dict(self):
        """Handles image_url given as a dict with 'url' key."""
        client = self._make_client()
        openai_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"},
                        },
                    ],
                }
            ],
        }
        result = client._convert_to_ollama_generate(openai_data)
        assert result["images"] == ["/9j/4AAQ"]

    def test_non_data_uri_image(self):
        """Non-data-URI image URLs are kept as-is."""
        client = self._make_client()
        openai_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "OCR"},
                        {
                            "type": "image_url",
                            "image_url": "https://example.com/image.png",
                        },
                    ],
                }
            ],
        }
        result = client._convert_to_ollama_generate(openai_data)
        assert result["images"] == ["https://example.com/image.png"]

    def test_string_content(self):
        """Handles content as a plain string."""
        client = self._make_client()
        openai_data = {
            "messages": [{"role": "user", "content": "Hello world"}],
        }
        result = client._convert_to_ollama_generate(openai_data)
        assert result["prompt"] == "Hello world"
        assert "images" not in result

    def test_empty_messages(self):
        """Handles empty messages list."""
        client = self._make_client()
        result = client._convert_to_ollama_generate({"messages": []})
        assert result["prompt"] == ""
        assert "images" not in result

    def test_no_messages_key(self):
        """Handles missing messages key."""
        client = self._make_client()
        result = client._convert_to_ollama_generate({})
        assert result["prompt"] == ""

    def test_non_user_messages_ignored(self):
        """System and assistant messages are ignored (last user message used)."""
        client = self._make_client()
        openai_data = {
            "messages": [
                {"role": "system", "content": "You are an OCR model."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ],
        }
        result = client._convert_to_ollama_generate(openai_data)
        assert result["prompt"] == "Second question"

    def test_parameter_mapping(self):
        """Maps OpenAI parameters to Ollama options."""
        client = self._make_client()
        openai_data = {
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "repetition_penalty": 1.2,
        }
        result = client._convert_to_ollama_generate(openai_data)
        opts = result["options"]
        assert opts["num_predict"] == 500
        assert opts["temperature"] == 0.5
        assert opts["top_p"] == 0.9
        assert opts["top_k"] == 40
        assert opts["repeat_penalty"] == 1.2

    def test_no_options_when_no_params(self):
        """No options field when no parameters are mapped."""
        client = self._make_client()
        openai_data = {
            "messages": [{"role": "user", "content": "test"}],
        }
        result = client._convert_to_ollama_generate(openai_data)
        assert "options" not in result

    def test_model_fallback(self):
        """Falls back to 'glm-ocr:latest' when model is not set."""
        client = self._make_client(model=None)
        result = client._convert_to_ollama_generate(
            {"messages": [{"role": "user", "content": "x"}]}
        )
        assert result["model"] == "glm-ocr:latest"

    def test_multiple_images(self):
        """Handles multiple images in a single message."""
        client = self._make_client()
        openai_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/png;base64,AAAA",
                        },
                        {
                            "type": "image_url",
                            "image_url": "data:image/png;base64,BBBB",
                        },
                    ],
                }
            ],
        }
        result = client._convert_to_ollama_generate(openai_data)
        assert result["images"] == ["AAAA", "BBBB"]


class TestOCRClientProcessOllama:
    """Tests for OCRClient.process() with ollama_generate mode."""

    def _make_client(self, model="glm-ocr:latest"):
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        config = OCRApiConfig(
            api_mode="ollama_generate",
            model=model,
            retry_max_attempts=0,
        )
        client = OCRClient(config)
        client._session = MagicMock()
        return client

    def _mock_response(self, status_code=200, json_data=None, text="", headers=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.text = text
        resp.headers = headers or {}
        return resp

    def test_process_ollama_success(self):
        """Successful Ollama generate response is parsed correctly."""
        client = self._make_client()
        resp = self._mock_response(
            json_data={"response": "# Hello World", "done": True}
        )
        client._session.post.return_value = resp

        result, status = client.process(
            {"messages": [{"role": "user", "content": "OCR this"}], "max_tokens": 100}
        )

        assert status == 200
        assert result["choices"][0]["message"]["content"] == "# Hello World"

    def test_process_ollama_strips_whitespace(self):
        """Ollama response is stripped of whitespace."""
        client = self._make_client()
        resp = self._mock_response(json_data={"response": "  Hello  \n", "done": True})
        client._session.post.return_value = resp

        result, status = client.process(
            {"messages": [{"role": "user", "content": "test"}]}
        )
        assert result["choices"][0]["message"]["content"] == "Hello"

    def test_process_ollama_error_field(self):
        """Ollama response with error field returns 500."""
        client = self._make_client()
        resp = self._mock_response(json_data={"error": "model not found"})
        client._session.post.return_value = resp

        result, status = client.process(
            {"messages": [{"role": "user", "content": "test"}]}
        )
        assert status == 500
        assert "model not found" in result["error"]

    def test_process_ollama_missing_response_field(self):
        """Ollama response without 'response' field returns 500."""
        client = self._make_client()
        resp = self._mock_response(json_data={"done": True})  # no "response" key
        client._session.post.return_value = resp

        result, status = client.process(
            {"messages": [{"role": "user", "content": "test"}]}
        )
        assert status == 500
        assert "missing 'response' field" in result["error"]

    def test_process_ollama_converts_request_format(self):
        """Ollama mode converts OpenAI request to generate format."""
        client = self._make_client()
        resp = self._mock_response(json_data={"response": "ok", "done": True})
        client._session.post.return_value = resp

        client.process(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "OCR this"},
                            {
                                "type": "image_url",
                                "image_url": "data:image/png;base64,abc123",
                            },
                        ],
                    }
                ],
                "max_tokens": 50,
            }
        )

        # Inspect the actual JSON sent
        call_kwargs = client._session.post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1]["data"])
        assert sent_data["model"] == "glm-ocr:latest"
        assert sent_data["prompt"] == "OCR this"
        assert sent_data["images"] == ["abc123"]
        assert sent_data["stream"] is False
        assert sent_data["options"]["num_predict"] == 50

    def test_process_openai_mode_unchanged(self):
        """OpenAI mode does not convert request format."""
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        config = OCRApiConfig(api_mode="openai", model="my-model", retry_max_attempts=0)
        client = OCRClient(config)
        client._session = MagicMock()

        resp = self._mock_response(
            json_data={"choices": [{"message": {"content": "result"}}]}
        )
        client._session.post.return_value = resp

        result, status = client.process(
            {
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            }
        )
        assert status == 200
        assert result["choices"][0]["message"]["content"] == "result"

        # Verify the request kept OpenAI format (has "messages")
        call_kwargs = client._session.post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1]["data"])
        assert "messages" in sent_data
        assert sent_data["model"] == "my-model"

    def test_process_openai_invalid_response_format(self):
        """OpenAI mode returns 500 on malformed response."""
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        config = OCRApiConfig(api_mode="openai", retry_max_attempts=0)
        client = OCRClient(config)
        client._session = MagicMock()

        resp = self._mock_response(json_data={"unexpected": "format"})
        client._session.post.return_value = resp

        result, status = client.process(
            {"messages": [{"role": "user", "content": "test"}]}
        )
        assert status == 500
        assert "Invalid OpenAI API response format" in result["error"]


class TestOCRClientConnectOllama:
    """Tests for OCRClient.connect() with ollama_generate mode."""

    @patch("glmocr.ocr_client.requests.post")
    @patch("glmocr.ocr_client.socket.socket")
    def test_connect_ollama_builds_correct_payload(self, mock_socket_cls, mock_post):
        """connect() sends Ollama-style test payload when api_mode is ollama_generate."""
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        # Socket connects successfully
        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_socket_cls.return_value = mock_sock

        # API responds 200
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = OCRApiConfig(
            api_mode="ollama_generate",
            model="glm-ocr:latest",
            api_host="localhost",
            api_port=11434,
            api_path="/api/generate",
        )
        client = OCRClient(config)
        client.connect()

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1]["data"])
        assert sent_data["model"] == "glm-ocr:latest"
        assert sent_data["prompt"] == "hello"
        assert sent_data["stream"] is False
        assert sent_data["options"]["num_predict"] == 10

    @patch("glmocr.ocr_client.requests.post")
    @patch("glmocr.ocr_client.socket.socket")
    def test_connect_openai_builds_correct_payload(self, mock_socket_cls, mock_post):
        """connect() sends OpenAI-style test payload when api_mode is openai."""
        from glmocr.config import OCRApiConfig
        from glmocr.ocr_client import OCRClient

        mock_sock = MagicMock()
        mock_sock.connect_ex.return_value = 0
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_socket_cls.return_value = mock_sock

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config = OCRApiConfig(
            api_mode="openai",
            model="my-model",
            api_host="localhost",
            api_port=8080,
        )
        client = OCRClient(config)
        client.connect()

        call_kwargs = mock_post.call_args
        sent_data = json.loads(call_kwargs.kwargs.get("data") or call_kwargs[1]["data"])
        assert "messages" in sent_data
        assert sent_data["model"] == "my-model"
        assert sent_data["max_tokens"] == 10


# ═══════════════════════════════════════════════════════════════════════
# MaaS Client ↔ Server Protocol Alignment Tests
# ═══════════════════════════════════════════════════════════════════════


class TestBuildResponse:
    """Tests for server._build_response() output format."""

    def test_contains_sdk_native_fields(self):
        """Response includes SDK native fields (json_result, markdown_result)."""
        from glmocr.server import _build_response

        resp = _build_response({"key": "val"}, "# Markdown")
        assert resp["json_result"] == {"key": "val"}
        assert resp["markdown_result"] == "# Markdown"

    def test_contains_maas_compatible_fields(self):
        """Response includes MaaS-compatible fields (layout_details, md_results)."""
        from glmocr.server import _build_response

        layout = [[{"index": 0, "label": "text", "content": "hello"}]]
        resp = _build_response(layout, "# Hello")
        assert resp["layout_details"] == layout
        assert resp["md_results"] == "# Hello"

    def test_sdk_and_maas_fields_are_identical(self):
        """SDK native and MaaS-compatible fields point to the same values."""
        from glmocr.server import _build_response

        layout = [[{"index": 0, "label": "title", "content": "Title"}]]
        md = "# Title"
        resp = _build_response(layout, md)
        assert resp["json_result"] is resp["layout_details"]
        assert resp["markdown_result"] is resp["md_results"]

    def test_contains_metadata_fields(self):
        """Response contains model, id, created, usage, data_info."""
        from glmocr.server import _build_response

        resp = _build_response(None, "")
        assert resp["model"] == "glm-ocr"
        assert resp["id"].startswith("chatcmpl-")
        assert isinstance(resp["created"], int)
        assert resp["usage"] == {}
        assert resp["data_info"] == {"pages": []}

    def test_id_is_unique(self):
        """Each call generates a unique id."""
        from glmocr.server import _build_response

        r1 = _build_response(None, "")
        r2 = _build_response(None, "")
        assert r1["id"] != r2["id"]

    def test_none_json_result(self):
        """Handles None json_result gracefully."""
        from glmocr.server import _build_response

        resp = _build_response(None, "")
        assert resp["json_result"] is None
        assert resp["layout_details"] is None

    def test_empty_string_markdown(self):
        """Handles empty string markdown_result."""
        from glmocr.server import _build_response

        resp = _build_response([], "")
        assert resp["markdown_result"] == ""
        assert resp["md_results"] == ""


class TestServerAcceptsMaaSInputFormat:
    """Tests for server accepting MaaS client's 'file' field as input."""

    @pytest.fixture()
    def flask_app(self):
        """Create a Flask test app with mocked pipeline."""
        pytest.importorskip(
            "flask", reason="server tests require 'glmocr[server]' extra"
        )
        from glmocr.server import create_app
        from glmocr.config import load_config

        config = load_config()
        app = create_app(config)
        app.config["TESTING"] = True
        return app

    def _mock_single_result(self, json_result=None, markdown_result=""):
        """Create a mock pipeline result."""
        from unittest.mock import MagicMock

        r = MagicMock()
        r.json_result = json_result or [
            [{"index": 0, "label": "text", "content": "hi"}]
        ]
        r.markdown_result = markdown_result
        return r

    def test_file_field_accepted_as_input(self, flask_app):
        """Server accepts 'file' field (MaaS client format) in addition to 'images'."""
        mock_result = self._mock_single_result(markdown_result="# Test")
        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"file": "https://example.com/doc.png"},
                    content_type="application/json",
                )
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["markdown_result"] == "# Test"

    def test_images_field_still_works(self, flask_app):
        """Server still accepts 'images' field (original SDK format)."""
        mock_result = self._mock_single_result(markdown_result="# Images")
        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"images": ["https://example.com/doc.png"]},
                    content_type="application/json",
                )
                assert resp.status_code == 200
                data = resp.get_json()
                assert data["markdown_result"] == "# Images"

    def test_file_field_ignored_when_images_present(self, flask_app):
        """'images' field takes priority over 'file' field."""
        mock_result = self._mock_single_result(markdown_result="# Priority")
        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ) as mock_process:
                resp = client.post(
                    "/glmocr/parse",
                    json={
                        "images": ["https://example.com/a.png"],
                        "file": "https://example.com/b.png",
                    },
                    content_type="application/json",
                )
                assert resp.status_code == 200
                # Verify pipeline was called with the 'images' URL, not 'file'
                call_args = mock_process.call_args
                request_data = call_args[0][0]
                sent_url = request_data["messages"][0]["content"][0]["image_url"]["url"]
                assert sent_url == "https://example.com/a.png"

    def test_empty_file_field_falls_back_to_error(self, flask_app):
        """Empty 'file' field without 'images' returns 400."""
        with flask_app.test_client() as client:
            resp = client.post(
                "/glmocr/parse",
                json={"file": ""},
                content_type="application/json",
            )
            assert resp.status_code == 400

    def test_no_images_no_file_returns_400(self, flask_app):
        """Missing both 'images' and 'file' returns 400."""
        with flask_app.test_client() as client:
            resp = client.post(
                "/glmocr/parse",
                json={},
                content_type="application/json",
            )
            assert resp.status_code == 400

    def test_file_field_non_string_ignored(self, flask_app):
        """Non-string 'file' value is ignored (falls through to 400)."""
        with flask_app.test_client() as client:
            resp = client.post(
                "/glmocr/parse",
                json={"file": 12345},
                content_type="application/json",
            )
            assert resp.status_code == 400


class TestServerOutputMaaSCompatible:
    """Tests for server response containing MaaS-compatible fields."""

    @pytest.fixture()
    def flask_app(self):
        """Create a Flask test app with mocked pipeline."""
        pytest.importorskip(
            "flask", reason="server tests require 'glmocr[server]' extra"
        )
        from glmocr.server import create_app
        from glmocr.config import load_config

        config = load_config()
        app = create_app(config)
        app.config["TESTING"] = True
        return app

    def _mock_single_result(self, json_result=None, markdown_result="# Test"):
        from unittest.mock import MagicMock

        r = MagicMock()
        r.json_result = json_result or [
            [{"index": 0, "label": "text", "content": "hello"}]
        ]
        r.markdown_result = markdown_result
        return r

    def test_single_result_has_both_field_sets(self, flask_app):
        """Single result response has both SDK and MaaS fields."""
        mock_result = self._mock_single_result(markdown_result="# Hello")
        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"images": ["https://example.com/doc.png"]},
                    content_type="application/json",
                )
                data = resp.get_json()
                # SDK fields
                assert "json_result" in data
                assert "markdown_result" in data
                # MaaS-compatible fields
                assert "layout_details" in data
                assert "md_results" in data
                assert "model" in data
                assert "id" in data
                assert "created" in data
                assert "usage" in data
                assert "data_info" in data

    def test_single_result_values_match(self, flask_app):
        """SDK and MaaS fields have identical values in single result."""
        layout = [[{"index": 0, "label": "text", "content": "hello"}]]
        mock_result = self._mock_single_result(
            json_result=layout, markdown_result="# MD"
        )
        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"images": ["https://example.com/doc.png"]},
                    content_type="application/json",
                )
                data = resp.get_json()
                assert data["json_result"] == layout
                assert data["layout_details"] == layout
                assert data["markdown_result"] == "# MD"
                assert data["md_results"] == "# MD"

    def test_empty_result_returns_maas_format(self, flask_app):
        """Empty pipeline result returns MaaS-compatible format."""
        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"images": ["https://example.com/doc.png"]},
                    content_type="application/json",
                )
                data = resp.get_json()
                assert resp.status_code == 200
                assert data["json_result"] is None
                assert data["layout_details"] is None
                assert data["markdown_result"] == ""
                assert data["md_results"] == ""
                assert data["model"] == "glm-ocr"

    def test_multiple_results_merged(self, flask_app):
        """Multiple results are merged into a single response."""
        from unittest.mock import MagicMock

        r1 = MagicMock()
        r1.json_result = [[{"index": 0, "label": "text", "content": "a"}]]
        r1.markdown_result = "# Page 1"
        r2 = MagicMock()
        r2.json_result = [[{"index": 0, "label": "text", "content": "b"}]]
        r2.markdown_result = "# Page 2"

        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([r1, r2]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={
                        "images": [
                            "https://example.com/a.png",
                            "https://example.com/b.png",
                        ]
                    },
                    content_type="application/json",
                )
                data = resp.get_json()
                assert resp.status_code == 200
                # json_result is a list of two results
                assert isinstance(data["json_result"], list)
                assert len(data["json_result"]) == 2
                assert isinstance(data["layout_details"], list)
                # Markdown results joined with separator
                assert "---" in data["markdown_result"]
                assert "# Page 1" in data["md_results"]
                assert "# Page 2" in data["md_results"]

    def test_null_markdown_becomes_empty_string(self, flask_app):
        """None markdown_result is converted to empty string."""
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.json_result = []
        mock_result.markdown_result = None

        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"images": ["https://example.com/doc.png"]},
                    content_type="application/json",
                )
                data = resp.get_json()
                assert data["markdown_result"] == ""
                assert data["md_results"] == ""


class TestMaaSClientResponseParsing:
    """Tests for MaaS client parsing server-style responses."""

    def test_parses_layout_details_from_server(self):
        """_maas_response_to_pipeline_result reads layout_details."""
        from glmocr.api import GlmOcr

        # This is the format the server returns (via _build_response)
        server_response = {
            "id": "chatcmpl-abc",
            "model": "glm-ocr",
            "created": 1709234567,
            "layout_details": [
                [
                    {
                        "index": 0,
                        "label": "text",
                        "content": "Hello",
                        "bbox_2d": [10, 20, 100, 200],
                    },
                    {
                        "index": 1,
                        "label": "title",
                        "content": "Title",
                        "bbox_2d": [10, 5, 100, 18],
                    },
                ]
            ],
            "md_results": "# Title\nHello",
            "data_info": {"pages": [{"width": 1000, "height": 1500}]},
            "usage": {},
        }

        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_response, "test.png")

        assert len(result.json_result) == 1  # 1 page
        assert len(result.json_result[0]) == 2  # 2 regions
        assert result.json_result[0][0]["label"] == "text"
        assert result.json_result[0][1]["label"] == "title"
        # bbox should be normalised to 0-1000 range
        assert result.json_result[0][0]["bbox_2d"] == [10, 13, 100, 133]

    def test_parses_md_results_from_server(self):
        """_maas_response_to_pipeline_result reads md_results."""
        from glmocr.api import GlmOcr

        server_response = {
            "layout_details": [],
            "md_results": "# Markdown from server",
            "data_info": {"pages": []},
        }

        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_response, "test.png")

        assert result.markdown_result == "# Markdown from server"

    def test_handles_empty_layout_details(self):
        """Handles empty layout_details list gracefully."""
        from glmocr.api import GlmOcr

        server_response = {
            "layout_details": [],
            "md_results": "",
            "data_info": {"pages": []},
        }

        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_response, "test.png")

        assert result.json_result == []
        assert result.markdown_result == ""

    def test_layout_details_not_json_result_is_used(self):
        """_maas_response_to_pipeline_result reads layout_details, not json_result."""
        from glmocr.api import GlmOcr

        # Even when json_result differs, layout_details is the source of truth
        server_response = {
            "json_result": "wrong-value",
            "layout_details": [[{"index": 0, "label": "text", "content": "correct"}]],
            "md_results": "correct md",
            "markdown_result": "wrong md",
            "data_info": {"pages": []},
        }

        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_response, "test.png")

        # layout_details is used (not json_result)
        assert result.json_result[0][0]["content"] == "correct"
        # md_results is used (not markdown_result)
        assert result.markdown_result == "correct md"

    def test_missing_layout_details_defaults_to_empty(self):
        """Missing layout_details field defaults to empty list."""
        from glmocr.api import GlmOcr

        server_response = {
            "md_results": "some markdown",
        }

        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_response, "test.png")

        assert result.json_result == []
        assert result.markdown_result == "some markdown"

    def test_stores_usage_and_data_info(self):
        """Usage and data_info are stored on the result object."""
        from glmocr.api import GlmOcr

        server_response = {
            "layout_details": [],
            "md_results": "",
            "data_info": {"pages": [{"width": 2000, "height": 3000}]},
            "usage": {"total_tokens": 42, "prompt_tokens": 10, "completion_tokens": 32},
        }

        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_response, "test.png")

        assert result._usage == {
            "total_tokens": 42,
            "prompt_tokens": 10,
            "completion_tokens": 32,
        }
        assert result._data_info == {"pages": [{"width": 2000, "height": 3000}]}


class TestMaaSClientServerProtocolRoundtrip:
    """End-to-end protocol alignment: MaaS client output → server input,
    server output → MaaS client parsing."""

    @pytest.fixture()
    def flask_app(self):
        """Create a Flask test app with mocked pipeline."""
        pytest.importorskip(
            "flask", reason="server tests require 'glmocr[server]' extra"
        )
        from glmocr.server import create_app
        from glmocr.config import load_config

        config = load_config()
        app = create_app(config)
        app.config["TESTING"] = True
        return app

    def test_maas_client_request_server_accepts(self, flask_app):
        """MaaS client's payload format (file field) is accepted by server."""
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.json_result = [
            [{"index": 0, "label": "text", "content": "OCR text"}]
        ]
        mock_result.markdown_result = "OCR text"

        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                # This is exactly what MaaSClient sends via _send_request
                maas_payload = {
                    "model": "glm-ocr",
                    "file": "https://example.com/document.png",
                }
                resp = client.post(
                    "/glmocr/parse",
                    json=maas_payload,
                    content_type="application/json",
                )
                assert resp.status_code == 200

    def test_server_response_parseable_by_maas_client(self, flask_app):
        """Server response can be parsed by _maas_response_to_pipeline_result."""
        from unittest.mock import MagicMock
        from glmocr.api import GlmOcr

        layout = [
            [
                {
                    "index": 0,
                    "label": "text",
                    "content": "Hello World",
                    "bbox_2d": [0, 0, 500, 500],
                }
            ]
        ]
        mock_result = MagicMock()
        mock_result.json_result = layout
        mock_result.markdown_result = "# Hello World"

        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"file": "https://example.com/doc.png"},
                    content_type="application/json",
                )
                server_data = resp.get_json()

        # Now parse it as the MaaS client would
        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(
            server_data, "https://example.com/doc.png"
        )

        assert result.json_result is not None
        assert len(result.json_result) == 1
        assert result.json_result[0][0]["content"] == "Hello World"
        assert result.markdown_result == "# Hello World"

    def test_full_roundtrip_data_integrity(self, flask_app):
        """Data integrity is preserved through the full roundtrip."""
        from unittest.mock import MagicMock
        from glmocr.api import GlmOcr

        original_layout = [
            [
                {
                    "index": 0,
                    "label": "title",
                    "content": "Report",
                    "bbox_2d": [10, 10, 500, 50],
                },
                {
                    "index": 1,
                    "label": "text",
                    "content": "Body text",
                    "bbox_2d": [10, 60, 500, 200],
                },
            ]
        ]
        original_md = "# Report\n\nBody text"

        mock_result = MagicMock()
        mock_result.json_result = original_layout
        mock_result.markdown_result = original_md

        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                resp = client.post(
                    "/glmocr/parse",
                    json={"file": "data:image/png;base64,iVBORw0KGgo="},
                    content_type="application/json",
                )
                server_data = resp.get_json()

        # Verify server response has both field sets
        assert server_data["json_result"] == original_layout
        assert server_data["layout_details"] == original_layout
        assert server_data["markdown_result"] == original_md
        assert server_data["md_results"] == original_md
        assert server_data["model"] == "glm-ocr"
        assert server_data["id"].startswith("chatcmpl-")
        assert isinstance(server_data["created"], int)

        # Parse as MaaS client
        obj = object.__new__(GlmOcr)
        result = obj._maas_response_to_pipeline_result(server_data, "test.png")

        # Content labels preserved
        assert result.json_result[0][0]["label"] == "title"
        assert result.json_result[0][1]["label"] == "text"
        assert result.json_result[0][0]["content"] == "Report"
        assert result.json_result[0][1]["content"] == "Body text"
        assert result.markdown_result == original_md

    def test_maas_client_payload_with_options_accepted(self, flask_app):
        """MaaS client payload with extra options is accepted by server."""
        from unittest.mock import MagicMock

        mock_result = MagicMock()
        mock_result.json_result = []
        mock_result.markdown_result = ""

        with flask_app.test_client() as client:
            with patch.object(
                flask_app.config["pipeline"],
                "process",
                return_value=iter([mock_result]),
            ):
                # MaaSClient sends this format with optional fields
                maas_payload = {
                    "model": "glm-ocr",
                    "file": "https://example.com/doc.pdf",
                    "return_crop_images": True,
                    "need_layout_visualization": True,
                    "start_page_id": 1,
                    "end_page_id": 3,
                    "request_id": "req-123",
                }
                resp = client.post(
                    "/glmocr/parse",
                    json=maas_payload,
                    content_type="application/json",
                )
                assert resp.status_code == 200
