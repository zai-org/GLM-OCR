from unittest.mock import patch

import pytest

from glmocr.api import GlmOcr


@pytest.mark.parametrize("source", ["environment", "dotenv"])
def test_explicit_selfhosted_mode_survives_an_api_key(source, monkeypatch, tmp_path):
    monkeypatch.delenv("GLMOCR_API_KEY", raising=False)
    monkeypatch.delenv("GLMOCR_MODE", raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    options = {}
    if source == "environment":
        monkeypatch.setenv("GLMOCR_MODE", "selfhosted")
    else:
        env_file = tmp_path / ".env"
        env_file.write_text("GLMOCR_MODE=selfhosted\n", encoding="utf-8")
        options["env_file"] = str(env_file)
    with patch("glmocr.pipeline.Pipeline") as pipeline:
        with GlmOcr(**options) as parser:
            assert parser.config_model.pipeline.maas.enabled is False
            pipeline.assert_called_once()


def test_constructor_mode_still_overrides_environment(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "test-key")
    monkeypatch.setenv("GLMOCR_MODE", "selfhosted")
    with GlmOcr(mode="maas") as parser:
        assert parser.config_model.pipeline.maas.enabled is True
