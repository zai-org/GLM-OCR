import base64
from io import BytesIO

import pytest
from PIL import Image

from glmocr.config import MaaSApiConfig
from glmocr.maas_client import MaaSClient


@pytest.mark.parametrize("variant", ["short", "unpadded", "wrapped", "prefixed"])
def test_prepare_base64_image_is_a_canonical_data_uri(variant):
    buffer = BytesIO()
    Image.new("RGBA", (1, 1)).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    assert len(encoded) < 128
    assert encoded.endswith("=")
    source = {
        "short": encoded,
        "unpadded": encoded.rstrip("="),
        "wrapped": "\n".join([encoded[:40], encoded[40:]]),
        "prefixed": "<|base64|>" + encoded,
    }[variant]
    client = MaaSClient(MaaSApiConfig(api_key="test-key"))
    result = client._prepare_file(source)
    assert result == "data:image/png;base64," + encoded
    assert base64.b64decode(result.split(",", 1)[1], validate=True) == image_bytes


def test_short_filename_is_not_base64():
    assert not MaaSClient._looks_like_base64("document")


def test_existing_file_and_url_inputs(tmp_path):
    document = tmp_path / "document.pdf"
    contents = b"%PDF-1.4\n"
    document.write_bytes(contents)
    client = MaaSClient(MaaSApiConfig(api_key="test-key"))
    assert client._prepare_file(document) == (
        "data:application/pdf;base64," + base64.b64encode(contents).decode("ascii")
    )
    url = "https://example.com/document.pdf?signature=abc"
    assert client._prepare_file(url) == url
