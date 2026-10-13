"""Security and preview regressions against real HTTP routes and task records."""

import json
import os
import shutil
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote, urlsplit
from uuid import uuid4

import pytest
from PIL import Image


def assert_hidden(response):
    assert response.status_code == 404
    assert response.json() == {"detail": "Image not found"}


def test_registered_preview(image_request, artifact):
    response = image_request()
    assert response.status_code == 200
    assert response.content == artifact.content
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, no-store"
    assert "authorization" not in response.request.headers
    assert "cookie" not in response.request.headers


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/proc/self/environ",
        "/app/../etc/hostname",
        "/app/app/utils/config.py",
        "/app/data/tasks.db",
        "",
        "\x00",
    ],
)
def test_report_paths_are_blocked(client, output_dir, path):
    assert_hidden(client.get("/api/v1/tasks/file", params={"path": path}))


@pytest.mark.parametrize(
    "name",
    [
        "tasks.db",
        "config.py",
        "ocr_result.json",
        "merged.json",
        "input.png",
        "images/page_0001.png",
        "split_1_0002.png",
        "split_1_0001.svg",
    ],
)
def test_non_artifacts_inside_output_are_blocked(client, artifact, output_dir, name):
    path = (output_dir if name == "tasks.db" else artifact.directory) / name
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(artifact.content)
    assert_hidden(client.get("/api/v1/tasks/file", params={"path": str(path)}))


def test_prefix_lookalike_is_not_within_root(client, artifact, output_dir):
    outside = output_dir.with_name(output_dir.name + "_backup")
    shutil.copytree(artifact.directory, outside / artifact.task_id)
    path = outside / artifact.task_id / artifact.image.name
    assert_hidden(client.get("/api/v1/tasks/file", params={"path": str(path)}))


@pytest.mark.parametrize("encoding", ["plain", "encoded", "double_encoded"])
def test_traversal_is_blocked(client, artifact, output_dir, encoding):
    path = f"{artifact.directory}/../{artifact.task_id}/{artifact.image.name}"
    if encoding == "plain":
        response = client.get("/api/v1/tasks/file", params={"path": path})
    else:
        encoded = quote(path, safe="").replace("..", "%2e%2e")
        if encoding == "double_encoded":
            encoded = quote(encoded, safe="")
        response = client.get(f"/api/v1/tasks/file?path={encoded}")
    assert_hidden(response)


def test_legacy_relative_paths_still_work(client, artifact, output_dir, monkeypatch):
    from app.utils.config import settings

    monkeypatch.chdir(output_dir.parent)
    monkeypatch.setattr(settings, "OUTPUT_DIR", output_dir.name)
    relative = artifact.image.relative_to(output_dir.parent)
    artifact.manifest.write_text(
        json.dumps(
            {
                "success": True,
                "ref_image_paths": [str(relative)],
            }
        ),
        encoding="utf-8",
    )
    response = client.get("/api/v1/tasks/file", params={"path": str(relative)})
    assert response.status_code == 200
    assert response.content == artifact.content


def test_configured_output_symlink_is_supported(
    client,
    artifact,
    output_dir,
    tmp_path,
    monkeypatch,
):
    from app.utils.config import settings

    alias = tmp_path / "configured-storage"
    alias.symlink_to(output_dir, target_is_directory=True)
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(alias))
    for root in (alias, output_dir):
        response = client.get(
            "/api/v1/tasks/file",
            params={
                "path": str(root / artifact.task_id / artifact.image.name),
            },
        )
        assert response.status_code == 200
        assert response.content == artifact.content


def test_matching_basename_in_other_task_is_not_registration(image_request, artifact):
    other = artifact.directory.parent / str(uuid4()) / artifact.image.name
    artifact.manifest.write_text(
        json.dumps(
            {
                "success": True,
                "ref_image_paths": [str(other)],
            }
        ),
        encoding="utf-8",
    )
    assert_hidden(image_request())


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        [],
        {},
        {"success": False, "ref_image_paths": []},
        {"success": True, "ref_image_paths": "not a list"},
        {"success": True, "ref_image_paths": [None, {}, "/etc/passwd"]},
        {"success": True, "ref_image_paths": []},
    ],
)
def test_unregistered_image_is_blocked(image_request, artifact, manifest):
    artifact.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    assert_hidden(image_request())


def test_missing_or_broken_manifest_is_blocked(image_request, artifact):
    artifact.manifest.write_text("{broken JSON", encoding="utf-8")
    assert_hidden(image_request())
    artifact.manifest.unlink()
    assert_hidden(image_request())


@pytest.mark.parametrize("kind", ["text", "jpeg", "truncated_png"])
def test_disguised_non_png_is_not_returned(image_request, artifact, kind):
    if kind == "jpeg":
        Image.new("RGB", (2, 2)).save(artifact.image, format="JPEG")
    else:
        artifact.image.write_bytes(
            b"private configuration" if kind == "text" else artifact.content[:20]
        )
    assert_hidden(image_request())


@pytest.mark.parametrize("target_kind", ["image", "manifest", "directory", "loop"])
def test_symlink_escape_is_blocked(image_request, artifact, tmp_path, target_kind):
    outside = tmp_path / "outside"
    if target_kind == "directory":
        artifact.directory.rename(outside)
        artifact.directory.symlink_to(outside, target_is_directory=True)
    else:
        path = artifact.manifest if target_kind == "manifest" else artifact.image
        if target_kind != "loop":
            outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(path if target_kind == "loop" else outside)
    assert_hidden(image_request())


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX special file")
def test_fifo_is_rejected_without_blocking(image_request, artifact):
    artifact.image.unlink()
    os.mkfifo(artifact.image)
    assert_hidden(image_request())


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX no-follow open")
@pytest.mark.parametrize("target_kind", ["image", "directory"])
def test_symlink_swap_after_resolution_is_blocked(
    image_request,
    artifact,
    tmp_path,
    monkeypatch,
    target_kind,
):
    target = artifact.image if target_kind == "image" else artifact.directory
    outside = tmp_path / "swapped-outside"
    original_resolve = Path.resolve
    swapped = False

    def racing_resolve(path, *args, **kwargs):
        nonlocal swapped
        result = original_resolve(path, *args, **kwargs)
        if path == target and not swapped:
            swapped = True
            target.rename(outside)
            target.symlink_to(outside, target_is_directory=target_kind == "directory")
        return result

    monkeypatch.setattr(Path, "resolve", racing_resolve)
    assert_hidden(image_request())
    assert swapped


@pytest.mark.parametrize(
    "task_status", ["pending", "processing", "failed", "cancelled"]
)
def test_unfinished_tasks_cannot_supply_manifests(
    image_request,
    artifact,
    complete_task,
    task_status,
):
    complete_task(artifact.task_id, artifact.result, task_status)
    assert_hidden(image_request())


def test_unknown_task_is_blocked(client, artifact, output_dir):
    unknown = str(uuid4())
    shutil.copytree(artifact.directory, output_dir / unknown)
    assert_hidden(client.get(f"/api/v1/tasks/{unknown}/images/{artifact.image.name}"))


def test_completed_task_without_registered_result_is_blocked(
    image_request,
    artifact,
    complete_task,
):
    complete_task(artifact.task_id, "")
    assert_hidden(image_request())


def test_internal_error_does_not_disclose_paths(image_request, monkeypatch):
    from app.api import tasks

    def fail(*args):
        raise OSError("private /deployment/configuration/path")

    monkeypatch.setattr(tasks, "read_task_image", fail)
    response = image_request()
    assert response.status_code == 500
    assert response.json() == {"detail": "Failed to read task image"}


@pytest.mark.parametrize(
    "task_id,image_id",
    [
        ("not-a-uuid", "split_1_0001.png"),
        (str(uuid4()), "tasks.db"),
        (str(uuid4()), "split_1_0001.png%00"),
        (str(uuid4()), "..%5Csplit_1_0001.png"),
    ],
)
def test_invalid_resource_ids_are_blocked(client, task_id, image_id):
    assert_hidden(client.get(f"/api/v1/tasks/{task_id}/images/{image_id}"))


class ImageSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.sources.append(dict(attrs)["src"])


@pytest.mark.parametrize("input_format", ["png", "pdf"])
@pytest.mark.parametrize("root_style", ["absolute", "relative", "parent_relative"])
def test_upload_pipeline_result_and_preview(
    client,
    output_dir,
    png_bytes,
    complete_task,
    monkeypatch,
    input_format,
    root_style,
):
    """Only the external model response is stubbed; all app processing is real."""
    from app.core.flows.base import ProcessingContext
    from app.core.flows.pipeline_flow import PipelineFlow
    from app.core.ocr_client import LayoutAndOCRClient
    from app.utils.config import settings

    if root_style != "absolute":
        workdir = output_dir.parent
        if root_style == "parent_relative":
            workdir = workdir / "backend-workdir"
            workdir.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setattr(
            settings, "OUTPUT_DIR", os.path.relpath(output_dir, workdir)
        )

    payload = png_bytes
    if input_format == "pdf":
        import fitz

        with fitz.open() as document:
            document.new_page(width=144, height=144)
            document.new_page(width=144, height=144)
            payload = document.tobytes()
    response = client.post(
        "/api/v1/tasks/upload",
        files={
            "file": (
                f"document.{input_format}",
                payload,
                "application/pdf" if input_format == "pdf" else "image/png",
            ),
        },
    )
    assert response.status_code == 201
    task_id = response.json()["data"]["task_id"]
    directory = output_dir / task_id
    model = AsyncMock(
        return_value=[
            {"label": "text", "bbox_2d": [0, 0, 500, 200], "content": "OCR test text"},
            {"label": "image", "bbox_2d": [0, 200, 500, 800], "content": None},
        ]
    )
    monkeypatch.setattr(LayoutAndOCRClient, "process_single_image", model)
    context = ProcessingContext(
        task_id=task_id,
        document_id=task_id,
        file_path=str(directory / f"document.{input_format}"),
        file_type=input_format,
        processing_mode="pipeline",
        ocr_config={},
        output_format="markdown",
    )
    result = client.portal.call(PipelineFlow(context).process)
    complete_task(task_id, result["json_output_path"])
    response = client.get(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()["data"]
    markdown = data["full_markdown"]
    assert "OCR test text" in markdown
    assert str(output_dir) not in markdown
    assert "?path=" not in markdown
    parser = ImageSources()
    parser.feed(markdown)
    expected_pages = 2 if input_format == "pdf" else 1
    assert model.await_count == expected_pages
    assert len(parser.sources) == expected_pages
    assert len(data["layout"]) == expected_pages * 2
    assert Path(result["md_output_path"]).read_text(encoding="utf-8") == markdown
    for source in parser.sources:
        assert source.startswith(f"http://testserver/api/v1/tasks/{task_id}/images/")
        preview = client.get(source)
        assert preview.status_code == 200
        with Image.open(BytesIO(preview.content)) as image:
            assert image.format == "PNG"
            image.verify()
        image_id = Path(urlsplit(source).path).name
        legacy = client.get(
            "/api/v1/tasks/file", params={"path": str(directory / image_id)}
        )
        assert legacy.status_code == 200
        assert legacy.content == preview.content


def test_merge_without_images_or_pages(client, output_dir):
    from app.core.flows.base import ProcessingContext
    from app.core.steps.merge_results import _merge_to_markdown

    context = ProcessingContext(
        task_id=str(uuid4()),
        document_id="test",
        file_path="unused",
        file_type="png",
        processing_mode="pipeline",
        ocr_config={},
        output_format="markdown",
    )
    progress = AsyncMock()
    _, result = client.portal.call(
        _merge_to_markdown, context, {}, str(output_dir), progress
    )
    assert json.loads(Path(result).read_text(encoding="utf-8"))["full_markdown"] == ""
    progress.assert_awaited_once_with(100.0, "Merged 0 pages")


def test_missing_crop_does_not_generate_broken_link(client, artifact):
    from app.core.flows.base import ProcessingContext
    from app.core.steps.merge_results import _merge_to_markdown

    context = ProcessingContext(
        task_id=artifact.task_id,
        document_id="test",
        file_path="unused",
        file_type="png",
        processing_mode="pipeline",
        ocr_config={},
        output_format="markdown",
    )
    ocr_results = {
        "pages": [
            {
                "layout": {
                    "blocks": [
                        {"layout_type": "image", "image_path": None, "content": None},
                        {"layout_type": "text", "content": "Text remains visible"},
                    ]
                }
            }
        ]
    }
    _, result = client.portal.call(
        _merge_to_markdown,
        context,
        ocr_results,
        str(artifact.directory),
        None,
    )
    markdown = json.loads(Path(result).read_text(encoding="utf-8"))["full_markdown"]
    assert "Text remains visible" in markdown
    assert "<img" not in markdown
    assert "None" not in markdown
