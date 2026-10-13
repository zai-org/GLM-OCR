"""Integration fixtures use the real app and a temporary SQLite database."""

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    sandbox = tmp_path_factory.mktemp("backend")
    with pytest.MonkeyPatch.context() as patch:
        # Set these before importing the app: never initialize the user's DB.
        patch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{sandbox / 'tasks.db'}")
        patch.setenv("OUTPUT_DIR", str(sandbox / "output"))
        patch.setenv("WORKER_COUNT", "0")
        patch.setenv("PUBLIC_API_BASE_URL", "http://testserver/api/v1")
        from app.main import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def output_dir(client, tmp_path, monkeypatch):
    from app.utils.config import settings

    directory = tmp_path / "output # & 中文"
    directory.mkdir()
    monkeypatch.setattr(settings, "OUTPUT_DIR", str(directory))
    return directory


@pytest.fixture
def png_bytes():
    stream = BytesIO()
    Image.new("RGB", (40, 30), "blue").save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def complete_task(client):
    def complete(task_id, result_path, task_status="completed"):
        async def update():
            from app.db.database import AsyncSessionLocal
            from app.repository.task import TaskRepository

            async with AsyncSessionLocal() as session:
                await TaskRepository(session).update_task_status(
                    task_id, task_status, result_file_path=str(result_path)
                )
                await session.commit()

        client.portal.call(update)

    return complete


@pytest.fixture
def artifact(client, output_dir, png_bytes, complete_task):
    response = client.post(
        "/api/v1/tasks/upload",
        files={"file": ("input.png", png_bytes, "image/png")},
    )
    assert response.status_code == 201
    task_id = response.json()["data"]["task_id"]
    directory = output_dir / task_id
    image = directory / "split_1_0001.png"
    image.write_bytes(png_bytes)
    manifest = directory / "ocr_result.json"
    manifest.write_text(
        json.dumps({"success": True, "ref_image_paths": [str(image)]}),
        encoding="utf-8",
    )
    result = directory / "merged.json"
    result.write_text('{"full_markdown": "existing result"}', encoding="utf-8")
    complete_task(task_id, result)
    return SimpleNamespace(
        task_id=task_id,
        directory=directory,
        image=image,
        manifest=manifest,
        result=result,
        content=png_bytes,
    )


@pytest.fixture(params=["legacy", "resource"])
def image_request(request, client, artifact):
    def get():
        if request.param == "legacy":
            return client.get(
                "/api/v1/tasks/file", params={"path": str(artifact.image)}
            )
        return client.get(
            f"/api/v1/tasks/{artifact.task_id}/images/{artifact.image.name}"
        )

    return get
