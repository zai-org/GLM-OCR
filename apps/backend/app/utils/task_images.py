"""Read only the PNG crops explicitly recorded by a task's OCR pipeline."""

import json
import os
import re
import stat
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from uuid import UUID

from PIL import Image

from app.utils.config import settings

_IMAGE_NAME = re.compile(r"split_[1-9][0-9]*_[0-9]{4,}\.png\Z")


class TaskImageNotFound(Exception):
    """The request does not identify a readable, registered task image."""


def validate_image_id(task_id: str, image_id: str) -> None:
    try:
        valid = str(UUID(task_id)) == task_id and _IMAGE_NAME.fullmatch(image_id)
    except (ValueError, AttributeError):
        valid = False
    if not valid:
        raise TaskImageNotFound


def image_id_from_path(path: str) -> tuple[str, str]:
    """Translate legacy paths without accepting traversal or arbitrary subtrees.

    Legacy results contain absolute paths (or paths relative to the backend's
    working directory), not paths relative to OUTPUT_DIR.
    """
    if not isinstance(path, str) or not path or "\x00" in path:
        raise TaskImageNotFound
    candidate = Path(path)
    if ".." in candidate.parts:
        raise TaskImageNotFound
    candidate = candidate.absolute()
    root = Path(settings.OUTPUT_DIR).absolute()
    try:
        # Accept both spellings when the administrator's OUTPUT_DIR is a symlink.
        for base in (root, root.resolve(strict=True)):
            if candidate.is_relative_to(base):
                parts = candidate.relative_to(base).parts
                if len(parts) == 2:
                    validate_image_id(*parts)
                    return parts[0], parts[1]
    except (OSError, RuntimeError, ValueError) as exc:
        raise TaskImageNotFound from exc
    raise TaskImageNotFound


@contextmanager
def _task_directory(root: Path, task_id: str):
    """Pin the directory on POSIX so a symlink swap cannot redirect a read."""
    task_dir = root / task_id
    if task_dir.is_symlink() or task_dir.resolve(strict=True) != task_dir:
        raise TaskImageNotFound
    if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW"):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        root_fd = os.open(root, flags)
        try:
            task_fd = os.open(task_id, flags, dir_fd=root_fd)
            try:
                yield task_dir, task_fd
            finally:
                os.close(task_fd)
        finally:
            os.close(root_fd)
    else:
        # Platforms without openat still reject symlinks and resolved escapes.
        yield task_dir, None


def _read_regular_file(task_dir: Path, task_fd: int | None, name: str) -> bytes:
    path = task_dir / name
    if path.is_symlink() or path.resolve(strict=True).parent != task_dir:
        raise TaskImageNotFound
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    fd = (
        os.open(name, flags, dir_fd=task_fd)
        if task_fd is not None
        else os.open(path, flags)
    )
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise TaskImageNotFound
        return source.read()


def read_task_image(task_id: str, image_id: str) -> bytes:
    """Use the generated manifest as an allowlist, then verify the PNG contents."""
    validate_image_id(task_id, image_id)
    try:
        root = Path(settings.OUTPUT_DIR).resolve(strict=True)
        with _task_directory(root, task_id) as (task_dir, task_fd):
            manifest = json.loads(
                _read_regular_file(task_dir, task_fd, "ocr_result.json")
            )
            if not isinstance(manifest, dict) or manifest.get("success") is not True:
                raise TaskImageNotFound
            references = manifest.get("ref_image_paths")
            if not isinstance(references, list):
                raise TaskImageNotFound
            registered = False
            for reference in references:
                if not isinstance(reference, str):
                    continue
                try:
                    # Stored pipeline paths can contain a relative OUTPUT_DIR
                    # such as ../data. Only normalize these internal references;
                    # the public legacy input still rejects parent traversal.
                    if image_id_from_path(os.path.abspath(reference)) == (
                        task_id,
                        image_id,
                    ):
                        registered = True
                        break
                except TaskImageNotFound:
                    continue
            if not registered:
                raise TaskImageNotFound
            content = _read_regular_file(task_dir, task_fd, image_id)
        with Image.open(BytesIO(content)) as image:
            if image.format != "PNG":
                raise TaskImageNotFound
            image.verify()
        return content
    except (
        OSError,
        ValueError,
        RuntimeError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as exc:
        raise TaskImageNotFound from exc
