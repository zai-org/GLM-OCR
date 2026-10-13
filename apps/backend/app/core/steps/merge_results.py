"""
结果合并步骤
"""

from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import json
import os
from html import escape

from app.core.flows.base import ProcessingContext
from app.utils.logger import logger
from app.utils.config import settings
from app.utils.task_images import TaskImageNotFound, image_id_from_path


class MergeResultsStepInput:
    ocr_result_path: str

    def __init__(self, ocr_result_path: str) -> None:
        self.ocr_result_path = ocr_result_path



async def merge_results(
    context: ProcessingContext,
    input: MergeResultsStepInput,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    合并OCR结果

    Args:
        context: 处理上下文
        input: MergeResultsStepInput，包含 ocr_result_path
        progress_callback: 进度回调函数

    Returns:
        Dict[str, Any]: 合并后的结果
    """
    task_id = context.task_id
    output_format = context.output_format
    output_dir = context.get_output_dir()
    result_path = input.ocr_result_path
    logger.info(f"[{task_id}] Starting result merge")

    try:
        if progress_callback:
            await progress_callback(0.0, "Initializing merge")

        # 从文件读取OCR结果
        ocr_results = {}
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                ocr_results.update(result_data)
        except Exception as e:
            logger.error(
                f"[{task_id}] Failed to read OCR result from {result_path}: {e}"
            )

        # 根据输出格式进行合并
        md_output_path, json_output_path = await _merge_to_markdown(
            context, ocr_results, output_dir, progress_callback
        )

        if progress_callback:
            await progress_callback(100.0, "Merge completed")

        result = {
            "md_output_path": md_output_path,
            "json_output_path": json_output_path,
            "output_files": [md_output_path, json_output_path],
            "metadata": {
                "format": output_format,
                "total_pages": len(ocr_results.get("pages", [])),
            },
        }

        logger.info(
            f"[{task_id}] Result merge completed: md_output_path:{md_output_path},json_output_path:{json_output_path}"
        )

        return result

    except Exception as e:
        logger.error(f"[{task_id}] Result merge failed: {e}")
        raise


async def _merge_to_markdown(
    context: ProcessingContext,
    ocr_results: Dict[str, Any],
    output_dir: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """合并为Markdown格式"""
    pages = ocr_results.get("pages", [])

    markdown_lines = []
    result = {}
    result["metadata"] = context.metadata
    merge_res_layout = []
    total_pages = len(pages)
    for page in pages:
        layout = page.get("layout", {}).get("blocks", [])
        for block in layout:
            text = block.get("content", "")
            layout_type = block.get("layout_type", "")
            if layout_type == "image":
                try:
                    image_path = block.get("image_path")
                    if isinstance(image_path, str):
                        image_path = os.path.abspath(image_path)
                    task_id, image_id = image_id_from_path(image_path)
                    if task_id != context.task_id:
                        raise TaskImageNotFound
                    image_url = (
                        f"{settings.PUBLIC_API_BASE_URL.rstrip('/')}/tasks/"
                        f"{task_id}/images/{image_id}"
                    )
                    text = (
                        '<div style="text-align: center;">'
                        f'<img src="{escape(image_url, quote=True)}" alt="Image"/></div>\n'
                    )
                except TaskImageNotFound:
                    # Failed crops have no image_path; do not emit a broken URL.
                    text = text or ""
            markdown_lines.append(f"{text}\n")
            merge_res_layout.append(
                {
                    "block_content": text,
                    "bbox": block.get("layout_box"),
                    "block_id": block.get("index"),
                    "page_index": block.get("page_index"),
                }
            )
    if progress_callback:
        progress = 100.0
        await progress_callback(progress, f"Merged {total_pages} pages")
    result["full_markdown"] = "".join(markdown_lines)
    result["layout"] = merge_res_layout
    # 写入文件
    md_output_path = str(Path(output_dir) / "result.md")
    with open(md_output_path, "w", encoding="utf-8") as f:
        f.writelines(markdown_lines)
    json_output_path = str(Path(output_dir) / "merged.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return md_output_path, json_output_path
