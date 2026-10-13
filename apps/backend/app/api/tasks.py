"""
任务相关API
"""

import json
import uuid
from pathlib import Path
from typing import Optional, List
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form, Response
from starlette.concurrency import run_in_threadpool

from app.schemas.response import ApiResponse, TaskData, TaskResultData
from app.core.task_manager import get_task_manager
from app.utils.logger import logger
from app.utils.upload_file_manager import file_upload_handler
from app.utils.config import settings
from app.utils.task_images import (
    TaskImageNotFound,
    image_id_from_path,
    read_task_image,
    validate_image_id,
)


router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.post(
    "/upload",
    response_model=ApiResponse[TaskData],
    status_code=status.HTTP_201_CREATED,
)
async def submit_task(
    file: UploadFile = File(..., description="要处理的文件"),
    processing_mode: str = Form("pipeline"),
    priority: int = Form(2, description="1=低,2=正常,3=高,4=紧急"),
    custom_url : str = Form(None, description=""),
    output_format: str = Form("markdown"),
):
    """
    提交新任务

    - **file**: 上传文件
    - **processing_mode**: 处理模式，默认pipeline
    - **priority**: 优先级 (1=低, 2=正常, 3=高, 4=紧急)
    - **ocr_config**: OCR配置（JSON字符串，可选）
    - **output_format**: 输出格式，默认markdown
    - **retry_config**: 重试配置（JSON字符串，可选）
    """
    try:
        # 生成document_id
        document_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        parsed_ocr_config = None
        # 解析配置
        if custom_url is not None :
            parsed_ocr_config = {"custom_url":custom_url}

        # 保存文件
        save_dir = str(Path(settings.OUTPUT_DIR) / task_id)
        saved_path = await file_upload_handler.save_to_path(
            file=file,
            filename=file.filename,
            upload_dir=save_dir,
        )
        saved_path_obj = Path(saved_path)
        file_size = saved_path_obj.stat().st_size
        file_type = saved_path_obj.suffix.lstrip(".").lower()

        # 提交任务
        task_manager = get_task_manager()
        await task_manager.submit_task(
            task_id=task_id,
            document_id=document_id,
            original_filename=file.filename,
            file_type=file_type,
            file_size=file_size,
            file_path=str(saved_path_obj),
            processing_mode=processing_mode,
            priority=priority,
            ocr_config=parsed_ocr_config,
            output_format=output_format,
        )

        return ApiResponse(
            success=True,
            data={
                "task_id": task_id,
                "document_id": document_id,
                "status": "pending",
                "processing_mode": processing_mode,
                "priority": priority,
                "created_at": datetime.now(UTC).isoformat(),
            },
            message="Task submitted successfully",
        )

    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}",
        )


@router.get("/file")
async def read_file(path: str):
    """兼容旧结果中的图片链接；仅允许访问已完成任务的裁剪图片。"""
    try:
        task_id, image_id = image_id_from_path(path)
    except TaskImageNotFound:
        raise HTTPException(status_code=404, detail="Image not found") from None
    return await _task_image_response(task_id, image_id)


@router.get("/{task_id}/images/{image_id}")
async def get_task_image(task_id: str, image_id: str):
    """获取任务生成并登记的裁剪图片，不接受服务器文件路径。"""
    return await _task_image_response(task_id, image_id)


async def _task_image_response(task_id: str, image_id: str):
    try:
        validate_image_id(task_id, image_id)
        task = await get_task_manager().get_task_status(task_id)
        # Uploaded files cannot supply their own manifest before OCR completes.
        if (
            not task
            or task.get("status") != "completed"
            or not task.get("result_file_path")
        ):
            raise TaskImageNotFound
        content = await run_in_threadpool(read_task_image, task_id, image_id)
        return Response(
            content=content,
            media_type="image/png",
            headers={
                "Content-Disposition": f'inline; filename="{image_id}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )
    except TaskImageNotFound:
        raise HTTPException(status_code=404, detail="Image not found") from None
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read task image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read task image",
        ) from None


@router.get("/{task_id}", response_model=ApiResponse[dict])
async def get_task_status(task_id: str):
    """
    获取任务状态

    - **task_id**: 任务ID
    """
    try:
        task_manager = get_task_manager()
        task_info = await task_manager.get_task_status(task_id)

        if not task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}",
            )

        # 如果有 result_file_path，读取并合并内容
        result_file_path = task_info.get("result_file_path")
        result_data = None
        if result_file_path:
            try:
                result_path = Path(result_file_path)
                if result_path.exists():
                    with open(result_path, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                        logger.info(f"Loaded result data for task {task_id}")
                else:
                    logger.warning(f"Result file not found: {result_file_path}")
            except Exception as e:
                logger.warning(f"Failed to read result file: {e}")

        # 构建响应数据
        response_data = {
            "task_id": task_info.get("task_id"),
            "document_id": task_info.get("document_id"),
            "status": task_info.get("status"),
            "progress": task_info.get("progress"),
            "current_step": task_info.get("current_step"),
            "created_at": task_info.get("created_at").isoformat() if task_info.get("created_at") else None,
            "started_at": task_info.get("started_at").isoformat() if task_info.get("started_at") else None,
            "completed_at": task_info.get("completed_at").isoformat() if task_info.get("completed_at") else None,
            "error_message": task_info.get("error_message"),
            "processing_mode": task_info.get("processing_mode"),
            "priority": task_info.get("priority"),
            "retry_count": task_info.get("retry_count"),
            "worker_id": task_info.get("worker_id"),
        }

        # 添加结果数据
        if result_data:
            response_data["metadata"]= result_data.get("metadata")
            response_data["full_markdown"] = result_data.get("full_markdown")
            response_data["layout"] = result_data.get("layout")

        return ApiResponse(
            success=True,
            data=response_data,
            message="Task status retrieved successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}",
        )


@router.delete("/{task_id}", response_model=ApiResponse[dict])
async def cancel_task(task_id: str):
    """
    取消任务

    - **task_id**: 任务ID
    """
    try:
        task_manager = get_task_manager()
        success = await task_manager.cancel_task(task_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found or cannot be cancelled: {task_id}",
            )

        return ApiResponse(
            success=True,
            data={
                "task_id": task_id,
                "status": "cancelled",
            },
            message="Task cancelled successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel task: {str(e)}",
        )


@router.get("/", response_model=ApiResponse[dict])
async def list_tasks(status: Optional[str] = None, limit: int = 100, offset: int = 0):
    """
    列出任务

    - **status**: 过滤状态 (pending, processing, completed, failed, cancelled)
    - **limit**: 返回数量限制
    - **offset**: 偏移量
    """
    try:
        task_manager = get_task_manager()
        tasks = await task_manager.list_tasks(status=status, limit=limit, offset=offset)

        return ApiResponse(
            success=True,
            data={
                "tasks": tasks,
                "total": len(tasks),
                "limit": limit,
                "offset": offset,
            },
            message="Tasks retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}",
        )
