
import json
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import httpx
import base64
from app.utils.config import settings
from app.utils.logger import logger
class ServiceRequestError(Exception):
    """服务请求错误"""

    pass


class ServiceResponseError(Exception):
    """服务响应错误"""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")
class LayoutAndOCRClient:
    """Layout分析和OCR识别服务的客户端。

    请求格式:
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image1_data}"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image2_data}"}},
                        {"type": "text", "text": "<|PIPELINE_DOCUMENT_RECOGNITION|>"}
                    ]
                }
            ]
        }

    响应格式:
        [
            [
                {
                    "index": 0,
                    "label": "text",
                    "bbox_2d": [100, 200, 800, 350],
                    "content": "这是第一段文本内容..."
                },
                {
                    "index": 1,
                    "label": "table",
                    "bbox_2d": [150, 400, 850, 700],
                    "content": "<td></td>..."  # 表格以html形式输出
                },
                {
                    "index": 2,
                    "label": "formula",
                    "bbox_2d": [200, 750, 600, 850],
                    "content": "\\[\n5 2 \\times 4\n\\]"  # 公式以LaTeX形式输出
                },
                {
                    "index": 3,
                    "label": "image",
                    "bbox_2d": [210, 400, 5500, 630],
                    "content": None  # 图像的content字段为None
                }
            ],
            # 第二张图像的识别结果...
        ]
    """

    # 默认的layout和OCR提示词
    DEFAULT_PROMPT = "<|PIPELINE_DOCUMENT_RECOGNITION|>"

    def __init__(self, service_url: Optional[str] = None, timeout: float = 120.0):
        """初始化LayoutAndOCR客户端。

        Args:
            service_url: 服务URL。如果为None，则从配置加载。
            timeout: 请求超时时间（秒）
        """
        self.service_url = service_url or getattr(settings, "layout_ocr_url", None)
        if not self.service_url:
            # 默认配置
            self.service_url = "http://localhost:5002/glmocr/parse"

        self.timeout = timeout
        self.async_headers = {"Content-Type": "application/json"}

    def _encode_image_to_base64(self, image_path: str) -> str:
        """将图片文件编码为base64格式。

        Args:
            image_path: 图片文件路径

        Returns:
            base64编码的图片URL字符串
        """

        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # 获取文件扩展名以确定MIME类型
        ext = Path(image_path).suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
        }.get(ext, "image/png")

        return f"data:{mime_type};base64,{image_data}"

    def _create_multi_image_payload(
        self,
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """创建多图片识别的请求载荷。

        Args:
            image_paths: 图片文件路径列表

        Returns:
            请求载荷字典
        """
        if not image_paths:
            raise ValueError("至少需要一张图片")

        # 添加所有图片
        image_base64_list = []
        for image_path in image_paths:
            image_base64 = self._encode_image_to_base64(image_path)
            image_base64_list.append(image_base64)

        return {"images": image_base64_list}

    def _create_single_image_payload(
        self,
        image_path: str
    ) -> Dict[str, Any]:
        """创建单图片识别的请求载荷。

        Args:
            image_path: 图片文件路径

        Returns:
            请求载荷字典
        """
        return self._create_multi_image_payload([image_path])

    async def process_images(
        self,
        image_paths: Union[str, List[str]],
        prompt: Optional[str] = None,
        custom_url: Optional[str] = None,
    ) -> List[List[Dict[str, Any]]]:
        """处理一张或多张图片，返回layout和OCR结果。

        Args:
            image_paths: 单个图片路径或图片路径列表
            prompt: 自定义提示词
            custom_url: 自定义服务URL，会覆盖初始化时的URL

        Returns:
            识别结果列表，每个元素对应一张图片的结果：
            [
                [
                    {
                        "index": 0,
                        "label": "text",
                        "bbox_2d": [100, 200, 800, 350],
                        "content": "文本内容"
                    },
                    ...
                ],
                # 第二张图片的结果...
            ]

        Raises:
            ServiceRequestError: 如果请求失败
            ServiceResponseError: 如果服务返回错误响应
        """
        url = custom_url if custom_url is not None else self.service_url
        # 统一处理为列表
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        logger.info(
            f"Processing {len(image_paths)} images with layout and OCR,url is:{url}"
        )

        # 创建请求载荷
        payload = self._create_multi_image_payload(image_paths)

        try:
            async with httpx.AsyncClient(
                headers=self.async_headers, timeout=self.timeout, verify=False
            ) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 200:
                    result = response.json()

                    # 提取内容
                    if "json_result" in result and len(result["json_result"]) > 0:
                        content = result["json_result"]

                        # 解析JSON结果
                        try:
                            # 检查 content 的类型，如果已经是列表则直接使用
                            if isinstance(content, list):
                                all_results = content
                            else:
                                all_results = json.loads(content)
                                
                            # 验证返回格式
                            if not isinstance(all_results, list):
                                raise ValueError("响应格式错误：期望返回列表")

                            # 验证每张图片的结果
                            for img_idx, img_result in enumerate(all_results):
                                if not isinstance(img_result, list):
                                    logger.warning(
                                        f"图片 {img_idx} 的结果格式错误，期望列表"
                                    )
                                    continue

                                # 验证每个block的格式
                                for block in img_result:
                                    if not isinstance(block, dict):
                                        continue
                                    # 确保必需字段存在
                                    if "index" not in block:
                                        block["index"] = 0
                                    if "label" not in block:
                                        block["label"] = "text"
                                    if "bbox_2d" not in block:
                                        block["bbox_2d"] = [0, 0, 0, 0]
                                    # content字段可能为None（如图像）

                            total_blocks = sum(len(img) for img in all_results)
                            logger.info(
                                f"Successfully processed {len(image_paths)} images, "
                                f"found {total_blocks} blocks total"
                            )

                            return all_results

                        except json.JSONDecodeError as e:
                            error_msg = f"Failed to parse response JSON: {str(e)}"
                            logger.error(f"Response content: {content[:500]}...")
                            raise ServiceRequestError(error_msg)
                    else:
                        error_msg = "Response missing 'choices' field"
                        logger.error(f"Response: {result}")
                        raise ServiceResponseError(response.status_code, error_msg)

                else:
                    error_msg = f"Service returned status code {response.status_code}"
                    try:
                        error_detail = response.json()
                        if isinstance(error_detail, dict):
                            error_detail = error_detail.get("error", {}).get(
                                "message", ""
                            )
                        elif isinstance(error_detail, str):
                            error_detail = error_detail
                        if error_detail:
                            error_msg += f": {error_detail}"
                    except Exception:
                        pass
                    logger.error(f"Service error: {error_msg}")
                    raise ServiceResponseError(response.status_code, error_msg)

        except httpx.RequestError as e:
            error_msg = f"Failed to make request to service: {str(e)}"
            logger.error(error_msg)
            raise ServiceRequestError(error_msg)
        except (KeyError, ValueError) as e:
            error_msg = f"Failed to parse service response: {str(e)}"
            logger.error(error_msg)
            raise ServiceRequestError(error_msg)

    async def process_single_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        custom_url: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """处理单张图片，返回layout和OCR结果。

        Args:
            image_path: 图片文件路径
            prompt: 自定义提示词
            custom_url: 自定义服务URL

        Returns:
            单张图片的识别结果列表
        """
        results = await self.process_images(image_path, prompt, custom_url)
        return results[0] if results else []
