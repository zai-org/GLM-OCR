"""PP-DocLayoutV3 layout detector."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessor,
)

from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
from glmocr.utils.logging import get_logger
from glmocr.utils.visualization_utils import draw_layout_boxes

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig

logger = get_logger(__name__)


class PPDocLayoutDetector(BaseLayoutDetector):
    """PP-DocLayoutV3 layout detector.

    Single instance, in-process batch inference. No multiprocessing workers.
    """

    def __init__(self, config: "LayoutConfig"):
        """Initialize.

        Args:
            config: LayoutConfig instance.
        """
        super().__init__(config)

        self.model_dir = config.model_dir
        self.cuda_visible_devices = config.cuda_visible_devices
        self._config_device = config.device  # explicit device override (may be None)

        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.batch_size = config.batch_size

        self.label_task_mapping = config.label_task_mapping
        self.id2label = getattr(config, "id2label", None)

        self._model = None
        self._image_processor = None
        self._device = None

    def _validate_runtime_config(self):
        """Validate layout config before model loading or post-processing."""
        if not self.model_dir:
            raise ValueError(
                "pipeline.layout.model_dir is required for self-hosted layout "
                "detection. Set it to a local checkpoint directory or a Hugging "
                "Face model id such as 'PaddlePaddle/PP-DocLayoutV3_safetensors'."
            )
        if self.label_task_mapping is not None and (
            not isinstance(self.label_task_mapping, dict) or not self.label_task_mapping
        ):
            raise ValueError(
                "pipeline.layout.label_task_mapping must be a non-empty mapping "
                "when provided."
            )

    def start(self):
        """Load model and processor once in the main process."""
        logger.debug("Initializing PP-DocLayoutV3...")
        self._validate_runtime_config()

        self._image_processor = PPDocLayoutV3ImageProcessor.from_pretrained(
            self.model_dir
        )
        self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(self.model_dir)
        self._model.eval()

        # Device selection priority:
        #   1. Explicit config.device ("cpu", "cuda", "cuda:N")
        #   2. Auto: cuda:{cuda_visible_devices} if CUDA available, else CPU
        if self._config_device is not None:
            self._device = self._config_device
        elif torch.cuda.is_available() and self.cuda_visible_devices:
            self._device = f"cuda:{self.cuda_visible_devices}"
        else:
            self._device = "cpu"
        self._model = self._model.to(self._device)
        if self.id2label is None:
            self.id2label = getattr(self._model.config, "id2label", None)
        if self.id2label is None:
            raise RuntimeError(
                "Missing id2label in both layout config and model config; "
                "please set pipeline.layout.id2label."
            )

        # Patch upstream _extract_polygon_points_by_masks to guard against
        # empty mask crops that crash cv2.resize with !ssize.empty().
        def _safe_extract(boxes, masks, scale_ratio):
            scale_w, scale_h = scale_ratio[0] / 4, scale_ratio[1] / 4
            mask_h, mask_w = masks.shape[1:]
            polygon_points = []

            for i in range(len(boxes)):
                x_min, y_min, x_max, y_max = boxes[i].astype(np.int32)
                box_w, box_h = x_max - x_min, y_max - y_min
                rect = np.array(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
                    dtype=np.float32,
                )

                if box_w <= 0 or box_h <= 0:
                    polygon_points.append(rect)
                    continue

                x_start = int(round((x_min * scale_w).item()))
                x_end = int(round((x_max * scale_w).item()))
                x_start, x_end = np.clip([x_start, x_end], 0, mask_w)
                y_start = int(round((y_min * scale_h).item()))
                y_end = int(round((y_max * scale_h).item()))
                y_start, y_end = np.clip([y_start, y_end], 0, mask_h)

                cropped_mask = masks[i, y_start:y_end, x_start:x_end]
                if cropped_mask.size == 0:
                    polygon_points.append(rect)
                    continue

                resized = cv2.resize(
                    cropped_mask.astype(np.uint8),
                    (box_w, box_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                polygon = self._image_processor._mask2polygon(resized)
                if polygon is not None and len(polygon) < 4:
                    polygon_points.append(rect)
                    continue
                if polygon is not None and len(polygon) > 0:
                    polygon = polygon + np.array([x_min, y_min])
                polygon_points.append(polygon)

            return polygon_points

        self._image_processor._extract_polygon_points_by_masks = _safe_extract
        if self.label_task_mapping is None:
            logger.warning(
                "layout.label_task_mapping is missing; defaulting all labels to text"
            )
            self.label_task_mapping = {"text": list(self.id2label.values())}
        logger.debug(f"PP-DocLayoutV3 loaded on device: {self._device}")

    def stop(self):
        """Unload model and processor."""
        if self._model is not None:
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
            self._model = None
        self._image_processor = None
        self._device = None
        logger.debug("PP-DocLayoutV3 stopped.")

    def _apply_per_class_threshold(self, raw_results: List[Dict]):
        """Filter detections by per-class confidence thresholds.

        For each detection, look up its class in threshold_by_class. Classes
        not listed fall back to self.threshold.

        Args:
            raw_results: List of dicts from post_process_object_detection,
                each with 'scores', 'labels', 'boxes' tensors and optional
                'order_seq' tensor and 'polygon_points' list.

        Returns:
            Filtered list in the same format.
        """
        # Build mapping for label name to class id lookup.
        label2id = {name: int(cls_id) for cls_id, name in self.id2label.items()}

        # Build a lookup mapping class_id (int) -> threshold (float).
        class_thresholds = {}
        for key, value in self.threshold_by_class.items():
            if isinstance(key, str):
                if key in label2id:
                    class_thresholds[label2id[key]] = float(value)
                else:
                    logger.warning(
                        "Unknown class name '%s' in threshold_by_class; "
                        "this entry will be ignored. Known classes: %s",
                        key,
                        ", ".join(sorted(label2id.keys())),
                    )
            else:
                class_thresholds[int(key)] = float(value)

        fallback = self.threshold

        filtered = []
        for result in raw_results:
            scores = result["scores"]
            labels = result["labels"]

            # Build a per-detection threshold tensor: use the per-class value
            # if defined, otherwise fall back to the global threshold.
            thresholds = torch.full_like(scores, fallback)
            for class_id, thresh in class_thresholds.items():
                thresholds[labels == class_id] = thresh

            keep = scores >= thresholds

            new_result = {
                "scores": scores[keep],
                "labels": labels[keep],
                "boxes": result["boxes"][keep],
            }
            if "order_seq" in result:
                new_result["order_seq"] = result["order_seq"][keep]
            if "polygon_points" in result:
                keep_list = keep.tolist()
                new_result["polygon_points"] = [
                    p for p, k in zip(result["polygon_points"], keep_list) if k
                ]
            filtered.append(new_result)
        return filtered

    def _empty_detection_result(self) -> Dict:
        """Return an empty detection result dict (no boxes)."""
        return {
            "scores": torch.tensor([], device=self._device),
            "labels": torch.tensor([], dtype=torch.long, device=self._device),
            "boxes": torch.tensor([], device=self._device).reshape(0, 4),
            "order_seq": torch.tensor([], dtype=torch.long, device=self._device),
        }

    def _run_detection_single_image(
        self, image: Image.Image, pre_threshold: float
    ) -> Dict:
        """Run model + post_process for a single image. Raises on error."""
        single_inputs = self._image_processor(images=[image], return_tensors="pt")
        single_inputs = {k: v.to(self._device) for k, v in single_inputs.items()}
        with torch.no_grad():
            single_outputs = self._model(**single_inputs)
        single_target = torch.tensor([image.size[::-1]], device=self._device)
        single_raw = self._image_processor.post_process_object_detection(
            single_outputs,
            threshold=pre_threshold,
            target_sizes=single_target,
        )
        return single_raw[0]

    def _post_process_chunk_with_fallback(
        self,
        chunk_pil: List[Image.Image],
        outputs,
        target_sizes,
        pre_threshold: float,
        chunk_start: int,
    ) -> List[Dict]:
        """Run batch post_process; on failure, retry image-by-image."""
        try:
            return self._image_processor.post_process_object_detection(
                outputs,
                threshold=pre_threshold,
                target_sizes=target_sizes,
            )
        except Exception as e:
            logger.warning(
                "Layout post_process failed for chunk (retrying image-by-image): %s",
                e,
            )
        raw_results = []
        for i, img in enumerate(chunk_pil):
            try:
                raw_results.append(self._run_detection_single_image(img, pre_threshold))
            except Exception as e2:
                logger.warning(
                    "Layout post_process failed for image %s in chunk: %s",
                    chunk_start + i,
                    e2,
                )
                raw_results.append(self._empty_detection_result())
        return raw_results

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        global_start_idx: int = 0,
        use_polygon: bool = False,
    ) -> tuple:
        """Batch-detect layout regions in-process.

        Args:
            images: List of PIL Images.
            save_visualization: Whether to generate visualization images.
            global_start_idx: Start index for visualization page numbering.
            use_polygon: Use polygon masks for visualization and cropping.

        Returns:
            Tuple of (results, vis_images) where *results* is
            ``List[List[Dict]]`` and *vis_images* is
            ``Dict[int, PIL.Image.Image]`` mapping global page index to
            the rendered layout visualization (empty dict when disabled).
        """
        if self._model is None:
            raise RuntimeError("Layout detector not started. Call start() first.")
        self._validate_runtime_config()

        num_images = len(images)
        pil_images = [
            img.convert("RGB") if img.mode != "RGB" else img for img in images
        ]
        all_paddle_format_results = []

        for chunk_start in range(0, num_images, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, num_images)
            chunk_pil = pil_images[chunk_start:chunk_end]

            inputs = self._image_processor(images=chunk_pil, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor(
                [img.size[::-1] for img in chunk_pil], device=self._device
            )
            if self.threshold_by_class:
                # Use the lowest threshold (per-class or global fallback)
                # so post-processing doesn't discard valid detections early.
                pre_threshold = min(
                    self.threshold, min(self.threshold_by_class.values())
                )
            else:
                pre_threshold = self.threshold

            raw_results = self._post_process_chunk_with_fallback(
                chunk_pil, outputs, target_sizes, pre_threshold, chunk_start
            )

            if self.threshold_by_class:
                raw_results = self._apply_per_class_threshold(raw_results)
            img_sizes = [img.size for img in chunk_pil]
            paddle_format_results = apply_layout_postprocess(
                raw_results=raw_results,
                id2label=self.id2label,
                img_sizes=img_sizes,
                layout_nms=self.layout_nms,
                layout_unclip_ratio=self.layout_unclip_ratio,
                layout_merge_bboxes_mode=self.layout_merge_bboxes_mode,
            )
            all_paddle_format_results.extend(paddle_format_results)

            if self._device.startswith("cuda") and chunk_end < num_images:
                del inputs, outputs, raw_results
                torch.cuda.empty_cache()

        vis_images: Dict[int, Image.Image] = {}
        if save_visualization:
            for img_idx, img_results in enumerate(all_paddle_format_results):
                vis_img = np.array(pil_images[img_idx])
                vis_images[global_start_idx + img_idx] = draw_layout_boxes(
                    image=vis_img,
                    boxes=img_results,
                    use_polygon=use_polygon,
                )

        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width, image_height = pil_images[img_idx].size
            results = []
            valid_index = 0
            for item in paddle_results:
                label = item["label"]
                score = item["score"]
                box = item["coordinate"]
                task_type = None
                for task_item, labels in self.label_task_mapping.items():
                    if isinstance(labels, list) and label in labels:
                        task_type = task_item
                        break
                if task_type is None or task_type == "abandon":
                    continue
                x1, y1, x2, y2 = box
                x1_norm = int(float(x1) / image_width * 1000)
                y1_norm = int(float(y1) / image_height * 1000)
                x2_norm = int(float(x2) / image_width * 1000)
                y2_norm = int(float(y2) / image_height * 1000)

                # Convert polygon_points to normalized list format
                poly_array = item["polygon_points"]
                polygon = [
                    [
                        int(float(point[0]) / image_width * 1000),
                        int(float(point[1]) / image_height * 1000),
                    ]
                    for point in poly_array
                ]

                results.append(
                    {
                        "index": valid_index,
                        "label": label,
                        "score": float(score),
                        "bbox_2d": [x1_norm, y1_norm, x2_norm, y2_norm],
                        "polygon": polygon,
                        "task_type": task_type,
                    }
                )
                valid_index += 1
            all_results.append(results)

        return all_results, vis_images
