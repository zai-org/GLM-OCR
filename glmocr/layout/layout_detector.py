"""PP-DocLayoutV3 layout detector.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Optional

import torch
import numpy as np
from PIL import Image
from transformers import (
    PPDocLayoutV3ForObjectDetection,
    PPDocLayoutV3ImageProcessorFast,
)

from glmocr.layout.base import BaseLayoutDetector
from glmocr.utils.layout_postprocess_utils import apply_layout_postprocess
from glmocr.utils.logging import get_logger
from glmocr.utils.visualization_utils import save_layout_visualization

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

        self.threshold = config.threshold
        self.threshold_by_class = config.threshold_by_class
        self.layout_nms = config.layout_nms
        self.layout_unclip_ratio = config.layout_unclip_ratio
        self.layout_merge_bboxes_mode = config.layout_merge_bboxes_mode
        self.batch_size = config.batch_size

        self.label_task_mapping = config.label_task_mapping
        self.id2label = config.id2label

        self._model = None
        self._image_processor = None
        self._device = None

    def start(self):
        """Load model and processor once in the main process."""
        logger.debug("Initializing PP-DocLayoutV3...")

        self._image_processor = PPDocLayoutV3ImageProcessorFast.from_pretrained(
            self.model_dir
        )
        self._model = PPDocLayoutV3ForObjectDetection.from_pretrained(self.model_dir)
        self._model.eval()

        if torch.cuda.is_available():
            self._device = (
                f"cuda:{self.cuda_visible_devices}"
                if self.cuda_visible_devices is not None
                else "cuda"
            )
        else:
            self._device = "cpu"
        self._model = self._model.to(self._device)
        if self.id2label is None:
            self.id2label = self._model.config.id2label
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

    def process(
        self,
        images: List[Image.Image],
        save_visualization: bool = False,
        visualization_output_dir: Optional[str] = None,
        global_start_idx: int = 0,
    ) -> List[List[Dict]]:
        """Batch-detect layout regions in-process.

        Args:
            images: List of PIL Images.
            save_visualization: Whether to also save visualization.
            visualization_output_dir: Where to save visualization outputs.
            global_start_idx: Start index for visualization filenames (layout_page{N}).

        Returns:
            List[List[Dict]]: Detection results per image.
        """
        if self._model is None:
            raise RuntimeError("Layout detector not started. Call start() first.")

        num_images = len(images)
        image_batch = []
        for image in images:
            image_width, image_height = image.size
            image_array = np.array(image.convert("RGB"))
            image_batch.append((image_array, image_width, image_height))

        pil_images = [Image.fromarray(img[0]) for img in image_batch]
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
            try:
                if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
                    pred_boxes = outputs.pred_boxes
                    if hasattr(outputs, "out_masks") and outputs.out_masks is not None:
                        mask_h, mask_w = outputs.out_masks.shape[-2:]
                    else:
                        mask_h, mask_w = 200, 200
                    min_norm_w = 1.0 / mask_w
                    min_norm_h = 1.0 / mask_h
                    box_wh = pred_boxes[..., 2:4]
                    valid_mask = (box_wh[..., 0] > min_norm_w) & (
                        box_wh[..., 1] > min_norm_h
                    )
                    if hasattr(outputs, "logits") and outputs.logits is not None:
                        invalid_mask = ~valid_mask
                        if invalid_mask.any():
                            outputs.logits.masked_fill_(
                                invalid_mask.unsqueeze(-1), -100.0
                            )
            except Exception as e:
                logger.warning("Pre-filter failed (%s), continuing...", e)

            if self.threshold_by_class:
                # Use the lowest threshold (per-class or global fallback)
                # so post-processing doesn't discard valid detections early.
                pre_threshold = min(
                    self.threshold, min(self.threshold_by_class.values())
                )
            else:
                pre_threshold = self.threshold

            raw_results = self._image_processor.post_process_object_detection(
                outputs,
                threshold=pre_threshold,
                target_sizes=target_sizes,
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

        saved_vis_paths = []
        if save_visualization and visualization_output_dir:
            vis_output_path = Path(visualization_output_dir)
            vis_output_path.mkdir(parents=True, exist_ok=True)
            for img_idx, img_results in enumerate(all_paddle_format_results):
                vis_img = np.array(pil_images[img_idx])
                save_filename = f"layout_page{global_start_idx + img_idx}.jpg"
                save_path = vis_output_path / save_filename
                save_layout_visualization(
                    image=vis_img,
                    boxes=img_results,
                    save_path=str(save_path),
                    show_label=True,
                    show_score=True,
                    show_index=True,
                )
                saved_vis_paths.append(str(save_path))

        all_results = []
        for img_idx, paddle_results in enumerate(all_paddle_format_results):
            image_width = image_batch[img_idx][1]
            image_height = image_batch[img_idx][2]
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

        return all_results
