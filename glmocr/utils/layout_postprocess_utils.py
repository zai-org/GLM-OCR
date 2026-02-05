"""Layout post-processing utilities."""

import numpy as np
from typing import List, Dict, Tuple, Union


def iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute the intersection coordinates
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)

    # Compute the area of intersection
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Compute the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    # Compute the IoU
    iou_value = inter_area / float(box1_area + box2_area - inter_area)

    return iou_value


def nms(boxes, iou_same=0.6, iou_diff=0.95):
    """Perform Non-Maximum Suppression (NMS) with different IoU thresholds for same and different classes."""
    # Extract class scores
    scores = boxes[:, 1]

    # Sort indices by scores in descending order
    indices = np.argsort(scores)[::-1]
    selected_boxes = []

    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = current_box[2:]

        selected_boxes.append(current)
        indices = indices[1:]

        filtered_indices = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff

            # If the IoU is below the threshold, keep the box
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    return selected_boxes


def is_contained(box1, box2):
    """Check if box1 is contained within box2."""
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height
    iou = intersect_area / box1_area if box1_area > 0 else 0
    return iou >= 0.8


def check_containment(boxes, preserve_indices=None, category_index=None, mode=None):
    """Check containment relationships among boxes.

    Args:
        boxes: Array of boxes
        preserve_indices: Set of class indices to always preserve (e.g., image, seal, chart)
        category_index: Category index for mode-specific filtering
        mode: Filtering mode ('large' or 'small')
    """
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Skip if box i is in preserve list (should never be marked as contained)
            if preserve_indices is not None and boxes[i][0] in preserve_indices:
                continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    return contains_other, contained_by_other


def unclip_boxes(boxes, unclip_ratio=None):
    """
    Expand bounding boxes from (x1, y1, x2, y2) format using an unclipping ratio.

    Parameters:
    - boxes: np.ndarray of shape (N, 6+), where each row is (cls_id, score, x1, y1, x2, y2, ...).
    - unclip_ratio: tuple of (width_ratio, height_ratio), or dict mapping cls_id to ratio, optional.

    Returns:
    - expanded_boxes: np.ndarray of shape (N, 6+), where each row is (cls_id, score, x1, y1, x2, y2, ...).
    """
    if unclip_ratio is None:
        return boxes

    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box[:6]
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]

                width = x2 - x1
                height = y2 - y1

                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2

                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2

                expanded_box = [class_id, score, new_x1, new_y1, new_x2, new_y2]
                if len(box) > 6:
                    expanded_box.extend(box[6:])
                expanded_boxes.append(expanded_box)
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)

    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]

        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2

        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        expanded_boxes = np.column_stack(
            (boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2)
        )
        if boxes.shape[1] > 6:
            expanded_boxes = np.column_stack((expanded_boxes, boxes[:, 6:]))
        return expanded_boxes


def apply_layout_postprocess(
    raw_results: List[Dict],
    id2label: Dict,
    img_sizes: List[Tuple[int, int]],
    layout_nms: bool = True,
    layout_unclip_ratio: Union[float, Tuple[float, float], Dict] = None,
    layout_merge_bboxes_mode: Union[str, Dict] = None,
) -> List[List[Dict]]:
    """
    Apply layout post-processing to raw detection results.

    Args:
        raw_results: List of dicts from transformers post_process_object_detection
                    Each dict has keys: 'scores', 'labels', 'boxes', 'order_seq', 'polygon_points'
        id2label: Dict mapping class id to label name
        img_sizes: List of (width, height) tuples for each image
        layout_nms: Whether to apply NMS
        layout_unclip_ratio: Unclip ratio for box expansion
        layout_merge_bboxes_mode: Mode for merging nested boxes ('union', 'large', 'small', or dict)

    Returns:
        List of lists, each containing dicts in PaddleOCR format:
        {
            'cls_id': int,
            'label': str,
            'score': float,
            'coordinate': [x1, y1, x2, y2],
            'order': int or None,
            'polygon_points': np.array
        }
    """
    all_labels = list(id2label.values())
    paddle_format_results = []

    for img_idx, result in enumerate(raw_results):
        scores = result["scores"].cpu().numpy()
        labels = result["labels"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()
        order_seq = result["order_seq"].cpu().numpy()
        polygon_points = result.get("polygon_points", [])
        img_size = img_sizes[img_idx]  # (width, height)

        # Build intermediate format: [cls_id, score, x1, y1, x2, y2, order]
        boxes_with_order = []
        for i in range(len(scores)):
            cls_id = int(labels[i])
            score = float(scores[i])
            x1, y1, x2, y2 = boxes[i]
            order = int(order_seq[i])
            boxes_with_order.append([cls_id, score, x1, y1, x2, y2, order])

        if len(boxes_with_order) == 0:
            paddle_format_results.append([])
            continue

        boxes_array = np.array(boxes_with_order)

        # Apply NMS
        if layout_nms:
            selected_indices = nms(boxes_array[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes_array = boxes_array[selected_indices]

        # Filter large images
        filter_large_image = True
        if filter_large_image and len(boxes_array) > 1:
            if img_size[0] > img_size[1]:
                area_thres = 0.82
            else:
                area_thres = 0.93
            image_index = all_labels.index("image") if "image" in all_labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            for box in boxes_array:
                label_index, score, xmin, ymin, xmax, ymax = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                else:
                    filtered_boxes.append(box)
            if len(filtered_boxes) > 0:
                boxes_array = np.array(filtered_boxes)

        # Apply merge_bboxes_mode
        if layout_merge_bboxes_mode:
            # Get indices for labels that should always be preserved
            preserve_labels = ["image", "seal", "chart"]
            preserve_indices = set()
            for label in preserve_labels:
                if label in all_labels:
                    preserve_indices.add(all_labels.index(label))

            if isinstance(layout_merge_bboxes_mode, str):
                assert layout_merge_bboxes_mode in [
                    "union",
                    "large",
                    "small",
                ], f"layout_merge_bboxes_mode must be one of ['union', 'large', 'small'], but got {layout_merge_bboxes_mode}"

                if layout_merge_bboxes_mode == "union":
                    pass
                else:
                    contains_other, contained_by_other = check_containment(
                        boxes_array[:, :6], preserve_indices
                    )
                    if layout_merge_bboxes_mode == "large":
                        boxes_array = boxes_array[contained_by_other == 0]
                    elif layout_merge_bboxes_mode == "small":
                        boxes_array = boxes_array[
                            (contains_other == 0) | (contained_by_other == 1)
                        ]

            elif isinstance(layout_merge_bboxes_mode, dict):
                keep_mask = np.ones(len(boxes_array), dtype=bool)
                for category_index, layout_mode in layout_merge_bboxes_mode.items():
                    assert layout_mode in [
                        "union",
                        "large",
                        "small",
                    ], f"layout_mode must be one of ['union', 'large', 'small'], but got {layout_mode}"

                    if layout_mode == "union":
                        pass
                    else:
                        if layout_mode == "large":
                            contains_other, contained_by_other = check_containment(
                                boxes_array[:, :6],
                                preserve_indices,
                                category_index,
                                mode=layout_mode,
                            )
                            keep_mask &= contained_by_other == 0
                        elif layout_mode == "small":
                            contains_other, contained_by_other = check_containment(
                                boxes_array[:, :6],
                                preserve_indices,
                                category_index,
                                mode=layout_mode,
                            )
                            keep_mask &= (contains_other == 0) | (
                                contained_by_other == 1
                            )
                boxes_array = boxes_array[keep_mask]

        if len(boxes_array) == 0:
            paddle_format_results.append([])
            continue

        # Sort by order
        sorted_idx = np.argsort(boxes_array[:, 6])
        boxes_array = boxes_array[sorted_idx]

        # Apply unclip_ratio
        if layout_unclip_ratio:
            if isinstance(layout_unclip_ratio, float):
                layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
            elif isinstance(layout_unclip_ratio, (tuple, list)):
                assert (
                    len(layout_unclip_ratio) == 2
                ), "layout_unclip_ratio length should be 2"
            elif isinstance(layout_unclip_ratio, dict):
                pass
            else:
                raise ValueError(
                    f"layout_unclip_ratio must be float, tuple, or dict, but got {type(layout_unclip_ratio)}"
                )
            boxes_array = unclip_boxes(boxes_array, layout_unclip_ratio)

        # Convert to PaddleOCR format
        img_width, img_height = img_size
        image_results = []
        for i, box_data in enumerate(boxes_array):
            cls_id = int(box_data[0])
            score = float(box_data[1])
            x1, y1, x2, y2 = box_data[2:6]
            order = int(box_data[6]) if box_data[6] > 0 else None
            label_name = id2label.get(cls_id, f"class_{cls_id}")

            # Clamp bbox to image boundaries
            x1 = max(0, min(float(x1), img_width))
            y1 = max(0, min(float(y1), img_height))
            x2 = max(0, min(float(x2), img_width))
            y2 = max(0, min(float(y2), img_height))

            # Skip invalid bbox
            if x1 >= x2 or y1 >= y2:
                continue

            # Since we may have filtered boxes, we need to match by coordinates
            poly = None
            if len(polygon_points) > 0:
                # Try to find matching polygon by comparing coordinates
                for orig_idx in range(len(boxes)):
                    if np.allclose(boxes[orig_idx], box_data[2:6], atol=1.0):
                        if orig_idx < len(polygon_points):
                            poly = polygon_points[orig_idx].astype(np.float32)
                        break

            if poly is None:
                # Fallback: convert box to 4-point polygon
                poly = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
                )
            else:
                # Clamp polygon points to image boundaries
                poly[:, 0] = np.clip(poly[:, 0], 0, img_width)
                poly[:, 1] = np.clip(poly[:, 1], 0, img_height)

            image_results.append(
                {
                    "cls_id": cls_id,
                    "label": label_name,
                    "score": score,
                    "coordinate": [int(x1), int(y1), int(x2), int(y2)],
                    "order": order,
                    "polygon_points": poly,
                }
            )

        paddle_format_results.append(image_results)

    return paddle_format_results
