"""Visualization utilities for layout detection and other tasks."""

from typing import List, Dict, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def get_colormap(rgb: bool = True) -> List[Tuple[int, int, int]]:
    """Get colormap for visualization.

    Args:
        rgb: If True, return RGB colors, otherwise return BGR colors.

    Returns:
        List of RGB or BGR color tuples.
    """
    # color palette - carefully selected for visual distinction
    color_list = np.array(
        [
            0xFF,
            0x00,
            0x00,  # Red
            0xCC,
            0xFF,
            0x00,  # Yellow-green
            0x00,
            0xFF,
            0x66,  # Spring green
            0x00,
            0x66,
            0xFF,  # Blue
            0xCC,
            0x00,
            0xFF,  # Purple
            0xFF,
            0x4D,
            0x00,  # Orange
            0x80,
            0xFF,
            0x00,  # Lime
            0x00,
            0xFF,
            0xB2,  # Turquoise
            0x00,
            0x1A,
            0xFF,  # Deep blue
            0xFF,
            0x00,
            0xE5,  # Magenta
            0xFF,
            0x99,
            0x00,  # Orange
            0x33,
            0xFF,
            0x00,  # Green
            0x00,
            0xFF,
            0xFF,  # Cyan
            0x33,
            0x00,
            0xFF,  # Indigo
            0xFF,
            0x00,
            0x99,  # Pink
            0xFF,
            0xE5,
            0x00,  # Yellow
            0x00,
            0xFF,
            0x1A,  # Bright green
            0x00,
            0xB2,
            0xFF,  # Sky blue
            0x80,
            0x00,
            0xFF,  # Violet
            0xFF,
            0x00,
            0x4D,  # Deep pink
        ],
        dtype=np.float32,
    )

    color_list = color_list.reshape((-1, 3))

    if not rgb:
        # Convert RGB to BGR
        color_list = color_list[:, ::-1]

    # Convert to list of tuples
    colormap = [tuple(map(int, color)) for color in color_list]
    return colormap


def font_colormap(color_index: int) -> Tuple[int, int, int]:
    """Get font color based on background color index.

    Args:
        color_index: Index of the background color.

    Returns:
        RGB color tuple for font.
    """
    # Dark color for text
    dark = (0x14, 0x0E, 0x35)  # Dark purple-blue
    # Light color for text
    light = (0xFF, 0xFF, 0xFF)  # White

    # Indices where light background colors require light text
    light_indices = [0, 3, 4, 8, 9, 13, 14, 18, 19]

    if color_index in light_indices:
        return light
    else:
        return dark


def get_default_font(font_size: int = 20) -> ImageFont.FreeTypeFont:
    """Get default font for text rendering.

    Args:
        font_size: Size of the font.

    Returns:
        ImageFont object.
    """
    try:
        # Get the path to the assets folder relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        custom_font_path = os.path.join(project_root, "resources", "PingFang.ttf")

        if os.path.exists(custom_font_path):
            return ImageFont.truetype(custom_font_path, font_size, encoding="utf-8")
    except Exception:
        pass

    # Fallback to PIL default font
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_polygon_masks(
    image: np.ndarray,
    boxes: List[Dict],
    label2color: dict,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw polygon masks on image with alpha blending.

    Args:
        image: Input image as numpy array (RGB format).
        boxes: List of detection boxes with polygon_points.
        label2color: Dictionary mapping labels to colors.
        alpha: Alpha value for blending (0-1).

    Returns:
        Image with masks drawn as numpy array.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, box_info in enumerate(boxes):
        polygon_points = box_info.get("polygon_points")
        if polygon_points is None or len(polygon_points) < 3:
            continue

        # Use the same color as label text
        label = box_info.get("label", "unknown")
        if label in label2color:
            color_mask = tuple(label2color[label])
        else:
            continue  # Skip if no color assigned

        draw.polygon(
            [(int(x), int(y)) for x, y in polygon_points],
            fill=color_mask + (int(alpha * 255),),
        )

    return np.array(Image.alpha_composite(base, overlay).convert("RGB"))


def draw_layout_boxes(
    image: np.ndarray,
    boxes: List[Dict],
    show_label: bool = True,
    show_score: bool = True,
    show_index: bool = True,
    thickness_ratio: float = 0.002,
    font_size_ratio: float = 0.018,
    use_polygon: bool = True,
    alpha: float = 0.5,
) -> Image.Image:
    """Draw layout detection boxes on image with high-quality visualization.

    Args:
        image: Input image as numpy array (RGB format).
        boxes: List of detection boxes, each box is a dict with keys:
            - 'coordinate': [xmin, ymin, xmax, ymax]
            - 'label': string label
            - 'score': confidence score (0-1)
            - 'polygon_points': optional list of [x, y] coordinates
            - 'order': optional reading order
        show_label: Whether to show label text.
        show_score: Whether to show confidence score.
        show_index: Whether to show index number.
        thickness_ratio: Line thickness as ratio of image size.
        font_size_ratio: Font size as ratio of image width.
        use_polygon: Whether to use polygon visualization when available.
        alpha: Alpha value for polygon mask blending (0-1).

    Returns:
        PIL Image with boxes drawn.
    """
    # Check if any box has polygon_points
    has_polygon = use_polygon and any(
        len(box_info.get("polygon_points", [])) >= 3 for box_info in boxes
    )

    # Convert to numpy array if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()

    if len(boxes) == 0:
        return Image.fromarray(img_array)

    color_list = get_colormap(rgb=True)
    num_colors = len(color_list)
    label2color = {}
    label2fontcolor = {}

    for i, box_info in enumerate(boxes):
        label = box_info.get("label", "unknown")
        if label not in label2color:
            color_index = i % num_colors
            label2color[label] = color_list[color_index]
            label2fontcolor[label] = font_colormap(color_index)

    # Draw polygon masks first if available
    if has_polygon:
        img_array = _draw_polygon_masks(img_array, boxes, label2color, alpha)

    # Convert to PIL Image for drawing text
    img = Image.fromarray(img_array)

    # Calculate font size and thickness based on image size
    img_width, img_height = img.size
    font_size = max(int(font_size_ratio * img_width) + 2, 12)
    draw_thickness = max(int(max(img.size) * thickness_ratio), 2)

    # Get font
    font = get_default_font(font_size)

    # Prepare drawing
    draw = ImageDraw.Draw(img)

    # Draw each box
    for i, box_info in enumerate(boxes):
        label = box_info.get("label", "unknown")
        bbox = box_info.get("coordinate", box_info.get("bbox", None))
        score = box_info.get("score", 1.0)
        polygon_points = box_info.get("polygon_points")

        if bbox is None:
            continue

        # Get color from pre-built mapping
        color = tuple(label2color.get(label, color_list[0]))
        font_color = tuple(label2fontcolor.get(label, (0, 0, 0)))

        # Parse bbox coordinates
        xmin, ymin, xmax, ymax = bbox[:4]
        xmin = max(0, min(int(xmin), img_width - 1))
        ymin = max(0, min(int(ymin), img_height - 1))
        xmax = max(0, min(int(xmax), img_width - 1))
        ymax = max(0, min(int(ymax), img_height - 1))

        # Only draw bbox outline if not using polygon masks
        if not has_polygon:
            rectangle = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
            draw.line(rectangle, width=draw_thickness, fill=color)

        # Determine text anchor position
        if has_polygon and polygon_points is not None and len(polygon_points) >= 3:
            # Find left-top and right-top points of polygon
            image_left_top = (0, 0)
            image_right_top = (img_width, 0)
            left_top = min(
                polygon_points,
                key=lambda p: (p[0] - image_left_top[0]) ** 2
                + (p[1] - image_left_top[1]) ** 2,
            )
            right_top = min(
                polygon_points,
                key=lambda p: (p[0] - image_right_top[0]) ** 2
                + (p[1] - image_right_top[1]) ** 2,
            )
            lx, ly = int(left_top[0]), int(left_top[1])
            rx, ry = int(right_top[0]), int(right_top[1])
        else:
            lx, ly = xmin, ymin
            rx, ry = xmax, ymin

        # Prepare label text
        text_parts = []
        if show_label:
            text_parts.append(label)
        if show_score:
            text_parts.append(f"{score:.2f}")

        if text_parts:
            text = " ".join(text_parts)

            # Calculate text size
            if font is not None:
                try:
                    bbox_text = draw.textbbox((0, 0), text, font=font)
                    tw = bbox_text[2] - bbox_text[0]
                    th = bbox_text[3] - bbox_text[1] + 4
                except AttributeError:
                    tw, th = draw.textsize(text, font=font)
            else:
                tw, th = len(text) * 8, 12

            # Draw label background and text
            if ly < th:
                draw.rectangle([(lx, ly), (lx + tw + 4, ly + th + 1)], fill=color)
                if font is not None:
                    draw.text((lx + 2, ly + 2), text, fill=font_color, font=font)
            else:
                draw.rectangle([(lx, ly - th), (lx + tw + 4, ly + 1)], fill=color)
                if font is not None:
                    draw.text((lx + 2, ly - th - 2), text, fill=font_color, font=font)

        # Draw order number on the right side
        if show_index:
            order_text = str(i + 1)
            text_position = (rx + 2, ry - font_size // 2)

            if img_width - rx < font_size * 1.2:
                text_position = (
                    int(rx - font_size * 1.1),
                    ry - font_size // 2,
                )

            if font is not None:
                draw.text(text_position, order_text, font=font, fill="red")

    return img


def save_layout_visualization(
    image: np.ndarray, boxes: List[Dict], save_path: str, **kwargs
) -> None:
    """Draw and save layout visualization.

    Args:
        image: Input image as numpy array (RGB format).
        boxes: List of detection boxes.
        save_path: Path to save the visualization.
        **kwargs: Additional arguments passed to draw_layout_boxes.
    """
    vis_img = draw_layout_boxes(image, boxes, **kwargs)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Save image (flatten alpha so RGBA inputs can be written as JPEG)
    from glmocr.parser_result.base import flatten_to_rgb

    flatten_to_rgb(vis_img).save(save_path, quality=95)
