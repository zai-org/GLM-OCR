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
        custom_font_path = os.path.join(project_root, "assets", "PingFang.ttf")

        if os.path.exists(custom_font_path):
            return ImageFont.truetype(custom_font_path, font_size, encoding="utf-8")
    except Exception:
        pass

    # Fallback to PIL default font
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_layout_boxes(
    image: np.ndarray,
    boxes: List[Dict],
    show_label: bool = True,
    show_score: bool = True,
    show_index: bool = True,
    thickness_ratio: float = 0.002,
    font_size_ratio: float = 0.018,
) -> Image.Image:
    """Draw layout detection boxes on image with high-quality visualization.

    Args:
        image: Input image as numpy array (RGB format).
        boxes: List of detection boxes, each box is a dict with keys:
            - 'coordinate': [xmin, ymin, xmax, ymax]
            - 'label': string label
            - 'score': confidence score (0-1)
        show_label: Whether to show label text.
        show_score: Whether to show confidence score.
        show_index: Whether to show index number.
        thickness_ratio: Line thickness as ratio of image size.
        font_size_ratio: Font size as ratio of image width.

    Returns:
        PIL Image with boxes drawn.
    """
    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        # Convert from RGB to PIL Image
        img = Image.fromarray(image)
    else:
        img = image

    if len(boxes) == 0:
        return img

    # Calculate font size and thickness based on image size
    img_width, img_height = img.size
    font_size = max(int(font_size_ratio * img_width) + 2, 12)
    draw_thickness = max(int(max(img.size) * thickness_ratio), 2)

    # Get font
    font = get_default_font(font_size)

    # Prepare drawing
    draw = ImageDraw.Draw(img)
    label2color = {}
    label2fontcolor = {}
    color_list = get_colormap(rgb=True)
    num_colors = len(color_list)

    # Draw each box
    for i, box_info in enumerate(boxes):
        label = box_info.get("label", "unknown")
        bbox = box_info.get("coordinate", box_info.get("bbox", None))
        score = box_info.get("score", 1.0)

        if bbox is None:
            continue

        # Assign color to label if not already assigned
        if label not in label2color:
            color_index = len(label2color) % num_colors
            label2color[label] = color_list[color_index]
            label2fontcolor[label] = font_colormap(color_index)

        color = tuple(label2color[label])
        font_color = tuple(label2fontcolor[label])

        # Parse bbox coordinates
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            # Ensure coordinates are within image bounds
            xmin = max(0, min(int(xmin), img_width - 1))
            ymin = max(0, min(int(ymin), img_height - 1))
            xmax = max(0, min(int(xmax), img_width - 1))
            ymax = max(0, min(int(ymax), img_height - 1))

            # Draw rectangle
            rectangle = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
        else:
            raise ValueError(
                f"Only support bbox format of [xmin, ymin, xmax, ymax], "
                f"got bbox of length {len(bbox)}."
            )

        # Draw bbox with specified thickness
        draw.line(rectangle, width=draw_thickness, fill=color)

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
                    # For PIL >= 10.0.0
                    bbox_text = draw.textbbox((0, 0), text, font=font)
                    tw = bbox_text[2] - bbox_text[0]
                    th = bbox_text[3] - bbox_text[1] + 4
                except AttributeError:
                    # For older PIL versions
                    tw, th = draw.textsize(text, font=font)
            else:
                # Rough estimation if font is not available
                tw, th = len(text) * 8, 12

            # Draw label background and text
            if ymin < th:
                # Draw below the top edge if not enough space above
                draw.rectangle(
                    [(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color
                )
                if font is not None:
                    draw.text((xmin + 2, ymin + 2), text, fill=font_color, font=font)
            else:
                # Draw above the bbox
                draw.rectangle(
                    [(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color
                )
                if font is not None:
                    draw.text((xmin + 2, ymin - th), text, fill=font_color, font=font)

        # Draw index number on the right side
        if show_index:
            index_text = str(i + 1)
            text_position = (xmax + 2, ymin - font_size // 2)

            # Adjust position if too close to right edge
            if img_width - xmax < font_size * 1.2:
                text_position = (
                    int(xmax - font_size * 1.1),
                    ymin - font_size // 2,
                )

            if font is not None:
                draw.text(text_position, index_text, font=font, fill="red")

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

    # Save image
    vis_img.save(save_path, quality=95)
