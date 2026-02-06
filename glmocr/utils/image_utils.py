"""Image processing utilities."""

import io
import cv2
import base64
import math
from io import BytesIO

import numpy as np
from PIL import Image


def smart_resize(
    t: int,
    h: int,
    w: int,
    t_factor: int = 1,
    h_factor: int = 28,
    w_factor: int = 28,
    min_pixels: int = 112 * 112,
    max_pixels: int = 14 * 14 * 4 * 15000,
):
    """
    Smart resize for images.

    Ensures:
    1. Height and width are divisible by the given factors
    2. Total pixels are within [min_pixels, max_pixels]
    3. Keeps aspect ratio as much as possible

    Args:
        t: Temporal dimension.
        h: Height.
        w: Width.
        t_factor: Temporal factor.
        h_factor: Height factor.
        w_factor: Width factor.
        min_pixels: Minimum pixels.
        max_pixels: Maximum pixels.

    Returns:
        (new_h, new_w)
    """
    assert t >= t_factor, "Temporal dimension must be greater than the factor."

    h_bar = round(h / h_factor) * h_factor
    w_bar = round(w / w_factor) * w_factor
    t_bar = round(t / t_factor) * t_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((t * h * w) / max_pixels)
        h_bar = math.floor(h / beta / h_factor) * h_factor
        w_bar = math.floor(w / beta / w_factor) * w_factor
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (t * h * w))
        h_bar = math.ceil(h * beta / h_factor) * h_factor
        w_bar = math.ceil(w * beta / w_factor) * w_factor

    return h_bar, w_bar


def load_image_to_base64(
    image_source,
    t_patch_size: int,
    max_pixels: int,
    image_format: str,
    patch_expand_factor: int = 1,
    min_pixels: int = 112 * 112,
):
    """Load an image and convert it to base64.

    Supported inputs:
    - PIL.Image.Image
    - Local file path (str)
    - data:image/... URL (str)
    - <|base64|>... blob (str)
    - <|tarpath|>... blob (str)
    - Raw bytes (bytes)

    Args:
        image_source: Image source.
        t_patch_size: Temporal patch size.
        max_pixels: Max pixels.
        image_format: Image format.
        patch_expand_factor: Patch expand factor.
        min_pixels: Min pixels.

    Returns:
        Base64-encoded image content.
    """
    import os

    def _try_decode_base64_to_image_bytes(s: str) -> bytes | None:
        # Remove whitespace/newlines and pad for base64.
        candidate = "".join(str(s).split())
        if len(candidate) < 32:
            return None

        # Strip optional "<|base64|>" prefix.
        if candidate.startswith("<|base64|>"):
            candidate = candidate[len("<|base64|>") :]

        # If it looks like a filename (has a short extension), skip.
        if "." in candidate and len(candidate.rsplit(".", 1)[-1]) <= 5:
            return None

        pad = (-len(candidate)) % 4
        if pad:
            candidate = candidate + ("=" * pad)

        try:
            return base64.b64decode(candidate, validate=True)
        except Exception:
            return None

    # Handle different input types
    if isinstance(image_source, Image.Image):
        # Already a PIL Image
        image = image_source
    elif isinstance(image_source, bytes):
        # Raw bytes
        image = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, str):
        if image_source.startswith("file://"):
            image_source = image_source[7:]

        if os.path.isfile(image_source):
            # Local file path (PDFs are handled via PageLoader)
            with open(image_source, "rb") as f:
                image_data = f.read()
            image = Image.open(io.BytesIO(image_data))
        elif image_source.startswith("data:image/"):
            # data:image/... URL
            image_data = base64.b64decode(image_source.split(",")[1])
            image = Image.open(io.BytesIO(image_data))
        else:
            # Raw base64 payload or <|base64|> blob
            decoded = _try_decode_base64_to_image_bytes(image_source)
            if decoded is None:
                raise ValueError(f"Invalid image source: {image_source}")
            image = Image.open(io.BytesIO(decoded))
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Original size
    w, h = image.size

    # Compute new size
    h_bar, w_bar = smart_resize(
        t=t_patch_size,
        h=h,
        w=w,
        t_factor=t_patch_size,
        h_factor=14 * 2 * patch_expand_factor,
        w_factor=14 * 2 * patch_expand_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Resize
    image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

    # Encode as bytes
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    buffered.seek(0)
    image_data = buffered.getvalue()

    # Convert bytes to base64
    base64_encoded_data = base64.b64encode(image_data)
    image_base64 = base64_encoded_data.decode("utf-8")

    return image_base64


def crop_image_region(image, bbox_2d, polygon=None, fill_color=255):
    """Crop an image region using bbox and optionally mask outside polygon.

    Args:
        image: PIL Image
        bbox_2d: [x1_norm, y1_norm, x2_norm, y2_norm] normalized to 0-1000
        polygon: List of [x, y] coordinates, normalized to 0-1000 (optional)
                 Example: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
                 If None or invalid, only bbox crop is performed.
        fill_color: Color to fill outside polygon (default 255 for white)

    Returns:
        PIL.Image.Image: Cropped region with optional polygon mask applied
    """
    image_width, image_height = image.size

    # De-normalize bbox to pixel coordinates
    x1_norm, y1_norm, x2_norm, y2_norm = bbox_2d
    x1 = int(x1_norm * image_width / 1000)
    y1 = int(y1_norm * image_height / 1000)
    x2 = int(x2_norm * image_width / 1000)
    y2 = int(y2_norm * image_height / 1000)

    # Simple bbox crop if polygon is invalid
    if not polygon or len(polygon) < 3:
        return image.crop((x1, y1, x2, y2))

    # Convert to numpy array once
    img_array = np.asarray(image)

    # Crop the bbox region
    img_crop = img_array[y1:y2, x1:x2]
    crop_height, crop_width = img_crop.shape[:2]

    # Pre-compute polygon coordinates
    scale_x = image_width / 1000
    scale_y = image_height / 1000
    polygon_pixels = np.empty((len(polygon), 2), dtype=np.int32)
    for i, point in enumerate(polygon):
        polygon_pixels[i, 0] = int(point[0] * scale_x) - x1
        polygon_pixels[i, 1] = int(point[1] * scale_y) - y1

    # Create mask
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pixels], 1)

    # Create output image with fill_color
    if len(img_crop.shape) == 3:
        output = np.full_like(img_crop, fill_color, dtype=np.uint8)
    else:
        output = np.full((crop_height, crop_width), fill_color, dtype=np.uint8)
    cv2.copyTo(img_crop, mask, output)

    # Convert back to PIL Image
    return Image.fromarray(output)


def image_tensor_to_base64(image_tensor, image_format):
    """Convert a torch image tensor to base64.

    Args:
        image_tensor: torch.Tensor, shape (C, H, W)
        image_format: Image format.

    Returns:
        Base64-encoded image.
    """

    if image_tensor.shape[0] != 3:
        raise ValueError("Input tensor is not a 3-channel image.")
    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = image_array.astype(np.uint8)
    image = Image.fromarray(image_array)
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# -----------------------------------------------------------------------------
# PDF rendering via pypdfium2
# -----------------------------------------------------------------------------

try:
    import pypdfium2 as _pdfium  # noqa: F401

    PYPDFIUM2_AVAILABLE = True
except ImportError:
    PYPDFIUM2_AVAILABLE = False


def _page_to_image(page, dpi: int = 200, max_width_or_height: int = 3500):
    """Render a PDF page to PIL Image (pypdfium2).

    Args:
        page: pypdfium2 PdfPage.
        dpi: Render DPI.
        max_width_or_height: Max width or height.

    Returns:
        (PIL.Image, scale_factor)
    """
    scale = dpi / 72.0
    width, height = page.get_size()
    long_side_length = max(width, height)
    if (long_side_length * scale) > max_width_or_height:
        scale = max_width_or_height / long_side_length
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil()
    try:
        bitmap.close()
    except Exception:
        pass
    return image, scale


def pdf_to_images_pil(
    pdf_path: str,
    dpi: int = 200,
    max_width_or_height: int = 3500,
    start_page_id: int = 0,
    end_page_id: int = None,
) -> list:
    """Convert PDF to list of PIL Images using pypdfium2 (single-process).

    Args:
        pdf_path: PDF file path.
        dpi: Render DPI.
        max_width_or_height: Max width or height.
        start_page_id: Start page index (0-based).
        end_page_id: End page index (inclusive); None = last page.

    Returns:
        List of PIL.Image.
    """
    if not PYPDFIUM2_AVAILABLE:
        raise ImportError(
            "PDF support requires pypdfium2. Install with: pip install pypdfium2"
        )
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(pdf_path)
        page_count = len(pdf)
        if end_page_id is None or end_page_id < 0:
            end_page_id = page_count - 1
        if end_page_id >= page_count:
            end_page_id = page_count - 1
        images = []
        for i in range(start_page_id, end_page_id + 1):
            page = pdf[i]
            try:
                image, _ = _page_to_image(
                    page, dpi=dpi, max_width_or_height=max_width_or_height
                )
                images.append(image)
            finally:
                page.close()
        return images
    finally:
        pdf.close()


def pdf_to_images_pil_iter(
    pdf_path: str,
    dpi: int = 200,
    max_width_or_height: int = 3500,
    start_page_id: int = 0,
    end_page_id: int = None,
):
    """Convert PDF to PIL Images one page at a time (generator).

    Use for streaming: each page is rendered and yielded immediately so
    downstream can start processing before the whole PDF is loaded.

    Args:
        pdf_path: PDF file path.
        dpi: Render DPI.
        max_width_or_height: Max width or height.
        start_page_id: Start page index (0-based).
        end_page_id: End page index (inclusive); None = last page.

    Yields:
        PIL.Image per page.
    """
    if not PYPDFIUM2_AVAILABLE:
        raise ImportError(
            "PDF support requires pypdfium2. Install with: pip install pypdfium2"
        )
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(pdf_path)
        page_count = len(pdf)
        if end_page_id is None or end_page_id < 0:
            end_page_id = page_count - 1
        if end_page_id >= page_count:
            end_page_id = page_count - 1
        for i in range(start_page_id, end_page_id + 1):
            page = pdf[i]
            try:
                image, _ = _page_to_image(
                    page, dpi=dpi, max_width_or_height=max_width_or_height
                )
                yield image
            finally:
                page.close()
    finally:
        pdf.close()
