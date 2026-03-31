"""Image asset export utilities.

Optional SDK-owned export of rendered and embedded image assets for layout image
regions. Embedded PDF images are matched geometrically; rendered crops remain the
fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from glmocr.utils.image_utils import crop_image_region, pdf_to_images_pil
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import fitz
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


def export_image_assets(
    json_result: List[list],
    markdown_result: str,
    source: str,
    *,
    enable_image_asset_export: bool,
    markdown_image_preference: str,
    image_match_iou_threshold: float,
    image_match_containment_threshold: float,
    rendered_image_dpi: int,
    rendered_images: Optional[Dict[str, Any]] = None,
) -> Tuple[List[list], str, Dict[str, Any]]:
    """Return updated JSON/Markdown plus exportable image assets.

    When disabled:
    - preserve existing rendered assets if already present (self-hosted)
    - otherwise create rendered crops under ``imgs_rendered/`` (MaaS / bbox-only path)

    When enabled:
    - export rendered crops under ``imgs_rendered/``
    - try to export matched embedded images under ``imgs_embedded/``
    - point JSON/Markdown to the preferred asset
    """
    has_images = any(
        region.get("label") == "image"
        for page in json_result
        if isinstance(page, list)
        for region in page
        if isinstance(region, dict)
    )
    if not has_images:
        return json_result, markdown_result, rendered_images or {}

    if not enable_image_asset_export:
        return _create_rendered_only_assets(
            json_result,
            markdown_result,
            source,
            rendered_image_dpi=rendered_image_dpi,
            rendered_images=rendered_images,
        )

    return _create_preferred_assets(
        json_result,
        markdown_result,
        source,
        markdown_image_preference=markdown_image_preference,
        image_match_iou_threshold=image_match_iou_threshold,
        image_match_containment_threshold=image_match_containment_threshold,
        rendered_image_dpi=rendered_image_dpi,
        rendered_images=rendered_images,
    )


def _create_rendered_only_assets(
    json_result: List[list],
    markdown_result: str,
    source: str,
    *,
    rendered_image_dpi: int,
    rendered_images: Optional[Dict[str, Any]] = None,
) -> Tuple[List[list], str, Dict[str, Any]]:
    if rendered_images:
        json_result, markdown_result, image_files = _preserve_rendered_assets(
            json_result, markdown_result, rendered_images
        )
        if not _has_pending_rendered_assets(json_result):
            return (
                _strip_internal_image_fields(json_result),
                markdown_result,
                image_files,
            )
    else:
        image_files = {}

    loaded_images = _load_render_pages(source, rendered_image_dpi)
    if not loaded_images:
        return (
            _strip_internal_image_fields(_fill_missing_image_asset_fields(json_result)),
            markdown_result,
            image_files,
        )

    image_counter = 0
    updated_json: List[list] = []

    for page_idx, page in enumerate(json_result):
        if not isinstance(page, list):
            updated_json.append(page)
            continue
        page_copy = []
        for region in page:
            if (
                not isinstance(region, dict)
                or region.get("label") != "image"
                or page_idx >= len(loaded_images)
            ):
                page_copy.append(region)
                continue

            region_copy = dict(region)
            region_copy.setdefault("image_path", None)
            region_copy.setdefault("rendered_image_path", None)
            region_copy.setdefault("embedded_image_path", None)
            region_copy.setdefault("image_asset_source", "rendered")
            bbox = region.get("bbox_2d")
            polygon = region.get("polygon")
            previous_image_path = region_copy.get("image_path")
            previous_embedded_path = region_copy.get("embedded_image_path")
            previous_asset_source = region_copy.get("image_asset_source", "rendered")
            if bbox and (
                region_copy.get("_needs_rendered_export")
                or not region_copy.get("rendered_image_path")
            ):
                try:
                    cropped = crop_image_region(loaded_images[page_idx], bbox, polygon)
                    rel_path = region_copy.get("rendered_image_path") or (
                        f"imgs_rendered/rendered_page{page_idx}_idx{region.get('index', image_counter)}.jpg"
                    )
                    image_files[rel_path] = cropped
                    region_copy["image_path"] = rel_path
                    region_copy["rendered_image_path"] = rel_path
                    region_copy["embedded_image_path"] = None
                    region_copy["image_asset_source"] = "rendered"
                    replace_region = dict(region)
                    if region_copy.get("_previous_image_path"):
                        replace_region["image_path"] = region_copy[
                            "_previous_image_path"
                        ]
                    markdown_result = _replace_markdown_image_reference(
                        markdown_result,
                        replace_region,
                        page_idx,
                        bbox,
                        rel_path,
                    )
                    image_counter += 1
                except Exception as e:
                    logger.warning(
                        "Failed to render image asset (page=%d, bbox=%s): %s",
                        page_idx,
                        bbox,
                        e,
                    )
                    region_copy["image_path"] = previous_image_path
                    region_copy["rendered_image_path"] = None
                    region_copy["embedded_image_path"] = previous_embedded_path
                    region_copy["image_asset_source"] = previous_asset_source
            region_copy.pop("_needs_rendered_export", None)
            region_copy.pop("_previous_image_path", None)
            page_copy.append(region_copy)
        updated_json.append(page_copy)

    return _strip_internal_image_fields(updated_json), markdown_result, image_files


def _create_preferred_assets(
    json_result: List[list],
    markdown_result: str,
    source: str,
    *,
    markdown_image_preference: str,
    image_match_iou_threshold: float,
    image_match_containment_threshold: float,
    rendered_image_dpi: int,
    rendered_images: Optional[Dict[str, Any]] = None,
) -> Tuple[List[list], str, Dict[str, Any]]:
    if rendered_images:
        rendered_json, markdown_result, image_files = _preserve_rendered_assets(
            json_result, markdown_result, rendered_images
        )
        if _has_pending_rendered_assets(rendered_json):
            rendered_pages = _load_render_pages(source, rendered_image_dpi)
        else:
            rendered_pages = []
        json_result = rendered_json
    else:
        rendered_pages = _load_render_pages(source, rendered_image_dpi)
        image_files = {}
    embedded_by_page = _inspect_embedded_pdf_images(source)
    updated_json: List[list] = []

    for page_idx, page in enumerate(json_result):
        if not isinstance(page, list):
            updated_json.append(page)
            continue

        image_regions = [
            region
            for region in page
            if isinstance(region, dict) and region.get("label") == "image"
        ]
        matches = _match_embedded_images(
            image_regions,
            embedded_by_page.get(page_idx, []),
            image_match_iou_threshold=image_match_iou_threshold,
            image_match_containment_threshold=image_match_containment_threshold,
        )

        page_copy = []
        for region in page:
            if not isinstance(region, dict) or region.get("label") != "image":
                page_copy.append(region)
                continue

            region_copy = dict(region)
            region_copy.setdefault("image_path", None)
            region_copy.setdefault("rendered_image_path", None)
            region_copy.setdefault("embedded_image_path", None)
            region_copy.setdefault("image_asset_source", "rendered")
            bbox = region.get("bbox_2d")
            polygon = region.get("polygon")
            rendered_rel_path = region_copy.get("rendered_image_path")
            embedded_rel_path = None
            previous_image_path = region_copy.get("image_path")
            render_failed = False
            rendered_asset_available = bool(
                rendered_rel_path and not region_copy.get("_needs_rendered_export")
            )

            if (
                bbox
                and (
                    rendered_rel_path is None
                    or region_copy.get("_needs_rendered_export")
                )
                and page_idx < len(rendered_pages)
            ):
                try:
                    rendered = crop_image_region(
                        rendered_pages[page_idx], bbox, polygon
                    )
                    rendered_rel_path = rendered_rel_path or (
                        f"imgs_rendered/rendered_page{page_idx}_idx{region.get('index', 0)}.jpg"
                    )
                    image_files[rendered_rel_path] = rendered
                    rendered_asset_available = True
                except Exception as e:
                    logger.warning(
                        "Failed to render fallback image asset (page=%d, bbox=%s): %s",
                        page_idx,
                        bbox,
                        e,
                    )
                    render_failed = True
                    rendered_rel_path = None

            match = matches.get(int(region.get("index", 0)))
            if match is not None:
                embedded_rel_path = f"imgs_embedded/embedded_page{page_idx}_idx{region.get('index', 0)}_xref{match['xref']}.{match['ext']}"
                image_files[embedded_rel_path] = match["image_bytes"]

            effective_rendered_path = (
                rendered_rel_path if rendered_asset_available else None
            )
            chosen_path = _choose_preferred_path(
                embedded_rel_path,
                effective_rendered_path,
                markdown_image_preference=markdown_image_preference,
            )

            if render_failed and embedded_rel_path is None:
                chosen_path = None
                effective_rendered_path = None
                original_markdown_path = (
                    region_copy.get("_previous_image_path") or previous_image_path
                )
                if bbox and original_markdown_path:
                    markdown_result = _replace_markdown_image_reference(
                        markdown_result,
                        {
                            "image_path": original_markdown_path,
                            "index": region.get("index", 0),
                        },
                        page_idx,
                        bbox,
                        "",
                    )
            elif (
                region_copy.get("_needs_rendered_export")
                and not rendered_asset_available
                and embedded_rel_path is None
            ):
                chosen_path = None
                original_markdown_path = (
                    region_copy.get("_previous_image_path") or previous_image_path
                )
                if bbox and original_markdown_path:
                    markdown_result = _replace_markdown_image_reference(
                        markdown_result,
                        {
                            "image_path": original_markdown_path,
                            "index": region.get("index", 0),
                        },
                        page_idx,
                        bbox,
                        "",
                    )

            if chosen_path is not None and bbox:
                replace_region = dict(region)
                if region_copy.get("_previous_image_path"):
                    replace_region["image_path"] = region_copy["_previous_image_path"]
                elif previous_image_path:
                    replace_region["image_path"] = previous_image_path
                markdown_result = _replace_markdown_image_reference(
                    markdown_result, replace_region, page_idx, bbox, chosen_path
                )
                region_copy["image_path"] = chosen_path
            if rendered_asset_available and effective_rendered_path is not None:
                region_copy["rendered_image_path"] = effective_rendered_path
            else:
                region_copy["rendered_image_path"] = None
            if render_failed and embedded_rel_path is None:
                rendered_rel_path = None
                region_copy["embedded_image_path"] = None
                region_copy["rendered_image_path"] = None
                region_copy["image_asset_source"] = "rendered"
                region_copy["image_path"] = None
            else:
                region_copy["embedded_image_path"] = embedded_rel_path
                if embedded_rel_path is None and not rendered_asset_available:
                    region_copy["image_asset_source"] = "rendered"
                    region_copy["image_path"] = None
                else:
                    region_copy["image_asset_source"] = (
                        "embedded" if chosen_path == embedded_rel_path else "rendered"
                    )
                if render_failed and embedded_rel_path is not None:
                    region_copy["rendered_image_path"] = None
            region_copy.pop("_needs_rendered_export", None)
            region_copy.pop("_previous_image_path", None)
            page_copy.append(region_copy)

        updated_json.append(page_copy)

    return _strip_internal_image_fields(updated_json), markdown_result, image_files


def _load_render_pages(source: str, rendered_image_dpi: int) -> List[Image.Image]:
    path = Path(source)
    try:
        if path.suffix.lower() == ".pdf" and path.is_file():
            return pdf_to_images_pil(
                str(path),
                dpi=rendered_image_dpi,
                max_width_or_height=6000,
            )
        if path.is_file():
            img = Image.open(str(path))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return [img]
    except Exception as e:
        logger.warning("Cannot load source %s for image asset export: %s", source, e)
    return []


def _inspect_embedded_pdf_images(source: str) -> Dict[int, List[Dict[str, Any]]]:
    if fitz is None:
        return {}
    path = Path(source)
    if path.suffix.lower() != ".pdf" or not path.is_file():
        return {}

    doc = fitz.open(str(path))
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            width = float(page.rect.width) or 1.0
            height = float(page.rect.height) or 1.0
            instances: List[Dict[str, Any]] = []
            for image in page.get_images(full=True):
                xref = int(image[0])
                try:
                    extracted = doc.extract_image(xref)
                    rects = page.get_image_rects(xref, transform=True)
                except Exception:
                    continue
                for placement_idx, placement in enumerate(rects):
                    rect = placement[0] if isinstance(placement, tuple) else placement
                    bbox_norm = [
                        rect.x0 / width,
                        rect.y0 / height,
                        rect.x1 / width,
                        rect.y1 / height,
                    ]
                    instances.append(
                        {
                            "xref": xref,
                            "ext": extracted.get("ext", "bin"),
                            "image_bytes": extracted.get("image", b""),
                            "width": int(extracted.get("width") or image[2] or 0),
                            "height": int(extracted.get("height") or image[3] or 0),
                            "bbox_norm": bbox_norm,
                            "placement_index": placement_idx,
                        }
                    )
            if instances:
                by_page[page_index] = instances
    finally:
        doc.close()
    return by_page


def _match_embedded_images(
    image_regions: List[Dict[str, Any]],
    embedded_instances: List[Dict[str, Any]],
    *,
    image_match_iou_threshold: float,
    image_match_containment_threshold: float,
) -> Dict[int, Dict[str, Any]]:
    candidates: List[Tuple[float, int, int]] = []
    for region in image_regions:
        bbox = region.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue
        region_idx = int(region.get("index", 0))
        region_bbox = [coord / 1000.0 for coord in bbox]
        region_ar = _bbox_aspect_ratio(region_bbox)
        for embedded_idx, embedded in enumerate(embedded_instances):
            embedded_bbox = embedded["bbox_norm"]
            iou = _bbox_iou(region_bbox, embedded_bbox)
            containment = _bbox_containment(region_bbox, embedded_bbox)
            if (
                iou < image_match_iou_threshold
                and containment < image_match_containment_threshold
            ):
                continue
            embedded_ar = _bbox_aspect_ratio(embedded_bbox)
            if not _aspect_ratio_plausible(region_ar, embedded_ar):
                continue
            area_ratio = _bbox_area_ratio(region_bbox, embedded_bbox)
            score = max(iou, containment) + area_ratio * 0.1
            candidates.append((score, region_idx, embedded_idx))

    candidates.sort(reverse=True)
    assigned_regions = set()
    assigned_embedded = set()
    matches: Dict[int, Dict[str, Any]] = {}
    for _, region_idx, embedded_idx in candidates:
        if region_idx in assigned_regions or embedded_idx in assigned_embedded:
            continue
        matches[region_idx] = embedded_instances[embedded_idx]
        assigned_regions.add(region_idx)
        assigned_embedded.add(embedded_idx)
    return matches


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    union = _bbox_area(a) + _bbox_area(b) - inter_area
    return inter_area / union if union > 0 else 0.0


def _bbox_containment(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    return max(inter_area / _bbox_area(a), inter_area / _bbox_area(b))


def _bbox_area(bbox: List[float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _bbox_area_ratio(a: List[float], b: List[float]) -> float:
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def _bbox_aspect_ratio(bbox: List[float]) -> float:
    x0, y0, x1, y1 = bbox
    width = max(1e-9, x1 - x0)
    height = max(1e-9, y1 - y0)
    return width / height


def _aspect_ratio_plausible(a: float, b: float) -> bool:
    ratio = max(a, b) / max(min(a, b), 1e-9)
    return ratio <= 2.0


def _choose_preferred_path(
    embedded_rel_path: Optional[str],
    rendered_rel_path: Optional[str],
    *,
    markdown_image_preference: str,
) -> Optional[str]:
    if markdown_image_preference == "rendered":
        return rendered_rel_path or embedded_rel_path
    if embedded_rel_path:
        return embedded_rel_path
    return rendered_rel_path


def _replace_markdown_image_reference(
    markdown_result: str,
    region: Dict[str, Any],
    page_idx: int,
    bbox: List[int],
    new_path: str,
) -> str:
    old_path = region.get("image_path")
    if old_path:
        if not new_path:
            import re

            pattern = re.compile(rf"!\[[^\]]*\]\({re.escape(old_path)}\)")
            return pattern.sub("", markdown_result, count=1)
        return markdown_result.replace(f"({old_path})", f"({new_path})", 1)
    old_tag = f"![](page={page_idx},bbox={bbox})"
    if not new_path:
        return markdown_result.replace(old_tag, "", 1)
    new_tag = f"![Image {page_idx}-{region.get('index', 0)}]({new_path})"
    return markdown_result.replace(old_tag, new_tag, 1)


def _preserve_rendered_assets(
    json_result: List[list],
    markdown_result: str,
    rendered_images: Dict[str, Any],
) -> Tuple[List[list], str, Dict[str, Any]]:
    updated_json: List[list] = []
    normalized_images: Dict[str, Any] = {}
    for page_idx, page in enumerate(json_result):
        if not isinstance(page, list):
            updated_json.append(page)
            continue
        page_copy = []
        for region in page:
            if not isinstance(region, dict) or region.get("label") != "image":
                page_copy.append(region)
                continue
            region_copy = dict(region)
            image_path = region_copy.get("image_path")
            rendered_path = region_copy.get("rendered_image_path") or image_path
            if rendered_path:
                filename = (
                    rendered_path.split("/", 1)[-1]
                    if isinstance(rendered_path, str)
                    else None
                )
                source_key = None
                if isinstance(rendered_path, str) and rendered_path in rendered_images:
                    source_key = rendered_path
                elif filename and filename in rendered_images:
                    source_key = filename
                if source_key:
                    normalized_images[rendered_path] = rendered_images[source_key]
                    region_copy["image_path"] = rendered_path
                    region_copy["rendered_image_path"] = rendered_path
                    region_copy["embedded_image_path"] = None
                    region_copy["image_asset_source"] = "rendered"
                    bbox = region_copy.get("bbox_2d")
                    if bbox:
                        markdown_result = _replace_markdown_image_reference(
                            markdown_result,
                            region,
                            page_idx,
                            bbox,
                            rendered_path,
                        )
                else:
                    previous_image_path = region_copy.get("image_path")
                    region_copy["image_path"] = rendered_path
                    region_copy["rendered_image_path"] = rendered_path
                    region_copy["embedded_image_path"] = None
                    region_copy["image_asset_source"] = "rendered"
                    region_copy["_needs_rendered_export"] = True
                    region_copy["_previous_image_path"] = previous_image_path
            else:
                region_copy.setdefault("image_path", None)
                region_copy.setdefault("rendered_image_path", None)
                region_copy.setdefault("embedded_image_path", None)
                region_copy.setdefault("image_asset_source", "rendered")
            page_copy.append(region_copy)
        updated_json.append(page_copy)
    return updated_json, markdown_result, normalized_images


def _has_pending_rendered_assets(json_result: List[list]) -> bool:
    for page in json_result:
        if not isinstance(page, list):
            continue
        for region in page:
            if isinstance(region, dict) and region.get("_needs_rendered_export"):
                return True
    return False


def _fill_missing_image_asset_fields(json_result: List[list]) -> List[list]:
    updated_json: List[list] = []
    for page in json_result:
        if not isinstance(page, list):
            updated_json.append(page)
            continue
        page_copy = []
        for region in page:
            if isinstance(region, dict) and region.get("label") == "image":
                region_copy = dict(region)
                region_copy.setdefault("image_path", None)
                region_copy.setdefault("rendered_image_path", None)
                region_copy.setdefault("embedded_image_path", None)
                region_copy.setdefault("image_asset_source", "rendered")
                page_copy.append(region_copy)
            else:
                page_copy.append(region)
        updated_json.append(page_copy)
    return updated_json


def _strip_internal_image_fields(json_result: List[list]) -> List[list]:
    updated_json: List[list] = []
    for page in json_result:
        if not isinstance(page, list):
            updated_json.append(page)
            continue
        page_copy = []
        for region in page:
            if isinstance(region, dict):
                region_copy = dict(region)
                region_copy.pop("_needs_rendered_export", None)
                region_copy.pop("_previous_image_path", None)
                page_copy.append(region_copy)
            else:
                page_copy.append(region)
        updated_json.append(page_copy)
    return updated_json
