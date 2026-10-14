"""Thread workers for the three-stage async pipeline.

Stage 1 (data_loading_worker):   Load pages from URLs → page_queue
Stage 2 (layout_worker):         Layout detection     → region_queue
Stage 3 (recognition_worker):    Parallel OCR         → recognition_results

Queue message formats
---------------------
page_queue::

    {"identifier": "image",     "page_idx": int, "unit_idx": int,
     "image": PIL.Image}
    {"identifier": "unit_done", "unit_idx": int}
    {"identifier": "done"}

region_queue::

    {"identifier": "region", "page_idx": int, "cropped_image": PIL.Image,
     "region": dict}
    {"identifier": "done"}
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from glmocr.pipeline._common import (
    IDENTIFIER_DONE,
    IDENTIFIER_IMAGE,
    IDENTIFIER_REGION,
    IDENTIFIER_UNIT_DONE,
)
from glmocr.pipeline._state import PipelineState
from glmocr.utils.image_utils import crop_image_region
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.dataloader import PageLoader
    from glmocr.layout.base import BaseLayoutDetector

logger = get_logger(__name__)


# ======================================================================
# Stage 1: Data Loading
# ======================================================================


def data_loading_worker(
    state: PipelineState,
    page_loader: "PageLoader",
    image_sources: List[Any],
) -> None:
    """Load pages from *image_sources* and push them onto ``state.page_queue``.

    *image_sources* may contain file paths (str), ``file://`` URLs, or raw
    ``bytes`` (image / PDF content).

    For each page that is loaded, ``state.register_page()`` is called
    **before** the page message is enqueued, so that the tracker's
    ``page → unit`` mapping is always available by the time Stage 3 calls
    ``on_region_done``.

    After all pages for each input URL (unit) have been enqueued, a
    ``IDENTIFIER_UNIT_DONE`` sentinel is sent so that the layout worker can
    call ``state.finalize_unit()`` without waiting for *all* units to finish.

    Units that produce zero pages (e.g. broken URLs) still receive a
    ``UNIT_DONE`` sentinel so the tracker can finalise them with
    ``region_count=0``.
    """
    num_units = len(image_sources)
    page_idx = 0
    unit_indices_list: List[int] = []
    prev_unit_idx: Optional[int] = None
    sent_unit_done: set = set()
    try:
        for page, unit_idx in page_loader.iter_pages_with_unit_indices(image_sources):
            if state.is_shutdown:
                break

            if prev_unit_idx is not None and unit_idx != prev_unit_idx:
                if not state.safe_put(
                    state.page_queue,
                    {
                        "identifier": IDENTIFIER_UNIT_DONE,
                        "unit_idx": prev_unit_idx,
                    },
                ):
                    break
                sent_unit_done.add(prev_unit_idx)

            state.register_page(page_idx, unit_idx)
            state.images_dict[page_idx] = page
            if not state.safe_put(
                state.page_queue,
                {
                    "identifier": IDENTIFIER_IMAGE,
                    "page_idx": page_idx,
                    "unit_idx": unit_idx,
                    "image": page,
                },
            ):
                break
            unit_indices_list.append(unit_idx)
            page_idx += 1
            state.num_images_loaded[0] = page_idx
            state.unit_indices_holder[0] = list(unit_indices_list)
            prev_unit_idx = unit_idx

        if not state.is_shutdown:
            if prev_unit_idx is not None:
                state.safe_put(
                    state.page_queue,
                    {
                        "identifier": IDENTIFIER_UNIT_DONE,
                        "unit_idx": prev_unit_idx,
                    },
                )
                sent_unit_done.add(prev_unit_idx)

            for u in range(num_units):
                if u not in sent_unit_done:
                    state.safe_put(
                        state.page_queue,
                        {
                            "identifier": IDENTIFIER_UNIT_DONE,
                            "unit_idx": u,
                        },
                    )

            state.safe_put(state.page_queue, {"identifier": IDENTIFIER_DONE})
    except Exception as e:
        logger.exception("Data loading worker error: %s", e)
        state.num_images_loaded[0] = page_idx
        state.unit_indices_holder[0] = list(unit_indices_list)
        state.record_exception("DataLoadingWorker", e)


# ======================================================================
# Stage 2: Layout Detection
# ======================================================================


def layout_worker(
    state: PipelineState,
    layout_detector: Optional["BaseLayoutDetector"],
    save_visualization: bool,
    use_polygon: bool = False,
    enable_layout: bool = True,
    direct_task_type: str = "text",
) -> None:
    """Consume pages, run layout detection in batches, push regions.

    When a ``IDENTIFIER_UNIT_DONE`` sentinel arrives from Stage 1, the
    current batch is flushed immediately (it contains the last pages for
    that unit) and ``state.finalize_unit()`` is called with the total
    region count for that unit.  This lets the main thread start emitting
    results for the completed unit without waiting for later units.
    """
    try:
        batch_images: List[Any] = []
        batch_page_indices: List[int] = []
        batch_unit_indices: List[int] = []
        global_start_idx = 0

        unit_page_indices: Dict[int, List[int]] = {}

        while True:
            if state.is_shutdown:
                break

            try:
                msg = state.page_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            identifier = msg["identifier"]

            if identifier == IDENTIFIER_IMAGE:
                unit_idx = msg["unit_idx"]
                batch_images.append(msg["image"])
                batch_page_indices.append(msg["page_idx"])
                batch_unit_indices.append(unit_idx)
                if unit_idx not in unit_page_indices:
                    unit_page_indices[unit_idx] = []
                unit_page_indices[unit_idx].append(msg["page_idx"])

                batch_size = layout_detector.batch_size if enable_layout else 1
                if len(batch_images) >= batch_size:
                    _flush_layout_batch(
                        state,
                        layout_detector,
                        batch_images,
                        batch_page_indices,
                        save_visualization,
                        global_start_idx,
                        use_polygon=use_polygon,
                        enable_layout=enable_layout,
                        direct_task_type=direct_task_type,
                    )
                    global_start_idx += len(batch_page_indices)
                    for pi in batch_page_indices:
                        state.images_dict.pop(pi, None)
                    batch_images, batch_page_indices, batch_unit_indices = [], [], []

            elif identifier == IDENTIFIER_UNIT_DONE:
                unit_idx = msg["unit_idx"]
                if batch_images:
                    _flush_layout_batch(
                        state,
                        layout_detector,
                        batch_images,
                        batch_page_indices,
                        save_visualization,
                        global_start_idx,
                        use_polygon=use_polygon,
                        enable_layout=enable_layout,
                        direct_task_type=direct_task_type,
                    )
                    global_start_idx += len(batch_page_indices)
                    for pi in batch_page_indices:
                        state.images_dict.pop(pi, None)
                    batch_images, batch_page_indices, batch_unit_indices = [], [], []

                pages_for_unit = unit_page_indices.get(unit_idx, [])
                region_count = sum(
                    len(state.layout_results_dict.get(pi, [])) for pi in pages_for_unit
                )
                state.finalize_unit(unit_idx, region_count)
                logger.debug(
                    "Unit %d finalised: %d pages, %d regions",
                    unit_idx,
                    len(pages_for_unit),
                    region_count,
                )

            elif identifier == IDENTIFIER_DONE:
                if batch_images:
                    _flush_layout_batch(
                        state,
                        layout_detector,
                        batch_images,
                        batch_page_indices,
                        save_visualization,
                        global_start_idx,
                        use_polygon=use_polygon,
                        enable_layout=enable_layout,
                        direct_task_type=direct_task_type,
                    )
                state.safe_put(state.region_queue, {"identifier": IDENTIFIER_DONE})
                break

    except Exception as e:
        logger.exception("Layout worker error: %s", e)
        state.record_exception("LayoutWorker", e)
    finally:
        state.drain_queue(state.page_queue)


def _flush_layout_batch(
    state: PipelineState,
    layout_detector: Optional["BaseLayoutDetector"],
    batch_images: List[Any],
    batch_page_indices: List[int],
    save_visualization: bool,
    global_start_idx: int,
    use_polygon: bool = False,
    enable_layout: bool = True,
    direct_task_type: str = "text",
) -> None:
    """Run layout detection on one batch and enqueue the resulting regions."""
    if not enable_layout:
        _flush_direct_ocr_batch(
            state,
            batch_images,
            batch_page_indices,
            global_start_idx,
            direct_task_type,
        )
        return

    if layout_detector is None:
        raise RuntimeError("layout_detector is required when enable_layout=True")

    try:
        layout_results, vis_images = layout_detector.process(
            batch_images,
            save_visualization=save_visualization,
            global_start_idx=global_start_idx,
            use_polygon=use_polygon,
        )
        if vis_images:
            state.layout_vis_images.update(vis_images)
    except Exception as e:
        logger.warning(
            "Layout detection failed for pages %s, skipping batch: %s",
            batch_page_indices,
            e,
        )
        for page_idx in batch_page_indices:
            state.layout_results_dict[page_idx] = []
        return

    for page_idx, image, layout_result in zip(
        batch_page_indices, batch_images, layout_results
    ):
        state.layout_results_dict[page_idx] = layout_result
        for region in layout_result:
            try:
                polygon = region.get("polygon") if use_polygon else None
                cropped = crop_image_region(image, region["bbox_2d"], polygon)
            except Exception as e:
                logger.warning(
                    "Failed to crop region on page %d (bbox=%s), skipping: %s",
                    page_idx,
                    region.get("bbox_2d"),
                    e,
                )
                region["content"] = ""
                state.add_recognition_result(page_idx, region)
                continue
            if not state.safe_put(
                state.region_queue,
                {
                    "identifier": IDENTIFIER_REGION,
                    "page_idx": page_idx,
                    "cropped_image": cropped,
                    "region": region,
                },
            ):
                return


def _flush_direct_ocr_batch(
    state: PipelineState,
    batch_images: List[Any],
    batch_page_indices: List[int],
    global_start_idx: int,
    task_type: str,
) -> None:
    """Treat each page as one caller-specified OCR region."""
    for offset, (page_idx, image) in enumerate(zip(batch_page_indices, batch_images)):
        region = {
            "index": global_start_idx + offset,
            "label": task_type,
            "score": 1.0,
            "bbox_2d": [0, 0, 1000, 1000],
            "polygon": None,
            "task_type": task_type,
        }
        state.layout_results_dict[page_idx] = [region]
        if not state.safe_put(
            state.region_queue,
            {
                "identifier": IDENTIFIER_REGION,
                "page_idx": page_idx,
                "cropped_image": image,
                "region": region,
            },
        ):
            return


# ======================================================================
# Stage 3: VLM Recognition
# ======================================================================


def recognition_worker(
    state: PipelineState,
    page_loader: "PageLoader",
    ocr_client: Any,
    max_workers: int,
) -> None:
    """Consume regions, run parallel OCR, store results."""
    executor = None
    try:
        concurrency = min(max_workers, 128)
        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures: Dict[Any, Dict[str, Any]] = {}
        processing_complete = False

        while True:
            if state.is_shutdown:
                break

            _collect_done_futures(futures, state)

            if len(futures) >= concurrency:
                _wait_for_any(futures)
                _collect_done_futures(futures, state)
                continue

            try:
                msg = state.region_queue.get(timeout=0.01)
            except queue.Empty:
                if processing_complete and not futures:
                    break
                if futures:
                    _wait_for_any(futures)
                continue

            identifier = msg["identifier"]

            if identifier == IDENTIFIER_REGION:
                if msg["region"]["task_type"] == "skip":
                    msg["region"]["content"] = None
                    bbox = msg["region"].get("bbox_2d")
                    if bbox and "cropped_image" in msg:
                        state.store_cropped_image(
                            msg["page_idx"], bbox, msg["cropped_image"]
                        )
                    state.add_recognition_result(msg["page_idx"], msg["region"])
                else:
                    req = page_loader.build_request_from_image(
                        msg["cropped_image"],
                        msg["region"]["task_type"],
                    )
                    del msg["cropped_image"]
                    future = executor.submit(ocr_client.process, req)
                    futures[future] = msg

            elif identifier == IDENTIFIER_DONE:
                processing_complete = True

        if not state.is_shutdown:
            for future in as_completed(futures.keys()):
                _handle_future_result(future, futures, state)
            executor.shutdown(wait=True)
        else:
            for f in list(futures):
                f.cancel()
            executor.shutdown(wait=False)

    except Exception as e:
        logger.exception("Recognition worker error: %s", e)
        state.record_exception("RecognitionWorker", e)
        if executor is not None:
            executor.shutdown(wait=False)
    finally:
        state.drain_queue(state.region_queue)


# ------------------------------------------------------------------
# Recognition helpers
# ------------------------------------------------------------------


def _collect_done_futures(
    futures: Dict[Any, Dict[str, Any]],
    state: PipelineState,
) -> None:
    for f in list(futures):
        if f.done():
            _handle_future_result(f, futures, state)


def _handle_future_result(
    future: Any,
    futures: Dict[Any, Dict[str, Any]],
    state: PipelineState,
) -> None:
    msg = futures.pop(future)
    region = msg["region"]
    page_idx = msg["page_idx"]
    try:
        response, status_code = future.result()
        if status_code == 200:
            content = response["choices"][0]["message"]["content"]
            region["content"] = content.strip() if content else ""
        else:
            logger.warning(
                "Recognition failed for page %d: HTTP %s", page_idx, status_code
            )
            region["content"] = None
    except Exception as e:
        logger.warning("Recognition failed for page %d: %s", page_idx, e)
        region["content"] = None
    state.add_recognition_result(page_idx, region)


def _wait_for_any(futures: Dict) -> None:
    done_list = [f for f in futures if f.done()]
    if not done_list:
        try:
            next(as_completed(futures.keys(), timeout=0.05))
        except Exception:
            pass
