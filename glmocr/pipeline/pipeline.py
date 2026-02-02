"""GLM-OCR Pipeline

Unified document parsing pipeline. Async by default: process() yields one result
per input unit (one image or one PDF). No separate sync API.

Extension options:
1. Replace components: custom LayoutDetector / ResultFormatter
2. Inherit: subclass Pipeline and override process()
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple, List, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.parser_result import PipelineResult
from glmocr.postprocess import ResultFormatter
from glmocr.utils.image_utils import crop_image_region
from glmocr.utils.image_utils import load_image_to_base64
from glmocr.utils.logging import get_logger, get_profiler

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector

logger = get_logger(__name__)
profiler = get_profiler(__name__)


@dataclass
class _AsyncPipelineState:
    """Shared state for the 3-thread layout path (loader -> layout -> recognition)."""

    page_queue: queue.Queue
    region_queue: queue.Queue
    ready_units_queue: queue.Queue
    recognition_results: List[Tuple[int, Dict]]
    results_lock: threading.Lock
    images_dict: Dict[int, Any]
    layout_results_dict: Dict[int, List]
    num_images_loaded: List[int]
    unit_indices_holder: List[Optional[List[int]]]
    unit_info_holder: List[Optional[Tuple]]
    units_put: set
    units_put_lock: threading.Lock
    count_lock: threading.Lock
    exceptions: List[Tuple[str, Exception]]
    exception_lock: threading.Lock


class Pipeline:
    """GLM-OCR pipeline.

    Unified processing flow:
    1. PageLoader: load images/PDF into pages
    2. (Optional) LayoutDetector: detect regions
    3. OCRClient: call OCR service
    4. ResultFormatter: format outputs

    Args:
        config: PipelineConfig instance.
        layout_detector: Custom layout detector (optional).
        result_formatter: Custom result formatter (optional).

    Example:
        from glmocr.config import load_config

        cfg = load_config()
        pipeline = Pipeline(cfg.pipeline)
        for result in pipeline.process(request_data):
            result.save(output_dir="./results")

        # Custom components
        pipeline = Pipeline(
            cfg.pipeline,
            layout_detector=MyLayoutDetector(cfg.pipeline.layout),
            result_formatter=MyFormatter(cfg.pipeline.result_formatter),
        )
    """

    def __init__(
        self,
        config: "PipelineConfig",
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional[ResultFormatter] = None,
    ):
        self.config = config
        self.enable_layout = config.enable_layout

        # Unified page loader
        self.page_loader = PageLoader(config.page_loader)

        # OCR client
        self.ocr_client = OCRClient(config.ocr_api)

        # Result formatter
        if result_formatter is not None:
            self.result_formatter = result_formatter
        else:
            self.result_formatter = ResultFormatter(config.result_formatter)

        # Layout detector (initialized only when enabled)
        if self.enable_layout:
            if layout_detector is not None:
                self.layout_detector = layout_detector
            else:
                from glmocr.layout import PPDocLayoutDetector

                if PPDocLayoutDetector is None:
                    from glmocr.layout import _raise_layout_import_error

                    _raise_layout_import_error()

                self.layout_detector = PPDocLayoutDetector(config.layout)
            self.max_workers = config.max_workers
        self._page_maxsize = getattr(config, "page_maxsize", 100)
        self._region_maxsize = getattr(config, "region_maxsize", 800)

    def _create_async_pipeline_state(
        self,
        page_maxsize: Optional[int],
        region_maxsize: Optional[int],
    ) -> _AsyncPipelineState:
        q1 = page_maxsize if page_maxsize is not None else self._page_maxsize
        q2 = region_maxsize if region_maxsize is not None else self._region_maxsize
        return _AsyncPipelineState(
            page_queue=queue.Queue(maxsize=q1),
            region_queue=queue.Queue(maxsize=q2),
            ready_units_queue=queue.Queue(),
            recognition_results=[],
            results_lock=threading.Lock(),
            images_dict={},
            layout_results_dict={},
            num_images_loaded=[0],
            unit_indices_holder=[None],
            unit_info_holder=[None],
            units_put=set(),
            units_put_lock=threading.Lock(),
            count_lock=threading.Lock(),
            exceptions=[],
            exception_lock=threading.Lock(),
        )

    def process(
        self,
        request_data: Dict[str, Any],
        save_layout_visualization: bool = False,
        layout_vis_output_dir: Optional[str] = None,
        page_maxsize: Optional[int] = None,
        region_maxsize: Optional[int] = None,
    ) -> Generator[PipelineResult, None, None]:
        """Process request with async three-stage flow; yield one result per input unit.

        Uses three threads: load pages -> layout detection -> recognition.
        Yields PipelineResult as each unit (one image or one PDF) completes.

        Args:
            request_data: Request payload containing messages.
            save_layout_visualization: Whether to save layout visualization.
            layout_vis_output_dir: Visualization output directory.
            page_maxsize: Max size for page_queue (page-level items).
            region_maxsize: Max size for region_queue (region-level items). Should be
                larger than page_maxsize since one page yields many regions.
                Defaults to page_maxsize * 8 if not set.

        Yields:
            PipelineResult per input URL (one image or one PDF).
        """

        if not self.enable_layout:
            image_urls = self._extract_image_urls(request_data)
            if not image_urls:
                request_data = self.page_loader.build_request(request_data)
                response, status_code = self.ocr_client.process(request_data)
                if status_code != 200:
                    raise Exception(
                        f"OCR request failed: {response}, status_code: {status_code}"
                    )
                content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                json_result, markdown_result = self.result_formatter.format_ocr_result(
                    content
                )
                yield PipelineResult(
                    json_result=json_result,
                    markdown_result=markdown_result,
                    original_images=[],
                    layout_vis_dir=layout_vis_output_dir,
                )
                return
            pages, unit_indices = self.page_loader.load_pages_with_unit_indices(
                image_urls
            )
            from copy import deepcopy

            base_request_data = deepcopy(request_data)
            cleaned_messages = []
            for msg in base_request_data.get("messages", []):
                if msg.get("role") != "user":
                    cleaned_messages.append(msg)
                    continue
                contents = msg.get("content", [])
                if isinstance(contents, list):
                    contents = [c for c in contents if c.get("type") != "image_url"]
                cleaned_messages.append({**msg, "content": contents})
            base_request_data["messages"] = cleaned_messages

            num_units = len(image_urls)
            original_inputs = [
                (url[7:] if url.startswith("file://") else url) for url in image_urls
            ]
            unit_contents: Dict[int, List[str]] = {u: [] for u in range(num_units)}
            for page_idx, page in enumerate(pages):
                u = (
                    (unit_indices or [0])[page_idx]
                    if page_idx < len(unit_indices or [])
                    else 0
                )
                img_b64 = load_image_to_base64(
                    page,
                    t_patch_size=self.page_loader.t_patch_size,
                    max_pixels=self.page_loader.max_pixels,
                    image_format=self.page_loader.image_format,
                    patch_expand_factor=self.page_loader.patch_expand_factor,
                    min_pixels=self.page_loader.min_pixels,
                )
                data_url = f"data:image/{self.page_loader.image_format.lower()};base64,{img_b64}"
                per_request = deepcopy(base_request_data)
                user_msg = None
                for m in per_request.get("messages", []):
                    if m.get("role") == "user" and isinstance(m.get("content"), list):
                        user_msg = m
                        break
                if user_msg is None:
                    per_request.setdefault("messages", []).append(
                        {"role": "user", "content": []}
                    )
                    user_msg = per_request["messages"][-1]
                user_msg["content"].append(
                    {"type": "image_url", "image_url": {"url": data_url}}
                )
                per_request = self.page_loader.build_request(per_request)
                response, status_code = self.ocr_client.process(per_request)
                if status_code != 200:
                    raise Exception(
                        f"OCR request failed: {response}, status_code: {status_code}"
                    )
                content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                unit_contents.setdefault(u, []).append(content)
            for u in range(num_units):
                contents_u = unit_contents.get(u, [])
                if len(contents_u) == 1:
                    (
                        json_result,
                        markdown_result,
                    ) = self.result_formatter.format_ocr_result(contents_u[0])
                else:
                    (
                        json_result,
                        markdown_result,
                    ) = self.result_formatter.format_multi_page_results(contents_u)
                yield PipelineResult(
                    json_result=json_result,
                    markdown_result=markdown_result,
                    original_images=[original_inputs[u]],
                    layout_vis_dir=layout_vis_output_dir,
                )
            return

        image_urls = self._extract_image_urls(request_data)
        if not image_urls:
            request_data = self.page_loader.build_request(request_data)
            response, status_code = self.ocr_client.process(request_data)
            if status_code != 200:
                raise Exception(
                    f"OCR request failed: {response}, status_code: {status_code}"
                )
            content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            json_result, markdown_result = self.result_formatter.format_ocr_result(
                content
            )
            yield PipelineResult(
                json_result=json_result,
                markdown_result=markdown_result,
                original_images=[],
                layout_vis_dir=layout_vis_output_dir,
            )
            return

        state = self._create_async_pipeline_state(page_maxsize, region_maxsize)

        def data_loading_thread() -> None:
            try:
                pages, unit_indices = self.page_loader.load_pages_with_unit_indices(
                    image_urls
                )
                state.num_images_loaded[0] = len(pages)
                state.unit_indices_holder[0] = unit_indices
                for img_idx, page in enumerate(pages):
                    state.images_dict[img_idx] = page
                    state.page_queue.put(("image", img_idx, page))
                state.page_queue.put(("done", None, None))
            except Exception as e:
                logger.exception("Data loading thread error: %s", e)
                with state.exception_lock:
                    state.exceptions.append(("DataLoadingThread", e))
                state.page_queue.put(("error", None, None))

        def layout_detection_thread() -> None:
            try:
                batch_images: List[Any] = []
                batch_indices: List[int] = []
                loading_complete = False
                global_start_idx = 0
                while True:
                    try:
                        item_type, img_idx, data = state.page_queue.get(timeout=1)
                    except queue.Empty:
                        if loading_complete and batch_images:
                            self._stream_process_layout_batch(
                                batch_images,
                                batch_indices,
                                state.region_queue,
                                state.images_dict,
                                state.layout_results_dict,
                                save_layout_visualization,
                                layout_vis_output_dir,
                                global_start_idx,
                            )
                            global_start_idx += len(batch_indices)
                            batch_images = []
                            batch_indices = []
                        continue
                    if item_type == "image":
                        batch_images.append(data)
                        batch_indices.append(img_idx)
                        if len(batch_images) >= self.layout_detector.batch_size:
                            self._stream_process_layout_batch(
                                batch_images,
                                batch_indices,
                                state.region_queue,
                                state.images_dict,
                                state.layout_results_dict,
                                save_layout_visualization,
                                layout_vis_output_dir,
                                global_start_idx,
                            )
                            global_start_idx += len(batch_indices)
                            batch_images = []
                            batch_indices = []
                    elif item_type == "done":
                        loading_complete = True
                        if batch_images:
                            self._stream_process_layout_batch(
                                batch_images,
                                batch_indices,
                                state.region_queue,
                                state.images_dict,
                                state.layout_results_dict,
                                save_layout_visualization,
                                layout_vis_output_dir,
                                global_start_idx,
                            )
                        state.region_queue.put(("done", None, None))
                        break
                    elif item_type == "error":
                        state.region_queue.put(("error", None, None))
                        break
            except Exception as e:
                logger.exception("Layout detection thread error: %s", e)
                with state.exception_lock:
                    state.exceptions.append(("LayoutDetectionThread", e))
                state.region_queue.put(("error", None, None))

        def maybe_notify_ready_units(img_idx: Optional[int] = None) -> None:
            """Notify when a unit is ready. O(1) when img_idx is given."""
            info = state.unit_info_holder[0]
            if info is None:
                return
            if img_idx is not None and len(info) >= 5:
                (
                    _,
                    unit_region_count,
                    unit_for_image,
                    unit_region_done_count,
                    c_lock,
                ) = info
                u = unit_for_image.get(img_idx)
                if u is None:
                    return
                with c_lock:
                    unit_region_done_count[u] += 1
                    if unit_region_done_count[u] >= unit_region_count[u]:
                        with state.units_put_lock:
                            if u not in state.units_put:
                                state.ready_units_queue.put(u)
                                state.units_put.add(u)
                return
            unit_image_indices, unit_region_count = info[:2]
            num_units = len(unit_region_count)
            with state.results_lock:
                rec = list(state.recognition_results)
            with state.units_put_lock:
                for u in range(num_units):
                    if u in state.units_put:
                        continue
                    if unit_region_count[u] == 0:
                        state.ready_units_queue.put(u)
                        state.units_put.add(u)
                        continue
                    count = sum(1 for (i, _) in rec if i in unit_image_indices[u])
                    if count >= unit_region_count[u]:
                        state.ready_units_queue.put(u)
                        state.units_put.add(u)

        def vlm_recognition_thread() -> None:
            try:
                executor = ThreadPoolExecutor(max_workers=min(self.max_workers, 128))
                futures: Dict[Any, Tuple[Dict, str, int]] = {}
                pending_skip: List[Tuple[Dict, str, int]] = []
                processing_complete = False
                while True:
                    for f in list(futures.keys()):
                        if f.done():
                            info, task, page_idx = futures.pop(f)
                            try:
                                response, status_code = f.result()
                                if status_code == 200:
                                    info["content"] = response["choices"][0]["message"][
                                        "content"
                                    ].strip()
                                else:
                                    info["content"] = ""
                            except Exception as e:
                                logger.warning("Recognition failed: %s", e)
                                info["content"] = ""
                            with state.results_lock:
                                state.recognition_results.append((page_idx, info))
                            maybe_notify_ready_units(page_idx)
                    try:
                        item_type, img_idx, data = state.region_queue.get(timeout=0.01)
                    except queue.Empty:
                        if processing_complete and len(futures) == 0:
                            for region, task_type, page_idx in pending_skip:
                                region["content"] = None
                                with state.results_lock:
                                    state.recognition_results.append((page_idx, region))
                                maybe_notify_ready_units(page_idx)
                            break
                        if futures:
                            done_list = [f for f in futures.keys() if f.done()]
                            if not done_list:
                                try:
                                    next(as_completed(futures.keys(), timeout=0.05))
                                except Exception:
                                    pass
                        continue
                    if item_type == "region":
                        cropped_image, region, task_type, page_idx = data
                        if task_type == "skip":
                            pending_skip.append((region, task_type, page_idx))
                        else:
                            req = self.page_loader.build_request_from_image(
                                cropped_image, task_type
                            )
                            future = executor.submit(self.ocr_client.process, req)
                            futures[future] = (region, task_type, page_idx)
                    elif item_type == "done":
                        processing_complete = True
                    elif item_type == "error":
                        break
                if futures:
                    for future in as_completed(futures.keys()):
                        info, task, page_idx = futures[future]
                        try:
                            response, status_code = future.result()
                            if status_code == 200:
                                info["content"] = response["choices"][0]["message"][
                                    "content"
                                ].strip()
                            else:
                                info["content"] = ""
                        except Exception as e:
                            logger.warning("Recognition failed: %s", e)
                            info["content"] = ""
                        with state.results_lock:
                            state.recognition_results.append((page_idx, info))
                        maybe_notify_ready_units(page_idx)
                executor.shutdown(wait=True)
            except Exception as e:
                logger.exception("VLM recognition thread error: %s", e)
                with state.exception_lock:
                    state.exceptions.append(("VLMRecognitionThread", e))

        t1 = threading.Thread(target=data_loading_thread, daemon=True)
        t2 = threading.Thread(target=layout_detection_thread, daemon=True)
        t3 = threading.Thread(target=vlm_recognition_thread, daemon=True)
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()

        num_images = state.num_images_loaded[0]
        unit_indices = state.unit_indices_holder[0]
        num_units = len(image_urls)
        original_inputs = [
            (url[7:] if url.startswith("file://") else url) for url in image_urls
        ]

        if num_images == 0:
            empty_json, empty_md = self.result_formatter.process([])
            for u in range(num_units):
                yield PipelineResult(
                    json_result=empty_json,
                    markdown_result=empty_md,
                    original_images=[original_inputs[u]],
                    layout_vis_dir=layout_vis_output_dir,
                )
            t3.join()
            with state.exception_lock:
                if state.exceptions:
                    raise RuntimeError(
                        "; ".join(f"{n}: {e}" for n, e in state.exceptions)
                    )
            return

        unit_image_indices: List[List[int]] = [[] for _ in range(num_units)]
        for img_idx in range(num_images):
            if unit_indices is not None and img_idx < len(unit_indices):
                u = unit_indices[img_idx]
                if u < num_units:
                    unit_image_indices[u].append(img_idx)
        unit_region_count = [
            sum(
                len(state.layout_results_dict.get(i, [])) for i in unit_image_indices[u]
            )
            for u in range(num_units)
        ]
        unit_for_image: Dict[int, int] = {
            i: u for u in range(num_units) for i in unit_image_indices[u]
        }
        unit_region_done_count: List[int] = [0] * num_units
        with state.results_lock:
            rec_init = list(state.recognition_results)
        for i, _ in rec_init:
            u = unit_for_image.get(i)
            if u is not None:
                unit_region_done_count[u] += 1
        state.unit_info_holder[0] = (
            unit_image_indices,
            unit_region_count,
            unit_for_image,
            unit_region_done_count,
            state.count_lock,
        )
        for u in range(num_units):
            if unit_region_done_count[u] >= unit_region_count[u]:
                state.ready_units_queue.put(u)
                state.units_put.add(u)

        emitted: set = set()
        while len(emitted) < num_units:
            u = state.ready_units_queue.get()
            if u in emitted:
                continue
            with state.results_lock:
                rec = list(state.recognition_results)
            count = sum(1 for (i, _) in rec if i in unit_image_indices[u])
            if count < unit_region_count[u]:
                state.ready_units_queue.put(u)
                continue
            img_to_idx = {i: k for k, i in enumerate(unit_image_indices[u])}
            grouped_u: List[List[Dict]] = [[] for _ in unit_image_indices[u]]
            for i, r in rec:
                if i in img_to_idx:
                    grouped_u[img_to_idx[i]].append(r)
            json_u, md_u = self.result_formatter.process(grouped_u)
            yield PipelineResult(
                json_result=json_u,
                markdown_result=md_u,
                original_images=[original_inputs[u]],
                layout_vis_dir=layout_vis_output_dir,
                layout_image_indices=unit_image_indices[u],
            )
            emitted.add(u)

        t3.join()
        with state.exception_lock:
            if state.exceptions:
                raise RuntimeError("; ".join(f"{n}: {e}" for n, e in state.exceptions))

    def _stream_process_layout_batch(
        self,
        batch_images: List[Any],
        batch_indices: List[int],
        region_queue: queue.Queue,
        images_dict: Dict[int, Any],
        layout_results_dict: Dict[int, List],
        save_visualization: bool,
        vis_output_dir: Optional[str],
        global_start_idx: int,
    ) -> None:
        """Run layout detection on a batch and push regions to queue2."""
        layout_results = self.layout_detector.process(
            batch_images,
            save_visualization=save_visualization and vis_output_dir is not None,
            visualization_output_dir=vis_output_dir,
            global_start_idx=global_start_idx,
        )
        for img_idx, image, layout_result in zip(
            batch_indices, batch_images, layout_results
        ):
            layout_results_dict[img_idx] = layout_result
            for region in layout_result:
                cropped = crop_image_region(image, region["bbox_2d"])
                region_queue.put(
                    (
                        "region",
                        img_idx,
                        (cropped, region, region["task_type"], img_idx),
                    )
                )

    def _extract_image_urls(self, request_data: Dict[str, Any]) -> List[str]:
        """Extract image URLs from request_data."""
        image_urls = []
        for msg in request_data.get("messages", []):
            if msg.get("role") == "user":
                contents = msg.get("content", [])
                if isinstance(contents, list):
                    for content in contents:
                        if content.get("type") == "image_url":
                            image_urls.append(content["image_url"]["url"])
        return image_urls

    def _prepare_regions(self, pages, layout_results) -> List[tuple]:
        """Prepare regions that need recognition."""
        regions = []
        with profiler.measure("crop_regions"):
            for page_idx, (page, layouts) in enumerate(zip(pages, layout_results)):
                for region in layouts:
                    cropped = crop_image_region(page, region["bbox_2d"])
                    regions.append((cropped, region, region["task_type"], page_idx))
        return regions

    def _recognize_regions(self, regions: List[tuple]) -> List[tuple]:
        """Recognize all regions in parallel."""
        results = []

        # Split skipped regions and regions to process
        to_process = []
        for img, info, task, page_idx in regions:
            if task == "skip":
                info["content"] = None
                results.append((page_idx, info))
            else:
                to_process.append((img, info, task, page_idx))

        if not to_process:
            return results

        # Build all requests first
        request_data_list = []
        with profiler.measure("build_region_requests"):
            for img, info, task, page_idx in to_process:
                request_data = self.page_loader.build_request_from_image(img, task)
                request_data_list.append((request_data, info, task, page_idx))

        # Run in parallel
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(to_process))
        ) as executor:
            futures = {}
            for request_data, info, task, page_idx in request_data_list:
                future = executor.submit(self.ocr_client.process, request_data)
                futures[future] = (info, task, page_idx)

            for future in as_completed(futures):
                info, task, page_idx = futures[future]
                try:
                    response, status_code = future.result()
                    if status_code == 200:
                        info["content"] = response["choices"][0]["message"][
                            "content"
                        ].strip()
                    else:
                        info["content"] = ""
                except Exception as e:
                    logger.warning("Recognition failed: %s", e)
                    info["content"] = ""
                results.append((page_idx, info))

        return results

    def start(self):
        """Start the pipeline."""
        logger.info("Starting Pipeline...")
        if self.enable_layout:
            self.layout_detector.start()
        self.ocr_client.start()
        logger.info("Pipeline started!")

    def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping Pipeline...")
        self.ocr_client.stop()
        if self.enable_layout:
            self.layout_detector.stop()
        logger.info("Pipeline stopped!")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
