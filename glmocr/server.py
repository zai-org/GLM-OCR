"""GLM-OCR SDK Flask service."""

import os
import shutil
import sys
import tempfile
import traceback
import multiprocessing
from typing import TYPE_CHECKING, List

try:
    from flask import Flask, request, jsonify

    _FLASK_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover
    Flask = None  # type: ignore
    request = None  # type: ignore
    jsonify = None  # type: ignore
    _FLASK_IMPORT_ERROR = e

from glmocr.api import _resolve_schema_template, _parse_json_from_text, _DEFAULT_EXTRACTION_PROMPT
from glmocr.pipeline import Pipeline
from glmocr.config import load_config
from glmocr.utils.logging import get_logger, configure_logging

if TYPE_CHECKING:
    from glmocr.config import GlmOcrConfig

logger = get_logger(__name__)

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""


def create_app(config: "GlmOcrConfig") -> Flask:
    """Create a Flask app.

    Args:
        config: GlmOcrConfig instance.

    Returns:
        Flask app instance.
    """
    if Flask is None:
        raise ImportError(
            "Flask server support requires the optional server extra. "
            "Install with: pip install 'glmocr[server]'"
        ) from _FLASK_IMPORT_ERROR

    app = Flask(__name__)

    # Create pipeline with typed config
    pipeline = Pipeline(config=config.pipeline)

    # Store pipeline and config in app.config
    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config

    def _build_messages(image_urls: List[str]) -> dict:
        """Build pipeline request_data from a list of image URL strings."""
        messages = [{"role": "user", "content": []}]
        for image_url in image_urls:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )
        return {"messages": messages}

    def _format_results(results):
        """Format pipeline results into a JSON response tuple."""
        if not results:
            return jsonify({"json_result": None, "markdown_result": ""}), 200
        if len(results) == 1:
            r = results[0]
            return (
                jsonify(
                    {
                        "json_result": r.json_result,
                        "markdown_result": r.markdown_result or "",
                    }
                ),
                200,
            )
        json_result = [r.json_result for r in results]
        markdown_result = "\n\n---\n\n".join(
            r.markdown_result or "" for r in results
        )
        return (
            jsonify(
                {
                    "json_result": json_result,
                    "markdown_result": markdown_result,
                }
            ),
            200,
        )

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
        """Document parsing endpoint.

        Accepts two content types:

        **application/json**::

            {
                "images": ["url1", "url2", ...],  # URLs or presigned URLs
            }

        **multipart/form-data**::

            files:  one or more file uploads (field name ``files``)
            urls:   one or more URL strings  (field name ``urls``)

        Response::

            {
                "json_result": {...},
                "markdown_result": "..."
            }
        """
        content_type = (request.content_type or "").split(";")[0].strip().lower()

        if content_type == "multipart/form-data":
            return _handle_multipart(pipeline)
        elif content_type == "application/json":
            return _handle_json(pipeline)
        else:
            return (
                jsonify(
                    {
                        "error": (
                            "Unsupported Content-Type. "
                            "Expected 'application/json' or 'multipart/form-data'."
                        )
                    }
                ),
                400,
            )

    def _handle_json(pipeline):
        """Handle application/json requests."""
        try:
            data = request.json
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        request_data = _build_messages(images)

        try:
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )
            return _format_results(results)
        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

    def _handle_multipart(pipeline):
        """Handle multipart/form-data requests (file uploads + URLs)."""
        from pathlib import Path as _Path

        uploaded_files = request.files.getlist("files")
        url_values = request.form.getlist("urls")

        if not uploaded_files and not url_values:
            return jsonify({"error": "No files or urls provided"}), 400

        temp_dir = None
        try:
            image_paths: List[str] = []

            # Save uploaded files to a temp directory
            if uploaded_files:
                temp_dir = tempfile.mkdtemp(prefix="glmocr_upload_")
                for idx, f in enumerate(uploaded_files):
                    filename = f.filename or f"upload_{idx}"
                    # Sanitise: keep only the basename to prevent path traversal
                    safe_name = _Path(filename).name or f"upload_{idx}"
                    save_path = os.path.join(temp_dir, f"{idx}_{safe_name}")
                    f.save(save_path)
                    image_paths.append(save_path)

            # Append any URL strings (presigned URLs, etc.)
            for url in url_values:
                url = url.strip()
                if url:
                    image_paths.append(url)

            if not image_paths:
                return jsonify({"error": "No valid files or urls provided"}), 400

            request_data = _build_messages(image_paths)

            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )
            return _format_results(results)

        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @app.route("/glmocr/extract", methods=["POST"])
    def extract():
        """Structured information extraction endpoint.

        Accepts a document image and a JSON schema, then returns structured
        data matching the schema.  The schema can be:

        - An **empty-value template** (GLM-OCR native format)
        - A **JSON Schema** (e.g. from Zod's ``zodToJsonSchema()``)

        **application/json**::

            {
                "images": ["url1"],
                "schema": {"invoice_no": "", "total": ""},
                "prompt": "..."          // optional
            }

        **multipart/form-data**::

            files:   file uploads  (field name ``files``)
            urls:    URL strings   (field name ``urls``)
            schema:  JSON string   (field name ``schema``, required)
            prompt:  string        (field name ``prompt``, optional)

        Response::

            {
                "data": { ... }          // extracted structured data
            }
        """
        content_type = (request.content_type or "").split(";")[0].strip().lower()

        if content_type == "multipart/form-data":
            return _handle_extract_multipart(pipeline)
        elif content_type == "application/json":
            return _handle_extract_json(pipeline)
        else:
            return (
                jsonify(
                    {
                        "error": (
                            "Unsupported Content-Type. "
                            "Expected 'application/json' or 'multipart/form-data'."
                        )
                    }
                ),
                400,
            )

    _SCHEMALESS_EXTRACTION_PROMPT = (
        "请将以下文档内容转换为结构化JSON格式输出。"
        "根据文档内容自动识别字段并组织为合理的JSON结构:"
    )

    def _build_extraction_prompt(schema_raw, prompt_override=None):
        """Resolve schema and build the full extraction prompt string."""
        import json as _json

        template = _resolve_schema_template(schema_raw)
        prefix = prompt_override or _DEFAULT_EXTRACTION_PROMPT
        return f"{prefix}\n{_json.dumps(template, ensure_ascii=False, indent=4)}"

    def _extraction_response(results, extraction_prompt):
        """Run extraction on pipeline results and return JSON response."""
        import json as _json

        extracted = []
        for r in results:
            text = r.markdown_result or ""
            try:
                data = _parse_json_from_text(text)
            except ValueError:
                data = None
            extracted.append(data)

        if len(extracted) == 1:
            return jsonify({"data": extracted[0]}), 200
        return jsonify({"data": extracted}), 200

    def _handle_extract_json(pipeline):
        """Handle JSON extraction requests."""
        try:
            data = request.json
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]
        schema_raw = data.get("schema")
        prompt_override = data.get("prompt")

        if not images:
            return jsonify({"error": "No images provided"}), 400

        if not schema_raw:
            # No schema: parse first, then convert markdown to JSON
            extraction_prompt = prompt_override or _SCHEMALESS_EXTRACTION_PROMPT
            return _handle_schemaless_extract(pipeline, images, extraction_prompt)

        try:
            extraction_prompt = _build_extraction_prompt(schema_raw, prompt_override)
        except (TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid schema: {e}"}), 400

        # Check if this server is backed by a MaaS-enabled GlmOcr
        maas_config = app.config["doc_config"].pipeline.maas
        if maas_config.enabled:
            return _handle_extract_maas(images, extraction_prompt)

        # Self-hosted: inject extraction prompt into pipeline request
        return _handle_extract_selfhosted(pipeline, images, extraction_prompt)

    def _handle_extract_maas(images, extraction_prompt):
        """Run extraction via MaaS API."""
        import json as _json

        from glmocr.maas_client import MaaSClient

        maas_config = app.config["doc_config"].pipeline.maas
        client = MaaSClient(maas_config)
        client.start()
        try:
            extracted = []
            for image in images:
                response = client.parse(image, prompt=extraction_prompt)
                logger.debug(
                    "MaaS extract response keys: %s", list(response.keys())
                )
                # Try md_results first, then fall back to content in choices
                md = response.get("md_results", "")
                if not md:
                    # Some MaaS responses use the chat-completion format
                    md = (
                        response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                if not md:
                    logger.warning(
                        "MaaS extract: no parseable text in response. "
                        "Response keys: %s, raw (truncated): %s",
                        list(response.keys()),
                        str(response)[:1000],
                    )
                try:
                    data = _parse_json_from_text(md)
                except ValueError as exc:
                    logger.warning("Extract JSON parse failed: %s", exc)
                    data = None
                extracted.append(data)

            if len(extracted) == 1:
                return jsonify({"data": extracted[0]}), 200
            return jsonify({"data": extracted}), 200
        except Exception as e:
            logger.error("Extract error: %s", e)
            return jsonify({"error": f"Extract error: {str(e)}"}), 500
        finally:
            client.stop()

    def _is_pdf_source(image_url):
        """Check if a source is a PDF (by extension or magic bytes)."""
        url_lower = image_url.lower()
        if url_lower.endswith(".pdf"):
            return True
        # Check file:// paths
        if url_lower.startswith("file://") and url_lower[7:].endswith(".pdf"):
            return True
        # Check if the file exists and starts with PDF magic bytes
        path = image_url
        if path.startswith("file://"):
            path = path[7:]
        try:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read(5) == b"%PDF-"
        except Exception:
            pass
        return False

    def _handle_extract_selfhosted(pipeline, images, extraction_prompt):
        """Run extraction via self-hosted OCR.

        For **single-page images**: send the image + extraction prompt
        directly to the VLM (bypasses layout detection).

        For **PDFs**: first run the normal parse pipeline (with layout
        detection) to get accurate markdown, then send the combined
        markdown text + extraction prompt to the VLM in a single call.
        This lets the model see the full document context.
        """
        try:
            extracted = []
            for image_url in images:
                if _is_pdf_source(image_url):
                    data = _extract_from_pdf(pipeline, image_url, extraction_prompt)
                else:
                    data = _extract_from_image(pipeline, image_url, extraction_prompt)
                extracted.append(data)

            if len(extracted) == 1:
                return jsonify({"data": extracted[0]}), 200
            return jsonify({"data": extracted}), 200
        except Exception as e:
            logger.error("Extract error: %s", e)
            return jsonify({"error": f"Extract error: {str(e)}"}), 500

    def _extract_from_image(pipeline, image_url, extraction_prompt):
        """Extract structured data from a single image."""
        pages = pipeline.page_loader.load_pages([image_url])
        if not pages:
            return None

        page = pages[0]
        req = pipeline.page_loader.build_request_from_image(
            page, task_type="text"
        )
        # Replace the default task prompt with our extraction prompt
        for msg in req.get("messages", []):
            if msg.get("role") == "user" and isinstance(
                msg.get("content"), list
            ):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        item["text"] = extraction_prompt

        response, status_code = pipeline.ocr_client.process(req)
        if status_code != 200:
            logger.error("OCR request failed (%s): %s", status_code, response)
            return None

        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ) or response.get("response", "")

        logger.debug(
            "Self-hosted extract raw content (truncated): %s",
            content[:500],
        )
        try:
            return _parse_json_from_text(content)
        except ValueError as exc:
            logger.warning("Extract JSON parse failed: %s", exc)
            return None

    def _extract_from_pdf(pipeline, image_url, extraction_prompt):
        """Extract structured data from a PDF.

        Two-phase approach:
        1. Parse the full PDF through the normal pipeline (layout +
           region OCR) to get high-quality markdown.
        2. Send the combined markdown + extraction prompt to the VLM
           in a single call so the model sees the complete document.
        """
        # Phase 1: standard parse to get markdown
        request_data = _build_messages([image_url])
        try:
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )
        except Exception as e:
            logger.error("PDF parse failed: %s", e)
            return None

        if not results:
            return None

        # Combine markdown from all results (usually one per PDF)
        full_markdown = "\n\n---\n\n".join(
            r.markdown_result or "" for r in results
        )

        if not full_markdown.strip():
            logger.warning("PDF parse produced empty markdown")
            return None

        logger.debug(
            "PDF parse markdown (truncated): %s", full_markdown[:500]
        )

        # Phase 2: send markdown + extraction prompt to VLM
        combined_prompt = (
            f"以下是文档内容:\n\n{full_markdown}\n\n{extraction_prompt}"
        )
        req = {
            "messages": [
                {
                    "role": "user",
                    "content": combined_prompt,
                }
            ],
            "temperature": 0.1,
            "top_p": pipeline.page_loader.top_p,
            "top_k": pipeline.page_loader.top_k,
            "repetition_penalty": pipeline.page_loader.repetition_penalty,
        }

        response, status_code = pipeline.ocr_client.process(req)
        if status_code != 200:
            logger.error(
                "Extraction VLM call failed (%s): %s", status_code, response
            )
            return None

        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        ) or response.get("response", "")

        logger.debug(
            "PDF extract raw content (truncated): %s", content[:500]
        )
        try:
            return _parse_json_from_text(content)
        except ValueError as exc:
            logger.warning("Extract JSON parse failed: %s", exc)
            return None

    def _handle_schemaless_extract(pipeline, images, extraction_prompt):
        """Extract without schema: parse to markdown, then convert to JSON.

        1. Run the normal parse pipeline to get markdown.
        2. Send the markdown + extraction prompt to the VLM to produce JSON.
        """
        try:
            request_data = _build_messages(images)
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )

            if not results:
                return jsonify({"data": None}), 200

            full_markdown = "\n\n---\n\n".join(
                r.markdown_result or "" for r in results
            )

            if not full_markdown.strip():
                logger.warning("Schemaless extract: parse produced empty markdown")
                return jsonify({"data": None}), 200

            logger.debug(
                "Schemaless extract markdown (truncated): %s",
                full_markdown[:500],
            )

            # Send markdown + prompt to VLM to convert to JSON
            combined_prompt = (
                f"以下是文档内容:\n\n{full_markdown}\n\n{extraction_prompt}"
            )
            req = {
                "messages": [
                    {
                        "role": "user",
                        "content": combined_prompt,
                    }
                ],
                "temperature": 0.1,
                "top_p": pipeline.page_loader.top_p,
                "top_k": pipeline.page_loader.top_k,
                "repetition_penalty": pipeline.page_loader.repetition_penalty,
            }

            response, status_code = pipeline.ocr_client.process(req)
            if status_code != 200:
                logger.error(
                    "Schemaless extract VLM call failed (%s): %s",
                    status_code,
                    response,
                )
                return jsonify({"error": "VLM extraction failed"}), 500

            content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or response.get("response", "")

            logger.debug(
                "Schemaless extract raw content (truncated): %s",
                content[:500],
            )

            try:
                data = _parse_json_from_text(content)
            except ValueError as exc:
                logger.warning("Schemaless extract JSON parse failed: %s", exc)
                data = None

            return jsonify({"data": data}), 200

        except Exception as e:
            logger.error("Schemaless extract error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Extract error: {str(e)}"}), 500

    def _handle_extract_multipart(pipeline):
        """Handle multipart/form-data extraction requests."""
        import json as _json
        from pathlib import Path as _Path

        uploaded_files = request.files.getlist("files")
        url_values = request.form.getlist("urls")
        schema_str = request.form.get("schema")
        prompt_override = request.form.get("prompt")

        if not uploaded_files and not url_values:
            return jsonify({"error": "No files or urls provided"}), 400

        if schema_str:
            try:
                schema_raw = _json.loads(schema_str)
            except _json.JSONDecodeError:
                return jsonify({"error": "schema must be valid JSON"}), 400

            try:
                extraction_prompt = _build_extraction_prompt(schema_raw, prompt_override)
            except (TypeError, ValueError) as e:
                return jsonify({"error": f"Invalid schema: {e}"}), 400
        else:
            extraction_prompt = prompt_override or _SCHEMALESS_EXTRACTION_PROMPT

        temp_dir = None
        try:
            image_paths: List[str] = []

            if uploaded_files:
                temp_dir = tempfile.mkdtemp(prefix="glmocr_extract_")
                for idx, f in enumerate(uploaded_files):
                    filename = f.filename or f"upload_{idx}"
                    safe_name = _Path(filename).name or f"upload_{idx}"
                    save_path = os.path.join(temp_dir, f"{idx}_{safe_name}")
                    f.save(save_path)
                    image_paths.append(save_path)

            for url in url_values:
                url = url.strip()
                if url:
                    image_paths.append(url)

            if not image_paths:
                return jsonify({"error": "No valid files or urls provided"}), 400

            if not schema_str:
                return _handle_schemaless_extract(pipeline, image_paths, extraction_prompt)

            maas_config = app.config["doc_config"].pipeline.maas
            if maas_config.enabled:
                return _handle_extract_maas(image_paths, extraction_prompt)
            return _handle_extract_selfhosted(pipeline, image_paths, extraction_prompt)

        except Exception as e:
            logger.error("Extract error: %s", e)
            return jsonify({"error": f"Extract error: {str(e)}"}), 500
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"}), 200

    return app


def main():
    """Main entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="GlmOcr Server")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    # Use spawn for multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    app = None

    try:
        config = load_config(args.config)

        # Configure logging
        log_level = args.log_level or config.logging.level
        configure_logging(level=log_level)

        # Create app with typed config
        app = create_app(config)

        # Start pipeline
        pipeline = app.config["pipeline"]
        pipeline.start()

        # Start Flask service
        server_config = config.server
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "GlmOcr Server starting on %s:%d...", server_config.host, server_config.port
        )
        logger.info("API endpoints: /glmocr/parse, /glmocr/extract")
        logger.info("=" * 60)
        logger.info("")

        app.run(
            debug=server_config.debug,
            host=server_config.host,
            port=server_config.port,
        )

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        # Stop pipeline
        if app is not None and "pipeline" in app.config:
            app.config["pipeline"].stop()


if __name__ == "__main__":
    main()
