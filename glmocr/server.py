"""GLM-OCR SDK Flask service."""

import base64
import os
import sys
import traceback
import multiprocessing
import urllib.request
from typing import TYPE_CHECKING, List, Tuple

try:
    from flask import Flask, request, jsonify

    _FLASK_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover
    Flask = None  # type: ignore
    request = None  # type: ignore
    jsonify = None  # type: ignore
    _FLASK_IMPORT_ERROR = e

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

    def _remote_url_to_data_uri(url: str) -> str:
        """Fetch a remote URL and return a data: URI (no temp file)."""
        req = urllib.request.Request(url, headers={"User-Agent": "glmocr/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            content_type = resp.headers.get_content_type() or "application/octet-stream"
            data = resp.read()
        b64 = base64.b64encode(data).decode()
        return f"data:{content_type};base64,{b64}"

    def _file_bytes_to_data_uri(file_bytes: bytes, content_type: str, filename: str = "") -> str:
        """Convert uploaded file bytes to a data: URI.

        Detects PDFs by magic bytes so the page_loader can route them correctly.
        """
        # Detect PDF by magic bytes (%PDF) regardless of declared content_type
        if file_bytes[:4] == b"%PDF" or "pdf" in content_type.lower():
            mime = "application/pdf"
        elif content_type and content_type.startswith("image/"):
            mime = content_type
        else:
            # Fallback: guess from filename extension
            import mimetypes

            guessed, _ = mimetypes.guess_type(filename or "")
            mime = guessed or "application/octet-stream"
        b64 = base64.b64encode(file_bytes).decode()
        return f"data:{mime};base64,{b64}"

    def _resolve_urls(raw_urls: List[str]) -> List[str]:
        """Convert http/https signed URLs to data: URIs; pass others through."""
        resolved = []
        for url in raw_urls:
            if url.startswith(("http://", "https://")):
                resolved.append(_remote_url_to_data_uri(url))
            else:
                resolved.append(url)
        return resolved

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
        """Document parsing endpoint.

        Accepts three Content-Type variants:

        1. application/json (original):
            {"images": ["file://...", "data:...", "https://signed-url..."]}

        2. multipart/form-data (file upload):
            files=<file1>&files=<file2>&urls=https://...&urls=https://...
            Field name "files" for uploaded files, "urls" for remote URLs.

        3. Mixed multipart: both "files" and "urls" fields together.

        Signed https:// URLs (in JSON or multipart) are fetched in-memory and
        converted to data: URIs — no temp files are written.

        Response:
            {"json_result": {...}, "markdown_result": "..."}
        """
        content_type = request.content_type or ""

        if content_type.startswith("multipart/form-data"):
            # --- multipart/form-data ---
            image_urls: List[str] = []

            # Uploaded files → in-memory data: URIs
            for uploaded in request.files.getlist("files"):
                file_bytes = uploaded.read()
                if not file_bytes:
                    continue
                data_uri = _file_bytes_to_data_uri(
                    file_bytes,
                    uploaded.mimetype or "",
                    uploaded.filename or "",
                )
                image_urls.append(data_uri)

            # Remote URLs in the form (signed URLs or plain URLs)
            for url in request.form.getlist("urls"):
                url = url.strip()
                if url:
                    image_urls.append(url)

            if not image_urls:
                return jsonify({"error": "No files or urls provided"}), 400

            # Resolve any http/https URLs to data: URIs
            try:
                image_urls = _resolve_urls(image_urls)
            except Exception as e:
                return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

        elif "application/json" in content_type:
            # --- application/json ---
            try:
                data = request.get_json(force=True)
            except Exception:
                return jsonify({"error": "Invalid JSON payload"}), 400

            images = data.get("images", [])
            if isinstance(images, str):
                images = [images]
            if not images:
                return jsonify({"error": "No images provided"}), 400

            # Resolve signed http/https URLs to data: URIs
            try:
                image_urls = _resolve_urls(images)
            except Exception as e:
                return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

        else:
            return (
                jsonify(
                    {
                        "error": "Unsupported Content-Type. Use application/json or multipart/form-data."
                    }
                ),
                415,
            )

        # Build pipeline request_data
        messages = [{"role": "user", "content": []}]
        for url in image_urls:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": url}})
        request_data = {"messages": messages}

        try:
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )
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
            # Multiple units: list of json_results, markdown separated by ---
            json_result = [r.json_result for r in results]
            markdown_result = "\n\n---\n\n".join(r.markdown_result or "" for r in results)
            return (
                jsonify({"json_result": json_result, "markdown_result": markdown_result}),
                200,
            )

        except Exception as e:
            logger.error("Parse error: %s", e)
            logger.debug(traceback.format_exc())
            return jsonify({"error": f"Parse error: {str(e)}"}), 500

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
        logger.info("GlmOcr Server starting on %s:%d...", server_config.host, server_config.port)
        logger.info("API endpoint: /glmocr/parse")
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
