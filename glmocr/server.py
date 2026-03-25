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
