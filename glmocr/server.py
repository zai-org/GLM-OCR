"""GLM-OCR SDK Flask service."""

import sys
import traceback
import multiprocessing
from typing import TYPE_CHECKING

from flask import Flask, request, jsonify

from glmocr.utils.logging import get_logger, configure_logging

if TYPE_CHECKING:
    from glmocr.config import GlmOcrConfig

logger = get_logger(__name__)


def create_app(config: "GlmOcrConfig") -> Flask:
    """Create a Flask app.

    Args:
        config: GlmOcrConfig instance.

    Returns:
        Flask app instance.
    """
    from glmocr.pipeline import Pipeline

    app = Flask(__name__)

    # Create pipeline with typed config
    pipeline = Pipeline(config=config.pipeline)

    # Store pipeline and config in app.config
    app.config["pipeline"] = pipeline
    app.config["doc_config"] = config

    @app.route("/glmocr/parse", methods=["POST"])
    def parse():
        """Document parsing endpoint.

        Request:
            {
                "images": ["url1", "url2", ...],  # image URLs (http/https/file/data)
            }

        Response:
            {
                "json_result": {...},
                "markdown_result": "..."
            }
        """
        # Validate Content-Type
        if not request.is_json:
            return (
                jsonify(
                    {"error": "Invalid Content-Type. Expected 'application/json'."}
                ),
                400,
            )

        # Parse JSON payload
        try:
            data = request.get_json(silent=False)
        except Exception:
            return jsonify({"error": "Invalid JSON payload"}), 400

        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400

        images = data.get("images", [])
        if isinstance(images, str):
            images = [images]

        if not images:
            return jsonify({"error": "No images provided"}), 400

        # Build pipeline request
        messages = [{"role": "user", "content": []}]
        for image_url in images:
            messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": image_url}}
            )

        request_data = {"messages": messages}

        try:
            # Pipeline.process() yields one result per input unit; merge for single response
            results = list(
                pipeline.process(
                    request_data,
                    save_layout_visualization=False,
                    layout_vis_output_dir=None,
                )
            )
            if not results:
                return (
                    jsonify({"json_result": None, "markdown_result": ""}),
                    200,
                )
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
            # Multiple units: merge json as list, markdown with separator
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
        from glmocr.config import load_config

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
