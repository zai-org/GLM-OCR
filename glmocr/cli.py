"""GLM OCR CLI.

Provides a command-line interface to run document parsing.
"""

import sys
import json
import re
import argparse
import threading
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from glmocr.api import GlmOcr
from glmocr.maas_client import MissingApiKeyError
from glmocr.utils.logging import get_logger, configure_logging

logger = get_logger(__name__)


def layout_device_type(value: str) -> str:
    """Validate --layout-device argument.

    Accepts:
      - "cpu"
      - "cuda"
      - "cuda:N" where N is a non-negative integer.
    """
    if value in ("cpu", "cuda"):
        return value
    if re.fullmatch(r"cuda:\d+", value):
        return value
    raise argparse.ArgumentTypeError(
        'Invalid layout device {!r}. Expected "cpu", "cuda", or "cuda:N".'.format(value)
    )


_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".pdf"}


def load_image_paths(input_path: str) -> Tuple[List[str], Optional[str]]:
    """Load image paths from a file or directory (recursively).

    When *input_path* is a directory the search is recursive — all supported
    image/PDF files in nested subdirectories are collected.

    Args:
        input_path: Input path (file or directory).

    Returns:
        A tuple ``(image_paths, input_root)``.
        *input_root* is the absolute directory path when the input is a
        directory (``None`` when it is a single file).  It is used by the
        caller to compute relative paths so that the output preserves the
        original directory hierarchy.
    """
    path = Path(input_path)

    if path.is_file():
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Not Supported Type: {path.suffix}")
        return [str(path.absolute())], None

    if path.is_dir():
        seen: set = set()
        image_paths: List[str] = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp", "*.pdf"]:
            for p in path.rglob(ext):
                abs_p = str(p.absolute())
                if abs_p not in seen:
                    seen.add(abs_p)
                    image_paths.append(abs_p)
            for p in path.rglob(ext.upper()):
                abs_p = str(p.absolute())
                if abs_p not in seen:
                    seen.add(abs_p)
                    image_paths.append(abs_p)
        image_paths.sort()
        if not image_paths:
            raise ValueError(
                f"Cannot find image or PDF files in directory: {input_path}"
            )
        return image_paths, str(path.absolute())

    raise ValueError(f"Path does not exist: {input_path}")


def _queue_stats_updater(glm_parser: GlmOcr, pbar: tqdm, stop: threading.Event):
    while not stop.wait(0.3):
        stats = glm_parser.get_queue_stats()
        if stats:
            pbar.set_postfix_str(
                f"Q1:{stats['page_queue_size']}/{stats['page_queue_maxsize']} "
                f"Q2:{stats['region_queue_size']}/{stats['region_queue_maxsize']}",
                refresh=True,
            )


def _auto_coerce(raw: str):
    """Coerce a CLI string to a Python scalar."""
    if raw.lower() in ("true", "yes"):
        return True
    if raw.lower() in ("false", "no"):
        return False
    if raw.lower() in ("null", "none", "~"):
        return None
    return raw


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="GlmOcr - Document Parsing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Parse a single image (uses ZHIPU_API_KEY from environment)
  glmocr parse image.png

    # Pass API key directly (no env setup needed)
  glmocr parse image.png --api-key sk-xxx

    # Parse all images in a directory
  glmocr parse ./images/

    # Disable layout detection (OCR-only): set pipeline.enable_layout=false
    glmocr parse image.png --config my_config.yaml

    # Specify output directory
  glmocr parse image.png --output ./output/

    # Print results to stdout only (no files written)
  glmocr parse image.png --api-key sk-xxx --stdout --no-save

    # Load API key from a specific .env file
  glmocr parse image.png --env-file /path/to/.env

    # Specify custom config file
  glmocr parse image.png --config config.yaml

    # Override config values via --set
  glmocr parse image.png --set pipeline.ocr_api.api_port 8080
  glmocr parse image.png --set pipeline.layout.use_polygon true --set pipeline.maas.enabled false
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    parse_parser = subparsers.add_parser("parse", help="Parse document images")
    parse_parser.add_argument(
        "input", type=str, help="Input image file or directory path"
    )
    parse_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parse_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to files (stdout can still be enabled)",
    )
    parse_parser.add_argument(
        "--no-layout-vis",
        action="store_true",
        help="Do not save layout visualization results",
    )
    parse_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to configuration file (YAML format)",
    )
    parse_parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON result only, do not output Markdown",
    )
    parse_parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output results to standard output (JSON format)",
    )
    parse_parser.add_argument(
        "--api-key",
        "-k",
        type=str,
        default=None,
        help="API key for MaaS mode (overrides ZHIPU_API_KEY env var)",
    )
    parse_parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["maas", "selfhosted"],
        help="Operation mode: 'maas' (cloud API, default) or 'selfhosted' (local vLLM/SGLang)",
    )
    parse_parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file to load ZHIPU_API_KEY and other settings from",
    )
    parse_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    parse_parser.add_argument(
        "--layout-device",
        type=layout_device_type,
        default=None,
        help='Device for layout model: "cpu", "cuda", or "cuda:N" (default: auto)',
    )
    parse_parser.add_argument(
        "--set",
        nargs=2,
        action="append",
        metavar=("KEY", "VALUE"),
        dest="config_overrides",
        help="Override a config value using dotted path, e.g. "
        "--set pipeline.ocr_api.api_port 8080",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    configure_logging(level=args.log_level)

    try:
        logger.info("Loading images: %s", args.input)
        image_paths, input_root = load_image_paths(args.input)
        logger.info("Found %d file(s)", len(image_paths))

        save_layout_vis = not args.no_layout_vis

        dotted_overrides: dict = {}
        for key, value in args.config_overrides or []:
            dotted_overrides[key] = _auto_coerce(value)

        ocr_error_count = 0
        with GlmOcr(
            config_path=args.config,
            api_key=args.api_key,
            mode=args.mode,
            env_file=args.env_file,
            layout_device=args.layout_device,
            _dotted=dotted_overrides,
        ) as glm_parser:
            total_files = len(image_paths)

            pbar = tqdm(
                total=total_files,
                desc="Parsing",
                unit="file",
                file=sys.stderr,
                dynamic_ncols=True,
            )

            stop_event = threading.Event()
            stats_thread = threading.Thread(
                target=_queue_stats_updater,
                args=(glm_parser, pbar, stop_event),
                daemon=True,
            )
            stats_thread.start()

            try:
                for result in glm_parser.parse(
                    image_paths,
                    stream=True,
                    save_layout_visualization=save_layout_vis,
                    preserve_order=False,
                ):
                    file_name = (
                        Path(result.original_images[0]).name
                        if result.original_images
                        else f"unit_{pbar.n + 1}"
                    )
                    pbar.update(1)

                    result_data = result.to_dict()
                    if "error" in result_data:
                        ocr_error_count += 1
                        tqdm.write(
                            f"Failed: {file_name}: {result_data['error']}",
                            file=sys.stderr,
                        )
                        continue

                    try:
                        if args.stdout:
                            stem = (
                                Path(result.original_images[0]).stem
                                if result.original_images
                                else file_name
                            )
                            print(f"\n=== {stem} - JSON Result ===")
                            print(
                                json.dumps(
                                    result.json_result,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                                if isinstance(result.json_result, (dict, list))
                                else result.json_result
                            )
                            if result.markdown_result and not args.json_only:
                                print(f"\n=== {stem} - Markdown Result ===")
                                print(result.markdown_result)

                        if not args.no_save:
                            save_dir = args.output
                            if input_root and result.original_images:
                                rel = Path(
                                    result.original_images[0]
                                ).parent.relative_to(input_root)
                                if str(rel) != ".":
                                    save_dir = str(Path(args.output) / rel)
                            result.save(
                                output_dir=save_dir,
                                save_layout_visualization=save_layout_vis,
                            )

                    except Exception as e:
                        tqdm.write(f"Failed: {file_name}: {e}", file=sys.stderr)
                        continue
            finally:
                stop_event.set()
                stats_thread.join(timeout=2)
                pbar.close()

        if ocr_error_count:
            logger.error("OCR failed for %d file(s).", ocr_error_count)
            sys.exit(1)
        logger.info("All done!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except MissingApiKeyError as e:
        logger.error(
            "%s\n\n"
            "  Quick fix:\n"
            "    export ZHIPU_API_KEY=sk-xxx           # set once in shell\n"
            "    glmocr parse image.png --api-key sk-xxx  # or pass directly\n\n"
            "  Get your free key at: https://open.bigmodel.cn",
            e,
        )
        logger.debug(traceback.format_exc())
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
