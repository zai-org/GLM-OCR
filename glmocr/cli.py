"""GLM OCR CLI.

Provides a command-line interface to run document parsing.
"""

import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import List

from glmocr.api import GlmOcr
from glmocr.utils.logging import get_logger, configure_logging

logger = get_logger(__name__)


def load_image_paths(input_path: str) -> List[str]:
    """Load image paths from a file or directory.

    PDF files are included as inputs (they will be expanded into page images later).

    Args:
        input_path: Input path (file or directory).

    Returns:
        List[str]: Image/PDF file paths.
    """
    path = Path(input_path)
    image_paths = []

    if path.is_file():
        # Single file
        suffix = path.suffix.lower()
        if suffix in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".pdf"]:
            image_paths.append(str(path.absolute()))
        else:
            raise ValueError(f"Not Supported Type: {path.suffix}")
    elif path.is_dir():
        # Directory: find all image and PDF files
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif", "*.webp", "*.pdf"]:
            image_paths.extend([str(p.absolute()) for p in path.glob(ext)])
            image_paths.extend([str(p.absolute()) for p in path.glob(ext.upper())])
        image_paths.sort()
        if not image_paths:
            raise ValueError(
                f"Cannot find image or PDF files in directory: {input_path}"
            )
    else:
        raise ValueError(f"Path does not exist: {input_path}")

    return image_paths


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="GlmOcr - Document Parsing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Parse a single image file
  glmocr parse image.png

    # Parse all images in a directory
  glmocr parse ./images/

    # Disable layout detection (OCR-only): set pipeline.enable_layout=false
    glmocr parse image.png --config my_config.yaml

    # Specify output directory
  glmocr parse image.png --output ./output/

    # Specify config file
  glmocr parse image.png --config config.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # parse command
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
        help="Do not save layout visualization results (only effective when enable_layout=true)",
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    configure_logging(level=args.log_level)

    try:
        # Load inputs
        logger.info("Loading images: %s", args.input)
        image_paths = load_image_paths(args.input)
        logger.info("Found %d file(s)", len(image_paths))

        # Use GlmOcr API
        save_layout_vis = not args.no_layout_vis

        with GlmOcr(config_path=args.config) as glm_parser:
            logger.info(
                "Using Pipeline (enable_layout=%s)...",
                "true" if glm_parser.enable_layout else "false",
            )

            # Process each file (parse() with str returns a single PipelineResult)
            total_files = len(image_paths)
            for idx, image_path in enumerate(image_paths, 1):
                file_name = Path(image_path).name
                logger.info("")
                logger.info("=== Parsing: %s (%d/%d) ===", file_name, idx, total_files)

                try:
                    result = glm_parser.parse(
                        image_path, save_layout_visualization=save_layout_vis
                    )
                    # Output
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

                    # Save to files by default (unless --no-save)
                    if not args.no_save:
                        result.save(
                            output_dir=args.output,
                            save_layout_visualization=save_layout_vis,
                        )

                except Exception as e:
                    logger.error("Failed: %s: %s", file_name, e)
                    logger.debug(traceback.format_exc())
                    continue

        logger.info("")
        logger.info("All done!")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Error: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
