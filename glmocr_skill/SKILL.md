---
name: glmocr
description: |
  Trigger when: (1) User wants to extract text, tables, formulas, or structured data from images/PDFs/scanned documents, (2) User mentions "OCR", "文字识别", "文档解析", (3) User has a document (screenshot, scanned page, invoice, paper, whiteboard photo) and needs its content in structured form, (4) User asks to parse, digitize, or extract content from a visual document.

  Invokes the GLM-OCR SDK (pip install glmocr) to parse documents via Zhipu's cloud API. No GPU required. Returns structured JSON (regions with labels + bounding boxes) and Markdown. Agent can operate entirely via CLI — no YAML files needed.

  NOT for: real-time camera feeds, audio transcription, or non-document images (photos, illustrations).
---

# OpenClaw Skill: glmocr

Parses documents (images, PDFs, scans) via the GLM-OCR SDK.

> **📌 On-demand**: This skill requires only `GLMOCR_API_KEY` in the environment. No YAML config files or GPU needed.

## ⚡ Quick Start

```bash
# Install
pip install glmocr

# Set API key (once)
export GLMOCR_API_KEY=sk-xxx
# or add to .env file in working directory:
echo "GLMOCR_API_KEY=sk-xxx" >> .env
```

```python
# One-liner
import glmocr
result = glmocr.parse("document.pdf")
print(result.markdown_result)
print(result.to_dict())
```

```bash
# CLI — pass API key directly (no env setup needed)
glmocr parse image.png --api-key sk-xxx

# Or load from a specific .env file
glmocr parse image.png --env-file /path/to/.env

# Or rely on env var / auto-discovered .env (set once, then omit)
glmocr parse image.png
glmocr parse ./scans/ --output ./output/ --stdout
```

---

## Configuration Priority

```
Constructor kwargs  >  os.environ  >  .env file  >  config.yaml  >  built-in defaults
```

Agents override everything via constructor kwargs or env vars — no YAML editing needed.

### Key Environment Variables

| Variable               | Description                            | Example     |
| ---------------------- | -------------------------------------- | ----------- |
| `GLMOCR_API_KEY`       | API key (required for MaaS)            | `sk-abc123` |
| `GLMOCR_MODEL`         | Model name                             | `glm-ocr`   |
| `GLMOCR_TIMEOUT`       | Request timeout (seconds)              | `600`       |
| `GLMOCR_ENABLE_LAYOUT` | Layout detection on/off                | `true`      |
| `GLMOCR_LOG_LEVEL`     | `DEBUG` / `INFO` / `WARNING` / `ERROR` | `INFO`      |

---

## Python API

### Convenience function (single call)

```python
import glmocr

# Single file → PipelineResult
result = glmocr.parse("invoice.png")

# Multiple files → list[PipelineResult]
results = glmocr.parse(["page1.png", "page2.png", "report.pdf"])
```

### Class-based (multiple calls / resource reuse)

```python
from glmocr import GlmOcr

parser = GlmOcr(api_key="sk-xxx")   # mode auto-set to "maas"
parser = GlmOcr(mode="maas")        # reads GLMOCR_API_KEY from env

# Always use as context manager or call .close()
with GlmOcr(api_key="sk-xxx") as parser:
    result = parser.parse("document.png")
    print(result.markdown_result)

parser.close()   # if not using `with`
```

### Constructor Parameters

| Parameter       | Type   | Description                                     |
| --------------- | ------ | ----------------------------------------------- |
| `api_key`       | `str`  | API key. Providing this auto-enables MaaS mode. |
| `api_url`       | `str`  | Override MaaS endpoint URL                      |
| `model`         | `str`  | Model name override                             |
| `timeout`       | `int`  | Request timeout in seconds (default: 600)       |
| `enable_layout` | `bool` | Enable layout detection                         |
| `log_level`     | `str`  | Logging level                                   |

---

## Working with `PipelineResult`

### Fields

```python
result.markdown_result    # str — full document as Markdown
result.json_result        # list[list[dict]] — structured regions per page
result.original_images    # list[str] — absolute paths of input images
```

### `json_result` structure

List of pages → list of regions per page:

```json
[
  [
    {
      "index": 0,
      "label": "title",
      "content": "Annual Report 2024",
      "bbox_2d": [100, 50, 900, 120]
    },
    {
      "index": 1,
      "label": "table",
      "content": "| Q1 | Q2 |\n|---|---|\n| 120 | 145 |",
      "bbox_2d": [100, 140, 900, 400]
    }
  ]
]
```

**Bounding boxes** (`bbox_2d`): `[x1, y1, x2, y2]` normalised to **0–1000** scale.

**Region labels**: `title`, `text`, `table`, `figure`, `formula`, `header`, `footer`, `page_number`, `reference`, `seal`

### Serialization

```python
# Dict (JSON-serializable, for passing to other tools)
d = result.to_dict()
# Keys: json_result, markdown_result, original_images, usage (MaaS), data_info (MaaS)

# JSON string
json_str = result.to_json()                 # pretty-printed, ensure_ascii=False
json_str = result.to_json(indent=None)      # compact single line

# Save to disk: writes <stem>/<stem>.json + <stem>/<stem>.md + layout_vis/
result.save(output_dir="./output")
result.save(output_dir="./output", save_layout_visualization=False)
```

### Error Handling

The SDK **does not raise** on MaaS errors — check `to_dict()` for an `"error"` key:

```python
result = parser.parse("image.png")
d = result.to_dict()
if "error" in d:
    # Handle failure
    print("OCR failed:", d["error"])
else:
    print(d["markdown_result"])
```

---

## CLI Reference

> **Agent-preferred interface**: use the CLI for most operations. Set `GLMOCR_API_KEY` in env once, then invoke as needed.

**Supported input formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.pdf`

### Basic usage

```bash
# Parse a single file → saves to ./output/<stem>/
# MaaS mode is the default; GLMOCR_API_KEY must be set (or use --api-key)
glmocr parse image.png

# Pass API key directly without any env setup
glmocr parse image.png --api-key sk-xxx

# Parse a directory → saves each file to ./output/<stem>/
glmocr parse ./scans/

# Use self-hosted vLLM/SGLang instead of cloud
glmocr parse image.png --mode selfhosted

# Specify output directory
glmocr parse image.png --output ./results/
```

### Read results in the terminal (agent-friendly)

```bash
# Print Markdown + JSON to stdout (and still save to disk)
glmocr parse image.png --stdout

# Print to stdout ONLY — do not write any files
glmocr parse image.png --stdout --no-save

# JSON only (no Markdown output)
glmocr parse image.png --stdout --json-only

# Pipe JSON into jq for structured extraction
glmocr parse image.png --stdout --json-only --no-save | jq '.[0] | map(select(.label=="table"))'
```

### Save control

```bash
# Skip layout visualization images (faster, smaller output)
glmocr parse image.png --no-layout-vis

# Parse and save only JSON + Markdown, skip layout vis
glmocr parse image.png --no-layout-vis --output ./results/
```

### Batch processing

```bash
# All images in a folder
glmocr parse ./invoice_scans/ --output ./parsed/ --no-layout-vis

# With progress visible in logs
glmocr parse ./docs/ --output ./parsed/ --log-level INFO
```

### Debugging

```bash
glmocr parse image.png --log-level DEBUG
```

### Full flag reference

| Flag              | Default    | Description                                           |
| ----------------- | ---------- | ----------------------------------------------------- |
| `--api-key / -k`  | env var    | API key for MaaS mode (overrides `GLMOCR_API_KEY`)    |
| `--mode`          | `maas`     | `maas` (cloud, default) or `selfhosted` (local GPU)   |
| `--env-file`      | auto       | Path to `.env` file (default: auto-discover from cwd) |
| `--output / -o`   | `./output` | Output directory                                      |
| `--stdout`        | off        | Print JSON + Markdown to stdout                       |
| `--no-save`       | off        | Skip writing files (use with `--stdout`)              |
| `--json-only`     | off        | stdout JSON only, no Markdown                         |
| `--no-layout-vis` | off        | Skip layout visualization images                      |
| `--config / -c`   | none       | Path to YAML config override                          |
| `--log-level`     | `INFO`     | `DEBUG` / `INFO` / `WARNING` / `ERROR`                |

---

## Typical Agent Workflow

```
receive document path / URL
       │
       ▼
glmocr.parse(path)            ← single call, handles PDF/image
       │
       ▼
result.to_dict()              ← safe to pass as tool output
       │
       ├── markdown_result    → hand to LLM for reading / summarization
       └── json_result        → structured extraction (tables, formulas, regions by label)
```

### Filter by label

```python
result = glmocr.parse("report.png")
regions = result.json_result[0]  # first page

tables = [r for r in regions if r["label"] == "table"]
formulas = [r for r in regions if r["label"] == "formula"]
body_text = [r for r in regions if r["label"] == "text"]
```

### Multi-page PDF → iterate pages

```python
with GlmOcr(api_key="sk-xxx") as parser:
    result = parser.parse("document.pdf")   # all pages in one PipelineResult
    for page_idx, page_regions in enumerate(result.json_result):
        print(f"Page {page_idx + 1}: {len(page_regions)} regions")
        for region in page_regions:
            print(f"  [{region['label']}] {region['content'][:60]}")
```

### Programmatic config (no env vars)

```python
from glmocr.config import GlmOcrConfig

cfg = GlmOcrConfig.from_env(
    api_key="sk-xxx",
    mode="maas",
    timeout=600,
    log_level="DEBUG",
)
```

---

## Output Directory Layout

After `result.save(output_dir)`:

```
output_dir/
  <image_stem>/
    <image_stem>.json         ← structured regions
    <image_stem>.md           ← full Markdown (with cropped figure images)
    imgs/                     ← cropped figures referenced in Markdown
    layout_vis/               ← layout detection overlay images (if enabled)
      <image_stem>.jpg
```

---

## Common Pitfalls

- **`GLMOCR_API_KEY` not set**: SDK defaults to MaaS mode. Without a key, `parse()` will fail with a clear error message and quick-fix instructions. Set via `export GLMOCR_API_KEY=sk-xxx`, add to a `.env` file, or pass `--api-key sk-xxx` to the CLI.
- **Large PDFs**: Default timeout is 600s. For very long documents increase with `timeout=1200`.
- **`result.json_result` is a string**: Happens when the model returns malformed JSON. The SDK preserves the raw string — parse or log it manually.
