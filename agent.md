# GLM-OCR SDK — Agent Integration Guide

This document describes how AI agents, MCP tool servers, and other
programmatic callers can integrate with the **GLM-OCR Python SDK** without
editing YAML files or understanding the internal pipeline.

---

## Quick Start

```python
import glmocr

# One-liner — uses GLMOCR_API_KEY from environment / .env file
result = glmocr.parse("document.pdf")
print(result.to_dict())
```

Or use the class-based API for multiple calls:

```python
from glmocr import GlmOcr

parser = GlmOcr(api_key="sk-xxx", mode="maas")
result = parser.parse("page.png")
print(result.to_json())
parser.close()            # or use:  with GlmOcr(...) as parser:
```

---

## Two Operational Modes

| Mode            | Value          | Requires GPU? | Description                                                        |
| --------------- | -------------- | ------------- | ------------------------------------------------------------------ |
| **MaaS**        | `"maas"`       | No            | Forwards requests to Zhipu's cloud API. Recommended for agents.    |
| **Self-hosted** | `"selfhosted"` | Yes           | Uses a local vLLM / SGLang service with optional layout detection. |

When `api_key` is provided without an explicit `mode`, the SDK **automatically
defaults to MaaS mode**.

---

## Configuration Priority

The SDK resolves every setting using this priority chain (highest wins):

```
Constructor kwargs  >  os.environ  >  .env file  >  config.yaml  >  built-in defaults
```

This means an agent can override any setting without touching files.

---

## Environment Variables

All variables use the prefix `GLMOCR_`. Place them in the shell environment
or in a `.env` file anywhere in the working-directory ancestry.

| Variable               | Maps to                         | Example                             |
| ---------------------- | ------------------------------- | ----------------------------------- |
| `GLMOCR_MODE`          | `pipeline.maas.enabled`         | `maas` or `selfhosted`              |
| `GLMOCR_API_KEY`       | `pipeline.maas.api_key`         | `sk-abc123`                         |
| `GLMOCR_API_URL`       | `pipeline.maas.api_url`         | `https://open.bigmodel.cn/...`      |
| `GLMOCR_MODEL`         | `pipeline.maas.model`           | `glm-ocr`                           |
| `GLMOCR_TIMEOUT`       | `pipeline.maas.request_timeout` | `600`                               |
| `GLMOCR_OCR_API_URL`   | `pipeline.ocr_api.api_url`      | `http://localhost:5002/v1/...`      |
| `GLMOCR_OCR_API_KEY`   | `pipeline.ocr_api.api_key`      | `token-xyz`                         |
| `GLMOCR_OCR_API_HOST`  | `pipeline.ocr_api.api_host`     | `localhost`                         |
| `GLMOCR_OCR_API_PORT`  | `pipeline.ocr_api.api_port`     | `5002`                              |
| `GLMOCR_OCR_MODEL`     | `pipeline.ocr_api.model`        | `glm-ocr-model`                     |
| `GLMOCR_ENABLE_LAYOUT` | `pipeline.enable_layout`        | `true` / `false`                    |
| `GLMOCR_LOG_LEVEL`     | `logging.level`                 | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### `.env` File Auto-Loading

The SDK walks up from the current working directory looking for a `.env` file.
Values from the `.env` file are merged with real environment variables, with
**real env vars always taking priority**.

```dotenv
# .env
GLMOCR_API_KEY=sk-my-secret-key
GLMOCR_MODE=maas
GLMOCR_LOG_LEVEL=DEBUG
```

---

## Constructor Parameters

`GlmOcr()` and the convenience `parse()` function accept these keyword
arguments. They map to the same settings as the environment variables but
with **higher priority**.

| Keyword         | Type   | Description                                               |
| --------------- | ------ | --------------------------------------------------------- |
| `config_path`   | `str`  | Path to a YAML config file (optional).                    |
| `api_key`       | `str`  | API key. Providing this without `mode` auto-enables MaaS. |
| `api_url`       | `str`  | MaaS API endpoint URL.                                    |
| `model`         | `str`  | Model name.                                               |
| `mode`          | `str`  | `"maas"` or `"selfhosted"`.                               |
| `timeout`       | `int`  | Request timeout in seconds.                               |
| `enable_layout` | `bool` | Enable layout detection.                                  |
| `log_level`     | `str`  | Logging level.                                            |

---

## `parse()` Return Type

The return type **mirrors the input type** for ergonomic usage:

```python
# Single file → single PipelineResult
result = parser.parse("image.png")
result.save("./output")

# Multiple files → list of PipelineResult
results = parser.parse(["img1.png", "doc.pdf"])
for r in results:
    r.save("./output")
```

Type checkers see proper `@overload` signatures — no casts needed.

---

## Result Serialization

Every `PipelineResult` can be serialized without touching the file system:

### `to_dict()`

Returns a JSON-serializable Python dict:

```python
d = result.to_dict()
# {
#   "json_result": [[{"index": 0, "label": "text", "content": "...", "bbox_2d": [...]}]],
#   "markdown_result": "# Page title\n...",
#   "original_images": ["/abs/path/to/image.png"],
#   "usage": {"total_tokens": 1234},      # present in MaaS mode
#   "data_info": {"pages": [...]},         # present in MaaS mode
# }
```

### `to_json(**kwargs)`

Returns a JSON string. Keyword arguments are forwarded to `json.dumps`.
Defaults: `ensure_ascii=False, indent=2`.

```python
json_str = result.to_json()                 # pretty-printed
json_str = result.to_json(indent=None)      # compact single line
```

### `save(output_dir)`

Writes JSON + Markdown files (with cropped images) to disk:

```python
result.save(output_dir="./output")
```

---

## Structured Output — `json_result`

`json_result` is a list of pages, each page a list of regions:

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
      "label": "text",
      "content": "Revenue grew 15% year-over-year...",
      "bbox_2d": [100, 140, 900, 400]
    }
  ]
]
```

**Coordinates** (`bbox_2d`) are normalised to a **0–1000** scale regardless
of the backend (MaaS or self-hosted).

**Labels**: `title`, `text`, `table`, `figure`, `formula`, `header`,
`footer`, `page_number`, `reference`, etc.

---

## Error Handling

When a MaaS request fails, the SDK returns a `PipelineResult` with an
`_error` attribute instead of raising:

```python
result = parser.parse("image.png")
d = result.to_dict()
if "error" in d:
    print("Parsing failed:", d["error"])
else:
    print(d["markdown_result"])
```

---

## MCP / Tool Integration Pattern

When wrapping GLM-OCR as an MCP tool:

```python
import json
import glmocr

def ocr_tool(image_path: str) -> str:
    """Parse a document and return structured JSON."""
    result = glmocr.parse(image_path)
    return result.to_json()
```

The tool only needs `GLMOCR_API_KEY` in the environment (or `.env` file).
No YAML configuration is required.

---

## Programmatic Config Construction

For advanced use cases you can build config objects directly:

```python
from glmocr.config import GlmOcrConfig

cfg = GlmOcrConfig.from_env(
    api_key="sk-xxx",
    mode="maas",
    timeout=600,
    log_level="DEBUG",
)
print(cfg.to_dict())
```

`from_env()` respects the full priority chain:
`kwargs > os.environ > .env > YAML > defaults`.

---

## Running Tests

```bash
python -m pytest glmocr/tests/test_unit.py -v
```

All tests run without network access or GPU. MaaS/Pipeline internals are
mocked where needed.
