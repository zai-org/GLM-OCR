# Self-hosted SDK Server + Client Mode

The GLM-OCR SDK supports a split **Server / Client** deployment: run the SDK Server on a GPU machine, and connect from any other machine over HTTP — no GPU required on the client side.

## Architecture

```
┌──────────────────────┐       HTTP        ┌──────────────────────────────┐
│  Client (no GPU)     │ ────────────────→ │  Server (GPU machine)        │
│                      │  POST /glmocr/parse│                              │
│  glmocr CLI / Python │ ←──────────────── │  python -m glmocr.server     │
│                      │  JSON response     │  (layout + OCR pipeline)    │
└──────────────────────┘                    └──────────────────────────────┘
```

The Server runs the full OCR pipeline (layout detection + parallel OCR). The Client calls it over HTTP with zero local computation.

## When to choose each deployment shape

| Deployment shape | Best for | Trade-off |
|---|---|---|
| MaaS API | Fastest first run, no GPU | Depends on external API availability |
| SDK Server + Client | Team-internal OCR service with a GPU server | You operate one always-on server |
| Direct local vLLM / SGLang | Single-machine experiments and benchmarking | Client and model runtime share the same machine resources |

If you already have a GPU box that serves several teammates, the SDK Server mode is usually the easiest way to expose a stable OCR endpoint without asking every client to install the self-hosted dependencies.

## Server Side

On the GPU machine, start the Server:

```bash
# 1. Install (includes selfhosted pipeline + server)
pip install "glmocr[selfhosted,server]"

# 2. Configure self-hosted mode and point to your local vLLM / SGLang
#    In config.yaml: set pipeline.maas.enabled: false and configure ocr_api

# 3. Start the server
python -m glmocr.server --config config.yaml
```

The server listens on `0.0.0.0:5002` by default, with the API endpoint at `/glmocr/parse`.

## Client Side

On any machine (no GPU needed), point the SDK's MaaS client at your self-hosted server:

```bash
pip install glmocr
```

Edit `config.yaml`:

```yaml
pipeline:
  maas:
    enabled: true
    api_url: http://<SERVER_IP>:<SERVER_PORT>/glmocr/parse
    api_key: any-string        # Self-hosted server does not validate API keys
    verify_ssl: false          # Internal networks typically lack HTTPS
```

Then use the CLI or Python API:

```bash
# CLI
glmocr parse document.png --config config.yaml

# Or set via environment variable
export ZHIPU_API_KEY=any-string
glmocr parse document.png
```

```python
# Python API
from glmocr import GlmOcr

with GlmOcr(
    mode="maas",
    api_url="http://<SERVER_IP>:5002/glmocr/parse",
    api_key="any-string",
) as parser:
    result = parser.parse("document.png")
    print(result.markdown_result)
```

## Protocol Details

The server accepts both input formats:

| Input format | Example |
|---|---|
| SDK native | `{"images": ["url1", "url2"]}` |
| MaaS compatible | `{"file": "url", "model": "glm-ocr"}` |

The server response includes both SDK and MaaS field sets:

```json
{
  "json_result": [...],
  "markdown_result": "...",
  "layout_details": [...],
  "md_results": "...",
  "data_info": {"pages": []},
  "usage": {},
  "model": "glm-ocr",
  "id": "chatcmpl-...",
  "created": 1709234567
}
```

This means the client can use the SDK CLI / Python API or send raw HTTP requests — both will parse the response correctly.
