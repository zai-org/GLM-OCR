# MLX Deployment Guide for GLM-OCR (Apple Silicon)

Run GLM-OCR natively on Apple Silicon Macs using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), which leverages the Metal GPU for efficient inference.

## Requirements

- Apple Silicon Mac (M series chip)
- macOS 14.0 (Sonoma) or later — required by the [MLX framework](https://ml-explore.github.io/mlx/build/html/install.html)
- Python 3.10+

## Why a Separate Environment?

mlx-vlm requires `transformers>=5.0.0rc3`; the GLM-OCR SDK's `layout` extra
(used for PP-DocLayout-V3 layout detection) requires `transformers>=5.3.0`.
These ranges now overlap, so a single environment can resolve both. This guide
still uses **two Python environments**, which keeps the inference server and the
SDK independently upgradable and is the layout this guide is tested against:

| Environment | Purpose                                         | Key dependency           |
| ----------- | ----------------------------------------------- | ------------------------ |
| **mlx-env** | Runs the mlx-vlm inference server               | `transformers>=5.0.0rc3` |
| **sdk-env** | Runs the GLM-OCR SDK (CLI / Python API / Flask) | `transformers` (stable)  |

The SDK talks to the mlx-vlm server over HTTP, so the two environments can
run side-by-side without any conflicts.

## Step 1 - Set Up the mlx-vlm Server Environment

Create an isolated environment for the server:

```bash
# Using conda
conda create -n mlx-env python=3.12 -y
conda activate mlx-env

# Or using venv
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
```

Install mlx-vlm. The `glm_ocr` model architecture ships in the PyPI release as
of **0.7.0**, so the plain install is enough:

```bash
pip install 'mlx-vlm>=0.7.0'
```

> **Note:** Older releases (up to 0.3.10) predate GLM-OCR support. If you are
> pinned to one of those, or want an unreleased fix, install from git instead:
> `pip install git+https://github.com/Blaizzy/mlx-vlm.git`

## Step 2 - Launch the mlx-vlm Server

With the `mlx-env` environment activated:

```bash
mlx_vlm.server --trust-remote-code
```

This starts an OpenAI-compatible API server on `http://localhost:8080` by
default. The first run downloads the model weights from Hugging Face
(`mlx-community/GLM-OCR-bf16`).

### Server Options

```bash
# Specify a custom port
mlx_vlm.server --trust-remote-code --port 9090
```

### Verify the Server

In a separate terminal, send a quick health-check request:

```bash
curl http://localhost:8080/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/GLM-OCR-bf16",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
    "max_tokens": 10
  }'
```

You should get a JSON response with `"choices"` in it.

## Step 3 - Set Up the GLM-OCR SDK Environment

In a **separate** terminal (keep the server running):

```bash
git clone https://github.com/zai-org/glm-ocr.git
cd glm-ocr

# Using conda
conda create -n sdk-env python=3.12 -y
conda activate sdk-env

# Or using venv
python3 -m venv .venv-sdk
source .venv-sdk/bin/activate
```

Install the GLM-OCR SDK with the `layout` extra:

```bash
pip install -e '.[layout]'
```

> **Important:** the `layout` extra is required. A bare `pip install -e .` omits
> `torch`, `torchvision` and `transformers`, and the pipeline then fails at
> startup with `ImportError: Layout detector dependencies are missing or
> incompatible`.

## Step 4 - Configure the SDK

Point the SDK at the mlx-vlm server. The key differences from vLLM/SGLang are
the `model` field, which is **required** for mlx-vlm, and `api_path` (no `/v1`
prefix):

```yaml
pipeline:
  maas:
    enabled: false

  ocr_api:
    api_host: localhost
    api_port: 8080 # Must match the mlx-vlm server port
    model: mlx-community/GLM-OCR-bf16 # Required for mlx-vlm
    api_path: /chat/completions # Remove /v1 prefix
```

> **Important — a custom config file replaces the defaults, it does not merge
> with them.** When you pass `--config my_config.yaml`, that file becomes the
> entire YAML baseline; the packaged `glmocr/config.yaml` is **not** layered
> underneath it (only `GLMOCR_*` environment variables and keyword overrides are
> applied on top). Any key you omit falls back to the built-in dataclass default,
> not to the packaged value.
>
> A file containing only the four `ocr_api` keys above therefore loses
> `pipeline.layout.model_dir` (which has no built-in default), and the run fails
> with `pipeline.layout.model_dir is required for self-hosted layout`. It also
> silently empties `layout.id2label`, `layout.label_task_mapping` and
> `page_loader.task_prompt_mapping`, which the pipeline needs to label and route
> detected regions.

So use one of these two approaches:

**Option A — edit the packaged config in place.** Apply the changes above
directly to `glmocr/config.yaml`; no `--config` flag is then needed.

**Option B — copy the packaged config, then edit the copy:**

```bash
cp glmocr/config.yaml my_config.yaml
# edit my_config.yaml: set maas.enabled=false and the four ocr_api keys above
glmocr parse examples/source/code.png --config my_config.yaml
```

> **Watch out:** `config.yaml` contains **two** `model:` keys — one under
> `pipeline.maas`, one under `pipeline.ocr_api`. The mlx-vlm model name belongs
> to `pipeline.ocr_api`. Editing the `maas` one instead leaves `ocr_api.model` as
> `glm-ocr`, which the mlx-vlm server rejects with an HTTP 400.

Do **not** hand-write a minimal `--config` file from scratch.

### Full Configuration Reference

Below is the `ocr_api` section with all available options and their defaults:

```yaml
pipeline:
  maas:
    enabled: false

  ocr_api:
    # Connection
    api_host: localhost
    api_port: 8080
    model: mlx-community/GLM-OCR-bf16 # Required for mlx-vlm

    # URL construction: {api_scheme}://{api_host}:{api_port}{api_path}
    api_scheme: null # null = auto (https if port 443, else http)
    api_path: /chat/completions # Remove /v1 prefix
    api_url: null # Full URL override (optional)

    # Authentication (not needed for local mlx-vlm)
    api_key: null
    headers: {}

    # SSL (disabled for local server)
    verify_ssl: false

    # Timeouts (seconds)
    connect_timeout: 30
    request_timeout: 120

    # Retry settings
    retry_max_attempts: 2
    retry_backoff_base_seconds: 0.5
    retry_backoff_max_seconds: 8.0
```

## Step 5 - Run GLM-OCR

With the mlx-vlm server running in one terminal and the SDK environment in
another:

```bash
# Parse a single image
glmocr parse examples/source/code.png

# Parse all files in a directory
glmocr parse examples/source/

# Parse with custom config
glmocr parse examples/source/code.png --config my_config.yaml

# Save output to a specific directory
glmocr parse examples/source/code.png --output ./results/
```

Or use the Python API:

```python
from glmocr import parse

result = parse("examples/source/code.png")
result.save(output_dir="./results")
```

## Troubleshooting

### `model glm_ocr not found` or similar import error

You are probably on an mlx-vlm release older than 0.7.0, which predates GLM-OCR
support. Upgrade:

```bash
pip install -U 'mlx-vlm>=0.7.0'
```

### `ImportError: Layout detector dependencies are missing or incompatible`

The SDK environment was installed without the `layout` extra. Reinstall with it:

```bash
pip install -e '.[layout]'
```

### `pipeline.layout.model_dir is required for self-hosted layout`

Your `--config` file is missing `pipeline.layout.model_dir`. A custom config
replaces the packaged defaults rather than merging with them, so it must be a
full copy of `glmocr/config.yaml` with your edits applied — see
[Step 4](#step-4---configure-the-sdk).

### `API server returned status code: 400`

The mlx-vlm server rejected the request, usually because `pipeline.ocr_api.model`
does not match the model the server is serving. It must be the full Hugging Face
id (e.g. `mlx-community/GLM-OCR-bf16`), not `glm-ocr`. Note that `config.yaml`
has a second, unrelated `model:` key under `pipeline.maas` — make sure you edited
the one under `pipeline.ocr_api`.

### `transformers` version conflict

If you see errors about incompatible `transformers` versions, make sure you
are using **separate environments** for the server and the SDK (see
[Why a Separate Environment?](#why-a-separate-environment)).

### Connection refused / timeout

- Confirm the mlx-vlm server is running (`mlx_vlm.server --trust-remote-code`).
- Check that `api_port` in your config matches the port the server is listening
  on (default: `8080`).
- Try the curl health-check command from [Step 2](#verify-the-server).

### Out of memory

GLM-OCR is a 0.9B-parameter model and should fit comfortably on 8 GB unified
memory. If you still hit OOM:

- Close other memory-intensive applications.
- Reduce `max_tokens` in `config.yaml` (e.g., `2048`).

### Slow first request

The first inference request is slower because mlx-vlm compiles the Metal
shaders and warms up the model. Subsequent requests will be significantly
faster.

## Architecture Overview

```
+------------------+        HTTP (OpenAI-compatible)        +-------------------+
|  GLM-OCR SDK     | -------------------------------------> |  mlx-vlm server   |
|  (sdk-env)       |    POST /chat/completions              |  (mlx-env)        |
|                  | <------------------------------------- |                   |
|  - Layout detect |        JSON response                   |  - GLM-OCR model  |
|  - Region crop   |                                        |  - Metal GPU      |
|  - Result format |                                        |                   |
+------------------+                                        +-------------------+
```

The SDK handles the full pipeline (layout detection, region cropping, parallel
OCR requests, result formatting) while the mlx-vlm server handles model
inference on the Metal GPU.
