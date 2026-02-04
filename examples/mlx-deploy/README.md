# MLX Deployment Guide for GLM-OCR (Apple Silicon)

Run GLM-OCR natively on Apple Silicon Macs using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), which leverages the Metal GPU for efficient inference.

## Requirements

- Apple Silicon Mac (M series chip)
- macOS 14.0 (Sonoma) or later — required by the [MLX framework](https://ml-explore.github.io/mlx/build/html/install.html)
- Python 3.10+

## Why a Separate Environment?

mlx-vlm currently requires `transformers>=5.0.0rc3`, which conflicts with the
`transformers` version pinned by the GLM-OCR SDK (used for PP-DocLayout-V3
layout detection). Until these version requirements converge in a future
release, **you need two Python environments**:

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

Install mlx-vlm **from git** (the `glm_ocr` model architecture is not yet
available in the latest PyPI release 0.3.10):

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git
```

> **Note:** Once a new mlx-vlm release ships with GLM-OCR support, you can
> switch to `pip install mlx-vlm` instead.

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

Install the GLM-OCR SDK:

```bash
pip install -e .

# Install transformers from source (required by the SDK)
pip install git+https://github.com/huggingface/transformers.git
```

## Step 4 - Configure the SDK

Edit `glmocr/config.yaml` (or create a custom config file) to point the SDK
at the mlx-vlm server. The key difference from vLLM/SGLang is the `model`
field, which is **required** for mlx-vlm, and api_path (no `/v1` prefix):

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

You may be using the PyPI release of mlx-vlm which doesn't include GLM-OCR
support yet. Install from git instead:

```bash
pip install git+https://github.com/Blaizzy/mlx-vlm.git
```

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
