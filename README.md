## GLM-OCR

<div align="center">
<img src=resources/logo.svg width="40%"/>
</div>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/QR7SARHRxK" target="_blank">Discord</a> community
    <br>
    📍 Use GLM-OCR's <a href="https://docs.z.ai/guides/vlm/glm-ocr" target="_blank">API</a>
</p>

### Model Introduction

GLM-OCR is a multimodal OCR model for complex document understanding, built on the GLM-V encoder–decoder architecture. It introduces Multi-Token Prediction (MTP) loss and stable full-task reinforcement learning to improve training efficiency, recognition accuracy, and generalization. The model integrates the CogViT visual encoder pre-trained on large-scale image–text data, a lightweight cross-modal connector with efficient token downsampling, and a GLM-0.5B language decoder. Combined with a two-stage pipeline of layout analysis and parallel recognition based on PP-DocLayout-V3, GLM-OCR delivers robust and high-quality OCR performance across diverse document layouts.

**Key Features**

- **State-of-the-Art Performance**: Achieves a score of 94.62 on OmniDocBench V1.5, ranking #1 overall, and delivers state-of-the-art results across major document understanding benchmarks, including formula recognition, table recognition, and information extraction.

- **Optimized for Real-World Scenarios**: Designed and optimized for practical business use cases, maintaining robust performance on complex tables, code-heavy documents, seals, and other challenging real-world layouts.

- **Efficient Inference**: With only 0.9B parameters, GLM-OCR supports deployment via vLLM, SGLang, and Ollama, significantly reducing inference latency and compute cost, making it ideal for high-concurrency services and edge deployments.

- **Easy to Use**: Fully open-sourced and equipped with a comprehensive [SDK](https://github.com/zai-org/GLM-OCR) and inference toolchain, offering simple installation, one-line invocation, and smooth integration into existing production pipelines.

### Download Model

| Model   | Download Links                                                                                                              | Precision |
| ------- | --------------------------------------------------------------------------------------------------------------------------- | --------- |
| GLM-OCR | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-OCR)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR) | BF16      |

## GLM-OCR SDK

We provide an SDK for using GLM-OCR more efficiently and conveniently.

### Install SDK

```bash
# Install from source
git clone https://github.com/zai-org/glm-ocr.git
cd glm-ocr && pip install -e .

# Install transformers from source
pip install git+https://github.com/huggingface/transformers.git
```

### Model Deployment

Two ways to use GLM-OCR:

#### Option 1: Zhipu MaaS API (Recommended for Quick Start)

Use the hosted cloud API – no GPU needed. The cloud service runs the complete GLM-OCR pipeline internally, so the SDK simply forwards your request and returns the result.

1. Get an API key from https://open.bigmodel.cn
2. Configure `config.yaml`:

```yaml
pipeline:
  maas:
    enabled: true # Enable MaaS mode
    api_key: your-api-key # Required
```

That's it! When `maas.enabled=true`, the SDK acts as a thin wrapper that:

- Forwards your documents to the Zhipu cloud API
- Returns the results directly (Markdown + JSON layout details)
- No local processing, no GPU required

Input note (MaaS): the upstream API accepts `file` as a URL or a `data:<mime>;base64,...` data URI.
If you have raw base64 without the `data:` prefix, wrap it as a data URI (recommended). The SDK will
auto-wrap local file paths / bytes / raw base64 into a data URI when calling MaaS.

API documentation: https://docs.bigmodel.cn/cn/guide/models/vlm/glm-ocr

#### Option 2: Self-host with vLLM / SGLang

Deploy the GLM-OCR model locally for full control. The SDK provides the complete pipeline: layout detection, parallel region OCR, and result formatting.

##### Using vLLM

Install vLLM:

```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
# Or use Docker
docker pull vllm/vllm-openai:nightly
```

Launch the service:

```bash
pip install git+https://github.com/huggingface/transformers.git

# Run with MTP for better performance
vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080 --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
```

##### Using SGLang

Install SGLang:

```bash
docker pull lmsysorg/sglang:dev
# Or build from source
pip install git+https://github.com/sgl-project/sglang.git#subdirectory=python
```

Launch the service:

```bash
pip install git+https://github.com/huggingface/transformers.git

# Run with MTP for better performance
python -m sglang.launch_server --model zai-org/GLM-OCR --port 8080 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
# Modify the speculative config base on your device
```

##### Update Configuration

After launching the service, configure `config.yaml`:

```yaml
pipeline:
  maas:
    enabled: false # Disable MaaS mode (default)
  ocr_api:
    api_host: localhost # or your vLLM/SGLang server address
    api_port: 8080
```

### SDK Usage Guide

#### CLI

```bash
# Parse a single image
glmocr parse examples/source/code.png

# Parse a directory
glmocr parse examples/source/

# Set output directory
glmocr parse examples/source/code.png --output ./results/

# Use a custom config
glmocr parse examples/source/code.png --config my_config.yaml

# Enable debug logging with profiling
glmocr parse examples/source/code.png --log-level DEBUG
```

#### Python API

```python
from glmocr import GlmOcr, parse

# Simple function
result = parse("image.png")
result = parse(["img1.png", "img2.jpg"])
result = parse("https://example.com/image.png")
result.save(output_dir="./results")

# Note: a list is treated as pages of a single document.

# Class-based API
with GlmOcr() as parser:
    result = parser.parse("image.png")
    print(result.json_result)
    result.save()
```

#### Flask Service

```bash
# Start service
python -m glmocr.server

# With debug logging
python -m glmocr.server --log-level DEBUG

# Call API
curl -X POST http://localhost:5002/glmocr/parse \
  -H "Content-Type: application/json" \
  -d '{"images": ["./example/source/code.png"]}'
```

Semantics:

- `images` can be a string or a list.
- A list is treated as pages of a single document.
- For multiple independent documents, call the endpoint multiple times (one document per request).

### Configuration

Full configuration in `glmocr/config.yaml`:

```yaml
# Server (for glmocr.server)
server:
  host: "0.0.0.0"
  port: 5002
  debug: false

# Logging
logging:
  level: INFO # DEBUG enables profiling

# Pipeline
pipeline:
  # OCR API connection
  ocr_api:
    api_host: localhost
    api_port: 8080
    api_key: null # or set API_KEY env var
    connect_timeout: 300
    request_timeout: 300

  # Page loader settings
  page_loader:
    max_tokens: 16384
    temperature: 0.01
    image_format: JPEG
    min_pixels: 12544
    max_pixels: 71372800

  # Result formatting
  result_formatter:
    output_format: both # json, markdown, or both

  # Layout detection (optional)
  enable_layout: false
```

See [config.yaml](glmocr/config.yaml) for all options.

### Output Formats

Here are two examples of output formats:

- JSON

```json
[[{ "index": 0, "label": "text", "content": "...", "bbox_2d": null }]]
```

- Markdown

```markdown
# Document Title

Body...

| Table | Content |
| ----- | ------- |
| ...   | ...     |
```

### Example of full pipeline

you can run example code like：

```bash
python examples/example.py
```

Output structure (one folder per input):

- `result.json` – structured OCR result
- `result.md` – Markdown result
- `imgs/` – cropped image regions (when layout mode is enabled)

### Modular Architecture

GLM-OCR uses composable modules for easy customization:

| Component             | Description                            |
| --------------------- | -------------------------------------- |
| `PageLoader`          | Preprocessing and image encoding       |
| `OCRClient`           | Calls the GLM-OCR model service        |
| `PPDocLayoutDetector` | PP-DocLayout layout detection          |
| `ResultFormatter`     | Post-processing, outputs JSON/Markdown |

You can extend the behavior by creating custom pipelines:

```python
from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.postprocess import ResultFormatter


class MyPipeline:
  def __init__(self, config):
    self.page_loader = PageLoader(config)
    self.ocr_client = OCRClient(config)
    self.formatter = ResultFormatter(config)

  def process(self, request_data):
    # Implement your own processing logic
    pass
```

## Acknowledgement

This project is inspired by the excellent work of the following projects and communities:

- [PP-DocLayout-V3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)

## License

The Code of this repo is under Apache License 2.0.

The GLM-OCR model is released under the MIT License.

The complete OCR pipeline integrates [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) for document layout analysis, which is licensed under the Apache License 2.0. Users should comply with both licenses when using this project.
