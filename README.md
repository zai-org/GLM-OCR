## GLM-OCR

[中文阅读](README_zh.md)

<div align="center">
<img src=resources/logo.svg width="40%"/>
</div>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/QR7SARHRxK" target="_blank">Discord</a> community
    <br>
    📖 Check out the GLM-OCR <a href="https://arxiv.org/abs/2603.10910" target="_blank">technical report</a>
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

### News & Updates

- **[2026.3.12]** GLM-OCR SDK now supports agent-friendly Skill mode — just `pip install glmocr` + set API key, ready to use via CLI or Python with no GPU or YAML config needed. See: [GLM-OCR Skill](skills/glmocr/SKILL.md)
- **[2026.3.12]** GLM-OCR Technical Report is now available. See: [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910)
- **[2026.2.12]** Fine-tuning tutorial based on LLaMA-Factory is now available. See: [GLM-OCR Fine-tuning Guide](examples/finetune/README.md)

### Download Model

| Model   | Download Links                                                                                                              | Precision |
| ------- | --------------------------------------------------------------------------------------------------------------------------- | --------- |
| GLM-OCR | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-OCR)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR) | BF16      |

## GLM-OCR SDK

We provide an SDK for using GLM-OCR more efficiently and conveniently.

### Install SDK

Choose the lightest installation that matches your scenario:

```bash
# Cloud / MaaS + local images / PDFs (fastest install)
pip install glmocr

# Self-hosted pipeline (layout detection)
pip install "glmocr[selfhosted]"

# Flask service support
pip install "glmocr[server]"
```

Install from source for development:

```bash
# Install from source
git clone https://github.com/zai-org/glm-ocr.git
cd glm-ocr
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e .
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

Install the self-hosted extra first:

```bash
pip install "glmocr[selfhosted]"
```

##### Using vLLM

Install vLLM:

```bash
docker pull vllm/vllm-openai:nightly
```

Or using with pip:

```bash
pip install -U "vllm>=0.17.0"
```

Launch the service:

```bash
pip install "transformers>=5.3.0"

vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080 --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}' --served-model-name glm-ocr
```

##### Using SGLang

Install SGLang:

```bash
docker pull lmsysorg/sglang:dev
```

Or using with pip:

```bash
pip install "sglang>=0.5.9"
```

Launch the service:

```bash
pip install "transformers>=5.3.0"

sglang serve --model zai-org/GLM-OCR --port 8080 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --served-model-name glm-ocr
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

#### Option 3: Ollama/MLX

For specialized deployment scenarios, see the detailed guides:

- **[Apple Silicon with mlx-vlm](examples/mlx-deploy/README.md)** - Optimized for Apple Silicon Macs
- **[Ollama Deployment](examples/ollama-deploy/README.md)** - Simple local deployment with Ollama

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

# Run layout detection on CPU (keep GPU free for OCR model)
glmocr parse examples/source/code.png --layout-device cpu

# Run layout detection on a specific GPU
glmocr parse examples/source/code.png --layout-device cuda:1

# Override any config value via --set (dotted path, repeatable)
glmocr parse examples/source/code.png --set pipeline.ocr_api.api_port 8080
glmocr parse examples/source/ --set pipeline.layout.use_polygon true --set logging.level DEBUG
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

# Extract printed page numbers from PP-DocLayoutV3 `number` regions
with GlmOcr(detect_printed_page_numbers=True) as parser:
    result = parser.parse("document.pdf")
    print(result.to_dict().get("page_metadata", []))

# Place layout model on CPU (useful when GPU is reserved for OCR)
with GlmOcr(layout_device="cpu") as parser:
    result = parser.parse("image.png")

# Place layout model on a specific GPU
with GlmOcr(layout_device="cuda:1") as parser:
    result = parser.parse("image.png")
```

#### Flask Service

Install the optional server dependency first:

```bash
pip install "glmocr[server]"
```

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

Configuration priority (highest to lowest):

1. CLI `--set` overrides
2. Python API keyword arguments
3. `GLMOCR_*` environment variables / `.env` file
4. YAML config file
5. Built-in defaults

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
    connect_timeout: 30
    request_timeout: 120

  # Page loader settings
  page_loader:
    max_tokens: 8192
    temperature: 0.0
    image_format: JPEG
    min_pixels: 12544
    max_pixels: 71372800

  # Result formatting
  result_formatter:
    output_format: both # json, markdown, or both
    detect_printed_page_numbers: false

  # Layout model device placement
  layout:
    # device: null   # null=auto, "cpu", "cuda", or "cuda:N"
```

See [config.yaml](glmocr/config.yaml) for all options.

Printed page number detection can be enabled in three ways:

```python
with GlmOcr(detect_printed_page_numbers=True) as parser:
    result = parser.parse("document.pdf")
```

```powershell
$env:GLMOCR_DETECT_PRINTED_PAGE_NUMBERS = 'true'
```

```yaml
pipeline:
  result_formatter:
    detect_printed_page_numbers: true
```

### Output Formats

Here are two examples of output formats:

- JSON

```json
[[{ "index": 0, "label": "text", "content": "...", "bbox_2d": null }]]
```

When printed page detection is enabled and printed-page data is actually found,
saved `paper.json` is wrapped as a top-level object and includes:

```json
{
  "json_result": [[{ "index": 0, "label": "text", "content": "...", "bbox_2d": null }]],
  "page_number_candidates": [
    {
      "page_index": 1,
      "label": "number",
      "content": "22",
      "layout_index": 0,
      "bbox_2d": [92, 26, 120, 41],
      "layout_score": 0.77,
      "numeric_like": true,
      "roman_like": false
    }
  ],
  "document_page_numbering": {
    "strategy": "visual_sequence",
    "confidence": 1.0,
    "sequence_type": "arabic",
    "page_offset": 21,
    "candidate_pages": 4
  },
  "page_metadata": [
    {
      "page_index": 1,
      "printed_page_label": "22",
      "printed_page_block_index": 0,
      "printed_page_bbox_2d": [92, 26, 120, 41],
      "printed_page_confidence": 0.77
    }
  ]
}
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

## Star History

<a href="https://www.star-history.com/?repos=zai-org%2FGLM-OCR&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=zai-org/GLM-OCR&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=zai-org/GLM-OCR&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=zai-org/GLM-OCR&type=date&legend=top-left" />
 </picture>
</a>

## Acknowledgement

This project is inspired by the excellent work of the following projects and communities:

- [PP-DocLayout-V3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)

## License

The Code of this repo is under Apache License 2.0.

The GLM-OCR model is released under the MIT License.

The complete OCR pipeline integrates [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) for document layout analysis, which is licensed under the Apache License 2.0. Users should comply with both licenses when using this project.

## Citation

If you find GLM-OCR useful in your research, please cite our technical report:

```bibtex
@misc{duan2026glmocrtechnicalreport,
      title={GLM-OCR Technical Report},
      author={Shuaiqi Duan and Yadong Xue and Weihan Wang and Zhe Su and Huan Liu and Sheng Yang and Guobing Gan and Guo Wang and Zihan Wang and Shengdong Yan and Dexin Jin and Yuxuan Zhang and Guohong Wen and Yanfeng Wang and Yutao Zhang and Xiaohan Zhang and Wenyi Hong and Yukuo Cen and Da Yin and Bin Chen and Wenmeng Yu and Xiaotao Gu and Jie Tang},
      year={2026},
      eprint={2603.10910},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.10910},
}
```
