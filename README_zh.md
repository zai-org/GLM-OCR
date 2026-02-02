## GLM-OCR

<div align="center">
<img src=resources/logo.svg width="40%"/>
</div>
<p align="center">
    👋 加入我们的 <a href="resources/WECHAT.md" target="_blank">微信群</a>
    <br>
    📍 使用 GLM-OCR 的 <a href="https://docs.z.ai/guides/image/glm-ocr" target="_blank">API</a>
</p>


### 模型介绍

GLM-OCR 是面向复杂文档理解的多模态 OCR 模型，基于 GLM-V 编解码架构构建。它引入 Multi-Token Prediction（MTP）损失与稳定的全任务强化学习，以提升训练效率、识别准确率与泛化能力。模型融合了在大规模图文数据上预训练的 CogViT 视觉编码器、具备高效 token 下采样能力的轻量跨模态连接器，以及 GLM-0.5B 语言解码器。配合基于 PP-DocLayout-V3 的两阶段流程（版面分析 + 并行识别），GLM-OCR 能在多种文档版式场景下提供稳定且高质量的 OCR 结果。

**主要特性**

- 业界领先的效果
在 OmniDocBench V1.5 上达到 94.62，排名 #1；并在主流文档理解基准上取得 SOTA 表现，包括公式识别、表格识别与信息抽取等任务。

- 面向真实业务场景优化
针对实际业务用例做了专项优化，在复杂表格、代码文档、印章等挑战性场景下也能保持稳定且准确的性能。

- 高效推理
仅 0.9B 参数，支持通过 vLLM 与 SGLang 部署，显著降低推理延迟与算力成本，适用于高并发与边缘部署。

- 易用
完整开源，提供配套 SDK 与推理工具链，支持一行命令调用，易于集成到现有系统。

### 下载模型

| 模型 | 下载链接 | 精度 |
|------|----------|------|
| GLM-OCR | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-OCR)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR) | BF16 |


## GLM-OCR SDK

我们提供了 SDK，帮助你更高效、更便捷地使用 GLM-OCR。

### 安装 SDK

```bash
pip install glmocr

# 或从源码安装
git clone https://github.com/zai-org/glm-ocr.git
cd glm-ocr && pip install -e .

# 从源码安装 transformers
pip install git+https://github.com/huggingface/transformers.git
```

### 模型服务部署

提供两种方式运行 GLM-OCR 模型服务：

#### 方式 1：智谱 MaaS API（推荐）

无需 GPU，直接使用托管 API：

1. 在 https://open.bigmodel.cn/ 获取 API Key
2. 配置 `config.yaml`：

```yaml
pipeline:
  ocr_api:
    api_host: open.bigmodel.cn
    api_port: 443
    api_scheme: https
    api_key: your-api-key
```

#### 方式 2：使用 vLLM / SGLang 自部署

你可以使用 vLLM 或 SGLang 部署一个 OpenAI 兼容服务。

##### 使用 vLLM

**安装 vLLM：**

```bash
pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
# 或使用 Docker
docker pull vllm/vllm-openai:nightly
```

**启动服务：**

```bash
pip install git+https://github.com/huggingface/transformers.git
vllm serve zai-org/GLM-OCR --allowed-local-media-path / --port 8080
```

##### 使用 SGLang

**安装 SGLang：**

```bash
docker pull lmsysorg/sglang:dev
# 或从源码安装
pip install git+https://github.com/sgl-project/sglang.git#subdirectory=python
```

**启动服务：**

```bash
pip install git+https://github.com/huggingface/transformers.git
python -m sglang.launch_server --model zai-org/GLM-OCR --port 8080
```

##### 更新配置

启动服务后，修改 `config.yaml`：

```yaml
pipeline:
  ocr_api:
    api_host: localhost # 或你的 vLLM/SGLang 服务地址
    api_port: 8080
```

### SDK 使用指南

#### CLI

```bash
# 解析单张图片
glmocr parse examples/source/code.png

# 解析目录
glmocr parse examples/source/

# 指定输出目录
glmocr parse examples/source/code.png --output ./results/

# 使用自定义配置
glmocr parse examples/source/code.png --config my_config.yaml

# 开启 debug 日志（包含 profiling）
glmocr parse examples/source/code.png --log-level DEBUG
```

#### Python API

```python
from glmocr import GlmOcr, parse

# 便捷函数
result = parse("image.png")
result = parse(["img1.png", "img2.jpg"])
result = parse("https://example.com/image.png")
result.save(output_dir="./results")

# 说明：传入 list 会被当作同一文档的多页

# 类接口
with GlmOcr() as parser:
    result = parser.parse("image.png")
    print(result.json_result)
    result.save()
```

#### Flask 服务

```bash
# 启动服务
python -m glmocr.server

# 开启 debug 日志
python -m glmocr.server --log-level DEBUG

# 调用 API
curl -X POST http://localhost:5002/glmocr/parse \
  -H "Content-Type: application/json" \
  -d '{"images": ["./example/source/code.png"]}'
```

语义说明：

- `images` 可以是 string 或 list。
- list 会被当作同一文档的多页处理。
- 如果要处理多个独立文档，请多次调用接口（一次请求一个文档）。

### 配置

完整配置见 `glmocr/config.yaml`：

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

更多选项请参考 [config.yaml](glmocr/config.yaml)。

### 输出格式

这里给出两种输出格式示例：

- JSON

```json
[[{ "index": 0, "label": "text", "content": "...", "bbox_2d": null }]]
```

- Markdown

```markdown
# 文档标题

正文...

| Table | Content |
| ----- | ------- |
| ...   | ...     |
```

### 完整流程示例

你可以运行示例代码：

```bash
python examples/example.py
```

输出结构（每个输入对应一个目录）：

- `result.json`：结构化 OCR 结果
- `result.md`：Markdown 结果
- `imgs/`：裁剪后的图片区域（启用 layout 模式时）

### 模块化架构

GLM-OCR 使用可组合模块，便于自定义扩展：

| 组件                  | 说明                         |
| --------------------- | ---------------------------- |
| `PageLoader`          | 预处理与图像编码             |
| `OCRClient`           | 调用 GLM-OCR 模型服务        |
| `PPDocLayoutDetector` | 基于 PP-DocLayout 的版面分析 |
| `ResultFormatter`     | 后处理与输出 JSON/Markdown   |

你也可以通过自定义 pipeline 扩展行为：

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
    # 实现你自己的处理逻辑
    pass
```


## 致谢

本项目受以下项目与社区的杰出工作启发：

- [PP-DocLayout-V3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)

## 开源协议

本仓库代码遵循 Apache License 2.0。

GLM-OCR 模型遵循 MIT License。

完整 OCR pipeline 集成了用于文档版面分析的 [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)，该组件遵循 Apache License 2.0。使用本项目时请同时遵守相关许可证。
