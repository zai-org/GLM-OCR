# Ollama Deployment Guide for GLM-OCR

This guide provides detailed instructions for deploying GLM-OCR using Ollama.

## Overview

Ollama provides a simple local deployment option for running GLM-OCR. However, due to limitations in Ollama's OpenAI-compatible API for vision requests, we recommend using Ollama's native `/api/generate` endpoint.

## Installation

### 1. Install Ollama

Download and install Ollama from the official website:

**macOS / Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from https://ollama.ai/download

### 2. Verify Installation

```bash
ollama --version
```

### 3. Pull the GLM-OCR Model

```bash
ollama pull glm-ocr:latest
```

This will download the GLM-OCR model.

### 4. Start Ollama Service

The Ollama service should start automatically after installation. If not:

```bash
ollama serve
```

The service will run on `http://localhost:11434` by default.

## Configuration

### SDK Configuration

Create or update your `config.yaml`:

```yaml
pipeline:
  maas:
    enabled: false
  
  ocr_api:
    api_host: localhost
    api_port: 11434
    api_path: /api/generate  # Use Ollama native endpoint
    model: glm-ocr:latest    # Required: specify model name
    api_mode: ollama_generate  # Required: use Ollama native format
  
  enable_layout: false  # Recommended for initial testing
```

### Configuration Options Explained

- **api_path**: `/api/generate` - Ollama's native endpoint (more stable for vision)
- **model**: `glm-ocr:latest` - Model name (required by Ollama)
- **api_mode**: `ollama_generate` - Enables Ollama-specific request/response format
- **enable_layout**: `false` - Disable layout detection if dependencies not installed

## Usage

### Command Line

```bash
# Parse a single image
glmocr parse examples/source/code.png --config config.yaml

# Parse with custom output directory
glmocr parse examples/source/code.png --output ./results/

# Enable debug logging
glmocr parse examples/source/code.png --log-level DEBUG
```

### Python API

```python
from glmocr import GlmOcr

# Initialize with custom config
with GlmOcr(config_path="config.yaml") as parser:
    result = parser.parse("image.png")
    print(result.markdown_result)
    result.save(output_dir="./results")
```

## Troubleshooting

### Issue: 502 Bad Gateway Errors

**Symptom:**
```
API server returned status code: 502, response: no body
```

**Solution:**
Ensure you're using Ollama's native API mode:
```yaml
ocr_api:
  api_path: /api/generate
  api_mode: ollama_generate
```

## Verification

### Check Model Status

```bash
# List installed models
ollama list

# View model details
ollama show glm-ocr:latest

# Check running models
ollama ps
```

### Test the API

```bash
# Test with a simple request (Linux/Mac)
curl http://localhost:11434/api/generate -d '{
  "model": "glm-ocr:latest",
  "prompt": "Hello",
  "stream": false
}'

# Windows PowerShell
Invoke-RestMethod -Uri http://localhost:11434/api/generate -Method Post -Body '{"model":"glm-ocr:latest","prompt":"Hello","stream":false}' -ContentType "application/json"
```

### Recommendations

- **For Testing/Personal Use**: Ollama is perfect
- **For Production**: Consider vLLM or SGLang for better performance and stability
- **For CPU-only**: Ollama is a good choice

## Advanced Configuration

### Custom Model Path

If you have a custom GLM-OCR model:

```bash
# Create a Modelfile
cat > Modelfile <<EOF
FROM /path/to/your/model
TEMPLATE {{ .Prompt }}
RENDERER glm-ocr
PARSER glm-ocr
PARAMETER temperature 0
EOF

# Create the model
ollama create my-glm-ocr -f Modelfile

# Use it in config
model: my-glm-ocr
```
## Uninstallation

```bash
# Remove the model
ollama rm glm-ocr:latest

# Uninstall Ollama (varies by OS)
# macOS/Linux: Remove /usr/local/bin/ollama
# Windows: Use the uninstaller
```

## Additional Resources

- [Ollama Official Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [GLM-OCR GitHub Repository](https://github.com/zai-org/GLM-OCR)
- [GLM-OCR API Documentation](https://docs.z.ai/guides/vlm/glm-ocr)
