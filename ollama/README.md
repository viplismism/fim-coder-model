# Ollama Deployment

Deploy your fine-tuned FIM model with Ollama for local inference.

## Prerequisites

- [Ollama](https://ollama.com) installed
- GGUF model file (see conversion steps below)

## Quick Start (if you have the GGUF)

```bash
cd ollama/
# Update Modelfile with correct path to your .gguf file
ollama create deepseek-coder-6.7b-fim -f Modelfile
ollama run deepseek-coder-6.7b-fim
```

## Full Pipeline: LoRA -> Merged -> GGUF -> Ollama

### Step 1: Merge LoRA (on GPU server)

```bash
# SSH to GPU server (H100 recommended)
git clone https://github.com/viplismism/fim-coder-model.git
cd fim-coder-model
pip install -r requirements.txt

# Merge LoRA adapter into base model
python utils/convert_to_gguf.py \
  --adapter viplismism/deepseek-coder-6.7b-fim \
  --base_model deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir merged_model \
  --skip_gguf
```

### Step 2: Convert to GGUF (on GPU server)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j && cd ..

# Convert to GGUF with Q5_K_M quantization (~22GB, high quality)
export LLAMA_CPP_PATH=./llama.cpp
python utils/convert_to_gguf.py \
  --adapter viplismism/deepseek-coder-6.7b-fim \
  --base_model deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir merged_model \
  --quantization q5_k_m
```

### Step 3: Download GGUF to Mac

```bash
# From your Mac
scp -P <port> user@server:/path/to/merged_model/deepseek-coder-6.7b-fim.q5_k_m.gguf ~/models/
```

### Step 4: Create Ollama Model

```bash
cd ~/models/
# Copy Modelfile and update the FROM path
ollama create deepseek-coder-6.7b-fim -f Modelfile
```

## Usage

### CLI
```bash
ollama run deepseek-coder-6.7b-fim "fn add(a: i32, b: i32) -> i32 {"
```

### FIM Mode (with suffix)
```bash
# Using the API with FIM tokens
curl http://localhost:11434/api/generate -d '{
  "model": "deepseek-coder-6.7b-fim",
  "prompt": "<｜fim▁begin｜>fn add(a: i32, b: i32) -> i32 {\n    <｜fim▁hole｜>\n}<｜fim▁end｜>",
  "stream": false
}'
```

### Python
```python
import ollama

response = ollama.generate(
    model='deepseek-coder-6.7b-fim',
    prompt='<｜fim▁begin｜>fn add(a: i32, b: i32) -> i32 {\n    <｜fim▁hole｜>\n}<｜fim▁end｜>'
)
print(response['response'])
```

## Quantization Options

| Type | Size | Quality | RAM Needed |
|------|------|---------|------------|
| Q4_K_M | ~18GB | Good | 24GB+ |
| Q5_K_M | ~22GB | Better | 28GB+ |
| Q6_K | ~26GB | High | 32GB+ |
| Q8_0 | ~34GB | Highest | 40GB+ |

For Mac M4 Max with 36GB RAM, use **Q5_K_M** for best quality/performance balance.

## VS Code Integration

Use with Continue.dev or other extensions:
```json
{
  "models": [{
    "title": "DeepSeek Coder 6.7B FIM",
    "provider": "ollama",
    "model": "deepseek-coder-6.7b-fim"
  }]
}
```
