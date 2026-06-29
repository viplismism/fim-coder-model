# Ollama Deployment

Deploy your fine-tuned FIM model with Ollama for local inference.

## Prerequisites

- [Ollama](https://ollama.com) installed
- GGUF model file (see conversion steps below)

## Quickest start: pull the published GGUF

Pre-built GGUF quants are published at
[viplismism/deepseek-coder-6.7b-fim-reth-v1-GGUF](https://huggingface.co/viplismism/deepseek-coder-6.7b-fim-reth-v1-GGUF)
(`q4_k_m` ~4 GB, `q8_0` ~7 GB) — no need to build anything:

```bash
huggingface-cli download viplismism/deepseek-coder-6.7b-fim-reth-v1-GGUF \
  fim-deepseek-6.7b-reth-v1.q4_k_m.gguf --local-dir .
ollama create fim-reth -f Modelfile      # point Modelfile FROM at the .gguf
ollama run fim-reth
```

## Build it yourself: LoRA -> Merged -> GGUF -> Ollama

### Step 1: Merge LoRA (on GPU server)

```bash
# SSH to GPU server (H100 recommended)
git clone https://github.com/viplismism/fim-coder-model.git
cd fim-coder-model
pip install -r requirements.txt

# Merge LoRA adapter into base model
python utils/convert_to_gguf.py \
  --adapter viplismism/deepseek-coder-6.7b-fim-reth-v1 \
  --base_model deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir merged_model \
  --skip_gguf
```

### Step 2: Convert to GGUF (on GPU server)

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j && cd ..

# Convert to GGUF with Q4_K_M quantization (~4GB, recommended default)
export LLAMA_CPP_PATH=./llama.cpp
python utils/convert_to_gguf.py \
  --adapter viplismism/deepseek-coder-6.7b-fim-reth-v1 \
  --base_model deepseek-ai/deepseek-coder-6.7b-base \
  --output_dir merged_model \
  --quantization q4_k_m
```

### Step 3: Download GGUF to Mac

```bash
# From your Mac
scp -P <port> user@server:/path/to/merged_model/*.q4_k_m.gguf ~/models/
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

## Quantization Options (DeepSeek-Coder-6.7B)

| Type | Size | Quality | RAM Needed |
|------|------|---------|------------|
| Q4_K_M | ~4 GB | Good — recommended default | 8 GB+ |
| Q5_K_M | ~4.8 GB | Better | 8 GB+ |
| Q8_0 | ~7 GB | Near-lossless | 12 GB+ |
| f16 | ~13 GB | Full precision | 16 GB+ |

The published repo ships **Q4_K_M** and **Q8_0**. Q4_K_M is the right default for laptops.

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
