# Ollama Deployment

Deploy your fine-tuned FIM model with Ollama for local inference.

## Prerequisites

- [Ollama](https://ollama.com) installed
- GGUF model file (see conversion steps below)

## Quick Start (if you have the GGUF)

```bash
cd ollama/
# Update Modelfile with correct path to your .gguf file
ollama create qwen-fim-32b -f Modelfile
ollama run qwen-fim-32b
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
  --adapter viplismism/qwen-fim-32B \
  --base_model Qwen/Qwen2.5-Coder-32B \
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
  --adapter viplismism/qwen-fim-32B \
  --output_dir merged_model \
  --quantization q5_k_m
```

### Step 3: Download GGUF to Mac

```bash
# From your Mac
scp -P <port> user@server:/path/to/merged_model/qwen-fim-32b.q5_k_m.gguf ~/models/
```

### Step 4: Create Ollama Model

```bash
cd ~/models/
# Copy Modelfile and update the FROM path
ollama create qwen-fim-32b -f Modelfile
```

## Usage

### CLI
```bash
ollama run qwen-fim-32b "fn add(a: i32, b: i32) -> i32 {"
```

### FIM Mode (with suffix)
```bash
# Using the API with FIM tokens
curl http://localhost:11434/api/generate -d '{
  "model": "qwen-fim-32b",
  "prompt": "<|fim_prefix|>fn add(a: i32, b: i32) -> i32 {\n    <|fim_suffix|>\n}<|fim_middle|>",
  "stream": false
}'
```

### Python
```python
import ollama

response = ollama.generate(
    model='qwen-fim-32b',
    prompt='<|fim_prefix|>fn add(a: i32, b: i32) -> i32 {\n    <|fim_suffix|>\n}<|fim_middle|>'
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
    "title": "Qwen FIM 32B",
    "provider": "ollama",
    "model": "qwen-fim-32b"
  }]
}
```
