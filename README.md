# FIM Coder Model

Fine-tune LLMs for Fill-in-the-Middle (FIM) code completion using AST-aware data generation.

## Features

- 🎯 **AST-based FIM data generation** - Extract function bodies, structs, impls from Rust codebases
- 🚀 **LoRA fine-tuning** - Efficient training with 4-bit quantization
- 📊 **Multi-GPU support** - Distributed training with `accelerate`
- 🔧 **Config-driven** - All parameters in `config.yaml`

## Quick Start

### 1. Setup

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# Build AST extractor
cd ast_extractor && cargo build --release && cd ..

# Clone target repo
git clone --depth 1 https://github.com/paradigmxyz/reth /tmp/reth

# Extract AST
./ast_extractor/target/release/ast_extractor /tmp/reth ./data/reth_ast.json

# Generate FIM training data
python3 datagen/datagen.py --ast data/reth_ast.json --output_prefix reth
```

### 3. Train

```bash
# Single GPU
python3 training/train.py

# Multi-GPU (4x H200)
accelerate launch --num_processes 4 training/train.py
```

### 4. Merge LoRA

```bash
python3 utils/merging.py --run_dir training/runs/<run_name>
```

### 5. Deploy with Ollama

```bash
ollama create my-fim-model -f training/runs/<run_name>/modelfile
```

## Project Structure

```
├── config.yaml          # Training configuration
├── requirements.txt     # Python dependencies
├── ast_extractor/       # Rust AST extraction tool
├── datagen/             # FIM dataset generation
├── training/            # Training scripts
├── inference/           # Evaluation scripts
└── utils/               # Merging & utilities
```

## Configuration

Edit `config.yaml` to customize:

- Model size (14B/32B)
- LoRA parameters (r, alpha, dropout)
- Training hyperparameters (epochs, lr, batch size)
- Data paths

## Supported Models

- Qwen2.5-Coder-32B (recommended)
- Qwen2.5-Coder-14B
- Qwen2.5-Coder-7B

## License

MIT
