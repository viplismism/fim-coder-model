# FIM Coder Model

A training framework for fine-tuning Large Language Models on Fill-in-the-Middle (FIM) code completion tasks using AST-aware data generation.

## Overview

This framework extracts semantic code boundaries (functions, structs, impl blocks) from Rust codebases using AST parsing, generates FIM training samples, and fine-tunes models using LoRA with 4-bit quantization for efficient multi-GPU training.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA PREPARATION                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐ │
│   │              │     │              │     │              │     │            │ │
│   │  Rust Repo   │────▶│ AST Extractor│────▶│  AST JSON    │────▶│  Datagen   │ │
│   │  (source)    │     │  (syn-based) │     │  (nodes +    │     │  (FIM      │ │
│   │              │     │              │     │   spans)     │     │  samples)  │ │
│   └──────────────┘     └──────────────┘     └──────────────┘     └─────┬──────┘ │
│                                                                        │        │
│                                                            ┌───────────┴──────┐ │
│                                                            │  train.jsonl     │ │
│                                                            │  test.jsonl      │ │
│                                                            └───────────┬──────┘ │
└────────────────────────────────────────────────────────────────────────┼────────┘
                                                                         │
┌────────────────────────────────────────────────────────────────────────┼────────┐
│                              MODEL TRAINING                            │        │
├────────────────────────────────────────────────────────────────────────┼────────┤
│                                                                        ▼        │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐ │
│   │              │     │              │     │              │     │            │ │
│   │  Base Model  │────▶│  LoRA Init   │────▶│  SFTTrainer  │◀────│  FIM Data  │ │
│   │  (Qwen2.5)   │     │  (4-bit QLoRA)│    │  (multi-GPU) │     │            │ │
│   │              │     │              │     │              │     │            │ │
│   └──────────────┘     └──────────────┘     └──────┬───────┘     └────────────┘ │
│                                                    │                            │
│                                                    ▼                            │
│                                             ┌──────────────┐                    │
│                                             │ LoRA Adapter │                    │
│                                             │  (final_model)│                   │
│                                             └──────┬───────┘                    │
│                                                    │                            │
└────────────────────────────────────────────────────┼────────────────────────────┘
                                                     │
┌────────────────────────────────────────────────────┼────────────────────────────┐
│                              DEPLOYMENT            │                            │
├────────────────────────────────────────────────────┼────────────────────────────┤
│                                                    ▼                            │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                    │
│   │              │     │              │     │              │                    │
│   │  LoRA Merge  │────▶│ Merged Model │────▶│   Ollama     │                    │
│   │  (full weights)    │ (deployable) │     │  Deployment  │                    │
│   │              │     │              │     │              │                    │
│   └──────────────┘     └──────────────┘     └──────────────┘                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## FIM Sample Format

The training data follows the Qwen FIM token format:

```
<|repo_name|>reth
<|file_sep|>crates/rpc/src/handler.rs
<|fim_prefix|>impl Handler {
    pub fn new(config: Config) -> Self {
        <|fim_suffix|>
    }
}
<|fim_middle|>Self { config, state: State::default() }<|endoftext|>
```

## Requirements

- Python 3.9+
- CUDA-capable GPU (80GB+ VRAM recommended for 32B model)
- Rust toolchain (for AST extractor)

## Installation

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt

# Build AST extractor
cd ast_extractor && cargo build --release && cd ..
```

## Usage

### Data Preparation

```bash
# Clone target repository
git clone --depth 1 https://github.com/paradigmxyz/reth /tmp/reth

# Extract AST nodes with spans
./ast_extractor/target/release/ast_extractor /tmp/reth ./data/reth_ast.json

# Generate FIM training samples
python3 datagen/datagen.py --ast data/reth_ast.json --output_prefix reth
```

### Training

```bash
# Single GPU
python3 training/train.py

# Multi-GPU with accelerate
accelerate launch --num_processes 4 training/train.py

# Override config parameters
python3 training/train.py --epochs 5 --lr 5e-5 --model_size 14B
```

### Post-Training

```bash
# Merge LoRA adapters into base model
python3 utils/merging.py --run_dir training/runs/<run_name>

# Deploy with Ollama
ollama create <model_name> -f training/runs/<run_name>/modelfile
```

## Configuration

All training parameters are defined in `config.yaml`:

| Section | Parameters |
|---------|------------|
| `model` | Base model selection, batch sizes, gradient accumulation |
| `lora` | Rank, alpha, dropout, target modules |
| `quantization` | 4-bit quantization settings |
| `training` | Epochs, learning rate, warmup, optimizer |
| `checkpointing` | Save frequency, evaluation intervals |
| `data` | Training/test file paths, repository name |

## Project Structure

```
├── config.yaml              # Training configuration
├── requirements.txt         # Python dependencies
├── ast_extractor/           # Rust-based AST extraction
│   ├── Cargo.toml
│   └── src/main.rs
├── datagen/
│   └── datagen.py           # FIM sample generation
├── training/
│   ├── train.py             # Main training script
│   └── runs/                # Training outputs
├── inference/
│   └── infer.py             # Model evaluation
└── utils/
    ├── merging.py           # LoRA adapter merging
    └── gen_modelfile.py     # Ollama modelfile generation
```

## Supported Base Models

| Model | Parameters | VRAM (4-bit) | Recommended GPUs |
|-------|------------|--------------|------------------|
| Qwen2.5-Coder-7B | 7B | ~8GB | 1x A100/H100 |
| Qwen2.5-Coder-14B | 14B | ~16GB | 2x A100/H100 |
| Qwen2.5-Coder-32B | 32B | ~36GB | 4x H100/H200 |

## License

MIT
