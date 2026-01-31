# FIM Coder Model

A training framework for fine-tuning Large Language Models on Fill-in-the-Middle (FIM) code completion tasks using AST-aware data generation.

## Overview

This framework extracts semantic code boundaries (functions, structs, impl blocks) from Rust codebases using AST parsing, generates FIM training samples, and fine-tunes models using LoRA with 4-bit quantization for efficient multi-GPU training.

## Architecture

### Data Preparation Pipeline

```
title: Data Preparation Pipeline

Rust Repo -> AST Extractor: Source files (*.rs)
note: Parses Rust code using syn crate to extract semantic boundaries

AST Extractor -> AST JSON: Nodes with spans
note AST Extractor, AST JSON: Extracts Function, Struct, Impl, Enum, Trait nodes with line number spans

AST JSON -> Datagen: Structured AST
Datagen -> Training Data: FIM samples

note Training Data: Outputs train.jsonl and test.jsonl with prefix/middle/suffix splits
```

[View Diagram](https://swimlanes.io/#ZVFBDoIwELzzir1DRI76Ag8GE8Gbo3dgobCmliJNEV/iFxSfxYEQEeZSmZ2dmYxnl6eijVhVQUQF6k1oiCQ6dFp6PpE6xwglCgWKV4Y2NmYxpDEnQ8NwSpXRAA0QBIxqPHCJTsJy6LmGADVoZ6BpCrfI7dJ3LiYnT7LWEnLhkfZBmgE9+Tn7N7bW6BXy1NlIGSwxf8dBPQlgpFtF5rfW7G0dUf+X2BvhPRJDQjL3Bx8=)

### Training Pipeline

```
title: Training Pipeline

Base Model -> LoRA Init: Load Qwen2.5-Coder
note Base Model: Qwen2.5-Coder-7B/14B/32B

LoRA Init -> LoRA Init: Apply 4-bit quantization (QLoRA)
note LoRA Init: BitsAndBytes NF4 quantization reduces VRAM by ~4x

LoRA Init => SFTTrainer: Attach LoRA adapters
note: Targets q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Training Data -> SFTTrainer: Load FIM samples
SFTTrainer -> SFTTrainer: Train with gradient accumulation
note SFTTrainer: Multi-GPU via accelerate/DeepSpeed

SFTTrainer -> LoRA Adapter: Save adapter weights
note LoRA Adapter: ~100MB vs 60GB full model
```

[View Diagram](https://swimlanes.io/#ZVLBboMwDL3vK3ITSfcFO0zambo/2C3qwUBIoyUJxelKv2hf0C/bQ6BVN8clku33/Ozn+cR0LZhpKFccNZgxoSNSqNE58flCsowjFCgkKJ5bymxiIshSTkbK0EIZowEaIAgYt7TnCr2E+bVjOALUo52FthPeIU+n6F3qpSfJaEKCXHikPVBmQI9Bxl+CbY2u0KZlhtthielb7uWTDYx0C2V+G83UtaH+n2qvQHgv2CckcXfw/fYDP2+r6vqXcBJNdzrp6Kg=)

### Deployment Pipeline

```
title: Deployment Pipeline

LoRA Adapter -> Merge Script: Load adapter
Base Model -> Merge Script: Load base weights

Merge Script -> Merge Script: Merge weights
note Merge Script: Combines LoRA deltas with base model

Merge Script -> Merged Model: Full model weights
note Merged Model: ~60GB for 32B model

Merged Model -> Modelfile Gen: Generate Ollama config
Modelfile Gen -> Ollama: Create deployment

note Ollama: Ready for inference with FIM tokens
```

[View Diagram](https://swimlanes.io/#ZY9BDsIgEEX3nGLWkrgPF7owxkT3egMUaGkqMIahNZ7Ei9iDGYrRhAXJ8P7/M5MbNKVkqqVScjRgp4Sezr3VKvCJNBlHqFAoULx2tPWJTSBLOBkFQyu1jwboABEwaulIFXoJy/vAMA9ao52HvhfhQZ5u0YXUS8+Q1YwEueBI5yjNgB79jL0G2lj0Enlub5Qctpjf46CfbWCgWyvzP2q2ro6o/1vtJQjvBXuPJO4OfrH94N+jrKqj3w==)

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

### Node Types Extracted

```
title: AST Node Types

_: Source Code Analysis

Rust File -> AST Extractor: Parse source

AST Extractor -> Function: fn declarations
AST Extractor -> FunctionBody: { ... } blocks  
AST Extractor -> Struct: struct definitions
AST Extractor -> Impl: impl blocks
AST Extractor -> Enum: enum definitions
AST Extractor -> Trait: trait definitions
AST Extractor -> Use: import statements
AST Extractor -> Const: const/static items

note: FunctionBody samples are most valuable for code completion training
```

[View Diagram](https://swimlanes.io/#ZY/BDoIwDIbvPkXPkugj6MGYMRF38QYbdbCxFSiEiS/hC/pgLIyJeld/++9r2wt0tSKqodZSacAOER2dfWdU4CtpCopQgFCgeO1pFxIXQ5pxMkqGdhrfTKAHDAfGHR24QS9h9Rg5lgFqcKuE9pPwCXlqYnBlTp5loyXkwgPtgTQDurZT9i/q1ugNeW6vYQ5rTO8wUI8ukNEtpPmfGt22pdF/id0d4T0h3yMJu4P1pw/+3FdN8/Ib)

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
