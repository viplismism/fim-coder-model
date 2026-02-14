#!/bin/bash
# Complete setup and training script for FIM Coder Model
# Usage: ./setup_and_train.sh [--model_size 14B|32B] [--epochs N] [--push_to_hub]

set -e

# Default values
MODEL_SIZE="14B"
NUM_EPOCHS=3
PUSH_TO_HUB=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --push_to_hub)
            PUSH_TO_HUB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "FIM Coder Model - Setup & Training"
echo "=========================================="
echo "Model: $MODEL_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Push to Hub: $PUSH_TO_HUB"
echo ""

# Step 1: Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo "[1/5] Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Step 2: Build AST extractor
echo "[2/5] Building AST extractor..."
cd "$(dirname "$0")"
if [ ! -f "ast_extractor/target/release/ast_extractor" ]; then
    cd ast_extractor
    cargo build --release
    cd ..
fi

# Step 3: Clone target repository if not exists
REPO_PATH="/tmp/reth"
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "[3/5] Cloning target repository (reth)..."
    rm -rf "$REPO_PATH"
    git clone --depth 1 https://github.com/paradigmxyz/reth "$REPO_PATH"
fi

# Step 4: Generate training data
echo "[4/5] Generating FIM training data..."
mkdir -p data

if [ ! -f "data/reth_ast.json" ]; then
    ./ast_extractor/target/release/ast_extractor "$REPO_PATH" ./data/reth_ast.json
fi

if [ ! -f "data/reth_train.jsonl" ]; then
    python3 datagen/datagen.py --ast data/reth_ast.json --output_prefix reth
fi

# Step 5: Update config for this run
echo "[5/5] Starting training..."

# Update config based on options
if [ "$PUSH_TO_HUB" = true ]; then
    sed -i 's/push_to_hub: false/push_to_hub: true/' config.yaml
    echo "✅ Will push to HuggingFace after training"
fi

# Run training
python3 training/train.py --model_size "$MODEL_SIZE" --epochs "$NUM_EPOCHS"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

if [ "$PUSH_TO_HUB" = true ]; then
    echo "Model saved locally. To push to HuggingFace:"
    echo "1. Login: huggingface-cli login"
    echo "2. Enable push_to_hub in config.yaml"
    echo "3. Run: python3 training/train.py --model_size $MODEL_SIZE"
fi
