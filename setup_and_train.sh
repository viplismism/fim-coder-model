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

# Step 1: Install system dependencies
echo "[1/6] Setting up Python environment..."
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip

# Create and activate venv
python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Step 2: Install Rust
echo "[2/6] Installing Rust..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Step 3: Build AST extractor
echo "[3/6] Building AST extractor..."
cd ast_extractor
cargo build --release
cd ..

# Step 4: Clone target repository
echo "[4/6] Cloning target repository (reth)..."
if [ ! -d "/tmp/reth/.git" ]; then
    git clone --depth 1 https://github.com/paradigmxyz/reth /tmp/reth
fi

# Step 5: Generate training data
echo "[5/6] Generating FIM training data..."
mkdir -p data
./ast_extractor/target/release/ast_extractor /tmp/reth ./data/reth_ast.json
python3 datagen/datagen.py --ast data/reth_ast.json --output_prefix reth

# Step 6: Start training
echo "[6/6] Starting training..."
python3 training/train.py --model_size "$MODEL_SIZE" --epochs "$NUM_EPOCHS"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
