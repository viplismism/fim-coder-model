#!/usr/bin/env python3
"""Check that a model tokenizer preserves code whitespace exactly."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from utils.tokenizer_whitespace import assert_tokenizer_preserves_code_whitespace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-coder-6.7b-base")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    assert_tokenizer_preserves_code_whitespace(tokenizer)
    print(f"OK: {args.model} preserves code whitespace")


if __name__ == "__main__":
    main()
