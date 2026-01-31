#!/usr/bin/env python3
"""Merge LoRA adapters into base model for deployment."""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge(run_dir: str, base_model: str = "Qwen/Qwen2.5-Coder-32B"):
    run_path = Path(run_dir)
    lora_path = run_path / "final_model"
    output_path = run_path / "merged_model"
    
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA not found: {lora_path}")
    
    print(f"Base: {base_model}\nLoRA: {lora_path}\nOutput: {output_path}\n")
    
    print("[1/4] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    
    print("[2/4] Loading LoRA...")
    model = PeftModel.from_pretrained(model, str(lora_path))
    
    print("[3/4] Merging...")
    model = model.merge_and_unload()
    
    print("[4/4] Saving...")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(output_path)
    
    size = sum(f.stat().st_size for f in output_path.iterdir() if f.is_file()) / (1024**3)
    print(f"\nSaved: {output_path} ({size:.2f} GB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-32B")
    args = parser.parse_args()
    merge(args.run_dir, args.base_model)
