#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and prepare for GGUF conversion.
Run this on a GPU server with enough VRAM (H100/A100).
"""

import os
import argparse
import torch
from pathlib import Path


def merge_lora_adapter(
    adapter_path: str,
    base_model: str,
    output_dir: str,
    push_to_hub: bool = False,
    hub_model_id: str = None
):
    """Merge LoRA adapter into base model and save."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter: {adapter_path}")
    
    # Load in full precision for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    
    # Load and merge LoRA
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("Merging weights...")
    model = model.merge_and_unload()
    
    # Save merged model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Optionally push to HuggingFace
    if push_to_hub and hub_model_id:
        print(f"Pushing to HuggingFace: {hub_model_id}")
        model.push_to_hub(hub_model_id, safe_serialization=True)
        tokenizer.push_to_hub(hub_model_id)
    
    print("Merge complete!")
    return output_path


def convert_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m"
):
    """Convert merged model to GGUF format using llama.cpp."""
    import subprocess
    
    # Check if llama.cpp is available
    llama_cpp_path = os.environ.get("LLAMA_CPP_PATH", "llama.cpp")
    convert_script = Path(llama_cpp_path) / "convert_hf_to_gguf.py"
    quantize_bin = Path(llama_cpp_path) / "build" / "bin" / "llama-quantize"
    
    if not convert_script.exists():
        print("llama.cpp not found. Clone it first:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && make -j")
        return None
    
    # Convert to GGUF (fp16)
    gguf_fp16 = Path(output_path).with_suffix(".fp16.gguf")
    print(f"Converting to GGUF (fp16): {gguf_fp16}")
    
    subprocess.run([
        "python3", str(convert_script),
        model_path,
        "--outfile", str(gguf_fp16),
        "--outtype", "f16"
    ], check=True)
    
    # Quantize
    gguf_quant = Path(output_path).with_suffix(f".{quantization}.gguf")
    print(f"Quantizing to {quantization}: {gguf_quant}")
    
    subprocess.run([
        str(quantize_bin),
        str(gguf_fp16),
        str(gguf_quant),
        quantization.upper()
    ], check=True)
    
    # Clean up fp16 version
    gguf_fp16.unlink()
    
    print(f"GGUF ready: {gguf_quant}")
    return gguf_quant


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to GGUF")
    parser.add_argument("--adapter", default="viplismism/deepseek-coder-6.7b-fim-reth-v1",
                       help="LoRA adapter path (HuggingFace or local)")
    parser.add_argument("--base_model", default="deepseek-ai/deepseek-coder-6.7b-base",
                       help="Base model name")
    parser.add_argument("--output_dir", default="merged_model",
                       help="Output directory for merged model")
    parser.add_argument("--quantization", default="q5_k_m",
                       choices=["q4_k_m", "q5_k_m", "q6_k", "q8_0"],
                       help="GGUF quantization type")
    parser.add_argument("--push_to_hub", action="store_true",
                       help="Push merged model to HuggingFace")
    parser.add_argument("--hub_model_id", default=None,
                       help="HuggingFace model ID for pushing")
    parser.add_argument("--skip_gguf", action="store_true",
                       help="Skip GGUF conversion (merge only)")
    args = parser.parse_args()
    
    # Step 1: Merge LoRA
    merged_path = merge_lora_adapter(
        adapter_path=args.adapter,
        base_model=args.base_model,
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    
    # Step 2: Convert to GGUF
    if not args.skip_gguf:
        gguf_output = Path(args.output_dir) / f"deepseek-coder-6.7b-fim.{args.quantization}.gguf"
        convert_to_gguf(
            model_path=str(merged_path),
            output_path=str(gguf_output),
            quantization=args.quantization
        )


if __name__ == "__main__":
    main()
