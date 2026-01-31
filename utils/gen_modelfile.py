#!/usr/bin/env python3
"""Generate Ollama modelfile from training run."""

import argparse
from pathlib import Path

MODELFILE_TEMPLATE = '''FROM {model_path}

TEMPLATE """{{{{ .Prompt }}}}"""

PARAMETER temperature {temperature}
PARAMETER num_predict {num_predict}
PARAMETER num_ctx {num_ctx}
PARAMETER stop <|fim_prefix|>
PARAMETER stop <|fim_suffix|>
PARAMETER stop <|fim_middle|>
PARAMETER stop <|endoftext|>
PARAMETER stop <|repo_name|>
PARAMETER stop <|file_sep|>
'''

def generate_modelfile(
    run_dir: str,
    output: str = None,
    model_name: str = None,
    temperature: float = 0.2,
    num_predict: int = 128,
    num_ctx: int = 4096,
    use_merged: bool = True,
):
    run_path = Path(run_dir)
    
    # Determine model path
    if use_merged:
        model_path = run_path / "merged_model"
        if not model_path.exists():
            model_path = run_path / "final_model"
    else:
        model_path = run_path / "final_model"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Generate modelfile content
    content = MODELFILE_TEMPLATE.format(
        model_path=model_path.resolve(),
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
    )
    
    # Determine output path
    if output:
        output_path = Path(output)
    else:
        # Auto-name based on run directory
        run_name = run_path.name
        output_path = run_path.parent.parent.parent / f"{run_name}.modelfile"
    
    output_path.write_text(content)
    print(f"✅ Generated: {output_path}")
    print(f"   Model: {model_path}")
    
    # Print ollama command
    name = model_name or run_path.name.replace("_", "-").lower()
    print(f"\n📦 To create Ollama model:")
    print(f"   ollama create {name} -f {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Ollama modelfile from training run")
    parser.add_argument("run_dir", help="Path to training run directory")
    parser.add_argument("--output", "-o", help="Output modelfile path (optional)")
    parser.add_argument("--name", "-n", help="Model name for ollama create command")
    parser.add_argument("--temperature", "-t", type=float, default=0.2)
    parser.add_argument("--num_predict", type=int, default=128)
    parser.add_argument("--num_ctx", type=int, default=4096)
    parser.add_argument("--lora", action="store_true", help="Use final_model (LoRA) instead of merged")
    
    args = parser.parse_args()
    
    generate_modelfile(
        run_dir=args.run_dir,
        output=args.output,
        model_name=args.name,
        temperature=args.temperature,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
        use_merged=not args.lora,
    )
