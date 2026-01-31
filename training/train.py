#!/usr/bin/env python3
"""FIM model fine-tuning with LoRA on Qwen2.5-Coder."""

import os, time, argparse, warnings, yaml
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import wandb
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_lora_config(cfg: dict) -> LoraConfig:
    """Create LoRA config from configuration dict."""
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )


def get_bnb_config(cfg: dict) -> BitsAndBytesConfig:
    """Create BitsAndBytes quantization config."""
    quant_cfg = cfg["quantization"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=dtype_map.get(quant_cfg["compute_dtype"], torch.bfloat16),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )


def format_fim(ex, repo="hyperswitch"):
    """Format example into FIM format."""
    return {"text": f"<|repo_name|>{repo}\n<|file_sep|>{ex['filePath']}\n"
                    f"<|fim_prefix|>{ex['prefix']}<|fim_suffix|>{ex['suffix']}<|fim_middle|>{ex['middle']}<|endoftext|>"}


def fmt_time(s):
    """Format seconds into human readable string."""
    return f"{int(s//3600)}h{int(s%3600//60)}m" if s >= 3600 else f"{int(s//60)}m{int(s%60)}s" if s >= 60 else f"{int(s)}s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--model_size", default=None, choices=["14B", "32B"], help="Override model size from config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    args = parser.parse_args()

    # Load config
    project_root = Path(__file__).parent.parent
    config_path = Path(args.config) if args.config else project_root / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = load_config(config_path)
    
    # CLI overrides
    model_size = args.model_size or cfg["model"]["size"]
    num_epochs = args.epochs if args.epochs is not None else cfg["training"]["num_epochs"]
    learning_rate = args.lr if args.lr is not None else cfg["training"]["learning_rate"]

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank in [-1, 0]
    
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://", world_size=world_size, rank=local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model config
    model_cfg = cfg["model"]["models"][model_size]
    model_name = model_cfg["name"]
    batch_size = model_cfg["per_device_batch_size"]
    grad_acc = model_cfg["gradient_accumulation_steps"]
    
    run_name = f"fim-{model_size}-{int(time.time())}"
    run_dir = Path(__file__).parent / "runs" / run_name
    
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        # Save config copy to run directory
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        
        if cfg["wandb"]["enabled"]:
            wandb.init(
                project=cfg["wandb"]["project"],
                name=run_name,
                config={
                    "model_size": model_size,
                    "model_name": model_name,
                    "epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "grad_acc": grad_acc,
                    **cfg["lora"],
                    **cfg["training"],
                }
            )

    # Tokenizer & model
    tok_cfg = cfg["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tok_cfg["padding_side"]
    tokenizer.model_max_length = tok_cfg["model_max_length"]

    bnb = get_bnb_config(cfg)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, trust_remote_code=True,
        torch_dtype=torch.bfloat16, use_cache=False,
        device_map={"": local_rank} if local_rank != -1 else "auto",
    )
    model = get_peft_model(prepare_model_for_kbit_training(model), get_lora_config(cfg))
    
    if is_main:
        model.print_trainable_parameters()

    # Data
    data_cfg = cfg["data"]
    ds = load_dataset("json", data_files={
        "train": str(project_root / data_cfg["train_file"]),
        "test": str(project_root / data_cfg["test_file"]),
    })
    repo_name = data_cfg["repo_name"]
    train_ds = ds["train"].map(lambda x: format_fim(x, repo_name))
    eval_ds = ds["test"].map(lambda x: format_fim(x, repo_name))
    
    if is_main:
        print(f"\nConfig: {config_path}")
        print(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,} | Model: {model_name}")
        print(f"Epochs: {num_epochs} | LR: {learning_rate} | Batch: {batch_size} | GradAcc: {grad_acc}")

    # Training args
    train_cfg = cfg["training"]
    ckpt_cfg = cfg["checkpointing"]
    dist_cfg = cfg["distributed"]
    
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_name,
        # Batch & accumulation
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        # Training hyperparams
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        optim=train_cfg["optim"],
        bf16=train_cfg["bf16"],
        seed=train_cfg["seed"],
        # Checkpointing & eval
        save_steps=ckpt_cfg["save_steps"],
        save_total_limit=ckpt_cfg["save_total_limit"],
        eval_steps=ckpt_cfg["eval_steps"],
        eval_strategy=ckpt_cfg["eval_strategy"],
        logging_steps=ckpt_cfg["logging_steps"],
        load_best_model_at_end=ckpt_cfg["load_best_model_at_end"] if is_main else False,
        metric_for_best_model=ckpt_cfg["metric_for_best_model"],
        # Distributed
        ddp_find_unused_parameters=dist_cfg["ddp_find_unused_parameters"],
        local_rank=local_rank,
        # Reporting
        report_to="wandb" if (is_main and cfg["wandb"]["enabled"]) else "none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=lambda x: x["text"],
    )

    if local_rank != -1:
        dist.barrier()

    # Train
    start = time.time()
    try:
        trainer.train(resume_from_checkpoint=args.resume)
        if is_main:
            print(f"\nTraining complete in {fmt_time(time.time() - start)}")
    except KeyboardInterrupt:
        if is_main:
            print(f"\nInterrupted after {fmt_time(time.time() - start)}")

    # Save
    if is_main:
        final_dir = run_dir / "final_model"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Saved to {final_dir}")        
        # Generate modelfile template
        modelfile_cfg = cfg.get("modelfile", {})
        modelfile_content = f'''FROM {final_dir.resolve()}

TEMPLATE """{{{{{{ .Prompt }}}}}}"""

PARAMETER temperature {modelfile_cfg.get("temperature", 0.2)}
PARAMETER num_predict {modelfile_cfg.get("num_predict", 128)}
PARAMETER num_ctx {modelfile_cfg.get("num_ctx", 4096)}
PARAMETER stop <|fim_prefix|>
PARAMETER stop <|fim_suffix|>
PARAMETER stop <|fim_middle|>
PARAMETER stop <|endoftext|>
PARAMETER stop <|repo_name|>
PARAMETER stop <|file_sep|>
'''
        modelfile_path = run_dir / "modelfile"
        modelfile_path.write_text(modelfile_content)
        print(f"✅ Modelfile: {modelfile_path}")
        print(f"   Run: ollama create {run_name} -f {modelfile_path}")
                if cfg["wandb"]["enabled"]:
            wandb.finish()

    if local_rank != -1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()