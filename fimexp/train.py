#!/usr/bin/env python3
"""LoRA FIM fine-tuning on DeepSeek-Coder-6.7B base, PSM only; both arms use this exact code path.

Loss: token-average negative log-likelihood over the answer (middle) tokens plus the final EOS;
prefix, suffix, BOS and FIM sentinels are context only (label -100). Development evaluation reads the
validation split; the test split is never loaded here.

Each run writes logs/<timestamp>_<arm>_<smoke|train>_s<seed>/:
  config.yaml  manifest.json  metrics.jsonl  console.log  code.diff  requirements.lock  checkpoints/
Usage:
  python -m fimexp.train --data_version <v> --arm ast                       # full run (1 epoch)
  python -m fimexp.train --data_version <v> --arm ast --smoke_updates 5     # smoke: longest sequences first
"""

import argparse, datetime, json, math, random, shutil, subprocess, sys, time, traceback
from pathlib import Path

import torch
import yaml

from . import sources
from .build_data import code_revision
from .model import load_base, load_tokenizer
from .psm import IGNORE, PSM

LOGS = sources.ROOT / "logs"
REQUIRED = {
    "lora": ["r", "alpha", "dropout", "target_modules"],
    "training": ["num_train_epochs", "per_device_batch_size", "gradient_accumulation_steps", "learning_rate",
                 "lr_scheduler_type", "warmup_ratio", "weight_decay", "max_grad_norm", "optim", "seed",
                 "eval_steps", "save_steps", "checkpoint_policy"],
}


def missing_settings(cfg) -> list[str]:
    return [f"{sec}.{k}" for sec, keys in REQUIRED.items() for k in keys if cfg[sec].get(k) is None]


class FIMDataset(torch.utils.data.Dataset):
    def __init__(self, rows, psm, max_tokens):
        self.rows, self.psm, self.max_tokens = rows, psm, max_tokens

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        b = self.psm.build(r["prefix"], r["suffix"], r["middle"], add_eos=True)
        if len(b["input_ids"]) != r["tokens"]["total_with_eos"] or len(b["input_ids"]) > self.max_tokens:
            raise RuntimeError(f"{r['id']}: token length {len(b['input_ids'])} disagrees with the data manifest")
        return {"input_ids": b["input_ids"], "labels": b["labels"]}


class Collator:
    """Right-pad; padded positions are masked from attention and loss."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            k = len(x["input_ids"])
            ids[i, :k] = torch.tensor(x["input_ids"]); labels[i, :k] = torch.tensor(x["labels"]); attn[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class UpdateCounter:
    """Counts what the model actually receives in training steps (the dataloader prefetches one batch ahead,
    so counting in the collator over-counts by one example per log)."""

    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.reset()

    def reset(self):
        self.examples = self.tokens = self.answer_tokens = self.max_len = 0

    def add(self, inputs):
        n = inputs["input_ids"].shape[1]
        if n > self.max_tokens:
            raise RuntimeError(f"training input of {n} tokens exceeds max_seq_tokens {self.max_tokens}")
        self.examples += inputs["input_ids"].shape[0]
        self.tokens += int(inputs["attention_mask"].sum())
        self.answer_tokens += int((inputs["labels"] != IGNORE).sum())
        self.max_len = max(self.max_len, n)


def make_trainer_class(counter):
    from transformers import Trainer

    class CountingTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            counter.add(inputs)
            return super().training_step(model, inputs, num_items_in_batch)

    return CountingTrainer


def make_callback(path, counter, wandb_run=None):
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        """One JSON line per optimizer-update log and per evaluation; stops on non-finite values."""

        def __init__(self):
            self.f = open(path, "a")
            self.t_last = time.time()
            self.failed = None

        def on_log(self, args, state, control, logs=None, **kw):
            logs = dict(logs or {})
            now = time.time()
            rec = {"time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "update": state.global_step,
                   "epoch": state.epoch, **logs}
            if "loss" in logs:
                rec.update({"seconds_since_last_log": round(now - self.t_last, 2), "examples": counter.examples,
                            "tokens": counter.tokens, "answer_tokens": counter.answer_tokens, "max_seq_len": counter.max_len,
                            "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)})
                counter.reset()
                self.t_last = now
            self.f.write(json.dumps(rec) + "\n"); self.f.flush()
            if wandb_run is not None:
                scalars = {k: float(v) for k, v in rec.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "update"}
                wandb_run.log(scalars, step=state.global_step)
            for k in ("loss", "grad_norm", "eval_loss"):
                if k in logs and not math.isfinite(float(logs[k])):
                    self.failed = f"non-finite {k} at update {state.global_step}"
                    control.should_training_stop = True

    return MetricsCallback()


def git(*a):
    return subprocess.run(["git", "-C", str(sources.ROOT), *a], capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_version", required=True)
    ap.add_argument("--arm", required=True, choices=["ast", "random"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke_updates", type=int, default=None, help="smoke run: this many optimizer updates, longest sequences first")
    ap.add_argument("--timing_updates", type=int, default=None, help="timing run: this many updates on a fixed seeded sample; no eval/save/wandb")
    args = ap.parse_args()

    cfg = sources.load_config(args.config)
    missing = missing_settings(cfg)
    if missing:
        sys.exit("settings still to decide in configs/experiment.yaml: " + ", ".join(missing))
    t, lc = cfg["training"], cfg["lora"]
    if t["checkpoint_policy"] != "final":
        sys.exit("only checkpoint_policy: final is supported (the final checkpoint is the one evaluated)")
    timing = args.timing_updates is not None
    if timing and args.smoke_updates is not None:
        sys.exit("choose --smoke_updates or --timing_updates")
    smoke = args.smoke_updates is not None
    if timing:
        cfg["wandb"] = {**cfg.get("wandb", {}), "enabled": False}

    data_dir = sources.ROOT / cfg["data"]["output_dir"] / args.data_version
    data_manifest = json.loads((data_dir / "manifest.json").read_text())
    if data_manifest.get("inspection_build"):
        sys.exit(f"{data_dir} is an inspection build")
    stamp = datetime.datetime.now().astimezone()
    kind = "timing" if timing else "smoke" if smoke else "train"
    run_name = f"{stamp.strftime('%Y%m%dT%H%M%S%z')}_{args.arm}_{kind}_s{t['seed']}"
    run_dir = LOGS / run_name
    run_dir.mkdir(parents=True)
    console = open(run_dir / "console.log", "a")
    sys.stdout = Tee(sys.__stdout__, console); sys.stderr = Tee(sys.__stderr__, console)
    print(f"run {run_name}")

    pins = sources.verify_pins(cfg)
    tok = load_tokenizer(cfg)
    psm = PSM(tok, cfg["format"])
    max_tokens = cfg["data"]["max_seq_tokens"]
    train_rows = [json.loads(l) for l in open(data_dir / f"train_{args.arm}.jsonl")]
    val_rows = [json.loads(l) for l in open(data_dir / f"validation_{args.arm}.jsonl")]
    if smoke:
        per_update = t["per_device_batch_size"] * t["gradient_accumulation_steps"]
        n = args.smoke_updates * per_update
        longest = sorted(train_rows, key=lambda r: -r["tokens"]["total_with_eos"])[:per_update]
        ids = {r["id"] for r in longest}
        rest = random.Random(t["seed"]).sample([r for r in train_rows if r["id"] not in ids], n - per_update)
        train_rows = longest + rest                     # first update = the longest sequences
    if timing:  # fixed seeded sample: identical examples whenever this arm is timed (alone or concurrently)
        train_rows = random.Random(t["seed"]).sample(train_rows, args.timing_updates * t["per_device_batch_size"] * t["gradient_accumulation_steps"])
    subset = None
    if t.get("validation_subset"):  # fixed pair ids, identical for both arms and all models
        subset = json.loads((data_dir / t["validation_subset"]).read_text())
        keep = set(subset["pair_ids"])
        val_rows = [r for r in val_rows if r["pair_id"] in keep]
        if len(val_rows) != len(keep):
            sys.exit("validation subset does not match this data version")
    full_val_rows = [json.loads(l) for l in open(data_dir / f"validation_{args.arm}.jsonl")]
    train_ds, val_ds = FIMDataset(train_rows, psm, max_tokens), FIMDataset(val_rows, psm, max_tokens)

    per_update = t["per_device_batch_size"] * t["gradient_accumulation_steps"]
    updates = args.smoke_updates if smoke else args.timing_updates if timing else math.ceil(len(train_ds) / t["per_device_batch_size"] / t["gradient_accumulation_steps"]) * t["num_train_epochs"]
    warmup = math.ceil(t["warmup_ratio"] * updates)

    from peft import LoraConfig, get_peft_model
    from transformers import TrainingArguments
    targs = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"), run_name=run_name, seed=t["seed"], data_seed=t["seed"],
        num_train_epochs=t["num_train_epochs"], max_steps=updates if (smoke or timing) else -1,
        per_device_train_batch_size=t["per_device_batch_size"], per_device_eval_batch_size=t["per_device_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"], learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=warmup, weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"], optim=t["optim"], bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"], gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no" if timing else "steps", eval_steps=updates if smoke else t["eval_steps"],
        save_strategy="no" if timing else "steps", save_steps=updates if smoke else t["save_steps"], save_only_model=False,
        prediction_loss_only=True, remove_unused_columns=False, logging_steps=1, logging_first_step=True,
        report_to=t["report_to"], train_sampling_strategy="sequential" if (smoke or timing) else "random",
    )

    effective = {**cfg, "run": {"name": run_name, "arm": args.arm, "kind": kind,
                                "data_version": args.data_version, "train_examples": len(train_ds), "validation_examples": len(val_ds),
                                "examples_per_update": per_update, "optimizer_updates": updates, "warmup_updates": warmup,
                                "last_update_examples": len(train_ds) - (math.ceil(len(train_ds) / per_update) - 1) * per_update}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(effective, sort_keys=False, allow_unicode=True))
    (run_dir / "code.diff").write_text(git("diff", "HEAD"))
    shutil.copy(sources.ROOT / "env" / "requirements.lock", run_dir / "requirements.lock")
    from importlib.metadata import version as pkg_version
    manifest = {"run_name": run_name, "started": stamp.isoformat(timespec="seconds"), "timezone": str(stamp.tzinfo),
                "code": code_revision(), "code_untracked": git("ls-files", "--others", "--exclude-standard").split(),
                "dependency_lock_sha256": sources.sha256_file(sources.ROOT / "env" / "requirements.lock"),
                "packages": {p: pkg_version(p) for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers")},
                "sources": pins, "model": cfg["model"], "base_precision": cfg["base_precision"],
                "data": {"version": args.data_version, "manifest_created": data_manifest["created_utc"],
                         "manifest_sha256": sources.sha256_file(data_dir / "manifest.json"),
                         "split_sha256": sources.sha256_file(data_dir / "split.json"),
                         "train_file_sha256": sources.sha256_file(data_dir / f"train_{args.arm}.jsonl"),
                         "validation_subset_sha256": sources.sha256_file(data_dir / t["validation_subset"]) if subset else None,
                         "validation_examples_during_training": len(val_rows)},
                "smoke_train_ids": [r["id"] for r in train_rows] if (smoke or timing) else None,
                "gpu": torch.cuda.get_device_name(0)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    torch.manual_seed(t["seed"])
    model = load_base(cfg)
    lora = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    unexpected = [n for n, _ in trainable if "lora_A" not in n and "lora_B" not in n]
    manifest["trainable"] = {"tensors": len(trainable), "params": sum(c for _, c in trainable), "unexpected": unexpected,
                             "frozen_params": sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if unexpected:
        sys.exit(f"unexpected trainable parameters: {unexpected[:5]}")
    print(f"trainable {manifest['trainable']['params']:,} params in {len(trainable)} LoRA tensors; "
          f"{updates} optimizer updates, warmup {warmup}")

    from . import wandb_log
    wrun = wandb_log.init(cfg.get("wandb"), run_name, effective, run_dir)
    manifest["wandb"] = {"enabled": wrun is not None, "entity": cfg["wandb"].get("entity"), "project": cfg["wandb"].get("project"),
                         "settings": wandb_log.PRIVATE_SETTINGS, "run_id": wrun.id if wrun else None}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    collator = Collator(tok.eos_token_id)
    counter = UpdateCounter(max_tokens)
    cb = make_callback(run_dir / "metrics.jsonl", counter, wrun)
    trainer = make_trainer_class(counter)(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                                          data_collator=collator, callbacks=[cb])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    status = "completed"
    try:
        trainer.train()
        if cb.failed:
            status = "stopped: " + cb.failed
        elif not (smoke or timing):  # full validation split once, at the end (own hole type)
            full = trainer.evaluate(eval_dataset=FIMDataset(full_val_rows, psm, max_tokens), metric_key_prefix="final_full_validation")
            manifest["final_full_validation"] = full
    except Exception:
        status = "failed"
        traceback.print_exc()
    final = run_dir / "checkpoints" / "final_adapter"
    if status == "completed" and not timing:
        model.save_pretrained(final)
    manifest.update({"finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "status": status,
                     "wall_seconds": round(time.time() - t0, 1),
                     "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "final_adapter": str(final.relative_to(sources.ROOT)) if status == "completed" and not timing else None})
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if wrun is not None:
        wrun.summary.update({"status": status, "wall_seconds": manifest["wall_seconds"], "peak_memory_gb": manifest["peak_memory_gb"]})
        wrun.finish(exit_code=0 if status == "completed" else 1)
    print(f"{status}; {manifest['wall_seconds']} s; peak {manifest['peak_memory_gb']} GB; logs in {run_dir}")
    if status != "completed":
        sys.exit(1)


if __name__ == "__main__":
    main()
