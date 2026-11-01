#!/usr/bin/env python3
"""Quantized-training checks for the declared LoRA setup, with and without prepare_model_for_kbit_training.

For each variant: target-module names exist, only LoRA params are trainable, base weights frozen, forward +
backward on the longest real training examples (finite loss/grads, time, memory), and a finite-difference
check of the adapter gradient. No optimizer step is taken. Writes checks/env/training_path.json.
Usage: python checks/env/check_training_path.py data/<version>
"""

import collections, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from fimexp import sources  # noqa: E402
from fimexp.model import load_tokenizer  # noqa: E402
from fimexp.psm import PSM  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=TARGETS)


def pick_examples(data_dir, psm):
    out = []
    for arm in ("ast", "random"):
        rows = [json.loads(l) for l in open(data_dir / f"train_{arm}.jsonl")]
        rows.sort(key=lambda r: -r["tokens"]["total_with_eos"])
        longest_mid = max(rows, key=lambda r: r["tokens"]["middle"])
        for r in rows[:2] + [longest_mid]:
            b = psm.build(r["prefix"], r["suffix"], r["middle"], add_eos=True)
            out.append({"id": r["id"], "arm": arm, "tokens": len(b["input_ids"]), "answer_tokens": b["n_answer"],
                        "input_ids": b["input_ids"], "labels": b["labels"]})
    return out


def run_variant(cfg, prepare, examples, short, precision="nf4"):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM
    from fimexp.model import bnb_config
    torch.cuda.reset_peak_memory_stats()
    if precision == "bf16":
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"],
                                                     dtype=torch.bfloat16, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], revision=cfg["model"]["revision"],
                                                     quantization_config=bnb_config(cfg), device_map={"": 0})
    leaf = collections.Counter(n.split(".")[-1] for n, m in model.named_modules() if not list(m.children()))
    target_counts = {t: leaf.get(t, 0) for t in TARGETS}
    if prepare:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    torch.manual_seed(0)  # identical LoRA init across variants, so their gradients are comparable
    model = get_peft_model(model, LoraConfig(**LORA))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    unexpected = [n for n, _ in trainable if "lora_A" not in n and "lora_B" not in n]
    by_kind = collections.Counter(("lora_A" if "lora_A" in n else "lora_B") + "." + n.split(".")[-3] for n, _ in trainable)
    rep = {"base_precision": precision, "prepare_model_for_kbit_training": prepare, "target_module_counts": target_counts,
           "trainable_tensors": len(trainable), "trainable_params": sum(p.numel() for _, p in trainable),
           "trainable_by_kind": dict(sorted(by_kind.items())), "unexpected_trainable": unexpected,
           "lora_param_dtypes": sorted({str(p.dtype) for _, p in trainable}),
           "frozen_params": sum(p.numel() for n, p in model.named_parameters() if not p.requires_grad),
           "base_dtypes": dict(collections.Counter(str(p.dtype) for n, p in model.named_parameters() if not p.requires_grad)),
           "examples": []}

    model.train()
    for ex in examples:
        ids = torch.tensor([ex["input_ids"]], device="cuda"); labels = torch.tensor([ex["labels"]], device="cuda")
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t0 = time.time()
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.cuda.synchronize()
        ga = [p.grad for n, p in trainable if "lora_A" in n and p.grad is not None]
        gb = [p.grad for n, p in trainable if "lora_B" in n and p.grad is not None]
        norm = lambda gs: torch.norm(torch.stack([g.float().norm() for g in gs])).item() if gs else None
        rep["examples"].append({"id": ex["id"], "arm": ex["arm"], "tokens": ex["tokens"], "answer_tokens": ex["answer_tokens"],
                                "loss": loss.item(), "loss_finite": bool(torch.isfinite(loss)),
                                "grads_finite": all(torch.isfinite(g).all().item() for g in ga + gb),
                                "lora_A_grad_norm": norm(ga), "lora_B_grad_norm": norm(gb),
                                "tensors_with_grad": len(ga) + len(gb), "fwd_bwd_seconds": round(time.time() - t0, 2)})
    rep["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    # Finite-difference check of the adapter gradient (eval mode: no dropout; no checkpointing needed)
    model.eval()
    ids = torch.tensor([short["input_ids"]], device="cuda"); labels = torch.tensor([short["labels"]], device="cuda")
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    params = [p for n, p in trainable if "lora_B" in n]
    g = [p.grad.float() for p in params]
    gnorm = torch.sqrt(sum((x ** 2).sum() for x in g)).item()
    dirs = [x / gnorm for x in g]  # unit direction along the gradient: expected slope = |g|
    def loss_at(step):
        with torch.no_grad():
            for p, d in zip(params, dirs):
                p.add_(step * d)
            l = model(input_ids=ids, labels=labels).loss.item()
            for p, d in zip(params, dirs):
                p.sub_(step * d)
            return l
    base_losses = [loss_at(0.0) for _ in range(3)]
    fd = []
    for eps in (0.01, 0.03, 0.1, 0.3):
        numeric = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
        fd.append({"eps": eps, "numeric_slope": numeric, "ratio_to_analytic": numeric / gnorm})
    rep["finite_difference_lora_B"] = {"analytic_slope_grad_norm": gnorm, "repeat_forward_losses": base_losses,
                                       "steps": fd, "example_tokens": len(short["input_ids"])}
    grads_snapshot = torch.cat([p.grad.float().flatten() for p in params]).cpu()
    del model
    torch.cuda.empty_cache()
    return rep, grads_snapshot


def main():
    data_dir = Path(sys.argv[1])
    cfg = sources.load_config()
    psm = PSM(load_tokenizer(cfg), cfg["format"])
    examples = pick_examples(data_dir, psm)
    rows = [json.loads(l) for l in open(data_dir / "train_ast.jsonl")]
    r = min((x for x in rows if x["tokens"]["total_with_eos"] >= 300), key=lambda x: x["tokens"]["total_with_eos"])
    b = psm.build(r["prefix"], r["suffix"], r["middle"], add_eos=True)
    short = {"input_ids": b["input_ids"], "labels": b["labels"]}
    out = {"lora": LORA, "data": str(data_dir), "variants": []}
    snaps = []
    variants = [("nf4", False), ("nf4", True), ("bf16", False)]
    if len(sys.argv) > 2:
        variants = [v for v in variants if v[0] in sys.argv[2].split(",")]
    for precision, prepare in variants:
        rep, snap = run_variant(cfg, prepare, examples, short, precision)
        out["variants"].append(rep); snaps.append(snap)
        print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1))
        for e in rep["examples"]:
            print(" ", e)
    names = [f"{p}{'+prep' if q else ''}" for p, q in variants]
    out["lora_B_grad_cosine"] = {f"{names[i]} vs {names[j]}": torch.nn.functional.cosine_similarity(snaps[i].double(), snaps[j].double(), dim=0).item()
                                 for i in range(len(snaps)) for j in range(i + 1, len(snaps))}
    print("lora_B gradient cosine:", out["lora_B_grad_cosine"])
    name = "training_path.json" if len(variants) == 3 else f"training_path_{'_'.join(names)}.json"
    (Path(__file__).parent / name).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
