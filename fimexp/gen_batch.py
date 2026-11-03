#!/usr/bin/env python3
"""Batched greedy FIM generation on ALL validation pairs (AST holes + random holes) for base / AST / random models.

Prompts are sorted by length and left-padded in batches. Generation budget per item:
min(max_new_cap, ceil(1.25 * answer_tokens) + 32) new tokens, identical for every model; hitting it = "no stop".
Usage: python -m fimexp.gen_batch --data_version rust-fim-v2 --out results/val_generations [--models base,ast,random]
VALIDATION ONLY; test, benchmark and the held-out set are not read.
"""

import argparse, json, math, time
from pathlib import Path

import torch

from . import sources
from .model import load_for_scoring, load_tokenizer
from .psm import PSM


def items_validation(cfg, version):
    d = sources.ROOT / cfg["data"]["output_dir"] / version
    out = []
    for arm in ("ast", "random"):
        for line in open(d / f"validation_{arm}.jsonl"):
            e = json.loads(line)
            out.append({"set": f"reth_{arm}", "id": e["id"], "pair_id": e["pair_id"], "crate": e["crate"], "path": e["path"],
                        "hole_chars": e["hole_chars"], "window_chars": e["window_chars"], "node_type": e.get("node_type"),
                        "prefix": e["prefix"], "suffix": e["suffix"], "middle": e["middle"], "answer_tokens": e["tokens"]["middle"]})
    return out


@torch.no_grad()
def generate_batch(model, psm, batch, cap):
    stops = [psm.ids[k] for k in ("eos", "fim_begin", "fim_hole", "fim_end")]
    prompts = [psm.build(it["prefix"], it["suffix"], "", add_eos=False)["input_ids"] for it in batch]
    budgets = [min(cap, math.ceil(1.25 * it["answer_tokens"]) + 32) for it in batch]
    n = max(map(len, prompts))
    pad = psm.ids["eos"]
    ids = torch.tensor([[pad] * (n - len(p)) + p for p in prompts], device=model.device)
    attn = torch.tensor([[0] * (n - len(p)) + [1] * len(p) for p in prompts], device=model.device)
    out = model.generate(input_ids=ids, attention_mask=attn, do_sample=False, max_new_tokens=max(budgets),
                         eos_token_id=stops, pad_token_id=pad)
    res = []
    for k, it in enumerate(batch):
        new = out[k, n:].tolist()[: budgets[k]]          # each item gets only its own budget
        cut = next((j for j, t in enumerate(new) if t in stops), None)
        stopped = cut is not None
        cut = cut if stopped else len(new)
        text = psm.tok.decode(new[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        res.append({"id": it["id"], "set": it["set"], "text": text, "new_tokens": cut, "stopped": stopped, "budget": budgets[k]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="base,ast,random")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_cap", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_answer_tokens", type=int, default=None, help="keep pairs whose AST and random answers are both <= this")
    ap.add_argument("--merge_adapter", action="store_true", help="merge LoRA into a bf16 copy for faster decoding")
    ap.add_argument("--write_items_only", action="store_true")
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapters", default=None, help="name=path,... (default: the 6.7B ast/random final adapters)")
    args = ap.parse_args()
    cfg = sources.load_config(args.config)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pins = sources.verify_pins(cfg)
    psm = PSM(load_tokenizer(cfg), cfg["format"])
    items = items_validation(cfg, args.data_version)
    if args.max_answer_tokens:
        by = {}
        for it in items:
            by.setdefault(it["pair_id"], []).append(it)
        keep = {p for p, v in by.items() if len(v) == 2 and all(x["answer_tokens"] <= args.max_answer_tokens for x in v)}
        items = [it for it in items if it["pair_id"] in keep]
    items = items[: args.limit]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["prefix"]) + len(items[i]["suffix"]))
    if (out / "items.jsonl").exists():
        saved = [json.loads(l)["id"] for l in open(out / "items.jsonl")]
        if saved != [i["id"] for i in items]:
            raise SystemExit("items.jsonl in the output dir was built with different selection settings")
    else:
        (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    adapters = {"base": None} | dict(kv.split("=", 1) for kv in args.adapters.split(",")) if args.adapters else {"base": None,
                "ast": str(sorted(sources.ROOT.glob("logs/*_ast_train_s42"))[-1] / "checkpoints" / "final_adapter"),
                "random": str(sorted(sources.ROOT.glob("logs/*_random_train_s42"))[-1] / "checkpoints" / "final_adapter")}
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": pins, "model": cfg["model"], "base_precision": cfg["base_precision"],
            "adapters": adapters, "split": "validation", "items": len(items), "batch_size": args.batch_size,
            "max_answer_tokens": args.max_answer_tokens, "merge_adapter": args.merge_adapter,
            "decoding": f"greedy, batch size {args.batch_size}; stop at EOS/FIM sentinel; budget=min({args.max_new_cap}, ceil(1.25*answer_tokens)+32)"}
    if not (out / "meta.json").exists():
        (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    if args.write_items_only:
        print(f"{len(items)} items written"); return
    for name in args.models.split(","):
        path = out / f"{name}.jsonl"
        done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
        todo = [i for i in order if items[i]["id"] not in done]
        if not todo:
            continue
        model = load_for_scoring(cfg, adapters[name])
        if adapters[name] and args.merge_adapter:
            model = model.merge_and_unload()
        t0 = time.time(); n_done = 0
        with open(path, "a") as f:
            for b in range(0, len(todo), args.batch_size):
                batch = [items[i] for i in todo[b:b + args.batch_size]]
                for r in generate_batch(model, psm, batch, args.max_new_cap):
                    r.update({"exact_match": r["text"] == next(it["middle"] for it in batch if it["id"] == r["id"])})
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush(); n_done += len(batch)
                print(f"{name} {n_done}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done", out)


if __name__ == "__main__":
    main()
