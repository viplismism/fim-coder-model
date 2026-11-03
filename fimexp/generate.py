#!/usr/bin/env python3
"""Greedy FIM completions from base / AST-trained / random-trained models on a fixed sample, for inspection.

Sets (fixed seeded sample, identical for every model):
  reth_ast     reth TEST AST holes, stratified across test packages
  reth_random  the matched random holes of the same pairs
  bench_add    Real-FIM-Eval Rust Add, spread across repos
Usage: python -m fimexp.generate --data_version rust-fim-v2 --out results/generations
This is qualitative inspection after the final evaluation; it does not select any setting.
"""

import argparse, collections, difflib, json, random, sys, time
from pathlib import Path

import torch

from . import sources
from .model import load_for_scoring, load_tokenizer
from .psm import PSM
from .score import bench_items

MIN_ANSWER_TOKENS, MAX_ANSWER_TOKENS = 3, 256


def stratified(rows, key, n, rng):
    by = collections.defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    groups = sorted(by)
    rng.shuffle(groups)
    for g in groups:
        rng.shuffle(by[g])
    out = []
    while len(out) < n and any(by.values()):
        for g in groups:
            if by[g] and len(out) < n:
                out.append(by[g].pop())
    return out


def select(cfg, version, n, psm, seed=20261102):
    d = sources.ROOT / cfg["data"]["output_dir"] / version
    ast = [json.loads(l) for l in open(d / "test_ast.jsonl")]
    rnd = {r["pair_id"]: r for r in map(json.loads, open(d / "test_random.jsonl"))}
    ok = lambda r: MIN_ANSWER_TOKENS <= r["tokens"]["middle"] <= MAX_ANSWER_TOKENS
    rng = random.Random(seed)
    picks = stratified([r for r in ast if ok(r) and ok(rnd[r["pair_id"]])], "crate", n, rng)
    items = []
    for r in picks:
        for arm, e in (("reth_ast", r), ("reth_random", rnd[r["pair_id"]])):
            items.append({"set": arm, "id": e["id"], "group": e["crate"], "path": e["path"], "node_type": e.get("node_type"),
                          "prefix": e["prefix"], "suffix": e["suffix"], "middle": e["middle"]})
    bench = [b for b in bench_items(cfg, "add") if MIN_ANSWER_TOKENS <= len(psm.encode(b["middle"])) <= MAX_ANSWER_TOKENS]
    for b in stratified(bench, "repo", n, rng):
        items.append({"set": "bench_add", "id": b["id"], "group": b["repo"], "path": None, "node_type": None,
                      "prefix": b["prefix"], "suffix": b["suffix"], "middle": b["middle"]})
    return items


@torch.no_grad()
def generate(model, psm, item):
    b = psm.build(item["prefix"], item["suffix"], "", add_eos=False)
    ids = torch.tensor([b["input_ids"]], device=model.device)
    n_answer = len(psm.encode(item["middle"]))
    stops = [psm.ids[k] for k in ("eos", "fim_begin", "fim_hole", "fim_end")]
    t0 = time.time()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                         max_new_tokens=min(512, 2 * n_answer + 32), eos_token_id=stops, pad_token_id=psm.ids["eos"])
    new = out[0, ids.shape[1]:].tolist()
    stopped = bool(new) and new[-1] in stops
    if stopped:
        new = new[:-1]
    text = psm.tok.decode(new, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {"text": text, "new_tokens": len(new), "stopped": stopped, "seconds": round(time.time() - t0, 2)}


def edit_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if (a or b) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_version", required=True)
    ap.add_argument("--per_set", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = sources.load_config()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    pins = sources.verify_pins(cfg)
    psm = PSM(load_tokenizer(cfg), cfg["format"])
    items = select(cfg, args.data_version, args.per_set, psm)
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    models = {"base": None,
              "ast": str(sorted(sources.ROOT.glob("logs/*_ast_train_s42"))[-1] / "checkpoints" / "final_adapter"),
              "random": str(sorted(sources.ROOT.glob("logs/*_random_train_s42"))[-1] / "checkpoints" / "final_adapter")}
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sources": pins, "model": cfg["model"], "base_precision": cfg["base_precision"],
            "adapters": models, "decoding": "greedy; stop at EOS or any FIM sentinel; max_new_tokens=min(512, 2*answer_tokens+32)",
            "sets": dict(collections.Counter(i["set"] for i in items)), "edit_template_note": "Edit not included (provisional)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    for name, adapter in models.items():
        print(f"loading {name}", flush=True)
        model = load_for_scoring(cfg, adapter)
        with open(out / f"{name}.jsonl", "w") as f:
            for k, it in enumerate(items):
                g = generate(model, psm, it)
                g.update({"id": it["id"], "set": it["set"], "exact_match": g["text"] == it["middle"],
                          "edit_similarity": round(edit_similarity(g["text"], it["middle"]), 4)})
                f.write(json.dumps(g, ensure_ascii=False) + "\n"); f.flush()
                if k % 10 == 0:
                    print(f"  {name} {k + 1}/{len(items)}  {g['seconds']} s  exact={g['exact_match']}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("done", out)


if __name__ == "__main__":
    main()
