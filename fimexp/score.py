#!/usr/bin/env python3
"""Teacher-forced scoring of the correct answer (no generation).

Character-level perplexity = exp(-sum of answer-token log probs / answer characters); the EOS is not scored.
Tasks:
  bench_add, bench_edit       Real-FIM-Eval Rust (PSM; Edit uses the inferred conflict-merge layout)
  reth_ast, reth_random       held-out reth infilling from data/<version>/<split>_<arm>.jsonl
  reth_l2r                    ordinary left-to-right on whole held-out reth files (line-bounded windows)
Held-out reth tasks read --split validation by default; --split test requires --final.
Usage: python -m fimexp.score --tasks bench_add,reth_ast --data_version v1 [--adapter runs/x/final] --out runs/eval/name
"""

import argparse, itertools, json, sys, time
from pathlib import Path

import torch

from . import sources
from .build_data import code_revision
from .holes import SourceText, line_token_counts
from .model import load_for_scoring, load_tokenizer
from .psm import PSM, char_perplexity


@torch.no_grad()
def answer_logprobs(model, b) -> list[float]:
    """Log prob of each answer token given everything before it (positions n_prompt-1 .. end-2)."""
    ids = torch.tensor([b["input_ids"]], device=model.device)
    n_answer = b["n_answer"]
    logits = model(input_ids=ids, logits_to_keep=n_answer + 1).logits[0, :-1].float()
    targets = ids[0, -n_answer:]
    return torch.log_softmax(logits, dim=-1).gather(1, targets[:, None])[:, 0].tolist()


def edit_layout(r, t):
    prefix = r["prompt"] + t["before_removed"] + r["to_remove"] + t["after_removed"]
    return prefix, t["suffix_start"] + r["suffix"]


def bench_items(cfg, split, lang=None):
    """Real-FIM-Eval items of one language. The prompt is language-neutral: only code plus the three FIM markers
    (and, for Edit, the conflict-merge markers); the language is never named."""
    t = cfg["evaluation"]["edit_template"]
    for r in sources.benchmark_rows(cfg, lang):
        if r["split"] != split:
            continue
        prefix, suffix = (r["prompt"], r["suffix"]) if split == "add" else edit_layout(r, t)
        yield {"id": f"{split}:{r['index']}", "repo": r["repo"], "lang": r["lang"], "prefix": prefix, "suffix": suffix,
               "middle": r["canonical_solution"]}


def reth_fim_items(cfg, version, split, arm, pair_ids=None):
    path = sources.ROOT / cfg["data"]["output_dir"] / version / f"{split}_{arm}.jsonl"
    for line in open(path):
        e = json.loads(line)
        if pair_ids is not None and e["pair_id"] not in pair_ids:
            continue
        yield {"id": e["id"], "crate": e["crate"], "prefix": e["prefix"], "suffix": e["suffix"], "middle": e["middle"]}


def l2r_windows(src: SourceText, psm, limit: int):
    """Split a file at line boundaries into consecutive windows of <= `limit` tokens (BOS included).
    Returns (windows, skipped_lines); windows concatenate back to the file except for any single line
    that alone exceeds the limit."""
    counts = line_token_counts(src, psm)
    ls = src.line_starts + [len(src.text)]
    fits = lambda i, j: len(psm.build_l2r(src.text[ls[i]:ls[j]])["input_ids"]) <= limit
    windows, skipped, i, n = [], [], 0, len(counts)
    while i < n:
        j, used = i, 0
        while j < n and (j == i or used + counts[j] <= limit - 1):
            used += counts[j]; j += 1
        while j > i + 1 and not fits(i, j):  # per-line counts are estimates; verify exactly
            j -= 1
        if fits(i, j):
            windows.append(src.text[ls[i]:ls[j]])
        else:
            skipped.append(i)
        i = j
    return windows, skipped


def resolve_source(cfg, version, path):
    """Source file of a data record: reth for the Rust experiment data; pinned repos for multi-repo language data
    (paths are '<owner>/<repo>/<file>')."""
    man = json.loads((sources.ROOT / cfg["data"]["output_dir"] / version / "manifest.json").read_text())
    if "pins" not in man:
        return sources.ROOT / cfg["sources"]["reth"]["path"] / path
    repo = "/".join(path.split("/")[:2])
    return sources.ROOT / man["pins"]["repos"][repo]["path"] / "/".join(path.split("/")[2:])


def l2r_items(cfg, version, split, psm):
    """Whole files of a split as line-bounded left-to-right windows."""
    limit = cfg["evaluation"]["l2r_window_tokens"]
    split_info = json.loads((sources.ROOT / cfg["data"]["output_dir"] / version / "split.json").read_text())
    for path in split_info["files_by_split"][split]:
        windows, skipped = l2r_windows(SourceText(sources.read_source(resolve_source(cfg, version, path))), psm, limit)
        if skipped:
            print(f"warning: {path}: {len(skipped)} line(s) longer than {limit} tokens not scored", file=sys.stderr)
        crate = max((c for c in split_info["crate_to_split"] if path.startswith(c + "/")), key=len, default=None)
        for w, text in enumerate(windows):
            yield {"id": f"{path}#w{w}", "text": text, "crate": crate}


HELDOUT_VERSION = "heldout-v1"


def heldout_fim_items(cfg, arm, version=HELDOUT_VERSION):
    path = sources.ROOT / cfg["data"]["output_dir"] / version / f"heldout_{arm}.jsonl"
    for line in open(path):
        e = json.loads(line)
        yield {"id": e["id"], "crate": e["crate"], "prefix": e["prefix"], "suffix": e["suffix"], "middle": e["middle"]}


def heldout_l2r_items(cfg, psm, version=HELDOUT_VERSION):
    hdir = sources.ROOT / cfg["data"]["output_dir"] / version
    files = json.loads((hdir / "l2r_files.json").read_text())
    repo_paths = json.loads((hdir / "manifest.json").read_text()).get("repos", {})
    limit = cfg["evaluation"]["l2r_window_tokens"]
    for repo, rels in files.items():
        d = (sources.ROOT / repo_paths[repo]["path"]) if isinstance(repo_paths, dict) and repo in repo_paths and "path" in repo_paths[repo] \
            else sources.ROOT / "sources" / "heldout" / repo.replace("/", "__")
        for rel in rels:
            windows, _ = l2r_windows(SourceText(sources.read_source(d / rel)), psm, limit)
            for w, text in enumerate(windows):
                yield {"id": f"{repo}/{rel}#w{w}", "text": text, "crate": repo}


def run_task(model, psm, task, items, max_tokens, out_path):
    tot_lp = tot_chars = tot_tok = 0.0
    n = 0
    groups = {}
    with open(out_path, "w") as f:
        for it in items:
            if task in ("reth_l2r", "heldout_l2r"):
                b = psm.build_l2r(it["text"]); n_chars = len(it["text"])
            else:
                b = psm.build(it["prefix"], it["suffix"], it["middle"], add_eos=False); n_chars = len(it["middle"])
            if len(b["input_ids"]) > max_tokens:
                raise RuntimeError(f"{it['id']}: {len(b['input_ids'])} tokens exceeds context {max_tokens}; no silent truncation")
            lps = answer_logprobs(model, b)
            s = sum(lps)
            rec = {"id": it["id"], "n_chars": n_chars, "n_answer_tokens": len(lps), "n_tokens": len(b["input_ids"]),
                   "sum_logprob": s, "char_ppl": char_perplexity(s, n_chars)}
            rec.update({k: it[k] for k in ("repo", "crate") if k in it})
            f.write(json.dumps(rec) + "\n")
            tot_lp += s; tot_chars += n_chars; tot_tok += len(lps); n += 1
            g = groups.setdefault(rec.get("crate") or rec.get("repo") or "all", [0, 0.0, 0])
            g[0] += 1; g[1] += s; g[2] += n_chars
    return {"examples": n, "chars": tot_chars, "answer_tokens": tot_tok,
            "corpus_char_ppl": char_perplexity(tot_lp, tot_chars) if n else None,
            "token_nll": -tot_lp / tot_tok if n else None,
            "per_package": {k: {"examples": v[0], "chars": v[2], "corpus_char_ppl": char_perplexity(v[1], v[2])}
                            for k, v in sorted(groups.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--data_version", default=None)
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test (final reporting only)")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--lang", default=None, help="Real-FIM-Eval language for bench_* tasks (default: config, i.e. rust)")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N items per task (inspection)")
    ap.add_argument("--final_heldout", action="store_true", help="required for heldout_* tasks (LOCKED set; final run only)")
    ap.add_argument("--heldout_version", default=HELDOUT_VERSION, help="locked held-out dataset (default: the Rust heldout-v1)")
    ap.add_argument("--pair_subset", default=None, help="validation_subset.json in the data dir: score only these pair ids")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        sys.exit("test split is for final reporting only; pass --final")
    if any(t.startswith("heldout_") for t in args.tasks.split(",")) and not args.final_heldout:
        sys.exit("heldout_* tasks score the LOCKED fresh held-out set; only in the final run, with --final_heldout")
    cfg = sources.load_config(args.config)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    pins = sources.verify_pins(cfg)
    tok = load_tokenizer(cfg)
    psm = PSM(tok, cfg["format"])
    model = load_for_scoring(cfg, args.adapter)
    out.mkdir(parents=True)
    ctx = cfg["evaluation"]["context_tokens"]
    summary = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": pins, "code": code_revision(),
               "model": cfg["model"], "adapter": args.adapter,
               "base_precision": cfg["base_precision"], "data_version": args.data_version, "split": args.split,
               "benchmark_lang": args.lang or cfg["sources"]["benchmark"]["lang"],
               "pair_subset": args.pair_subset, "edit_template": cfg["evaluation"]["edit_template"],
               "rules": {"metric": "exp(-sum answer-token logprob / answer chars); EOS not scored",
                         "context": f"full context up to {ctx} tokens, error otherwise"}, "tasks": {}}
    for task in args.tasks.split(","):
        if task in ("bench_add", "bench_edit"):
            items = bench_items(cfg, task.split("_")[1], args.lang)
        elif task in ("reth_ast", "reth_random"):
            ids = None
            if args.pair_subset:
                ids = set(json.loads((sources.ROOT / cfg["data"]["output_dir"] / args.data_version / args.pair_subset).read_text())["pair_ids"])
            items = reth_fim_items(cfg, args.data_version, args.split, task.split("_")[1], ids)
        elif task == "reth_l2r":
            items = l2r_items(cfg, args.data_version, args.split, psm)
        elif task in ("heldout_ast", "heldout_random"):
            items = heldout_fim_items(cfg, task.split("_")[1], args.heldout_version)
        elif task == "heldout_l2r":
            items = heldout_l2r_items(cfg, psm, args.heldout_version)
        else:
            sys.exit(f"unknown task {task}")
        if args.limit:
            items = itertools.islice(items, args.limit)
        summary["tasks"][task] = run_task(model, psm, task, items, ctx, out / f"{task}.jsonl")
        if task == "bench_edit":
            summary["tasks"][task]["status"] = "PROVISIONAL: Edit prompt layout inferred from the dataset figure, not confirmed"
        print(task, summary["tasks"][task], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
