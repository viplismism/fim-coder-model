#!/usr/bin/env python3
"""Repo-specific evaluation: generate FIM completions, then score them.

  generate (GPU): python gen_eval.py generate <config.yaml> <name> <examples.jsonl> <out_dir> [--adapter A] [--limit N]
      Greedy decoding, max 256 new tokens, stops at the model family's own end/FIM tokens (PSM.stop_ids). The
      prompt is the model's documented FIM layout with in-file context only (fimexp/psm.py). No language is named.
      Writes <out_dir>/<name>.completions.jsonl (completion, prompt tokens, generated tokens, seconds).
  ppl (GPU):      python gen_eval.py ppl <config.yaml> <name> <examples.jsonl> <out_dir> [--adapter A]
      Teacher-forced character perplexity of the REAL added code (same metric as the AST experiment:
      exp(-sum answer-token logprob / answer chars), end token not scored). Writes <out_dir>/<name>.ppl.jsonl.
  score (CPU):    python gen_eval.py score <repo_dir> <examples.jsonl> <completions.jsonl> <out.jsonl> [workers]
      exact match (all examples): completion == reference after normalising line ends (trailing spaces per line
      removed, leading/trailing blank lines removed);
      build+tests (sensitive examples only): the file at the example's commit with the completion in the hole
      (a final newline is added if missing, since holes are whole lines) -> `go build ./...` in the module, then
      `go test -count=1 <package>`; pass = both succeed.
Examples are the usable set (reference passes build+tests at its commit), with the `sensitive` flag from exec_check.
"""
import argparse, collections, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "repo_specific"))
MAX_NEW = 256


def norm_lines(s):
    rows = [l.rstrip() for l in s.replace("\r\n", "\n").split("\n")]
    while rows and not rows[0]:
        rows.pop(0)
    while rows and not rows[-1]:
        rows.pop()
    return "\n".join(rows)


def generate(args):
    import torch, yaml
    from fimexp.model import load_for_scoring, load_tokenizer
    from fimexp.psm import PSM
    cfg = yaml.safe_load(open(args.config))
    tok = load_tokenizer(cfg); psm = PSM(tok, cfg["format"])
    ctx = cfg["evaluation"]["context_tokens"]
    model = load_for_scoring(cfg, args.adapter)
    exs = [json.loads(l) for l in open(args.examples)][: args.limit or None]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.completions.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for e in exs:
            if e["id"] in done:
                continue
            ids = psm.prompt(e["prefix"], e["suffix"])
            if len(ids) + MAX_NEW > ctx:
                f.write(json.dumps({"id": e["id"], "skipped_over_context": len(ids)}) + "\n"); continue
            t0 = time.time()
            x = torch.tensor([ids], device=model.device)
            with torch.no_grad():
                y = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=MAX_NEW, do_sample=False,
                                   eos_token_id=psm.stop_ids, pad_token_id=psm.stop_ids[0])
            gen = y[0, len(ids):].tolist()
            stopped = bool(gen) and gen[-1] in psm.stop_ids
            if stopped:
                gen = gen[:-1]
            comp = tok.decode(gen, skip_special_tokens=False)
            f.write(json.dumps({"id": e["id"], "model": args.name, "adapter": args.adapter, "completion": comp,
                                "prompt_tokens": len(ids), "gen_tokens": len(gen), "stopped": stopped,
                                "seconds": round(time.time() - t0, 2)}) + "\n"); f.flush()
            print(e["id"], len(ids), len(gen), round(time.time() - t0, 1), flush=True)


def ppl(args):
    import torch, yaml
    from fimexp.model import load_for_scoring, load_tokenizer
    from fimexp.psm import PSM, char_perplexity
    from fimexp.score import answer_logprobs
    cfg = yaml.safe_load(open(args.config))
    tok = load_tokenizer(cfg); psm = PSM(tok, cfg["format"]); ctx = cfg["evaluation"]["context_tokens"]
    model = load_for_scoring(cfg, args.adapter)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}.ppl.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    with open(path, "a") as f:
        for e in map(json.loads, open(args.examples)):
            if e["id"] in done:
                continue
            b = psm.build(e["prefix"], e["suffix"], e["middle"], add_eos=False)
            if len(b["input_ids"]) > ctx:
                f.write(json.dumps({"id": e["id"], "skipped_over_context": len(b["input_ids"])}) + "\n"); continue
            with torch.no_grad():
                lps = answer_logprobs(model, b)
            s_ = sum(lps); n = len(e["middle"])
            f.write(json.dumps({"id": e["id"], "model": args.name, "n_chars": n, "n_answer_tokens": len(lps),
                                "sum_logprob": s_, "char_ppl": char_perplexity(s_, n)}) + "\n"); f.flush()


def score(args):
    from concurrent.futures import ThreadPoolExecutor
    import shutil, subprocess, tempfile
    from exec_check import check, P
    exs = {json.loads(l)["id"]: json.loads(l) for l in open(args.examples)}
    comps = {c["id"]: c for c in map(json.loads, open(args.completions))}
    by_c = collections.defaultdict(list)
    for i, c in comps.items():
        by_c[exs[i]["commit"]].append(i)

    def do(commit, ids):
        res = []
        wt = Path(tempfile.mkdtemp(prefix=f"score_{commit[:10]}_", dir=P / "sources_repo_specific")); shutil.rmtree(wt)
        subprocess.run(["git", "-C", args.repo, "worktree", "add", "-q", "--detach", str(wt), commit], check=True)
        try:
            for i in ids:
                e, c = exs[i], comps[i]
                r = {"id": i, "is_test_file": e["is_test_file"], "sensitive": e["sensitive"], "hole_lines": e["hole_lines"]}
                if "completion" not in c:
                    res.append({**r, "skipped": True}); continue
                comp = c["completion"] if c["completion"].endswith("\n") else c["completion"] + "\n"
                r["exact_match"] = norm_lines(comp) == norm_lines(e["middle"])
                if e["sensitive"]:
                    mod_dir = wt / (e["module"] if e["module"] != "." else "")
                    pkg = "./" + os.path.relpath(os.path.dirname(wt / e["path"]), mod_dir)
                    f = wt / e["path"]; orig = f.read_text()
                    f.write_text(e["prefix"] + comp + e["suffix"])
                    ch = check(mod_dir, pkg); f.write_text(orig)
                    r.update({"build": ch["build"], "test": ch["test"], "build_test_pass": ch["build"] and ch["test"],
                              "log": ch["log"][-400:]})
                res.append(r)
        finally:
            subprocess.run(["git", "-C", args.repo, "worktree", "remove", "--force", str(wt)])
        return res

    with open(args.out, "w") as fo, ThreadPoolExecutor(args.workers) as pool:
        for rs in pool.map(lambda kv: do(*kv), sorted(by_c.items())):
            for r in rs:
                fo.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("config"); g.add_argument("name"); g.add_argument("examples")
    g.add_argument("out_dir"); g.add_argument("--adapter"); g.add_argument("--limit", type=int)
    q = sub.add_parser("ppl"); q.add_argument("config"); q.add_argument("name"); q.add_argument("examples")
    q.add_argument("out_dir"); q.add_argument("--adapter")
    s = sub.add_parser("score"); s.add_argument("repo"); s.add_argument("examples"); s.add_argument("completions")
    s.add_argument("out"); s.add_argument("workers", type=int, nargs="?", default=4)
    a = ap.parse_args()
    {"generate": generate, "ppl": ppl, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
