#!/usr/bin/env python3
"""Summary of the facet-rs/facet comparison (cutoff 2026-01-01): Qwen2.5-Coder-14B base vs two LoRA fine-tunes
trained with identical settings on the cutoff snapshot, one on AST holes and one on random holes of identical lengths.
Metrics (fixed in advance):
  pass@1      = build+tests pass rate of the greedy completion (`cargo build -p <crate> --all-targets` + `cargo test
                -p <crate>`), on the SENSITIVE examples (reference passes, deleting the hole breaks build or tests);
  exact match = normalised-lines equality, on all usable examples;
  char ppl    = teacher-forced character perplexity of the real added code (exp(-sum logprob / chars)).
Only examples every model has (none skipped for context) are used, so all models see the same set.
Splits: source vs test files; authorship: human / ai_tagged (an AI co-author or "generated with" tag in the commit
messages) / bot_only (bot author, no AI tag; AI takes precedence). Extra pass@1 split: examples where no model hit the
256-token limit (stop effect check).
Comparisons: ast-base, random-base, ast-random. Paired bootstrap 95% CIs resampling COMMITS, B=2000, fixed seed.
Rule: A beats B iff diff > 0 and CI lower bound > 0 (for ppl: diff < 0 and CI upper bound < 0).
Optional 4th column (--extra NAME, e.g. qwen32b-base, added 2026-11-23 at the user's request): a model scored on
the same examples. The common set above is still defined by the three main models, so their numbers do not change;
under metrics[...]["extra"] the extra model and the three others are compared on the examples it has (n given), with
diffs extra-base, ast-extra, random-extra. "stop_check": for each model, on the pass@1 examples where it hit the
256-token limit, how many each model passes.
Usage: python -m eval.summarize_facet <scores_dir> <out.json> [base ast random] [--extra NAME]
"""
import collections, json, math, random, sys
from pathlib import Path

B, SEED = 2000, 20261121


def group(r):
    return "ai_tagged" if r.get("ai_tagged") else ("bot_only" if r.get("bot_author") else "human")


PARTS = {"all": lambda r: True, "source": lambda r: not r["is_test_file"], "test": lambda r: r["is_test_file"],
         "human": lambda r: group(r) == "human", "ai_tagged": lambda r: group(r) == "ai_tagged",
         "bot_only": lambda r: group(r) == "bot_only"}


def boot(ids, stat, pairs, lower_better=False):
    by_c = collections.defaultdict(list)
    for i in ids:
        by_c[i.split(":")[0]].append(i)
    cs = sorted(by_c)
    out = {}
    if not cs:
        return out
    rng = random.Random(SEED)
    d = {p: [] for p in pairs}
    for _ in range(B):
        smp = [i for c in (cs[rng.randrange(len(cs))] for _ in cs) for i in by_c[c]]
        for a, b in pairs:
            d[(a, b)].append(stat(a, smp) - stat(b, smp))
    for (a, b), xs in d.items():
        xs.sort(); v = stat(a, ids) - stat(b, ids)
        ci = [round(xs[int(0.025 * B)], 4), round(xs[int(0.975 * B) - 1], 4)]
        win = (v < 0 and ci[1] < 0) if lower_better else (v > 0 and ci[0] > 0)
        out[f"{a} - {b}"] = {"value": round(v, 4), "ci": ci, ("better" if lower_better else "beats"): win}
    return out


def main():
    argv = sys.argv[1:]
    extra = None
    if "--extra" in argv:
        k = argv.index("--extra"); extra = argv[k + 1]; del argv[k:k + 2]
    d, out = Path(argv[0]), Path(argv[1])
    base, ast, rnd = argv[2:5] if len(argv) > 4 else ("qwen14b-base", "facet-ast", "facet-random")
    names = [base, ast, rnd]
    pairs = [(ast, base), (rnd, base), (ast, rnd)]
    S = {n: {r["id"]: r for r in map(json.loads, open(d / f"{n}.scores.jsonl"))} for n in names}
    C = {n: {c["id"]: c for c in map(json.loads, open(d / f"{n}.completions.jsonl"))} for n in names}
    if extra:
        S[extra] = {r["id"]: r for r in map(json.loads, open(d / f"{extra}.scores.jsonl"))}
        C[extra] = {c["id"]: c for c in map(json.loads, open(d / f"{extra}.completions.jsonl"))}
        xok = lambda i: i in S[extra] and not S[extra][i].get("skipped") and i in C[extra] and "completion" in C[extra][i]
        xnames = names + [extra]; xpairs = [(extra, base), (ast, extra), (rnd, extra)]
    ids = sorted(i for i in set.intersection(*(set(S[n]) for n in names)) if not any(S[n][i].get("skipped") for n in names))
    ref = S[base]
    res = {"models": {"base": base, "ast": ast, "random": rnd, **({"extra": extra} if extra else {})}, "examples_common": len(ids),
           "skipped_any_model": len(set.union(*(set(S[n]) for n in names))) - len(ids),
           "bootstrap": {"resample": "commits", "B": B, "seed": SEED}, "metrics": {}}
    for metric, sel in (("build_test_pass", lambda r: r.get("sensitive")), ("exact_match", lambda r: True)):
        parts = dict(PARTS)
        if metric == "build_test_pass":
            parts["neither_hit_limit"] = lambda r, i=None: True  # filtered below
        for part, pf in parts.items():
            ex = [i for i in ids if sel(ref[i]) and pf(ref[i])]
            if part == "neither_hit_limit":
                ex = [i for i in ex if all(C[n][i].get("stopped") for n in names)]
            rate = lambda n, idx: sum(bool(S[n][i].get(metric)) for i in idx) / max(1, len(idx))
            res["metrics"][f"{metric}/{part}"] = {"n": len(ex), "commits": len({i.split(':')[0] for i in ex}),
                                                  "rate": {n: round(rate(n, ex), 4) for n in names},
                                                  "diffs": boot(ex, rate, pairs)}
            if extra:
                xe = [i for i in ex if xok(i) and (part != "neither_hit_limit" or C[extra][i].get("stopped"))]
                res["metrics"][f"{metric}/{part}"]["extra"] = {
                    "n": len(xe), "commits": len({i.split(':')[0] for i in xe}),
                    "rate": {n: round(rate(n, xe), 4) for n in xnames}, "diffs": boot(xe, rate, xpairs)}
    PP = {n: {r["id"]: r for r in map(json.loads, open(d / f"{n}.ppl.jsonl")) if "sum_logprob" in r}
          for n in names if (d / f"{n}.ppl.jsonl").exists()}
    if len(PP) == 3:
        pids = [i for i in ids if all(i in PP[n] for n in names)]
        cp = lambda n, idx: math.exp(-sum(PP[n][i]["sum_logprob"] for i in idx) / max(1, sum(PP[n][i]["n_chars"] for i in idx)))
        for part, pf in PARTS.items():
            ex = [i for i in pids if pf(ref[i])]
            res["metrics"][f"char_ppl/{part}"] = {"n": len(ex), "commits": len({i.split(':')[0] for i in ex}),
                                                  "char_ppl": {n: round(cp(n, ex), 4) for n in names} if ex else {},
                                                  "diffs": boot(ex, cp, pairs, lower_better=True)}
    res["generation"] = {n: {"mean_seconds": round(sum(c.get("seconds", 0) for c in C[n].values()) / max(1, len(C[n])), 2),
                             "hit_256_token_limit": sum(1 for i in ids if not C[n][i].get("stopped")),
                             "skipped_over_context": sum(1 for c in C[n].values() if "skipped_over_context" in c)} for n in names}
    allm = names + [extra] if extra else names
    pid = [i for i in ids if ref[i].get("sensitive") and (not extra or xok(i))]
    res["stop_check"] = {"examples": len(pid), "by_model_that_hit_limit": {
        h: {"n": len(hs), "passes": {n: sum(bool(S[n][i].get("build_test_pass")) for i in hs) for n in allm}}
        for h in allm for hs in [[i for i in pid if not C[h][i].get("stopped")]]}}
    if extra:
        res["generation"][extra] = {
            "mean_seconds": round(sum(c.get("seconds", 0) for c in C[extra].values()) / max(1, len(C[extra])), 2),
            "hit_256_token_limit": sum(1 for i in ids if xok(i) and not C[extra][i].get("stopped")),
            "skipped_over_context": sum(1 for c in C[extra].values() if "skipped_over_context" in c)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    for k, m in res["metrics"].items():
        print(f"{k:32s} n={m['n']:5d}", m.get("rate", m.get("char_ppl")), {p: (x["value"], x["ci"]) for p, x in m["diffs"].items()})


if __name__ == "__main__":
    main()
