#!/usr/bin/env python3
"""Summarise the Rust (reth) AST vs random experiment from per-example perplexity logs (eval.perplexity outputs).

  final:   tasks reth_ast, reth_random, reth_l2r (reth test split, grouped by crate) and bench_add, bench_edit
           (Real-FIM-Eval Rust, grouped by repo; Edit is provisional: inferred prompt layout)
  heldout: tasks heldout_ast, heldout_random, heldout_l2r on the locked 15-repo Rust held-out set, grouped by repo

Character perplexity per task = exp(-sum logprob / sum chars). Paired bootstrap 95% CIs (B = 2000) on the differences,
resampling examples and, separately, groups. Logic is identical to the summaries in results/multilang/rust_*.json.
Usage: python -m eval.summarize_rust final|heldout <base_dir> <ast_dir> <random_dir> <out.json>
"""
import collections, json, math, random, sys


def final(E):
    out = {"eval_dirs": E, "tasks": {}}
    rng = random.Random(0)
    for t in ("reth_ast", "reth_random", "reth_l2r", "bench_add", "bench_edit"):
        d = {n: {r["id"]: r for r in map(json.loads, open(f"{E[n]}/{t}.jsonl"))} for n in E}
        ids = sorted(d["base"]); assert all(sorted(d[n]) == ids for n in E)
        chars = [d["base"][i]["n_chars"] for i in ids]
        grp_key = "repo" if t.startswith("bench") else "crate"
        grp = [d["base"][i].get(grp_key) or "?" for i in ids]
        lp = {n: [d[n][i]["sum_logprob"] for i in ids] for n in E}
        def ppl(n, idx): return math.exp(-sum(lp[n][i] for i in idx) / sum(chars[i] for i in idx))
        full = list(range(len(ids)))
        groups = collections.defaultdict(list)
        for k, g in enumerate(grp): groups[g].append(k)
        gl = list(groups)
        def ci(a, b, cluster):
            ds = []
            for _ in range(2000):
                if cluster:
                    s = [k for g in (rng.choice(gl) for _ in gl) for k in groups[g]]
                else:
                    s = [rng.randrange(len(ids)) for _ in ids]
                ds.append(ppl(a, s) - ppl(b, s))
            ds.sort(); return [round(ds[50], 4), round(ds[1949], 4)]
        res = {"n": len(ids), "groups": len(gl), "ppl": {n: round(ppl(n, full), 4) for n in E}, "diff": {}}
        for a, b in (("ast", "base"), ("random", "base"), ("ast", "random")):
            res["diff"][f"{a}-{b}"] = {"value": round(ppl(a, full) - ppl(b, full), 4), "ci_examples": ci(a, b, False), "ci_groups": ci(a, b, True)}
        per = {g: {n: round(ppl(n, idx), 4) for n in E} | {"n": len(idx)} for g, idx in groups.items()}
        res["ast_better_than_random_groups"] = f"{sum(1 for v in per.values() if v['ast'] < v['random'])}/{len(per)}"
        res["per_group"] = dict(sorted(per.items(), key=lambda x: -x[1]["n"]))
        if t == "bench_edit": res["status"] = "PROVISIONAL (inferred prompt layout)"
        out["tasks"][t] = res
        print(f"{t}: n={res['n']} | base {res['ppl']['base']} ast {res['ppl']['ast']} random {res['ppl']['random']} | ast-random {res['diff']['ast-random']}")
    return out


def heldout(E):
    rng = random.Random(0); out = {"eval_dirs": E, "tasks": {}}
    for t in ("heldout_ast", "heldout_random", "heldout_l2r"):
        d = {n: {r["id"]: r for r in map(json.loads, open(f"{E[n]}/{t}.jsonl"))} for n in E}
        ids = sorted(d["base"]); assert all(sorted(d[n]) == ids for n in E)
        ch = [d["base"][i]["n_chars"] for i in ids]; grp = [d["base"][i]["crate"] for i in ids]
        lp = {n: [d[n][i]["sum_logprob"] for i in ids] for n in E}
        ppl = lambda n, idx: math.exp(-sum(lp[n][i] for i in idx) / sum(ch[i] for i in idx))
        groups = collections.defaultdict(list)
        for k, g in enumerate(grp): groups[g].append(k)
        gl = list(groups); full = list(range(len(ids)))
        def ci(a, b, cluster):
            ds = sorted(ppl(a, s) - ppl(b, s) for s in (([k for g in (rng.choice(gl) for _ in gl) for k in groups[g]]) if cluster else [rng.randrange(len(ids)) for _ in ids] for _ in range(2000)))
            return [round(ds[50], 4), round(ds[1949], 4)]
        p = {n: ppl(n, full) for n in E}
        bits = {n: math.log2(p[n]) for n in E}
        res = {"n": len(ids), "repos": len(gl), "ppl": {n: round(v, 4) for n, v in p.items()},
               "uncertainty_reduction_vs_base": {n: round(1 - bits[n] / bits["base"], 4) for n in ("ast", "random")},
               "diff": {f"{a}-{b}": {"value": round(p[a] - p[b], 4), "ci_examples": ci(a, b, False), "ci_repos": ci(a, b, True)}
                        for a, b in (("ast", "base"), ("random", "base"), ("ast", "random"))},
               "per_repo": {g: {n: round(ppl(n, idx), 4) for n in E} | {"n": len(idx)} for g, idx in sorted(groups.items())}}
        res["ast_better_than_random_repos"] = f"{sum(v['ast'] < v['random'] for v in res['per_repo'].values())}/{len(gl)}"
        res["tuned_better_than_base_repos"] = {n: f"{sum(v[n] < v['base'] for v in res['per_repo'].values())}/{len(gl)}" for n in ("ast", "random")}
        out["tasks"][t] = res
        print(f"{t}: n={res['n']} | base {res['ppl']['base']} ast {res['ppl']['ast']} random {res['ppl']['random']} | ast-random {res['diff']['ast-random']}")
    return out


if __name__ == "__main__":
    mode, b, a, r, out = sys.argv[1:6]
    res = {"final": final, "heldout": heldout}[mode]({"base": b, "ast": a, "random": r})
    json.dump(res, open(out, "w"), indent=1)
