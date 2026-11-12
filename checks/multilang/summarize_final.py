#!/usr/bin/env python3
"""Summarise one language's FINAL evaluation (base / ast / random), same layout as results/final_eval_summary.json.
Aggregate char perplexity per task = exp(-sum logp / sum chars). Paired bootstrap 95% CIs on perplexity differences,
resampling examples and, separately, groups (benchmark repo / held-out repo). Primary rule (fixed in advance):
AST beats random on Real-FIM-Eval Add only if the ast-random point estimate < 0 AND the repo-level CI upper bound < 0.
Edit is reported but labelled provisional.
Usage: python checks/multilang/summarize_final.py <lang> <base_dir> <ast_dir> <random_dir> <out.json>
"""

import json, math, random, sys
from pathlib import Path

B, SEED = 2000, 20261101
TASKS = ("bench_add", "bench_edit", "heldout_ast", "heldout_random", "heldout_l2r")


def load(d, task):
    p = Path(d) / f"{task}.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return {r["id"]: r for r in rows}


def ppl(rows):
    return math.exp(-sum(r["sum_logprob"] for r in rows) / sum(r["n_chars"] for r in rows))


def main():
    lang, dirs, out = sys.argv[1], dict(zip(("base", "ast", "random"), sys.argv[2:5])), Path(sys.argv[5])
    res = {"lang": lang, "eval_dirs": dirs, "bootstrap_resamples": B, "seed": SEED, "tasks": {}}
    for task in TASKS:
        data = {m: load(d, task) for m, d in dirs.items()}
        ids = sorted(set.intersection(*(set(v) for v in data.values()))) if all(data.values()) else []
        if not ids:
            continue
        grp = lambda i: data["base"][i].get("repo") or data["base"][i].get("crate") or "?"
        groups = sorted({grp(i) for i in ids})
        by_g = {g: [i for i in ids if grp(i) == g] for g in groups}
        P = lambda m, idx: ppl([data[m][i] for i in idx])
        point = {m: P(m, ids) for m in dirs}
        pairs = (("ast", "base"), ("random", "base"), ("ast", "random"))
        rng = random.Random(SEED)
        bs_e = {p: [] for p in pairs}; bs_g = {p: [] for p in pairs}
        for _ in range(B):
            s = [ids[rng.randrange(len(ids))] for _ in ids]
            v = {m: P(m, s) for m in dirs}
            for a, b in pairs:
                bs_e[(a, b)].append(v[a] - v[b])
            gs = [groups[rng.randrange(len(groups))] for _ in groups]
            s = [i for g in gs for i in by_g[g]]
            v = {m: P(m, s) for m in dirs}
            for a, b in pairs:
                bs_g[(a, b)].append(v[a] - v[b])
        ci = lambda xs: [round(sorted(xs)[int(0.025 * B)], 4), round(sorted(xs)[int(0.975 * B) - 1], 4)]
        per_group = {g: {**{m: round(P(m, by_g[g]), 4) for m in dirs}, "n": len(by_g[g])}
                     for g in sorted(groups, key=lambda g: -len(by_g[g]))}
        t = {"n": len(ids), "groups": len(groups), "ppl": {m: round(x, 4) for m, x in point.items()},
             "less_uncertainty_vs_base": {m: round(1 - math.log(point[m]) / math.log(point["base"]), 4) for m in ("ast", "random")},
             "diff": {f"{a}-{b}": {"value": round(point[a] - point[b], 4), "ci_examples": ci(bs_e[(a, b)]),
                                   "ci_groups": ci(bs_g[(a, b)])} for a, b in pairs},
             "ast_better_than_random_groups": f"{sum(per_group[g]['ast'] < per_group[g]['random'] for g in groups)}/{len(groups)}",
             "per_group": per_group}
        if task == "bench_edit":
            t["note"] = "provisional"
        res["tasks"][task] = t
    add = res["tasks"].get("bench_add")
    if add:
        d = add["diff"]["ast-random"]
        res["primary"] = {"rule": "AST beats random iff Add ast-random < 0 and repo-level CI upper bound < 0",
                          "ast_minus_random": d["value"], "ci_groups": d["ci_groups"],
                          "ast_beats_random": d["value"] < 0 and d["ci_groups"][1] < 0}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "ppl", "diff")} for k, v in res["tasks"].items()}, indent=1))
    print(json.dumps(res.get("primary"), indent=1))


if __name__ == "__main__":
    main()
