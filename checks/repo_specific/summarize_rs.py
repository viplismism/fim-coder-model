#!/usr/bin/env python3
"""Summary of the repo-specific comparison (telekom/controlplane, cutoff 2026-05-01).
Metrics: pass@1 = build+tests pass rate (greedy, one completion); exact match; character perplexity of the real code
(teacher-forced; Qwen-14B base and fine-tuned only). Primary: build+tests pass rate on the sensitive examples (632 after the common-context rule). Secondary: exact match
on all common examples (891). Both split by source vs test files. Paired bootstrap 95% CIs resampling COMMITS
(examples from one commit move together), B=2000, seed fixed. Rule fixed in advance: the fine-tuned model beats
another model iff the difference (fine-tuned - other) > 0 and the CI lower bound > 0.
Usage: python summarize_rs.py <scores_dir> <out.json> <finetuned_name> <other names...>
"""
import collections, json, random, sys
from pathlib import Path

B, SEED = 2000, 20261118


def main():
    d, out, ft, others = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
    names = [ft] + others
    S = {n: {r["id"]: r for r in map(json.loads, open(d / f"{n}.scores.jsonl"))} for n in names if (d / f"{n}.scores.jsonl").exists()}
    C = {n: {c["id"]: c for c in map(json.loads, open(d / f"{n}.completions.jsonl"))} for n in S}
    ids = sorted(set.intersection(*(set(v) for v in S.values())))
    commit = lambda i: i.split(":")[0]
    res = {"models": list(S), "examples_common": len(ids), "bootstrap": {"resample": "commits", "B": B, "seed": SEED}, "metrics": {}}
    for metric, sel in (("build_test_pass", lambda r: r.get("sensitive")), ("exact_match", lambda r: True)):
        for part, pf in (("all", lambda r: True), ("source", lambda r: not r["is_test_file"]), ("test", lambda r: r["is_test_file"])):
            ex = [i for i in ids if sel(S[ft][i]) and pf(S[ft][i]) and not any(S[n][i].get("skipped") for n in S)]
            by_c = collections.defaultdict(list)
            for i in ex:
                by_c[commit(i)].append(i)
            cs = sorted(by_c)
            val = lambda n, idx: sum(bool(S[n][i].get(metric)) for i in idx) / max(1, len(idx))
            m = {"n": len(ex), "commits": len(cs), "rate": {n: round(val(n, ex), 4) for n in S}}
            if ft in S:
                rng = random.Random(SEED); diffs = {o: [] for o in S if o != ft}
                for _ in range(B):
                    smp = [i for c in (cs[rng.randrange(len(cs))] for _ in cs) for i in by_c[c]]
                    for o in diffs:
                        diffs[o].append(val(ft, smp) - val(o, smp))
                ci = lambda xs: [round(sorted(xs)[int(0.025 * B)], 4), round(sorted(xs)[int(0.975 * B) - 1], 4)]
                m["finetuned_minus"] = {o: {"value": round(val(ft, ex) - val(o, ex), 4), "ci": ci(v),
                                            "finetuned_beats": (val(ft, ex) - val(o, ex)) > 0 and ci(v)[0] > 0} for o, v in diffs.items()}
            res["metrics"][f"{metric}/{part}"] = m
    # character perplexity of the real added code (teacher-forced), where available (Qwen-14B base and fine-tuned)
    PP = {n: {r["id"]: r for r in map(json.loads, open(d / f"{n}.ppl.jsonl")) if "sum_logprob" in r}
          for n in names if (d / f"{n}.ppl.jsonl").exists()}
    if PP:
        import math
        pids = sorted(set(ids) & set.intersection(*(set(v) for v in PP.values())))
        cp = lambda n, idx: math.exp(-sum(PP[n][i]["sum_logprob"] for i in idx) / sum(PP[n][i]["n_chars"] for i in idx))
        for part, pf in (("all", lambda i: True), ("source", lambda i: not S[ft][i]["is_test_file"]), ("test", lambda i: S[ft][i]["is_test_file"])):
            ex = [i for i in pids if pf(i)]
            by_c = collections.defaultdict(list)
            for i in ex:
                by_c[commit(i)].append(i)
            cs = sorted(by_c)
            m = {"n": len(ex), "commits": len(cs), "char_ppl": {n: round(cp(n, ex), 4) for n in PP}}
            if ft in PP:
                rng = random.Random(SEED); diffs = {o: [] for o in PP if o != ft}
                for _ in range(B):
                    smp = [i for c in (cs[rng.randrange(len(cs))] for _ in cs) for i in by_c[c]]
                    for o in diffs:
                        diffs[o].append(cp(ft, smp) - cp(o, smp))
                ci = lambda xs: [round(sorted(xs)[int(0.025 * B)], 4), round(sorted(xs)[int(0.975 * B) - 1], 4)]
                m["finetuned_minus"] = {o: {"value": round(cp(ft, ex) - cp(o, ex), 4), "ci": ci(v),
                                            "less_uncertainty": round(1 - math.log(cp(ft, ex)) / math.log(cp(o, ex)), 4),
                                            "finetuned_better": (cp(ft, ex) - cp(o, ex)) < 0 and ci(v)[1] < 0} for o, v in diffs.items()}
            res["metrics"][f"char_ppl/{part}"] = m
    res["generation"] = {n: {"mean_seconds": round(sum(c.get("seconds", 0) for c in C[n].values()) / max(1, len(C[n])), 2),
                             "hit_256_token_limit": sum(1 for c in C[n].values() if c.get("completion") is not None and not c.get("stopped")),
                             "skipped_over_context": sum(1 for c in C[n].values() if "skipped_over_context" in c)} for n in S}
    out.write_text(json.dumps(res, indent=1) + "\n")
    for k, m in res["metrics"].items():
        print(k, m["n"], m.get("rate", m.get("char_ppl")), {o: (x["value"], x["ci"]) for o, x in m.get("finetuned_minus", {}).items()})


if __name__ == "__main__":
    main()
