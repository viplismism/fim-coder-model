#!/usr/bin/env python3
"""Diagnosis of AST-trained vs random-trained on VALIDATION scores only (test/benchmark untouched).

Buckets: hole size (answer tokens), hole type (code vs comment-only), boundary type (mid-line start/end,
syntax-node-aligned start/end). Per bucket: count, char perplexity of base/AST/random, AST-random with a
paired bootstrap 95% CI (resampling examples). Also boundary statistics of the TRAINING holes per arm.
Usage: python checks/diagnosis/diagnose.py
"""

import collections, glob, json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fimexp import sources  # noqa: E402
from fimexp.holes import COMMENT_TYPES, SourceText, parser  # noqa: E402

DATA = ROOT / "data" / "rust-fim-v2"
SIZE_BUCKETS = [(1, 8), (9, 32), (33, 128), (129, 512), (513, 10**9)]
B = 2000


def size_bucket(n):
    for lo, hi in SIZE_BUCKETS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"


class FileInfo:
    """Node start/end byte sets and comment ranges for one source file."""

    def __init__(self, text):
        self.src = SourceText(text)
        tree = parser().parse(self.src.data)
        self.starts, self.ends, self.comments = set(), set(), []
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n.parent is not None and n.is_named:
                self.starts.add(n.start_byte); self.ends.add(n.end_byte)
                if n.type in COMMENT_TYPES:
                    self.comments.append((n.start_byte, n.end_byte))
            stack.extend(n.children)

    def features(self, hs, he):
        """hs/he are char offsets of the hole."""
        t = self.src.text
        mid = t[hs:he]
        ls = t.rfind("\n", 0, hs) + 1
        le = t.find("\n", he); le = len(t) if le == -1 else le
        start_midline = t[ls:hs].strip() != ""
        end_midline = t[he:le].strip() != ""
        lead = len(mid) - len(mid.lstrip()); trail = len(mid) - len(mid.rstrip())
        bs = self.src.byte_of_char(hs + lead); be = self.src.byte_of_char(he - trail)
        start_node = bs in self.starts
        end_node = be in self.ends
        nonws = [self.src.byte_of_char(i) for i in range(hs, he) if not t[i].isspace()]
        comment_only = bool(nonws) and all(any(a <= b < z for a, z in self.comments) for b in nonws)
        return {"start_midline": start_midline, "end_midline": end_midline, "start_node": start_node,
                "end_node": end_node, "comment_only": comment_only}


def line_class(f):
    s, e = f["start_midline"], f["end_midline"]
    return "whole lines" if not s and not e else "starts mid-line" if s and not e else "ends mid-line" if e and not s else "both mid-line"


def node_class(f):
    s, e = f["start_node"], f["end_node"]
    return "both ends node-aligned" if s and e else "one end mid-node" if s or e else "both ends mid-node"


def ppl(lp, ch, idx):
    return math.exp(-sum(lp[i] for i in idx) / sum(ch[i] for i in idx))


def bucket_stats(idx, lp, ch, rng):
    out = {"n": len(idx)}
    for m in ("base", "ast", "random"):
        out[m] = round(ppl(lp[m], ch, idx), 4)
    d = ppl(lp["ast"], ch, idx) - ppl(lp["random"], ch, idx)
    boots = []
    for _ in range(B):
        s = [idx[rng.randrange(len(idx))] for _ in idx]
        boots.append(ppl(lp["ast"], ch, s) - ppl(lp["random"], ch, s))
    boots.sort()
    out["ast_minus_random"] = round(d, 4)
    out["ci95"] = [round(boots[int(0.025 * B)], 4), round(boots[int(0.975 * B) - 1], 4)]
    out["verdict"] = "AST better" if out["ci95"][1] < 0 else "random better" if out["ci95"][0] > 0 else "no clear difference"
    return out


def main():
    cfg = sources.load_config()
    reth = ROOT / cfg["sources"]["reth"]["path"]
    evals = {m: sorted(glob.glob(str(ROOT / f"logs/*_{m}_eval_validation")))[-1] for m in ("base", "ast", "random")}
    files = {}
    def info(path):
        if path not in files:
            files[path] = FileInfo(sources.read_source(reth / path))
        return files[path]

    rng = random.Random(0)
    report = {"scope": "validation split only (518 pairs per hole type); test and benchmark not used",
              "eval_dirs": evals, "bootstrap_resamples": B, "sets": {}}
    for task, arm in (("reth_ast", "ast"), ("reth_random", "random")):
        rows = {r["id"]: r for r in map(json.loads, open(DATA / f"validation_{arm}.jsonl"))}
        sc = {m: {r["id"]: r for r in map(json.loads, open(Path(evals[m]) / f"{task}.jsonl"))} for m in evals}
        ids = sorted(rows)
        lp = {m: [sc[m][i]["sum_logprob"] for i in ids] for m in evals}
        ch = [sc["base"][i]["n_chars"] for i in ids]
        feats = []
        for i in ids:
            r = rows[i]
            f = info(r["path"]).features(*r["hole_chars"])
            f["size"] = size_bucket(r["tokens"]["middle"])
            f["type"] = "comment" if (r["is_comment"] if arm == "ast" else f["comment_only"]) else "code"
            feats.append(f)
        res = {"overall": bucket_stats(list(range(len(ids))), lp, ch, rng)}
        for name, key in (("hole size (answer tokens)", lambda f: f["size"]), ("hole type", lambda f: f["type"]),
                          ("line boundaries", line_class), ("node boundaries", node_class)):
            groups = collections.defaultdict(list)
            for k, f in enumerate(feats):
                groups[key(f)].append(k)
            order = [f"{lo}-{hi}" if hi < 10**9 else f"{lo}+" for lo, hi in SIZE_BUCKETS] if "size" in name else sorted(groups)
            res[name] = {g: bucket_stats(groups[g], lp, ch, rng) for g in order if g in groups}
        report["sets"][task] = res

    # Training-hole boundary statistics per arm (what each model was trained on)
    train = {}
    for arm in ("ast", "random"):
        c = collections.Counter(); n = 0
        for line in open(DATA / f"train_{arm}.jsonl"):
            r = json.loads(line)
            f = info(r["path"]).features(*r["hole_chars"])
            c[line_class(f)] += 1; c[node_class(f)] += 1; c["comment" if (r["is_comment"] if arm == "ast" else f["comment_only"]) else "code"] += 1
            n += 1
        train[arm] = {k: round(v / n, 3) for k, v in sorted(c.items())} | {"n": n}
    report["training_holes"] = train
    out = Path(__file__).parent / "report.json"
    out.write_text(json.dumps(report, indent=1) + "\n")

    for task, res in report["sets"].items():
        print(f"\n=== {task}  overall: n={res['overall']['n']} base {res['overall']['base']} ast {res['overall']['ast']} random {res['overall']['random']} "
              f"AST-random {res['overall']['ast_minus_random']:+.4f} {res['overall']['ci95']}")
        for name, groups in res.items():
            if name == "overall":
                continue
            print(f"  {name}:")
            for g, s in groups.items():
                print(f"    {g:24s} n={s['n']:4d}  base {s['base']:.4f}  ast {s['ast']:.4f}  random {s['random']:.4f}  "
                      f"AST-random {s['ast_minus_random']:+.4f} [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  {s['verdict']}")
    print("\n=== training holes (share of examples)")
    for arm, s in train.items():
        print(f"  {arm}: {s}")


if __name__ == "__main__":
    main()
