#!/usr/bin/env python3
"""Screening datasets for arms R, S, A, M (checks/screening/PLAN.md) from the rust-fim-v2 split.

Per slot (train file x example index), common random numbers decide each arm's hole:
  S: 90% single-node, 10% random (length of the slot's single-node hole)
  A: 90% aligned-span, 10% random (length of the slot's aligned-span hole)
  M: 45% single-node, 45% aligned-span, 10% random (length of the single-node or aligned hole, 50/50)
  R: random hole with the same file and character length as the slot's M hole (shared control)
A slot is kept only if every arm's example fits the context rule (answers never truncated). Plus 25% left-to-right
windows (identical across arms). Validation sets: aligned-span holes from the validation files (new, fixed);
the existing rust-fim-v2 validation AST/random holes are reused as they are.
Usage: python -m fimexp.build_variants --version screen-v1
"""

import argparse, collections, json, math, random, sys, time

from . import sources
from .build_data import code_revision, make_example, quantiles
from .holes import SourceText, Windower, aligned_span, eligible_nodes, line_token_counts, n_examples_for, parser
from .model import load_tokenizer
from .psm import PSM
from .score import l2r_windows

ARMS = ("R", "S", "A", "M")
SIZE_CLASSES = [(1, 8, "tiny"), (9, 32, "small"), (33, 128, "medium"), (129, 512, "large"), (513, 10**9, "very_large")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--base_version", default="rust-fim-v2")
    ap.add_argument("--l2r_share", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--l2r_window_tokens", type=int, default=None, help="left-to-right window size (default: max_seq_tokens)")
    ap.add_argument("--aligned_mode", choices=["matched", "windowed", "literal"], default="matched",
                    help="windowed: random center + random length <= aligned_max_tokens (adapted); literal: endpoints anywhere in the file")
    ap.add_argument("--aligned_max_tokens", type=int, default=3000)
    args = ap.parse_args()
    cfg = sources.load_config(); dcfg = cfg["data"]
    out = sources.ROOT / dcfg["output_dir"] / args.version
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    pins = sources.verify_pins(cfg)
    base = sources.ROOT / dcfg["output_dir"] / args.base_version
    split = json.loads((base / "split.json").read_text())
    reth = sources.ROOT / cfg["sources"]["reth"]["path"]
    psm = PSM(load_tokenizer(cfg), cfg["format"]); win = Windower(psm, dcfg["max_seq_tokens"]); ts = parser()
    exempt = set(dcfg["keyword_text_exempt_types"]); crate_of = split["crate_to_split"]
    c = t = 0
    for line in open(base / "train_ast.jsonl"):
        r = json.loads(line); c += len(r["middle"]); t += r["tokens"]["middle"]
    chars_per_token = c / t
    max_span_chars = round(args.aligned_max_tokens * chars_per_token) if args.aligned_mode == "windowed" else None
    l2r_tokens = args.l2r_window_tokens or dcfg["max_seq_tokens"]
    out.mkdir(parents=True)
    w = {a: open(out / f"train_{a}.jsonl", "w") for a in ARMS}
    w_val = open(out / "validation_aligned.jsonl", "w")
    drops, kinds, kept_slots = collections.Counter(), {a: collections.Counter() for a in ARMS}, 0
    sizes = collections.defaultdict(list)  # (arm, hole kind) -> answer tokens
    slots_tried = collections.Counter()
    l2r_pool, val_n = [], 0

    def crate_for(path):
        return max((c for c in crate_of if path.startswith(c + "/")), key=len)

    for split_name in ("train", "validation"):
        for path in split["files_by_split"][split_name]:
            text = sources.read_source(reth / path); data = text.encode()
            tree = ts.parse(data)
            if tree.root_node.has_error:
                drops[f"{split_name}: parse-error file"] += 1; continue
            src = SourceText(text); nodes = eligible_nodes(tree, src, exempt)
            if not nodes:
                continue
            f = {"path": path, "crate": crate_for(path)}
            line_tok = None
            for k in range(n_examples_for(len(data), dcfg["bytes_per_example"])):
                slots_tried[split_name] += 1
                rng = random.Random(f"{args.seed}:{path}:{k}")
                single = rng.choices(nodes, weights=[x["end_byte"] - x["start_byte"] for x in nodes])[0]
                sh = (src.char_of_byte(single["start_byte"]), src.char_of_byte(single["end_byte"]), single["type"], "single", single["is_comment"])
                target = (sh[1] - sh[0]) if args.aligned_mode == "matched" else None
                al = aligned_span(tree, src, random.Random(f"{args.seed}:{path}:{k}:aligned"), exempt,
                                  max_span_chars=max_span_chars, target_chars=target)
                if al is None:
                    drops[f"{split_name}: no aligned span"] += 1; continue
                if split_name == "validation":
                    holes = {"V": (al[0], al[1], al[2], "aligned", None)}
                else:
                    u, u2 = rng.random(), rng.random()
                    rand_start = lambda L: rng.randint(0, len(text) - L)
                    def rnd(length):
                        s0 = rand_start(length); return (s0, s0 + length, None, "random", None)
                    alh = (al[0], al[1], al[2], "aligned", None)
                    S = sh if u < 0.9 else rnd(sh[1] - sh[0])
                    A = alh if u < 0.9 else rnd(alh[1] - alh[0])
                    M = sh if u < 0.45 else alh if u < 0.9 else rnd((sh if u2 < 0.5 else alh)[1] - (sh if u2 < 0.5 else alh)[0])
                    R = rnd(M[1] - M[0])
                    holes = {"R": R, "S": S, "A": A, "M": M}
                line_tok = line_tok or line_token_counts(src, psm)
                built, ok = {}, True
                for arm, (s, e, t, kind, isc) in holes.items():
                    mt = len(psm.encode(text[s:e]))
                    wnd = win.window(src, line_tok, s, e, mt) if win.fixed_tokens(text[s:e]) <= dcfg["max_seq_tokens"] else None
                    if wnd is None:
                        ok = False; break
                    ex = make_example(arm, f"{path}#{k}", split_name, f, src, (s, e), wnd, psm, t, isc)
                    ex["hole_kind"] = kind
                    built[arm] = ex
                if not ok:
                    drops[f"{split_name}: slot dropped (an arm's answer does not fit)"] += 1; continue
                if split_name == "validation":
                    w_val.write(json.dumps(built["V"], ensure_ascii=False) + "\n"); val_n += 1
                    sizes[("validation", "aligned")].append(built["V"]["tokens"]["middle"])
                else:
                    kept_slots += 1
                    for arm in ARMS:
                        kinds[arm][built[arm]["hole_kind"]] += 1
                        sizes[(arm, built[arm]["hole_kind"])].append(built[arm]["tokens"]["middle"])
                        w[arm].write(json.dumps(built[arm], ensure_ascii=False) + "\n")
            if split_name == "train":
                for i, t in enumerate(l2r_windows(src, psm, l2r_tokens)[0]):
                    l2r_pool.append({"id": f"{path}#l2r{i}", "kind": "l2r", "path": path, "crate": f["crate"], "split": "train", "text": t})
    n_l2r = round(kept_slots * args.l2r_share / (1 - args.l2r_share))
    l2r = random.Random(args.seed).sample(l2r_pool, min(n_l2r, len(l2r_pool)))
    for r in l2r:
        r["tokens"] = {"total_with_eos": len(psm.build_l2r(r["text"])["input_ids"])}
    for arm in ARMS:
        for r in l2r:
            w[arm].write(json.dumps({**r, "arm": arm}, ensure_ascii=False) + "\n")
        w[arm].close()
    w_val.close()
    manifest = {"version": args.version, "base_version": args.base_version, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plan": "checks/screening/PLAN.md", "sources": pins, "code": code_revision(), "seed": args.seed,
                "l2r_window_tokens": l2r_tokens, "l2r_pool": len(l2r_pool),
                "aligned_span": {"mode": args.aligned_mode, "max_tokens": args.aligned_max_tokens if max_span_chars else None,
                                 "chars_per_token": round(chars_per_token, 4), "max_span_chars": max_span_chars,
                                 "rule": {"matched": "length-matched: center ~ U[0,n], target length = the same slot's single-node hole length (chars), "
                                                     "then snapped to the best-IoU run of direct named children of the smallest containing node",
                                          "windowed": "windowed: center ~ U[0,n], length ~ U[1, max_span_chars], then snapped as above",
                                          "literal": "paper-literal: endpoints ~ U over the whole file, then snapped as above"}[args.aligned_mode]},
                "slots_tried": dict(slots_tried),
                "slot_drop_rate": {k: round(sum(v for d, v in drops.items() if d.startswith(k) and "slot dropped" in d) / slots_tried[k], 4) for k in slots_tried},
                "answer_tokens": {f"{a}/{k}": quantiles(v) for (a, k), v in sorted(sizes.items())},
                "size_classes": {f"{a}/{k}": {name: sum(lo <= x <= hi for x in v) for lo, hi, name in SIZE_CLASSES} for (a, k), v in sorted(sizes.items())},
                "fim_slots_per_arm": kept_slots, "l2r_examples_per_arm": len(l2r), "l2r_share": round(len(l2r) / (kept_slots + len(l2r)), 4),
                "hole_kinds": {a: dict(kinds[a]) for a in ARMS}, "validation_aligned": val_n, "drops": dict(drops),
                "split_sha256": sources.sha256_file(base / "split.json")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("aligned_span", "slots_tried", "slot_drop_rate", "fim_slots_per_arm", "l2r_examples_per_arm",
                                               "l2r_share", "hole_kinds", "validation_aligned", "drops")}, indent=1))
    for k, sc in manifest["size_classes"].items():
        tot = sum(sc.values()) or 1
        print(f"  size classes {k:22s} " + "  ".join(f"{n} {100*v/tot:4.1f}%" for n, v in sc.items()))
    for k, q in manifest["answer_tokens"].items():
        print(f"  answer tokens {k:22s} n={q['n']:5d} p10 {q['p10']:5d} p50 {q['p50']:5d} p90 {q['p90']:5d} max {q['max']:5d}")


if __name__ == "__main__":
    main()
