#!/usr/bin/env python3
"""Build the two-arm dataset (AST single-node holes + matched random holes) for one language from pinned repos.

Same recipe as the Rust experiment (rust-fim-v2): one eligible node per example, probability proportional to byte size;
a random hole of the same character length in the same file; PSM; <= max_seq_tokens incl. BOS/FIM/EOS; context trimmed at
line boundaries; answers never truncated; pairs kept or dropped together; examples per file = ceil(bytes / 2048).
Multi-repo additions: training-pool repos split by repo (seeded) into train / validation; pairs subsampled by
water-filling so no repo dominates (targets: train 7,549 and validation 518 pairs, the sizes of the Rust run);
held-out repos give 100 pairs each in a separate, locked dataset.
Usage: python -m fimexp.build_lang_data --lang python --pins checks/multilang/python_pins.json --version python-fim-v1
"""

import argparse, collections, hashlib, json, math, random, sys, time

from . import sources
from .build_data import code_revision, make_example, quantiles
from .holes import SourceText, Windower, line_token_counts, n_examples_for, parser, sample_file
from .langs import get
from .model import load_tokenizer
from .psm import PSM

GENERATED = ("@generated", "do not edit", "automatically generated", "auto-generated")
SIZE_CLASSES = [(1, 8, "tiny"), (9, 32, "small"), (33, 128, "medium"), (129, 512, "large"), (513, 10**9, "very_large")]


def water_fill(available: dict, target: int) -> dict:
    """Per-repo counts summing to min(target, total available), as equal as possible."""
    take = {r: 0 for r in available}; left = min(target, sum(available.values()))
    open_ = {r for r, n in available.items() if n > 0}
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for r in sorted(open_):
            add = min(share, available[r] - take[r], left)
            take[r] += add; left -= add
            if take[r] >= available[r]:
                open_.discard(r)
            if left == 0:
                break
    return take


def eligible_files(lang, pins, kinds, exclude=frozenset()):
    cfg_l = get(lang); out, excluded = [], collections.Counter()
    for repo, p in sorted(pins["repos"].items()):
        if p["kind"] not in kinds:
            continue
        d = sources.ROOT / p["path"]
        for f in sorted(x for x in d.rglob("*") if x.is_file() and x.suffix in cfg_l["extensions"] and ".git" not in x.parts):
            data = f.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                excluded["not utf-8"] += 1; continue
            if len(data) > 300_000:
                excluded["file > 300 KB"] += 1; continue
            if any(g in text[:400].lower() for g in GENERATED):
                excluded["generated header"] += 1; continue
            if not text.strip():
                excluded["empty"] += 1; continue
            if f"{repo}/{f.relative_to(d).as_posix()}" in exclude:
                excluded["decontamination (checks/multilang/DECONTAMINATION.md)"] += 1; continue
            out.append((repo, f.relative_to(d).as_posix(), text))
    return out, excluded


def build_pairs(lang, files, psm, dcfg, split_of):
    windower = Windower(psm, dcfg["max_seq_tokens"]); ts = parser(lang); exempt = get(lang)["exempt_types"]
    by_repo = collections.defaultdict(list); drops = collections.Counter(); parse_err = []
    for repo, rel, text in files:
        data = text.encode()
        tree = ts.parse(data)
        if tree.root_node.has_error:
            parse_err.append(f"{repo}/{rel}"); continue
        src = SourceText(text); line_tok = None
        f = {"path": f"{repo}/{rel}", "crate": repo}
        for p in sample_file(f["path"], text, tree, n_examples_for(len(data), dcfg["bytes_per_example"]), dcfg["seed"], exempt, lang):
            mids = {a: text[p[a][0]:p[a][1]] for a in ("ast", "random")}
            mtok = {a: len(psm.encode(m)) for a, m in mids.items()}
            if max(windower.fixed_tokens(m) for m in mids.values()) > dcfg["max_seq_tokens"]:
                drops["pair: middle alone exceeds max_seq_tokens"] += 1; continue
            line_tok = line_tok or line_token_counts(src, psm)
            wins = {a: windower.window(src, line_tok, *p[a], mtok[a]) for a in ("ast", "random")}
            if any(w is None for w in wins.values()):
                drops["pair: hole lines alone exceed max_seq_tokens"] += 1; continue
            pid = f"{f['path']}#{p['k']}"
            s = split_of(repo)
            ea = make_example("ast", pid, s, f, src, p["ast"], wins["ast"], psm, p["node_type"], p["is_comment"])
            er = make_example("random", pid, s, f, src, p["random"], wins["random"], psm)
            assert len(ea["middle"]) == len(er["middle"]) and data[ea["hole_bytes"][0]:ea["hole_bytes"][1]].decode() == ea["middle"]
            by_repo[repo].append((ea, er))
    return by_repo, drops, parse_err


def write_split(out, name, pairs):
    for arm, k in (("ast", 0), ("random", 1)):
        with open(out / f"{name}_{arm}.jsonl", "w") as fh:
            for pr in sorted(pairs, key=lambda x: x[0]["pair_id"]):
                fh.write(json.dumps(pr[k], ensure_ascii=False) + "\n")


def stats(pairs):
    a = [p[0] for p in pairs]
    sizes = [e["tokens"]["middle"] for e in a]
    return {"pairs": len(a), "repos": dict(collections.Counter(e["crate"] for e in a)),
            "comment_share": round(sum(bool(e["is_comment"]) for e in a) / max(1, len(a)), 4),
            "middle_tokens": quantiles(sizes), "total_tokens_with_eos": quantiles([e["tokens"]["total_with_eos"] for e in a]),
            "size_classes": {n: sum(lo <= x <= hi for x in sizes) for lo, hi, n in SIZE_CLASSES},
            "node_types_top": dict(collections.Counter(e["node_type"] for e in a).most_common(12))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--pins", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--n_validation_repos", type=int, default=2)
    ap.add_argument("--train_pairs", type=int, default=7549)
    ap.add_argument("--validation_pairs", type=int, default=518)
    ap.add_argument("--heldout_pairs_per_repo", type=int, default=100)
    ap.add_argument("--l2r_files_per_repo", type=int, default=10)
    ap.add_argument("--split_seed", type=int, default=20261101)
    ap.add_argument("--exclude_files", default=None, help="JSON list of '<owner>/<repo>/<file>' removed by decontamination")
    ap.add_argument("--reuse_heldout", default=None, help="existing locked held-out version to keep unchanged (no rebuild)")
    args = ap.parse_args()
    cfg = sources.load_config(); dcfg = cfg["data"]
    out = sources.ROOT / dcfg["output_dir"] / args.version
    hout = sources.ROOT / dcfg["output_dir"] / f"{args.version}-heldout"
    if args.reuse_heldout:
        hout = sources.ROOT / dcfg["output_dir"] / args.reuse_heldout
        if not (hout / "manifest.json").exists():
            sys.exit(f"{hout} not found")
    for d in ((out,) if args.reuse_heldout else (out, hout)):
        if d.exists():
            sys.exit(f"{d} exists; refusing to overwrite")
    exclude = frozenset(json.loads(open(args.exclude_files).read())) if args.exclude_files else frozenset()
    pins = json.loads(open(args.pins).read())
    for repo, p in pins["repos"].items():
        if sources.git_head(sources.ROOT / p["path"]) != p["commit"]:
            sys.exit(f"{repo} is not at its pinned commit")
    pool = sorted(r for r, p in pins["repos"].items() if p["kind"] == "train")
    order = pool[:]; random.Random(args.split_seed).shuffle(order)
    val_repos = sorted(order[:args.n_validation_repos]); train_repos = sorted(order[args.n_validation_repos:])
    split_of = lambda r: "validation" if r in val_repos else "train" if r in train_repos else "heldout"
    psm = PSM(load_tokenizer(cfg), cfg["format"])
    rng = random.Random(args.split_seed)
    t0 = time.time()

    files, excl = eligible_files(args.lang, pins, {"train"}, exclude)
    by_repo, drops, perr = build_pairs(args.lang, files, psm, dcfg, split_of)
    out.mkdir(parents=True)
    split_stats = {}
    for name, repos, target in (("train", train_repos, args.train_pairs), ("validation", val_repos, args.validation_pairs)):
        take = water_fill({r: len(by_repo.get(r, [])) for r in repos}, target)
        chosen = [pr for r in repos for pr in rng.sample(by_repo.get(r, []), take[r])]
        write_split(out, name, chosen)
        split_stats[name] = stats(chosen) | {"available_by_repo": {r: len(by_repo.get(r, [])) for r in repos}, "taken_by_repo": take}
    files_by_split = collections.defaultdict(list)
    for repo, rel, _ in files:
        files_by_split[split_of(repo)].append(f"{repo}/{rel}")
    (out / "split.json").write_text(json.dumps({"rule": "training-pool repos sorted, shuffled once with split_seed; first n are validation",
                                                "split_seed": args.split_seed, "train_repos": train_repos, "validation_repos": val_repos,
                                                "crate_to_split": {r: split_of(r) for r in pool},
                                                "files_by_split": dict(files_by_split)}, indent=1) + "\n")
    manifest = {"version": args.version, "lang": args.lang, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "recipe": "two arms as rust-fim-v2: single-node AST holes + matched random holes", "code": code_revision(),
                "pins": pins, "pins_sha256": sources.sha256_file(args.pins), "data_settings": {k: dcfg[k] for k in ("seed", "bytes_per_example", "max_seq_tokens")},
                "targets": {"train_pairs": args.train_pairs, "validation_pairs": args.validation_pairs},
                "decontamination": {"exclude_files": args.exclude_files, "exclude_sha256": sources.sha256_file(args.exclude_files) if args.exclude_files else None,
                                    "removed": sorted(exclude)},
                "heldout": {"version": args.reuse_heldout or f"{args.version}-heldout", "rebuilt": not args.reuse_heldout},
                "file_exclusions": dict(excl), "parse_error_files": perr, "dropped_pairs": dict(drops), "splits": split_stats,
                "tokenizer": cfg["model"], "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    if args.reuse_heldout:
        print(json.dumps({"train_repos": train_repos, "validation_repos": val_repos, "file_exclusions": dict(excl),
                          "train": {k: split_stats["train"][k] for k in ("pairs", "comment_share", "size_classes")},
                          "validation": {k: split_stats["validation"][k] for k in ("pairs", "comment_share")},
                          "train_available_by_repo": split_stats["train"]["available_by_repo"], "train_taken_by_repo": split_stats["train"]["taken_by_repo"],
                          "heldout_reused_unchanged": args.reuse_heldout}, indent=1))
        return

    hfiles, hexcl = eligible_files(args.lang, pins, {"heldout"})
    hby, hdrops, hperr = build_pairs(args.lang, hfiles, psm, dcfg, split_of)
    hout.mkdir(parents=True)
    hrepos = sorted(r for r, p in pins["repos"].items() if p["kind"] == "heldout")
    hpairs = [pr for r in hrepos for pr in (rng.sample(hby.get(r, []), args.heldout_pairs_per_repo) if len(hby.get(r, [])) > args.heldout_pairs_per_repo else hby.get(r, []))]
    for arm, k in (("ast", 0), ("random", 1)):
        with open(hout / f"heldout_{arm}.jsonl", "w") as fh:
            for pr in sorted(hpairs, key=lambda x: x[0]["pair_id"]):
                fh.write(json.dumps(pr[k], ensure_ascii=False) + "\n")
    hf = collections.defaultdict(list)
    for repo, rel, _ in hfiles:
        hf[repo].append(rel)
    (hout / "l2r_files.json").write_text(json.dumps({r: sorted(rng.sample(sorted(v), min(args.l2r_files_per_repo, len(v)))) for r, v in sorted(hf.items())}, indent=1) + "\n")
    hman = {"version": f"{args.version}-heldout", "LOCKED": "score only once, in the final evaluation", "lang": args.lang,
            "repos": {r: pins["repos"][r] for r in hrepos}, "file_exclusions": dict(hexcl), "parse_error_files": hperr,
            "dropped_pairs": dict(hdrops), "stats": stats(hpairs), "pairs_available_by_repo": {r: len(hby.get(r, [])) for r in hrepos}}
    (hout / "manifest.json").write_text(json.dumps(hman, indent=1, ensure_ascii=False) + "\n")
    hman["file_sha256"] = {p.name: sources.sha256_file(p) for p in sorted(hout.iterdir())}
    (hout / "manifest.json").write_text(json.dumps(hman, indent=1, ensure_ascii=False) + "\n")
    for p in hout.iterdir():
        p.chmod(0o444)
    hout.chmod(0o555)
    print(json.dumps({"train_repos": train_repos, "validation_repos": val_repos, "file_exclusions": dict(excl), "parse_error_files": len(perr),
                      "dropped_pairs": dict(drops), "train": {k: split_stats["train"][k] for k in ("pairs", "comment_share", "size_classes")},
                      "validation": {k: split_stats["validation"][k] for k in ("pairs", "comment_share")},
                      "heldout": {k: hman["stats"][k] for k in ("pairs", "comment_share", "size_classes")}}, indent=1))


if __name__ == "__main__":
    main()
