#!/usr/bin/env python3
"""Fresh LOCKED held-out set from Rust repos never used in this project (see checks/heldout/).

Same hole pipeline as training data (single-node AST holes + matched character-length random holes, PSM,
<= max_seq_tokens, pairs dropped together); package = repo; at most `--per_repo` pairs per repo (seeded).
Also a left-to-right file list (at most `--l2r_files_per_repo` files per repo). The output directory is made
read-only and must only be scored once, in the final run (`fimexp.score --final_heldout`).
Usage: python -m fimexp.build_heldout --version heldout-v1
"""

import argparse, collections, hashlib, json, random, sys, time
from pathlib import Path

from . import sources
from .build_data import code_revision, make_example, quantiles
from .holes import SourceText, Windower, line_token_counts, n_examples_for, parser, sample_file
from .model import load_tokenizer
from .psm import PSM

sys.path.insert(0, str(sources.ROOT / "checks" / "heldout"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--per_repo", type=int, default=100)
    ap.add_argument("--l2r_files_per_repo", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261103)
    args = ap.parse_args()
    from overlap import heldout_files
    cfg = sources.load_config(); dcfg = cfg["data"]
    out = sources.ROOT / dcfg["output_dir"] / args.version
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    picked_path = sources.ROOT / "checks" / "heldout" / "picked.json"
    overlap_path = sources.ROOT / "checks" / "heldout" / "overlap_report.json"
    picked = json.loads(picked_path.read_text())["picked"]
    for p in picked:  # verify pinned commits
        d = sources.ROOT / "sources" / "heldout" / p["repo"].replace("/", "__")
        if sources.git_head(d) != p["commit"]:
            sys.exit(f"{p['repo']} checkout is not at pinned commit {p['commit']}")
    files, excluded = heldout_files(picked)
    psm = PSM(load_tokenizer(cfg), cfg["format"])
    windower = Windower(psm, dcfg["max_seq_tokens"]); ts = parser(); exempt = set(dcfg["keyword_text_exempt_types"])
    rng = random.Random(args.seed)
    by_repo = collections.defaultdict(list); drops = collections.Counter(); parse_err = []
    for repo, rel, text in files:
        data = text.encode()
        tree = ts.parse(data)
        if tree.root_node.has_error:
            parse_err.append(f"{repo}/{rel}"); continue
        src = SourceText(text); line_tok = None
        f = {"path": f"{repo}/{rel}", "crate": repo}
        for p in sample_file(f["path"], text, tree, n_examples_for(len(data), dcfg["bytes_per_example"]), dcfg["seed"], exempt):
            mids = {a: text[p[a][0]:p[a][1]] for a in ("ast", "random")}
            mtok = {a: len(psm.encode(m)) for a, m in mids.items()}
            if max(windower.fixed_tokens(m) for m in mids.values()) > dcfg["max_seq_tokens"]:
                drops["pair: middle alone exceeds max_seq_tokens"] += 1; continue
            line_tok = line_tok or line_token_counts(src, psm)
            wins = {a: windower.window(src, line_tok, *p[a], mtok[a]) for a in ("ast", "random")}
            if any(w is None for w in wins.values()):
                drops["pair: hole lines alone exceed max_seq_tokens"] += 1; continue
            pid = f"{f['path']}#{p['k']}"
            by_repo[repo].append((make_example("ast", pid, "heldout", f, src, p["ast"], wins["ast"], psm, p["node_type"], p["is_comment"]),
                                  make_example("random", pid, "heldout", f, src, p["random"], wins["random"], psm)))
    out.mkdir(parents=True)
    kept = {r: (rng.sample(v, args.per_repo) if len(v) > args.per_repo else v) for r, v in sorted(by_repo.items())}
    for arm, k in (("ast", 0), ("random", 1)):
        with open(out / f"heldout_{arm}.jsonl", "w") as fh:
            for r in sorted(kept):
                for pair in sorted(kept[r], key=lambda x: x[0]["pair_id"]):
                    fh.write(json.dumps(pair[k], ensure_ascii=False) + "\n")
    files_by_repo = collections.defaultdict(list)
    for repo, rel, _ in files:
        files_by_repo[repo].append(rel)
    l2r = {r: sorted(rng.sample(sorted(v), min(args.l2r_files_per_repo, len(v)))) for r, v in sorted(files_by_repo.items())}
    (out / "l2r_files.json").write_text(json.dumps(l2r, indent=1) + "\n")
    ast_rows = [p[0] for v in kept.values() for p in v]
    manifest = {
        "version": args.version, "LOCKED": "score only in the final run, with fimexp.score --final_heldout",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code": code_revision(),
        "repos": picked, "selection": json.loads(picked_path.read_text())["rule"],
        "overlap_report_sha256": sources.sha256_file(overlap_path),
        "overlap_summary": {k: v for k, v in json.loads(overlap_path.read_text()).items() if k in ("minhash_verified", "files_kept")},
        "file_exclusions": collections.Counter(r for _, _, r in excluded), "parse_error_files": parse_err,
        "dropped_pairs": dict(drops), "seed": args.seed, "per_repo_cap": args.per_repo,
        "pairs_available_by_repo": {r: len(v) for r, v in sorted(by_repo.items())},
        "pairs_kept_by_repo": {r: len(v) for r, v in kept.items()}, "pairs_total": len(ast_rows),
        "comment_share_ast": round(sum(e["is_comment"] for e in ast_rows) / len(ast_rows), 4),
        "middle_tokens_ast": quantiles([e["tokens"]["middle"] for e in ast_rows]),
        "l2r_files": sum(len(v) for v in l2r.values()),
        "versions": {"tokenizer": cfg["model"], "data_settings": {k: dcfg[k] for k in ("seed", "bytes_per_example", "max_seq_tokens")}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=dict) + "\n")
    manifest["file_sha256"] = {p.name: sources.sha256_file(p) for p in sorted(out.iterdir())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=dict) + "\n")
    for p in out.iterdir():
        p.chmod(0o444)
    out.chmod(0o555)
    print(json.dumps({k: manifest[k] for k in ("pairs_total", "pairs_kept_by_repo", "comment_share_ast", "middle_tokens_ast", "dropped_pairs",
                                               "file_exclusions", "parse_error_files", "l2r_files")}, indent=1, default=dict))


if __name__ == "__main__":
    main()
