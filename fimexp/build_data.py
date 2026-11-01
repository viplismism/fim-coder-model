#!/usr/bin/env python3
"""Build matched AST / random FIM datasets from the pinned reth snapshot.

Usage: python -m fimexp.build_data --version v1 [--max_files_per_split N]
Writes data/<version>/ and refuses to overwrite an existing directory.
"""

import argparse, collections, hashlib, json, statistics, subprocess, sys, time
from importlib.metadata import version as pkg_version

from . import sources
from .model import load_tokenizer
from .holes import SourceText, Windower, line_token_counts, n_examples_for, parser, sample_file
from .psm import PSM
from .split import crate_split, package_count_split


def code_revision() -> dict:
    """Git commit plus content hashes of the pipeline code and config (pins uncommitted code too)."""
    run = lambda *a: subprocess.run(["git", "-C", str(sources.ROOT), *a], capture_output=True, text=True).stdout.strip()
    files = sorted((sources.ROOT / "fimexp").glob("*.py")) + [sources.ROOT / "configs" / "experiment.yaml"]
    return {"commit": run("rev-parse", "HEAD"), "dirty_files": run("status", "--porcelain").splitlines(),
            "sha256": {str(f.relative_to(sources.ROOT)): sources.sha256_file(f) for f in files}}


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
    return {"n": len(xs), "min": xs[0], "p10": q(0.1), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1],
            "mean": round(statistics.fmean(xs), 2)}


def make_example(arm, pair_id, split, f, src, hole, win, psm, node_type=None, is_comment=None):
    s, e = hole
    ws, we, total = win
    prefix, middle, suffix = src.text[ws:s], src.text[s:e], src.text[e:we]
    assert prefix + middle + suffix == src.text[ws:we], "window reconstruction failed"
    return {"id": f"{pair_id}:{arm}", "pair_id": pair_id, "arm": arm, "split": split, "path": f["path"], "crate": f["crate"],
            "node_type": node_type, "is_comment": is_comment,
            "hole_chars": [s, e], "hole_bytes": [src.byte_of_char(s), src.byte_of_char(e)], "window_chars": [ws, we],
            "prefix": prefix, "middle": middle, "suffix": suffix,
            "tokens": {"prefix": len(psm.encode(prefix)), "middle": len(psm.encode(middle)),
                       "suffix": len(psm.encode(suffix)), "total_with_eos": total}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max_files_per_split", type=int, default=None, help="small inspection build")
    args = ap.parse_args()

    cfg = sources.load_config(args.config)
    out = sources.ROOT / cfg["data"]["output_dir"] / args.version
    if out.exists():
        sys.exit(f"{out} exists; refusing to overwrite")
    pins = sources.verify_pins(cfg)
    dcfg = cfg["data"]

    files = sources.reth_files(cfg)
    if cfg["split"]["unit"] == "package_count":
        split = package_count_split(files, cfg["split"]["fractions"], cfg["split"]["seed"])
    else:
        split = crate_split(files, cfg["split"]["fractions"], cfg["split"]["seed"], cfg["split"].get("max_heldout_crate_fraction"))
    tok = load_tokenizer(cfg)
    psm = PSM(tok, cfg["format"])
    windower = Windower(psm, dcfg["max_seq_tokens"])
    ts_parser = parser()
    exempt = set(dcfg["keyword_text_exempt_types"])
    reth_root = sources.ROOT / cfg["sources"]["reth"]["path"]

    by_split = collections.defaultdict(list)
    for f in files:
        if f["crate"] is not None:
            by_split[split["crate_to_split"][f["crate"]]].append(f)

    out.mkdir(parents=True)
    writers = {(s, a): open(out / f"{s}_{a}.jsonl", "w") for s in cfg["split"]["fractions"] for a in ("ast", "random")}
    drops, parse_error_files, file_hashes = collections.Counter(), [], {}
    kept = collections.defaultdict(list)
    t0 = time.time()
    for s in cfg["split"]["fractions"]:
        flist = by_split[s][: args.max_files_per_split] if args.max_files_per_split else by_split[s]
        for f in flist:
            data = (reth_root / f["path"]).read_bytes()
            text = data.decode("utf-8")
            file_hashes[f["path"]] = hashlib.sha256(data).hexdigest()
            tree = ts_parser.parse(data)
            if tree.root_node.has_error:
                parse_error_files.append(f["path"])
                continue
            src = SourceText(text)
            line_tok = None
            for p in sample_file(f["path"], text, tree, n_examples_for(f["bytes"], dcfg["bytes_per_example"]), dcfg["seed"], exempt):
                pair_id = f"{f['path']}#{p['k']}"
                mids = {a: text[p[a][0]:p[a][1]] for a in ("ast", "random")}
                mtok = {a: len(psm.encode(m)) for a, m in mids.items()}
                cap = dcfg["max_middle_tokens"]
                if cap is not None and max(mtok.values()) > cap:
                    drops["pair: middle over max_middle_tokens"] += 1
                    continue
                if max(windower.fixed_tokens(m) for m in mids.values()) > dcfg["max_seq_tokens"]:
                    drops["pair: middle alone exceeds max_seq_tokens"] += 1
                    continue
                if line_tok is None:
                    line_tok = line_token_counts(src, psm)
                wins = {a: windower.window(src, line_tok, *p[a], mtok[a]) for a in ("ast", "random")}
                if any(w is None for w in wins.values()):
                    drops["pair: hole lines alone exceed max_seq_tokens"] += 1
                    continue
                ex_ast = make_example("ast", pair_id, s, f, src, p["ast"], wins["ast"], psm, p["node_type"], p["is_comment"])
                ex_rnd = make_example("random", pair_id, s, f, src, p["random"], wins["random"], psm)
                assert len(ex_ast["middle"]) == len(ex_rnd["middle"])
                assert data[ex_ast["hole_bytes"][0]:ex_ast["hole_bytes"][1]].decode("utf-8") == ex_ast["middle"]
                for ex in (ex_ast, ex_rnd):
                    writers[(s, ex["arm"])].write(json.dumps(ex, ensure_ascii=False) + "\n")
                    kept[(s, ex["arm"])].append(ex)
    for w in writers.values():
        w.close()

    stats = {}
    for (s, a), exs in sorted(kept.items()):
        st = {"examples": len(exs),
              "middle_chars": quantiles([len(e["middle"]) for e in exs]),
              "middle_tokens": quantiles([e["tokens"]["middle"] for e in exs]),
              "total_tokens_with_eos": quantiles([e["tokens"]["total_with_eos"] for e in exs])}
        if a == "ast":
            st["comment_only_share"] = sum(e["is_comment"] for e in exs) / len(exs) if exs else None
            st["node_types"] = dict(collections.Counter(e["node_type"] for e in exs).most_common())
        stats[f"{s}/{a}"] = st

    split_record = {**split, "seed": cfg["split"]["seed"], "fractions": cfg["split"]["fractions"],
                    "files_by_split": {s: [f["path"] for f in by_split[s]] for s in by_split}}
    (out / "split.json").write_text(json.dumps(split_record, indent=1) + "\n")
    manifest = {
        "version": args.version, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inspection_build": args.max_files_per_split is not None, "max_files_per_split": args.max_files_per_split,
        "sources": pins, "code": code_revision(), "config": cfg,
        "versions": {p: pkg_version(p) for p in ("tree-sitter", "tree-sitter-rust", "transformers", "tokenizers")},
        "tokenizer": {"model": cfg["model"]["name"], "revision": cfg["model"]["revision"]},
        "seeds": {"split": cfg["split"]["seed"], "data": dcfg["seed"]},
        "split_summary": {k: split[k] for k in ("rule", "bytes", "fraction_of_bytes", "crates", "files", "largest_crate_share", "forced_to_train", "excluded_files") if k in split},
        "filtering": {"parse_error_files": sorted(parse_error_files), "dropped_pairs": dict(drops),
                      "rule": "pairs are kept or dropped together; context trimmed at line boundaries; answers never truncated"},
        "file_sha256": file_hashes, "stats": stats, "seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("split_summary", "filtering", "stats")}, indent=1)[:5000])


if __name__ == "__main__":
    main()
