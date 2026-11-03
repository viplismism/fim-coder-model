#!/usr/bin/env python3
"""Overlap check for the fresh held-out repos: vs all of reth (training data source), vs Real-FIM-Eval
(all Rust examples), and across held-out repos. Same method as checks/neardup (datatrove MinHash on files +
30-line chunks, exact verification, Jaccard >= 0.5 inspected) plus exact 5-line windows.
Usage: python checks/heldout/overlap.py <picked.json> <out_dir>
"""

import collections, hashlib, json, shutil, struct, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "neardup"))
from neardup import RustWordTokenizer, build_docs, chunks  # noqa: E402
from datatrove.executor import LocalPipelineExecutor  # noqa: E402
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupSignature  # noqa: E402
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.utils.hashing import HashConfig  # noqa: E402
from datatrove.utils.text import TextNormConfig  # noqa: E402
from fimexp import sources  # noqa: E402

GENERATED = ("@generated", "do not edit", "automatically generated", "auto-generated")
MAX_FILE_BYTES = 300_000


def heldout_files(picked):
    """(repo, relpath, text) for every eligible .rs file; returns also the excluded list with reasons."""
    out, excluded = [], []
    for p in picked:
        d = ROOT / "sources" / "heldout" / p["repo"].replace("/", "__")
        for f in sorted(x for x in d.rglob("*.rs") if ".git" not in x.parts):
            rel = f.relative_to(d).as_posix()
            data = f.read_bytes()
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                excluded.append((p["repo"], rel, "not utf-8")); continue
            if len(data) > MAX_FILE_BYTES:
                excluded.append((p["repo"], rel, "file > 300 KB")); continue
            if any(g in text[:400].lower() for g in GENERATED):
                excluded.append((p["repo"], rel, "generated header")); continue
            out.append((p["repo"], rel, text))
    return out, excluded


def win_set(text):
    rows = [l.strip() for l in text.split("\n") if l.strip()]
    return {hashlib.md5("\n".join(rows[i:i + 5]).encode()).hexdigest() for i in range(len(rows) - 4)
            if sum(len(x) for x in rows[i:i + 5]) >= 150}


def main():
    picked = json.loads(Path(sys.argv[1]).read_text())["picked"]
    out_dir = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    st = json.loads((ROOT / "checks" / "neardup" / "settings.json").read_text())
    cfg = sources.load_config()
    pins = sources.verify_pins(cfg)
    split_map = {c: "reth" for c in sources.reth_crates(cfg)}
    reth, bench = build_docs(cfg, split_map, st)
    files, excluded = heldout_files(picked)
    held = []
    for repo, rel, text in files:
        meta = {"source": "heldout", "repo": repo, "path": rel}
        held.append({"text": text, "metadata": {**meta, "unit": "file"}})
        for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):
            held.append({"text": t, "metadata": {**meta, "unit": "chunk", "lines": [a + 1, b + 1]}})

    mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                       hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
    tok = RustWordTokenizer()
    sig = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tok)
    min_tokens = st["min_doc_ngrams"] + st["n_grams"] - 1
    groups = [[d for d in g if len(tok.word_tokenize(d["text"])) >= min_tokens] for g in (reth, bench, held)]
    work = Path(tempfile.mkdtemp(prefix="heldout_overlap_"))
    try:
        (work / "input").mkdir()
        for k, docs in enumerate(groups):
            with open(work / "input" / f"{k:05d}.jsonl", "w") as f:
                for i, d in enumerate(docs):
                    f.write(json.dumps({"id": f"{k}-{i}", "text": d["text"], "metadata": {}}) + "\n")
        LocalPipelineExecutor([JsonlReader(str(work / "input")), MinhashDedupSignature(output_folder=str(work / "sigs"), config=mh, language=tok)],
                              tasks=3, logging_dir=str(work / "l1")).run()
        LocalPipelineExecutor([MinhashDedupBuckets(input_folder=str(work / "sigs"), output_folder=str(work / "b"), config=mh)],
                              tasks=mh.num_buckets, logging_dir=str(work / "l2")).run()
        LocalPipelineExecutor([MinhashDedupCluster(input_folder=str(work / "b"), output_folder=str(work / "c"), config=mh, save_cluster_id=True)],
                              tasks=1, logging_dir=str(work / "l3")).run()
        clusters = collections.defaultdict(list)
        for k, docs in enumerate(groups):
            p = work / "c" / f"{k:06d}.clusters"
            if p.exists():
                for doc, cid in struct.iter_unpack("<2I", p.read_bytes()):
                    clusters[cid].append(docs[doc])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    cache = {}
    def sh(d):
        return cache.setdefault(id(d), set(sig.get_shingles(d["text"]).ravel().tolist()))
    def jac(a, b):
        A, B = sh(a), sh(b)
        return len(A & B) / max(1, len(A | B))
    brief = lambda d: {k: d["metadata"].get(k) for k in ("source", "repo", "path", "unit", "lines", "refs") if d["metadata"].get(k)} | {"text_head": d["text"][:400]}
    hits = {"vs_reth": [], "vs_benchmark": [], "across_heldout_repos": []}
    for docs in clusters.values():
        H = [d for d in docs if d["metadata"]["source"] == "heldout"]
        for h in H:
            for key, others in (("vs_reth", [d for d in docs if d["metadata"]["source"] == "reth"]),
                                ("vs_benchmark", [d for d in docs if d["metadata"]["source"] == "bench"]),
                                ("across_heldout_repos", [d for d in H if d["metadata"]["repo"] != h["metadata"]["repo"]])):
                if others:
                    best = max(others, key=lambda o: jac(h, o)); j = jac(h, best)
                    if j >= st["verify_jaccard"]:
                        hits[key].append({"jaccard": round(j, 3), "heldout": brief(h), "other": brief(best)})
    # exact 5-line windows shared with reth / benchmark
    reth_w = set().union(*(win_set(d["text"]) for d in reth if d["metadata"]["unit"] == "file"))
    bench_w = set().union(*(win_set(d["text"]) for d in bench if d["metadata"]["unit"] in ("post_file", "pre_file")))
    per_repo = collections.defaultdict(lambda: {"windows": 0, "in_reth": 0, "in_benchmark": 0, "files": 0})
    for repo, rel, text in files:
        w = win_set(text); r = per_repo[repo]
        r["files"] += 1; r["windows"] += len(w); r["in_reth"] += len(w & reth_w); r["in_benchmark"] += len(w & bench_w)
    report = {"inputs": pins, "heldout_repos": picked, "file_rules": {"max_file_bytes": MAX_FILE_BYTES, "generated_markers": GENERATED},
              "files_kept": len(files), "files_excluded": excluded,
              "docs": {"reth": len(groups[0]), "benchmark": len(groups[1]), "heldout": len(groups[2])},
              "minhash_verified": {k: len(v) for k, v in hits.items()}, "verified_items": hits,
              "exact_5line_windows": dict(per_repo)}
    (out_dir / "overlap_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("files_kept", "docs", "minhash_verified")}, indent=1))
    print("excluded files:", collections.Counter(r for _, _, r in excluded))
    for repo, r in sorted(per_repo.items()):
        print(f"  {repo:36s} files {r['files']:4d} windows {r['windows']:6d} in reth {r['in_reth']:3d} in benchmark {r['in_benchmark']:3d}")
    for k, v in hits.items():
        for h in sorted(v, key=lambda x: -x["jaccard"])[:6]:
            print(f"  [{k}] J={h['jaccard']} {h['heldout']['repo']}:{h['heldout']['path']} ~ {h['other'].get('repo') or h['other'].get('source')}:{h['other'].get('path')}")


if __name__ == "__main__":
    main()
