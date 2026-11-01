#!/usr/bin/env python3
"""Near-duplicate check (datatrove MinHash) between the pinned reth snapshot and Real-FIM-Eval Rust,
and across the reth train/validation/test splits.

Pipeline: build documents -> datatrove MinhashDedupSignature -> MinhashDedupBuckets -> MinhashDedupCluster
(cluster ids) -> exact verification of every mixed cluster with the same shingles -> report.
Settings live in settings.json; output goes to report.json / candidates.md in this folder.
Usage: python checks/neardup/neardup.py --split_json data/<version>/split.json
"""

import argparse, collections, hashlib, json, re, shutil, struct, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datatrove.data import Document  # noqa: E402
from datatrove.executor import LocalPipelineExecutor  # noqa: E402
from datatrove.pipeline.dedup.minhash import (  # noqa: E402
    MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupSignature)
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.utils.hashing import HashConfig  # noqa: E402
from datatrove.utils.text import TextNormConfig  # noqa: E402
from datatrove.utils.word_tokenizers import WordTokenizer  # noqa: E402

from fimexp import sources  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


class RustWordTokenizer(WordTokenizer):
    """Identifiers, numbers and single punctuation characters; whitespace dropped."""

    def word_tokenize(self, text):
        return TOKEN_RE.findall(text)

    def sent_tokenize(self, text):
        return text.split("\n")

    def span_tokenize(self, text):
        return [m.span() for m in TOKEN_RE.finditer(text)]


def chunks(text, lines, stride):
    """Windows of `lines` non-blank lines every `stride` non-blank lines; yields (first_line, last_line, text)."""
    rows = [(i, l) for i, l in enumerate(text.split("\n")) if l.strip()]
    if len(rows) <= lines:
        if rows:
            yield rows[0][0], rows[-1][0], "\n".join(l for _, l in rows)
        return
    for s in range(0, len(rows) - lines + stride, stride):
        w = rows[s:s + lines]
        if len(w) < lines // 2:
            break
        yield w[0][0], w[-1][0], "\n".join(l for _, l in w)


def build_docs(cfg, split_map, st):
    reth_root = ROOT / cfg["sources"]["reth"]["path"]
    reth, bench = [], []
    for f in sources.reth_files(cfg):
        text = sources.read_source(reth_root / f["path"])
        meta = {"source": "reth", "path": f["path"], "split": split_map.get(f["crate"], "excluded")}
        reth.append({"text": text, "metadata": {**meta, "unit": "file"}})
        for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):
            reth.append({"text": t, "metadata": {**meta, "unit": "chunk", "lines": [a + 1, b + 1]}})
    seen = {}
    for r in sources.benchmark_rows(cfg):
        ref = f"{r['split']}:{r['index']}"
        texts = {"post_file": r["prompt"] + r["canonical_solution"] + r["suffix"]}
        if r["split"] == "edit":
            texts["pre_file"] = r["prompt"] + r["to_remove"] + r["suffix"]
        if sum(1 for l in r["canonical_solution"].split("\n") if l.strip()) >= st["min_middle_lines"]:
            texts["middle"] = r["canonical_solution"]
        for unit, text in texts.items():
            pieces = [(None, text)] if unit == "middle" else [(None, text)] + [((a + 1, b + 1), t) for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"])]
            for lines, t in pieces:
                u = unit if lines is None else unit + "_chunk"
                h = hashlib.sha1(t.encode()).hexdigest()
                if h in seen:  # same text from several examples (same file/commit): keep one doc, list refs
                    seen[h]["metadata"]["refs"].append(ref)
                    continue
                doc = {"text": t, "metadata": {"source": "bench", "unit": u, "repo": r["repo"], "path": r["path"],
                                               "refs": [ref], **({"lines": list(lines)} if lines else {})}}
                seen[h] = doc
                bench.append(doc)
    return reth, bench


def planted_controls(bench, n, seed, st):
    """Altered copies of benchmark files placed on the reth side: numbers changed, indentation re-flowed,
    ~10% of lines deleted. The check must find them, or a clean result means nothing."""
    import random
    rng = random.Random(seed)
    files = [d for d in bench if d["metadata"]["unit"] == "post_file"]
    out = []
    for d in rng.sample(files, n):
        lines = [l for l in d["text"].split("\n") if rng.random() > 0.1]
        text = "\n".join(re.sub(r"\d+", lambda m: str(int(m.group()) + 1), l).replace("    ", "  ") for l in lines)
        meta = {"source": "reth", "path": "CONTROL:" + d["metadata"]["path"], "split": "control", "control_of": d}
        out.append({"text": text, "metadata": {**meta, "unit": "file"}})
        for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):  # same units as real reth files
            out.append({"text": t, "metadata": {**meta, "unit": "chunk", "lines": [a + 1, b + 1]}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=Path, default=Path(__file__).with_name("settings.json"))
    ap.add_argument("--split_json", type=Path, required=True, help="split.json from a data build")
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()
    st = json.loads(args.settings.read_text())
    cfg = sources.load_config()
    pins = sources.verify_pins(cfg)
    split_map = json.loads(args.split_json.read_text())["crate_to_split"]

    mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                       hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
    tokenizer = RustWordTokenizer()
    sig_step = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tokenizer)

    t0 = time.time()
    reth, bench = build_docs(cfg, split_map, st)
    controls = planted_controls(bench, st["planted_controls"], cfg["split"]["seed"], st)
    reth = reth + controls
    min_tokens = st["min_doc_ngrams"] + st["n_grams"] - 1
    ok = lambda d: len(tokenizer.word_tokenize(d["text"])) >= min_tokens
    skipped = {"reth": sum(not ok(d) for d in reth), "bench": sum(not ok(d) for d in bench)}
    reth, bench = [d for d in reth if ok(d)], [d for d in bench if ok(d)]

    work = Path(tempfile.mkdtemp(prefix="neardup_"))
    try:
        (work / "input").mkdir()
        for stem, docs in (("00000", reth), ("00001", bench)):
            with open(work / "input" / f"{stem}.jsonl", "w") as f:
                for i, d in enumerate(docs):
                    f.write(json.dumps({"id": f"{stem}-{i}", "text": d["text"], "metadata": {k: v for k, v in d["metadata"].items() if k != "control_of"}}) + "\n")
        LocalPipelineExecutor([JsonlReader(str(work / "input")),
                               MinhashDedupSignature(output_folder=str(work / "sigs"), config=mh, language=tokenizer)],
                              tasks=2, logging_dir=str(work / "logs/sig")).run()
        LocalPipelineExecutor([MinhashDedupBuckets(input_folder=str(work / "sigs"), output_folder=str(work / "buckets"), config=mh)],
                              tasks=mh.num_buckets, logging_dir=str(work / "logs/buckets")).run()
        LocalPipelineExecutor([MinhashDedupCluster(input_folder=str(work / "buckets"), output_folder=str(work / "clusters"),
                                                   config=mh, save_cluster_id=True)],
                              tasks=1, logging_dir=str(work / "logs/cluster")).run()
        clusters = collections.defaultdict(list)
        for rank, docs in ((0, reth), (1, bench)):
            p = work / "clusters" / f"{rank:06d}.clusters"
            if p.exists():
                raw = p.read_bytes()
                for doc, cid in struct.iter_unpack("<2I", raw):
                    clusters[cid].append(docs[doc])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    shingle_cache = {}
    def shingles(d):
        k = id(d)
        if k not in shingle_cache:
            shingle_cache[k] = set(sig_step.get_shingles(d["text"]).ravel().tolist())
        return shingle_cache[k]
    def sims(a, b):
        A, B = shingles(a), shingles(b)
        inter = len(A & B)
        return {"jaccard": inter / len(A | B), "containment_in_reth": inter / len(A) if A else 0.0}

    bench_hits, split_hits = [], []
    for cid, docs in clusters.items():
        r_docs = [d for d in docs if d["metadata"]["source"] == "reth"]
        b_docs = [d for d in docs if d["metadata"]["source"] == "bench"]
        for b in b_docs:  # best reth partner for each benchmark doc in a mixed cluster
            if r_docs:
                best = max(r_docs, key=lambda r: sims(b, r)["jaccard"])
                bench_hits.append({"cluster": cid, "bench": b, "reth": best, **sims(b, best)})
        by_split = collections.defaultdict(list)
        for r in r_docs:
            if r["metadata"]["split"] == "control":
                continue
            by_split[r["metadata"]["split"]].append(r)
        names = sorted(by_split)
        for i, s1 in enumerate(names):
            for s2 in names[i + 1:]:
                pairs = [(a, b) for a in by_split[s1] for b in by_split[s2]][: st["max_pairs_per_cluster"]]
                if pairs:
                    a, b = max(pairs, key=lambda p: sims(*p)["jaccard"])
                    split_hits.append({"cluster": cid, "splits": [s1, s2], "a": a, "b": b, "cluster_size": len(docs), **sims(a, b)})

    def brief(d):
        m = d["metadata"]
        return {k: m[k] for k in ("source", "unit", "path", "split", "repo", "lines", "refs") if k in m} | {"text_head": d["text"][:600]}

    control_hits = [h for h in bench_hits if h["reth"]["metadata"]["split"] == "control" and h["jaccard"] >= st["verify_jaccard"]]
    found_controls = {h["reth"]["metadata"]["path"] for h in control_hits}
    bench_hits = [h for h in bench_hits if h["reth"]["metadata"]["split"] != "control"]
    verified = lambda hits: [h for h in hits if h["jaccard"] >= st["verify_jaccard"]]
    bench_v = sorted(verified(bench_hits), key=lambda h: -h["jaccard"])
    split_v = sorted(verified(split_hits), key=lambda h: -h["jaccard"])
    exact_dup_files = collections.defaultdict(set)
    for d in reth:
        if d["metadata"]["unit"] == "file":
            exact_dup_files[hashlib.sha1(d["text"].encode()).hexdigest()].add((d["metadata"]["split"], d["metadata"]["path"]))
    cross_split_exact = [sorted(v) for v in exact_dup_files.values() if len({s for s, _ in v}) > 1]

    # Verbatim 5-line windows (same rule as checks/overlap) shared between reth splits: lower-similarity copying
    # that MinHash at ~0.7 Jaccard would not flag.
    def win_set(text):
        rows = [l.strip() for l in text.split("\n") if l.strip()]
        return {hashlib.md5("\n".join(rows[i:i + 5]).encode()).hexdigest() for i in range(len(rows) - 4)
                if sum(len(x) for x in rows[i:i + 5]) >= 150}
    split_windows = collections.defaultdict(collections.Counter)
    for d in reth:
        m = d["metadata"]
        if m["unit"] == "file" and m["split"] in ("train", "validation", "test"):
            split_windows[m["split"]].update(win_set(d["text"]))
    train_w = set(split_windows["train"])
    window_overlap = {s: {"windows": len(split_windows[s]), "also_in_train": len(set(split_windows[s]) & train_w),
                          "share": round(len(set(split_windows[s]) & train_w) / max(1, len(split_windows[s])), 4)}
                      for s in ("validation", "test")}
    window_overlap["validation_test_shared"] = len(set(split_windows["validation"]) & set(split_windows["test"]))

    report = {
        "inputs": pins, "settings": st, "split_json": str(args.split_json.resolve().relative_to(ROOT)),
        "datatrove_minhash": {"n_grams": mh.n_grams, "num_buckets": mh.num_buckets, "hashes_per_bucket": mh.hashes_per_bucket,
                              "approx_jaccard_at_50pct_detection": round((1 / mh.num_buckets) ** (1 / mh.hashes_per_bucket), 3)},
        "coverage": {"reth_docs": collections.Counter(d["metadata"]["unit"] for d in reth),
                     "bench_docs": collections.Counter(d["metadata"]["unit"] for d in bench),
                     "skipped_short_docs": skipped},
        "clusters": len(clusters),
        "positive_control": {"planted": len({c["metadata"]["path"] for c in controls}), "found": len(found_controls),
                             "per_control": sorted(({"path": c["metadata"]["path"], "found": c["metadata"]["path"] in found_controls,
                                                     "file_jaccard_to_source": round(sims(c["metadata"]["control_of"], c)["jaccard"], 3)}
                                                    for c in controls if c["metadata"]["unit"] == "file"), key=lambda x: x["file_jaccard_to_source"])},
        "bench_vs_reth": {"candidates_in_mixed_clusters": len(bench_hits), "verified": len(bench_v),
                          "verified_items": [{"jaccard": round(h["jaccard"], 3), "containment": round(h["containment_in_reth"], 3),
                                              "bench": brief(h["bench"]), "reth": brief(h["reth"])} for h in bench_v[:200]]},
        "reth_cross_split": {"candidate_pairs": len(split_hits), "verified": len(split_v),
                             "by_split_pair": dict(collections.Counter("/".join(h["splits"]) for h in split_v)),
                             "exact_duplicate_files_across_splits": cross_split_exact,
                             "verbatim_5line_windows": window_overlap,
                             "verified_items": [{"jaccard": round(h["jaccard"], 3), "splits": h["splits"], "cluster_size": h["cluster_size"],
                                                 "a": brief(h["a"]), "b": brief(h["b"])} for h in split_v[:300]]},
        "seconds": round(time.time() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, default=list) + "\n")
    summary = {k: report[k] for k in ("coverage", "clusters", "datatrove_minhash", "positive_control")}
    summary["bench_vs_reth"] = {k: report["bench_vs_reth"][k] for k in ("candidates_in_mixed_clusters", "verified")}
    summary["reth_cross_split"] = {k: report["reth_cross_split"][k] for k in ("candidate_pairs", "verified", "by_split_pair")}
    summary["reth_cross_split"]["exact_duplicate_files_across_splits"] = len(cross_split_exact)
    summary["reth_cross_split"]["verbatim_5line_windows"] = window_overlap
    print(json.dumps(summary, indent=1, default=list))


if __name__ == "__main__":
    main()
