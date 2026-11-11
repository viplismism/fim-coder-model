#!/usr/bin/env python3
"""Overlap check for one language: training pool vs Real-FIM-Eval (that language), training pool vs held-out repos,
and train vs validation. Same method as checks/neardup: datatrove MinHash on whole files + 30-line chunks (+ benchmark
answers), exact verification (Jaccard >= 0.5), 20 planted altered controls; plus exact 5-line windows.
Usage: python checks/multilang/overlap_lang.py <lang> <data_version>
Writes checks/multilang/<lang>_overlap_report.json. A clean result is "no overlap detected", not proof of absence.
"""

import collections, hashlib, json, random, re, shutil, struct, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "neardup"))
from neardup import RustWordTokenizer, chunks  # noqa: E402  (regex tokens: identifiers, numbers, punctuation)
from datatrove.executor import LocalPipelineExecutor  # noqa: E402
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupSignature  # noqa: E402
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.utils.hashing import HashConfig  # noqa: E402
from datatrove.utils.text import TextNormConfig  # noqa: E402
from fimexp import sources  # noqa: E402
from fimexp.langs import get  # noqa: E402


def win_set(text):
    rows = [l.strip() for l in text.split("\n") if l.strip()]
    return {hashlib.md5("\n".join(rows[i:i + 5]).encode()).hexdigest() for i in range(len(rows) - 4) if sum(map(len, rows[i:i + 5])) >= 150}


def main():
    lang, version = sys.argv[1], sys.argv[2]
    st = json.loads((ROOT / "checks/neardup/settings.json").read_text())
    cfg = sources.load_config()
    data = ROOT / cfg["data"]["output_dir"] / version
    split = json.loads((data / "split.json").read_text())
    pins = json.loads((data / "manifest.json").read_text())["pins"]
    files = {}
    for rel in [f for s in ("train", "validation") for f in split["files_by_split"].get(s, [])]:
        repo = "/".join(rel.split("/")[:2])
        files[rel] = (split["crate_to_split"][repo], (ROOT / pins["repos"][repo]["path"] / "/".join(rel.split("/")[2:])).read_bytes().decode())
    held = {}
    for repo, p in pins["repos"].items():
        if p["kind"] == "heldout":
            d = ROOT / p["path"]
            for f in d.rglob("*"):
                if f.is_file() and f.suffix in get(lang)["extensions"] and ".git" not in f.parts and f.stat().st_size <= 300_000:
                    try:
                        held[f"{repo}/{f.relative_to(d).as_posix()}"] = f.read_bytes().decode()
                    except UnicodeDecodeError:
                        pass
    pool = []
    for rel, (s, text) in files.items():
        meta = {"source": "pool", "split": s, "path": rel}
        pool.append({"text": text, "metadata": {**meta, "unit": "file"}})
        for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):
            pool.append({"text": t, "metadata": {**meta, "unit": "chunk", "lines": [a + 1, b + 1]}})
    other = []
    for rel, text in held.items():
        meta = {"source": "heldout", "path": rel}
        other.append({"text": text, "metadata": {**meta, "unit": "file"}})
        for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):
            other.append({"text": t, "metadata": {**meta, "unit": "chunk"}})
    seen = set(); bench_files = []
    for r in sources.benchmark_rows(cfg, lang):
        texts = {"post_file": r["prompt"] + r["canonical_solution"] + r["suffix"]}
        if r["split"] == "edit":
            texts["pre_file"] = r["prompt"] + r["to_remove"] + r["suffix"]
        if sum(1 for l in r["canonical_solution"].split("\n") if l.strip()) >= st["min_middle_lines"]:
            texts["middle"] = r["canonical_solution"]
        for unit, text in texts.items():
            pieces = [text] if unit == "middle" else [text] + [t for _, _, t in chunks(text, st["chunk_lines"], st["chunk_stride"])]
            for t in pieces:
                h = hashlib.sha1(t.encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                doc = {"text": t, "metadata": {"source": "bench", "unit": unit, "repo": r["repo"], "path": r["path"]}}
                other.append(doc)
                if unit == "post_file" and t == text:
                    bench_files.append(doc)
    rng = random.Random(cfg["split"]["seed"])
    controls = []
    for d in rng.sample(bench_files, min(st["planted_controls"], len(bench_files))):
        lines = [l for l in d["text"].split("\n") if rng.random() > 0.1]
        text = "\n".join(re.sub(r"\d+", lambda m: str(int(m.group()) + 1), l).replace("    ", "  ") for l in lines)
        tag = "CONTROL:" + d["metadata"]["path"]
        controls.append(tag)
        pool.append({"text": text, "metadata": {"source": "pool", "split": "control", "path": tag, "unit": "file"}})
        for a, b, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):
            pool.append({"text": t, "metadata": {"source": "pool", "split": "control", "path": tag, "unit": "chunk"}})

    mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                       hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
    tok = RustWordTokenizer(); sig = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tok)
    ok = lambda d: len(tok.word_tokenize(d["text"])) >= st["min_doc_ngrams"] + st["n_grams"] - 1
    groups = [[d for d in pool if ok(d)], [d for d in other if ok(d)]]
    work = Path(tempfile.mkdtemp(prefix=f"overlap_{lang}_"))
    try:
        (work / "in").mkdir()
        for k, docs in enumerate(groups):
            with open(work / "in" / f"{k:05d}.jsonl", "w") as f:
                for i, d in enumerate(docs):
                    f.write(json.dumps({"id": f"{k}-{i}", "text": d["text"]}) + "\n")
        LocalPipelineExecutor([JsonlReader(str(work / "in")), MinhashDedupSignature(output_folder=str(work / "s"), config=mh, language=tok)], tasks=2, logging_dir=str(work / "l1")).run()
        LocalPipelineExecutor([MinhashDedupBuckets(input_folder=str(work / "s"), output_folder=str(work / "b"), config=mh)], tasks=mh.num_buckets, logging_dir=str(work / "l2")).run()
        LocalPipelineExecutor([MinhashDedupCluster(input_folder=str(work / "b"), output_folder=str(work / "c"), config=mh, save_cluster_id=True)], tasks=1, logging_dir=str(work / "l3")).run()
        clusters = collections.defaultdict(list)
        for k, docs in enumerate(groups):
            p = work / "c" / f"{k:06d}.clusters"
            if p.exists():
                for doc, cid in struct.iter_unpack("<2I", p.read_bytes()):
                    clusters[cid].append(docs[doc])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cache = {}
    sh = lambda d: cache.setdefault(id(d), set(sig.get_shingles(d["text"]).ravel().tolist()))
    jac = lambda a, b: len(sh(a) & sh(b)) / max(1, len(sh(a) | sh(b)))
    brief = lambda d: {k: d["metadata"].get(k) for k in ("source", "split", "repo", "path", "unit")} | {"text_head": d["text"][:300]}
    hits = collections.defaultdict(list); found_controls = set()
    for docs in clusters.values():
        P = [d for d in docs if d["metadata"]["source"] == "pool"]
        for x in P:
            for key, others in (("pool_vs_benchmark", [d for d in docs if d["metadata"]["source"] == "bench"]),
                                ("pool_vs_heldout", [d for d in docs if d["metadata"]["source"] == "heldout"]),
                                ("train_vs_validation", [d for d in P if {d["metadata"]["split"], x["metadata"]["split"]} == {"train", "validation"}])):
                if others:
                    best = max(others, key=lambda o: jac(x, o)); j = jac(x, best)
                    if j >= st["verify_jaccard"]:
                        if x["metadata"]["split"] == "control":
                            if key == "pool_vs_benchmark":
                                found_controls.add(x["metadata"]["path"])
                        else:
                            hits[key].append({"jaccard": round(j, 3), "a": brief(x), "b": brief(best)})
    wins = {s: set() for s in ("train", "validation")}
    for rel, (s, text) in files.items():
        wins[s] |= win_set(text)
    bench_w = set().union(*(win_set(d["text"]) for d in bench_files))
    held_w = set().union(*(win_set(t) for t in held.values()))
    pool_w = wins["train"] | wins["validation"]
    report = {"lang": lang, "data_version": version, "docs": {"pool": len(groups[0]), "benchmark_and_heldout": len(groups[1])},
              "positive_control": {"planted": len(controls), "found": len(found_controls)},
              "minhash_verified": {k: len(hits.get(k, [])) for k in ("pool_vs_benchmark", "pool_vs_heldout", "train_vs_validation")},
              "verified_items": {k: sorted(v, key=lambda h: -h["jaccard"])[:100] for k, v in hits.items()},
              "exact_5line_windows": {"pool_windows": len(pool_w), "shared_with_benchmark": len(pool_w & bench_w),
                                      "shared_with_heldout": len(pool_w & held_w), "train_validation_shared": len(wins["train"] & wins["validation"])}}
    (ROOT / "checks/multilang" / f"{lang}_overlap_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("docs", "positive_control", "minhash_verified", "exact_5line_windows")}, indent=1))


if __name__ == "__main__":
    main()
