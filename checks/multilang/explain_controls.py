"""Rebuild the 20 planted controls exactly as overlap_lang.py did (same documents, same seed) and explain each one:
file-level Jaccard to its source, best chunk Jaccard, and the LSH detection probability at those values."""
import hashlib, json, random, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "neardup"))
from neardup import RustWordTokenizer, chunks
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupSignature
from datatrove.utils.hashing import HashConfig
from datatrove.utils.text import TextNormConfig
from fimexp import sources
lang = sys.argv[1]
st = json.loads((ROOT / "checks/neardup/settings.json").read_text()); cfg = sources.load_config()
mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                   hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
tok = RustWordTokenizer(); sig = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tok)
sh = lambda t: set(sig.get_shingles(t).ravel().tolist()) if len(tok.word_tokenize(t)) >= st["n_grams"] else set()
jac = lambda a, b: (len(sh(a) & sh(b)) / len(sh(a) | sh(b))) if sh(a) and sh(b) else 0.0
p_detect = lambda j: 1 - (1 - j ** mh.hashes_per_bucket) ** mh.num_buckets
# same construction as overlap_lang.py: post_file docs in benchmark order, de-duplicated by text hash
seen, bench_files = set(), []
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
            if unit == "post_file" and t == text:
                bench_files.append({"text": t, "path": r["path"], "repo": r["repo"]})
rng = random.Random(cfg["split"]["seed"])
rows = []
for d in rng.sample(bench_files, min(st["planted_controls"], len(bench_files))):
    lines = [l for l in d["text"].split("\n") if rng.random() > 0.1]
    ctl = "\n".join(re.sub(r"\d+", lambda m: str(int(m.group()) + 1), l).replace("    ", "  ") for l in lines)
    fj = jac(ctl, d["text"])
    src_chunks = [t for _, _, t in chunks(d["text"], st["chunk_lines"], st["chunk_stride"])] or [d["text"]]
    ctl_chunks = [t for _, _, t in chunks(ctl, st["chunk_lines"], st["chunk_stride"])] or [ctl]
    best_chunk = max((jac(c, s) for c in ctl_chunks for s in src_chunks + [d["text"]]), default=0.0)
    deleted = 1 - len(lines) / max(1, len(d["text"].split("\n")))
    rows.append({"control": d["repo"] + ":" + d["path"], "tokens": len(tok.word_tokenize(d["text"])), "lines_deleted": round(deleted, 3),
                 "file_jaccard": round(fj, 3), "best_chunk_jaccard": round(best_chunk, 3),
                 "p_detect_file": round(p_detect(fj), 3), "p_detect_best_unit": round(p_detect(max(fj, best_chunk)), 3)})
rows.sort(key=lambda r: r["p_detect_best_unit"])
for r in rows:
    print(f"{r['p_detect_best_unit']:.3f}  file J {r['file_jaccard']:.3f}  best chunk J {r['best_chunk_jaccard']:.3f}  "
          f"deleted {r['lines_deleted']:.0%}  tokens {r['tokens']:6d}  {r['control']}")
(ROOT / "checks/multilang" / f"{lang}_controls_explained.json").write_text(json.dumps(rows, indent=1) + "\n")

# Which control was actually missed: compare MinHash band signatures (deterministic) of every control unit against the
# source file's units; a control is detected if any unit pair shares any band.
def bands(t):
    s = sig.get_shingles(t)
    return set() if len(s) == 0 else {(i, tuple(b)) for i, b in enumerate(sig.get_signature(s))}
rng = random.Random(cfg["split"]["seed"])
missed = []
for d in rng.sample(bench_files, min(st["planted_controls"], len(bench_files))):
    lines = [l for l in d["text"].split("\n") if rng.random() > 0.1]
    ctl = "\n".join(re.sub(r"\d+", lambda m: str(int(m.group()) + 1), l).replace("    ", "  ") for l in lines)
    cu = [ctl] + [t for _, _, t in chunks(ctl, st["chunk_lines"], st["chunk_stride"])]
    su = [d["text"]] + [t for _, _, t in chunks(d["text"], st["chunk_lines"], st["chunk_stride"])]
    sb = set().union(*(bands(t) for t in su))
    hit = any(bands(t) & sb for t in cu)
    if not hit:
        missed.append(d["repo"] + ":" + d["path"])
print("controls with no shared band (missed by LSH):", missed)
