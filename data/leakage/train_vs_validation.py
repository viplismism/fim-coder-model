#!/usr/bin/env python3
"""Train-vs-validation check (DECONTAMINATION.md addendum) for one built data version: a VALIDATION file is listed for
removal if it shares with any TRAINING file a non-licence exact 5-line window, or is a verified near-duplicate (whole
file, 30-line or 10-line chunk; shingle Jaccard >= 0.5 after licence-header removal). Training files are never removed.
Usage: python -m data.leakage.train_vs_validation <lang> <data_version>
Writes results/multilang/leakage/<lang>_train_vs_validation.json; prints the number of validation files to remove.
"""
import collections, json, shutil, struct, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); from data.leakage.decontaminate import COMMENT, JACCARD, LICENCE, strip_licence, units, windows  # noqa: E402
from data.leakage.neardup import RustWordTokenizer  # noqa: E402
from datatrove.executor import LocalPipelineExecutor  # noqa: E402
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupSignature  # noqa: E402
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.utils.hashing import HashConfig  # noqa: E402
from datatrove.utils.text import TextNormConfig  # noqa: E402


def main():
    lang, version = sys.argv[1], sys.argv[2]
    comment = COMMENT.get(lang, ("//", "/*", "*"))
    st = json.loads((ROOT / "configs/leakage/neardup_settings.json").read_text())
    data = ROOT / "work" / "datasets" / version
    split = json.loads((data / "split.json").read_text())
    pins = json.loads((data / "manifest.json").read_text())["pins"]["repos"]
    text = lambda rel: (ROOT / pins["/".join(rel.split("/")[:2])]["path"] / "/".join(rel.split("/")[2:])).read_bytes().decode()
    tr = {f: text(f) for f in split["files_by_split"]["train"]}
    va = {f: text(f) for f in split["files_by_split"]["validation"]}
    reasons = collections.defaultdict(set)
    tw = {}
    for f, t in tr.items():
        for h, w in windows(t).items():
            tw.setdefault(h, f)
    for f, t in va.items():
        for h, w in windows(t).items():
            if h in tw and not LICENCE.search(w):
                reasons[f].add(f"exact 5-line window with {tw[h]}")
    mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                       hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
    tok = RustWordTokenizer(); sig = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tok)
    min_tok = st["min_doc_ngrams"] + st["n_grams"] - 1
    U = {"T": [], "V": []}
    for side, files in (("T", tr), ("V", va)):
        for f, t in files.items():
            s, _ = strip_licence(t, comment)
            for u, x in units(s, st):
                if len(tok.word_tokenize(x)) >= min_tok:
                    U[side].append((f, u, x))
    work = Path(tempfile.mkdtemp(prefix=f"trainval_{lang}_"))
    order = [("T", i) for i in range(len(U["T"]))] + [("V", i) for i in range(len(U["V"]))]
    n = 8; shards = [order[k::n] for k in range(n)]
    try:
        (work / "in").mkdir()
        for k, sh in enumerate(shards):
            with open(work / "in" / f"{k:05d}.jsonl", "w") as fh:
                for side, i in sh:
                    fh.write(json.dumps({"id": f"{side}{i}", "text": U[side][i][2]}) + "\n")
        LocalPipelineExecutor([JsonlReader(str(work / "in")), MinhashDedupSignature(output_folder=str(work / "s"), config=mh, language=tok)], tasks=n, logging_dir=str(work / "l1")).run()
        LocalPipelineExecutor([MinhashDedupBuckets(input_folder=str(work / "s"), output_folder=str(work / "b"), config=mh)], tasks=mh.num_buckets, logging_dir=str(work / "l2")).run()
        LocalPipelineExecutor([MinhashDedupCluster(input_folder=str(work / "b"), output_folder=str(work / "c"), config=mh, save_cluster_id=True)], tasks=1, logging_dir=str(work / "l3")).run()
        clusters = collections.defaultdict(list)
        for k, sh in enumerate(shards):
            p = work / "c" / f"{k:06d}.clusters"
            if p.exists():
                for doc, cid in struct.iter_unpack("<2I", p.read_bytes()):
                    clusters[cid].append(sh[doc])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cache = {}
    shs = lambda side, i: cache.setdefault((side, i), set(sig.get_shingles(U[side][i][2]).ravel().tolist()))
    for members in clusters.values():
        T = [i for s, i in members if s == "T"]; V = [i for s, i in members if s == "V"]
        for i in V:
            for j in T[:300]:
                A, B = shs("V", i), shs("T", j)
                if len(A & B) / max(1, len(A | B)) >= JACCARD:
                    reasons[U["V"][i][0]].add(f"near-duplicate ({U['V'][i][1]}) of {U['T'][j][0]}"); break
    out = {"rule": "DECONTAMINATION.md addendum (train vs validation)", "data_version": version,
           "train_files": len(tr), "validation_files": len(va),
           "removed_validation_files": {k: sorted(v)[:5] for k, v in sorted(reasons.items())}}
    (ROOT / "results/multilang/leakage" / f"{lang}_train_vs_validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"train_files": len(tr), "validation_files": len(va), "remove": len(reasons)}))


if __name__ == "__main__":
    main()
