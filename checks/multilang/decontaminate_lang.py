#!/usr/bin/env python3
"""File-level decontamination of a language's training pool (train + validation repos) against its Real-FIM-Eval subset
and locked held-out repos. Rule: checks/multilang/DECONTAMINATION.md. Test sets are never modified.

Usage: python checks/multilang/decontaminate_lang.py <lang> <pins.json> [--pool_only <data_version>]
Writes checks/multilang/<lang>_decontamination.json (removed files + reasons, licence-header exceptions, planted-control
results) and <lang>_exclude_files.json (the removal list used by the data builder).
"""

import argparse, collections, hashlib, json, random, re, shutil, struct, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "neardup"))
from neardup import RustWordTokenizer, chunks  # noqa: E402
from datatrove.executor import LocalPipelineExecutor  # noqa: E402
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets, MinhashDedupCluster, MinhashDedupSignature  # noqa: E402
from datatrove.pipeline.readers import JsonlReader  # noqa: E402
from datatrove.utils.hashing import HashConfig  # noqa: E402
from datatrove.utils.text import TextNormConfig  # noqa: E402
from fimexp import sources  # noqa: E402
from fimexp.langs import get  # noqa: E402

LICENCE = re.compile(r"copyright|licensed under|\blicen[cs]e\b|spdx-license-identifier|permission is hereby granted|all rights reserved", re.I)
COMMENT = {"python": "#"}
GENERATED = ("@generated", "do not edit", "automatically generated", "auto-generated")
FINE_LINES, FINE_STRIDE, JACCARD = 10, 5, 0.5


def strip_licence(text, comment):
    lines = text.split("\n"); i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(comment) or lines[i].startswith("#!")):
        i += 1
    return ("\n".join(lines[i:]), True) if LICENCE.search("\n".join(lines[:i])) else (text, False)


def windows(text):
    rows = [l.strip() for l in text.split("\n") if l.strip()]
    return {hashlib.md5("\n".join(rows[i:i + 5]).encode()).hexdigest(): "\n".join(rows[i:i + 5])
            for i in range(len(rows) - 4) if sum(map(len, rows[i:i + 5])) >= 150}


def alter(text, rng):
    lines = [l for l in text.split("\n") if rng.random() > 0.1]
    return "\n".join(re.sub(r"\d+", lambda m: str(int(m.group()) + 1), l).replace("    ", "  ") for l in lines)


def units(text, st):
    yield "file", text
    for _, _, t in chunks(text, st["chunk_lines"], st["chunk_stride"]):
        yield "chunk30", t
    for _, _, t in chunks(text, FINE_LINES, FINE_STRIDE):
        yield "chunk10", t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang"); ap.add_argument("pins")
    ap.add_argument("--pool_only", default=None, help="check only the files of this built data version (post-rebuild verification)")
    args = ap.parse_args()
    lang = args.lang; comment = COMMENT.get(lang, "//"); exts = get(lang)["extensions"]
    st = json.loads((ROOT / "checks/neardup/settings.json").read_text())
    cfg = sources.load_config(); pins = json.loads(Path(args.pins).read_text())["repos"]

    def repo_files(kind):
        out = {}
        for repo, p in sorted(pins.items()):
            if p["kind"] != kind:
                continue
            d = ROOT / p["path"]
            for f in sorted(x for x in d.rglob("*") if x.is_file() and x.suffix in exts and ".git" not in x.parts):
                try:
                    t = f.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                out[f"{repo}/{f.relative_to(d).as_posix()}"] = t
        return out
    pool = repo_files("train")
    pool = {k: v for k, v in pool.items() if len(v.encode()) <= 300_000 and not any(g in v[:400].lower() for g in GENERATED) and v.strip()}
    if args.pool_only:
        split = json.loads((ROOT / "data" / args.pool_only / "split.json").read_text())
        keep = {f for s in ("train", "validation") for f in split["files_by_split"][s]}
        pool = {k: v for k, v in pool.items() if k in keep}
    held = repo_files("heldout")
    bench_rows = sources.benchmark_rows(cfg, lang)
    targets = []  # (label, text)
    for r in bench_rows:
        ref = f"{r['repo']}:{r['path']} [{r['split']}:{r['index']}]"
        targets.append((f"benchmark post-file {ref}", r["prompt"] + r["canonical_solution"] + r["suffix"]))
        if r["split"] == "edit":
            targets.append((f"benchmark pre-file {ref}", r["prompt"] + r["to_remove"] + r["suffix"]))
        if sum(1 for l in r["canonical_solution"].split("\n") if l.strip()) >= st["min_middle_lines"]:
            targets.append((f"benchmark answer {ref}", r["canonical_solution"]))
    targets += [(f"held-out {k}", v) for k, v in held.items()]
    seen = set(); tg = []
    for lab, t in targets:
        h = hashlib.sha1(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h); tg.append((lab, t))
    targets = tg

    # planted controls (pool side)
    rng = random.Random(cfg["split"]["seed"])
    post = [(lab, t) for lab, t in targets if lab.startswith("benchmark post-file")]
    controls = {}
    for lab, t in rng.sample(post, min(20, len(post))):
        controls[f"CONTROL-FILE:{lab}"] = {"text": alter(t, rng), "source": t, "kind": "whole file"}
    long_post = [(lab, t) for lab, t in post if sum(1 for l in t.split("\n") if l.strip()) >= 30]
    hosts = sorted(pool)
    frng = random.Random(cfg["split"]["seed"] + 1)
    for lab, t in frng.sample(long_post, min(20, len(long_post))):
        lines = t.split("\n"); n = frng.randint(12, 20); s = frng.randint(0, max(0, len(lines) - n))
        frag = "\n".join(lines[s:s + n])
        host = pool[frng.choice(hosts)].split("\n"); cut = len(host) // 2
        controls[f"CONTROL-FRAGMENT({n} lines):{lab}"] = {"text": "\n".join(host[:cut] + [alter(frag, frng)] + host[cut:]), "source": frag, "kind": "fragment"}
    if lang == "python":  # the previously missed short whole-file control must be among the whole-file controls
        assert any("aiohttp/examples/curl.py" in k for k in controls), "missed control not reproduced"

    # Rule 1: exact non-licence windows
    target_w = collections.defaultdict(set)
    for lab, t in targets:
        for h in windows(t):
            target_w[h].add(lab)
    reasons = collections.defaultdict(list); licence_exceptions = collections.defaultdict(list)
    def exact_check(name, text):
        hit = False
        for h, w in windows(text).items():
            if h in target_w:
                if LICENCE.search(w):
                    licence_exceptions[name].append(sorted(target_w[h])[0])
                else:
                    reasons[name].append(("exact 5-line window", sorted(target_w[h])[0])); hit = True
        return hit
    for name, text in pool.items():
        exact_check(name, text)
    control_hits = {k: set() for k in controls}
    for k, c in controls.items():
        for h, w in windows(c["text"]).items():
            if h in target_w and not LICENCE.search(w):
                control_hits[k].add("exact 5-line window")

    # Rules 2-3: MinHash on licence-stripped units (whole file, 30-line, 10-line chunks)
    mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                       hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
    tok = RustWordTokenizer(); sig = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tok)
    min_tok = st["min_doc_ngrams"] + st["n_grams"] - 1
    pool_units, target_units = [], []
    for name, text in list(pool.items()) + [(k, c["text"]) for k, c in controls.items()]:
        s, _ = strip_licence(text, comment)
        for u, t in units(s, st):
            if len(tok.word_tokenize(t)) >= min_tok:
                pool_units.append((name, u, t))
    for lab, text in targets:
        s, _ = strip_licence(text, comment)
        for u, t in (units(s, st) if not lab.startswith("benchmark answer") else [("answer", s)]):
            if len(tok.word_tokenize(t)) >= min_tok:
                target_units.append((lab, u, t))
    work = Path(tempfile.mkdtemp(prefix=f"decont_{lang}_"))
    n_files = 8
    try:
        (work / "in").mkdir()
        shards = [[] for _ in range(n_files)]
        for i, d in enumerate(pool_units):
            shards[i % (n_files // 2)].append(("P", i))
        for i, d in enumerate(target_units):
            shards[n_files // 2 + i % (n_files // 2)].append(("T", i))
        for k, sh in enumerate(shards):
            with open(work / "in" / f"{k:05d}.jsonl", "w") as f:
                for side, i in sh:
                    t = (pool_units if side == "P" else target_units)[i][2]
                    f.write(json.dumps({"id": f"{side}{i}", "text": t}) + "\n")
        LocalPipelineExecutor([JsonlReader(str(work / "in")), MinhashDedupSignature(output_folder=str(work / "s"), config=mh, language=tok)],
                              tasks=n_files, logging_dir=str(work / "l1")).run()
        LocalPipelineExecutor([MinhashDedupBuckets(input_folder=str(work / "s"), output_folder=str(work / "b"), config=mh)],
                              tasks=mh.num_buckets, logging_dir=str(work / "l2")).run()
        LocalPipelineExecutor([MinhashDedupCluster(input_folder=str(work / "b"), output_folder=str(work / "c"), config=mh, save_cluster_id=True)],
                              tasks=1, logging_dir=str(work / "l3")).run()
        clusters = collections.defaultdict(list)
        for k, sh in enumerate(shards):
            p = work / "c" / f"{k:06d}.clusters"
            if p.exists():
                for doc, cid in struct.iter_unpack("<2I", p.read_bytes()):
                    clusters[cid].append(sh[doc])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cache = {}
    def shs(side, i):
        key = (side, i)
        if key not in cache:
            cache[key] = set(sig.get_shingles((pool_units if side == "P" else target_units)[i][2]).ravel().tolist())
        return cache[key]
    for members in clusters.values():
        P = [i for s, i in members if s == "P"]; T = [i for s, i in members if s == "T"]
        if not P or not T:
            continue
        for i in P:
            best, bj = None, 0.0
            for j in T[:300]:
                A, B = shs("P", i), shs("T", j)
                jj = len(A & B) / max(1, len(A | B))
                if jj > bj:
                    best, bj = j, jj
            if bj >= JACCARD:
                name, unit, _ = pool_units[i]
                lab = target_units[best][0]
                rule = {"file": "near-duplicate (whole file)", "chunk30": "near-duplicate (30-line chunk)", "chunk10": "near-duplicate (10-line chunk)"}[unit]
                if name in controls:
                    control_hits[name].add(rule)
                else:
                    reasons[name].append((rule, lab, round(bj, 3)))

    removed = {k: sorted({r[0] for r in v}) for k, v in reasons.items()}
    by_repo = collections.Counter("/".join(k.split("/")[:2]) for k in removed)
    files_by_repo = collections.Counter("/".join(k.split("/")[:2]) for k in pool)
    ctl = {k: {"kind": c["kind"], "caught_by": sorted(control_hits[k])} for k, c in controls.items()}
    out = {"rule": "checks/multilang/DECONTAMINATION.md", "lang": lang, "pool_files": len(pool), "targets": len(targets),
           "units": {"pool": len(pool_units), "targets": len(target_units)},
           "removed_files": {k: {"rules": removed[k], "examples": [list(map(str, r)) for r in reasons[k][:3]]} for k in sorted(removed)},
           "removed_by_repo": {r: f"{by_repo.get(r, 0)}/{files_by_repo[r]}" for r in sorted(files_by_repo)},
           "removed_by_rule": dict(collections.Counter(r for v in removed.values() for r in v)),
           "licence_header_exceptions": {k: sorted(set(v))[:3] for k, v in sorted(licence_exceptions.items()) if k not in removed},
           "planted_controls": ctl,
           "planted_summary": {kind: {"planted": sum(1 for c in ctl.values() if c["kind"] == kind),
                                      "caught": sum(1 for c in ctl.values() if c["kind"] == kind and c["caught_by"]),
                                      **{rule: sum(1 for c in ctl.values() if c["kind"] == kind and rule in c["caught_by"])
                                         for rule in ("exact 5-line window", "near-duplicate (whole file)", "near-duplicate (30-line chunk)", "near-duplicate (10-line chunk)")}}
                               for kind in ("whole file", "fragment")},
           "missed_controls": sorted(k for k, c in ctl.items() if not c["caught_by"])}
    tag = f"_check_{args.pool_only}" if args.pool_only else ""
    (ROOT / "checks/multilang" / f"{lang}_decontamination{tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    if not args.pool_only:
        (ROOT / "checks/multilang" / f"{lang}_exclude_files.json").write_text(json.dumps(sorted(removed), indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("pool_files", "units", "removed_by_rule", "removed_by_repo", "planted_summary", "missed_controls")}, indent=1))
    print(f"removed files: {len(removed)}; licence-header exceptions logged: {len(out['licence_header_exceptions'])}")


if __name__ == "__main__":
    main()
