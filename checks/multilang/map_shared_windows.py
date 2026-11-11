"""Map every non-licence exact 5-line window shared between the training pool and the benchmark / held-out repos to the
files on both sides; for the benchmark, say whether the window touches the scored answer (middle) or only context."""
import collections, hashlib, json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from fimexp import sources
LIC = re.compile(r"copyright|licensed under|\blicen[cs]e\b|spdx-license-identifier|permission is hereby granted|all rights reserved", re.I)
lang, version = sys.argv[1], sys.argv[2]
pins = json.loads((ROOT / "data" / version / "manifest.json").read_text())["pins"]["repos"]
split = json.loads((ROOT / "data" / version / "split.json").read_text())
def norm_rows(t): return [l.strip() for l in t.split("\n") if l.strip()]
def wins(t):
    r = norm_rows(t)
    return {hashlib.md5("\n".join(r[i:i + 5]).encode()).hexdigest(): "\n".join(r[i:i + 5]) for i in range(len(r) - 4) if sum(map(len, r[i:i + 5])) >= 150}
src = lambda p: (ROOT / pins["/".join(p.split("/")[:2])]["path"] / "/".join(p.split("/")[2:])).read_text()
pool = collections.defaultdict(set); text_of = {}
for f in [x for s in ("train", "validation") for x in split["files_by_split"][s]]:
    for h, w in wins(src(f)).items():
        if not LIC.search(w): pool[h].add(f); text_of[h] = w
bench = collections.defaultdict(set); in_answer = collections.defaultdict(set)
for r in sources.benchmark_rows(sources.load_config(), lang):
    full = r["prompt"] + r["canonical_solution"] + r["suffix"]; mid = set(wins(r["canonical_solution"]))
    for h in wins(full):
        if h in pool:
            bench[h].add(f"{r['repo']}:{r['path']}")
            # window overlaps the answer if any of its lines are in the answer
            if set(text_of[h].split("\n")) & set(norm_rows(r["canonical_solution"])): in_answer[h].add(f"{r['split']}:{r['index']}")
held = collections.defaultdict(set)
for repo, p in pins.items():
    if p["kind"] != "heldout": continue
    d = ROOT / p["path"]
    for f in d.rglob("*.py"):
        try: t = f.read_text()
        except UnicodeDecodeError: continue
        for h in wins(t):
            if h in pool: held[h].add(f"{repo}/{f.relative_to(d).as_posix()}")
def summarise(other):
    pairs = collections.Counter()
    for h, o in other.items():
        for pf in pool[h]:
            for of in o: pairs[(pf, of)] += 1
    return pairs
out = {}
for name, other in (("benchmark", bench), ("heldout", held)):
    pairs = summarise(other)
    by_repo = collections.Counter(("/".join(pf.split("/")[:2]), of.split(":")[0] if name == "benchmark" else "/".join(of.split("/")[:2])) for (pf, of) in pairs)
    out[name] = {"windows": len(other), "file_pairs": [{"pool_file": pf, "other_file": of, "shared_windows": n} for (pf, of), n in pairs.most_common()],
                 "by_repo_pair": {f"{a} ~ {b}": n for (a, b), n in by_repo.most_common()}}
    if name == "benchmark":
        out[name]["windows_touching_scored_answers"] = {h[:8]: sorted(v) for h, v in in_answer.items()}
        out[name]["benchmark_examples_whose_answer_overlaps"] = sorted({x for v in in_answer.values() for x in v})
    print(f"\n== {name}: {len(other)} windows; by repo pair: {out[name]['by_repo_pair']}")
    for p in out[name]["file_pairs"][:10]: print(f"   {p['shared_windows']:3d}  {p['pool_file']}  ~  {p['other_file']}")
print("\nbenchmark examples whose SCORED ANSWER contains shared lines:", out["benchmark"]["benchmark_examples_whose_answer_overlaps"])
(ROOT / "checks/multilang" / f"{lang}_shared_windows_map.json").write_text(json.dumps(out, indent=1) + "\n")
