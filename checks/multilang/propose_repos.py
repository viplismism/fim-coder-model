#!/usr/bin/env python3
"""PROPOSE training-pool and fresh held-out repos per language from GitHub search metadata (read-only, no cloning).

Rule (fixed before running):
- Search: language:<L>, not a fork, not archived, sorted by stars (desc).
  - training pool: stars >= 2000, pushed >= 2025-01-01 (active), created before 2024-01-01 (established projects;
    also a clean time split from the 2024-25 held-out repos)  [v2 rule]
  - fresh held-out: created 2024-01-01..2025-06-30 (after the base model's data; >= 16 months of history), stars >= 500
- Keep: licence MIT, Apache-2.0, BSD-2-Clause or BSD-3-Clause; GitHub primary language == <L>;
  repo size between 2 MB and 300 MB (metadata `size`; exact code bytes are verified later with sparse clones).
- Exclude: every Real-FIM-Eval repo (all languages), every repo in heldout-v1, owners already picked (one repo per
  owner, across the training pool and held-out of the same language), and non-library repos by name/description
  keywords (awesome lists, tutorials, courses, books, interview/leetcode collections, lists/collections of APIs,
  prompts, workflows or skills, algorithm compilations)  [v2 rule].
- Take, in star order: 12 training-pool repos, then 10 held-out repos. Rust: reth is kept as one of the 12 (continuity
  with the earlier experiment); the other 11 come from the search.
Writes checks/multilang/proposed_repos_v2.json and .md (v1 = first rule, kept for transparency).
"""

import gzip, json, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANGS = {"rust": "Rust", "python": "Python", "typescript": "TypeScript", "go": "Go", "cpp": "C++"}
LICENCES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}
NON_LIBRARY = re.compile(r"awesome|tutorial|course|book|interview|leetcode|cheatsheet|roadmap|learn-|-learning|examples?$|exercises|"
                         r"\blist of\b|collective list|collection of|curated|workflows|prompts|\bskills?\b|all algorithms|algorithms implemented", re.I)
QUERIES = {
    "train": "language:{L} stars:>=2000 fork:false archived:false pushed:>=2025-01-01 created:<2024-01-01",
    "heldout": "language:{L} stars:>=500 fork:false archived:false created:2024-01-01..2025-06-30",
}
TAKE = {"train": 12, "heldout": 10}


def search(q, page):
    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page})
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["items"]


def main():
    bench = set()
    for s in ("add", "edit"):
        for l in gzip.open(ROOT / f"sources/real_fim_eval/{s}/test.jsonl.gz", "rt"):
            bench.add(json.loads(l)["repo"].lower())
    heldout_v1 = {p["repo"].lower() for p in json.loads((ROOT / "checks/heldout/picked.json").read_text())["picked"]}
    out = {"rule": __doc__, "searched": time.strftime("%Y-%m-%d"), "languages": {}}
    for lang, L in LANGS.items():
        owners, res = set(), {}
        if lang == "rust":
            owners.add("paradigmxyz")
        for kind in ("train", "heldout"):
            picked = [{"repo": "paradigmxyz/reth", "note": "kept for continuity with the earlier experiment"}] if (lang == "rust" and kind == "train") else []
            skipped = []
            for page in (1, 2, 3):
                if len(picked) >= TAKE[kind]:
                    break
                time.sleep(7)  # search API: 10 requests/minute unauthenticated
                for it in search(QUERIES[kind].format(L=L), page):
                    name = it["full_name"]; owner = name.split("/")[0].lower()
                    lic = (it.get("license") or {}).get("spdx_id")
                    why = None
                    if name.lower() in bench: why = "in Real-FIM-Eval"
                    elif name.lower() in heldout_v1: why = "in heldout-v1"
                    elif owner in owners: why = "owner already picked"
                    elif lic not in LICENCES: why = f"licence {lic}"
                    elif it.get("language") != L: why = f"primary language {it.get('language')}"
                    elif not (2_000 <= it["size"] <= 300_000): why = f"repo size {it['size'] // 1000} MB"
                    elif NON_LIBRARY.search(name) or NON_LIBRARY.search(it.get("description") or ""): why = "not a library/application repo"
                    if why:
                        skipped.append({"repo": name, "reason": why}); continue
                    owners.add(owner)
                    picked.append({"repo": name, "stars": it["stargazers_count"], "licence": lic, "created": it["created_at"][:10],
                                   "pushed": it["pushed_at"][:10], "size_mb": round(it["size"] / 1000, 1),
                                   "description": (it.get("description") or "")[:90]})
                    if len(picked) >= TAKE[kind]:
                        break
            res[kind] = {"picked": picked, "skipped_count": len(skipped),
                         "skipped_reasons": dict(sorted({s["reason"].split(" ")[0]: 0 for s in skipped}.items()))}
            for s in skipped:
                res[kind]["skipped_reasons"][s["reason"].split(" ")[0]] += 1
            print(f"{lang} {kind}: picked {len(picked)}, skipped {len(skipped)}", flush=True)
        out["languages"][lang] = res
    (ROOT / "checks/multilang/proposed_repos_v2.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["# Proposed repos per language (metadata only; verify code size with sparse clones before building)", ""]
    for lang, res in out["languages"].items():
        for kind, title in (("train", "training pool (10 train / 2 validation, split later by seeded shuffle)"), ("heldout", "fresh held-out (locked, scored once)")):
            lines += [f"## {lang}: {title}", "", "| # | repo | stars | licence | created | size MB | description |", "|---|---|---|---|---|---|---|"]
            for i, p in enumerate(res[kind]["picked"], 1):
                lines.append(f"| {i} | {p['repo']} | {p.get('stars', '')} | {p.get('licence', '')} | {p.get('created', '')} | {p.get('size_mb', '')} | {p.get('description', p.get('note', ''))} |")
            lines += ["", f"Skipped {res[kind]['skipped_count']} candidates: {res[kind]['skipped_reasons']}", ""]
    (ROOT / "checks/multilang/proposed_repos_v2.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
