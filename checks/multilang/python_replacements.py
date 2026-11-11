#!/usr/bin/env python3
"""Pick 2 replacement Python training-pool repos by the v3 rule (removed: Z4nzu/hackingtool, pathwaycom/serviette).
Walks the v3 training search in star order, skipping already-proposed repos; owners already used in the Python and
Rust v3 lists are excluded (one repo per owner across all lists). Writes checks/multilang/python_replacements.json."""
import json, sys, time, urllib.parse, gzip
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "checks" / "multilang"))
from propose_repos import LANGS, LICENCES, NON_LIBRARY, QUERIES  # noqa: E402
from propose_repos_v3 import get, MIN_BYTES, MAX_BYTES, MIN_SHARE  # noqa: E402

v3 = json.loads((ROOT / "checks/multilang/proposed_repos_v3_python_rust.json").read_text())["languages"]
proposed = {p["repo"].lower() for l in v3.values() for k in l.values() for p in k["picked"]}
owners = {r.split("/")[0] for r in proposed}
bench = set()
for s in ("add", "edit"):
    for l in gzip.open(ROOT / f"sources/real_fim_eval/{s}/test.jsonl.gz", "rt"):
        bench.add(json.loads(l)["repo"])
bench_full = {b.lower() for b in bench}; bench_names = {b.split("/")[1].lower() for b in bench}
heldout_v1 = {p["repo"].lower() for p in json.loads((ROOT / "checks/heldout/picked.json").read_text())["picked"]}
picked, skipped, page = [], [], 0
while len(picked) < 2 and page < 5:
    page += 1; time.sleep(7)
    for it in get("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
            {"q": QUERIES["train"].format(L="Python"), "sort": "stars", "order": "desc", "per_page": 100, "page": page}))["items"]:
        if len(picked) >= 2: break
        name = it["full_name"]; low = name.lower(); owner = low.split("/")[0]
        if low in proposed: continue
        lic = (it.get("license") or {}).get("spdx_id"); why = None
        if low in bench_full or low.split("/")[1] in bench_names: why = "benchmark name"
        elif low in heldout_v1: why = "heldout-v1"
        elif owner in owners: why = "owner used"
        elif lic not in LICENCES: why = f"licence {lic}"
        elif it.get("language") != "Python": why = "primary language"
        elif not (2_000 <= it["size"] <= 300_000): why = "repo size"
        elif NON_LIBRARY.search(name) or NON_LIBRARY.search(it.get("description") or ""): why = "not a library"
        if not why:
            langs = get(f"https://api.github.com/repos/{name}/languages"); b = langs.get("Python", 0); tot = sum(langs.values()) or 1
            if not (MIN_BYTES <= b <= MAX_BYTES): why = f"code {b/1e6:.2f} MB"
            elif b / tot < MIN_SHARE: why = f"share {b/tot:.0%}"
        if why: skipped.append((name, why)); continue
        owners.add(owner)
        picked.append({"repo": name, "stars": it["stargazers_count"], "licence": lic, "created": it["created_at"][:10],
                       "code_mb": round(b / 1e6, 2), "code_share": round(b / tot, 2), "description": (it.get("description") or "")[:90]})
        print("picked", picked[-1], flush=True)
(ROOT / "checks/multilang/python_replacements.json").write_text(json.dumps({"picked": picked, "skipped": skipped}, indent=1) + "\n")
print("skipped:", skipped[:15])
