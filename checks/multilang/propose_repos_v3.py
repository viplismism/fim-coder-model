#!/usr/bin/env python3
"""PROPOSE repos per language, v3 rule (read-only GitHub metadata, no cloning).

Same as v2 (propose_repos.py) plus:
1. One repo per owner across ALL languages and ALL lists (training pools and held-out sets).
2. Exclude a candidate whose repo NAME equals a Real-FIM-Eval repo name under any owner (catches renames/forks and
   same-name projects, e.g. Eugeny/tabby vs TabbyML/tabby).
3. Code-size check with GitHub's language-bytes API: the target language must have >= 300 KB and <= 30 MB of code
   and be >= 50% of the repo's code bytes.
Languages are processed in the order python, rust, typescript, go, cpp (python first: the planned pilot), training pool
before held-out, candidates in star order. GitHub core API: 60 requests/hour unauthenticated, so this paces itself.
Writes checks/multilang/proposed_repos_v3.json and .md.
"""

import gzip, json, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "checks" / "multilang"))
from propose_repos import LANGS, LICENCES, NON_LIBRARY, QUERIES, TAKE  # noqa: E402

ORDER = ("python", "rust", "typescript", "go", "cpp")
MIN_BYTES, MAX_BYTES, MIN_SHARE = 300_000, 30_000_000, 0.5


def get(url):
    while True:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                reset = int(e.headers.get("X-RateLimit-Reset", time.time() + 600))
                wait = max(30, reset - time.time() + 5)
                print(f"  rate limited; waiting {wait / 60:.1f} min", flush=True)
                time.sleep(wait)
                continue
            raise


def main():
    bench = set()
    for s in ("add", "edit"):
        for l in gzip.open(ROOT / f"sources/real_fim_eval/{s}/test.jsonl.gz", "rt"):
            bench.add(json.loads(l)["repo"])
    bench_full = {b.lower() for b in bench}
    bench_names = {b.split("/")[1].lower(): b for b in bench}
    heldout_v1 = {p["repo"].lower() for p in json.loads((ROOT / "checks/heldout/picked.json").read_text())["picked"]}
    owners = {"paradigmxyz"}                       # reth (kept in the Rust training pool)
    out = {"rule": __doc__, "searched": time.strftime("%Y-%m-%d"), "languages": {}}
    for lang in ORDER:
        L = LANGS[lang]; res = {}
        for kind in ("train", "heldout"):
            picked = [{"repo": "paradigmxyz/reth", "note": "kept for continuity with the earlier experiment"}] if (lang == "rust" and kind == "train") else []
            skipped = []
            page = 0
            while len(picked) < TAKE[kind] and page < 5:
                page += 1
                time.sleep(7)
                items = get("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                    {"q": QUERIES[kind].format(L=L), "sort": "stars", "order": "desc", "per_page": 100, "page": page}))["items"]
                for it in items:
                    if len(picked) >= TAKE[kind]:
                        break
                    name = it["full_name"]; owner = name.split("/")[0].lower(); rname = name.split("/")[1].lower()
                    lic = (it.get("license") or {}).get("spdx_id")
                    why = None
                    if name.lower() in bench_full: why = "in Real-FIM-Eval"
                    elif rname in bench_names: why = f"same name as Real-FIM-Eval repo {bench_names[rname]}"
                    elif name.lower() in heldout_v1: why = "in heldout-v1"
                    elif owner in owners: why = "owner already used (any language/list)"
                    elif lic not in LICENCES: why = f"licence {lic}"
                    elif it.get("language") != L: why = f"primary language {it.get('language')}"
                    elif not (2_000 <= it["size"] <= 300_000): why = f"repo size {it['size'] // 1000} MB"
                    elif NON_LIBRARY.search(name) or NON_LIBRARY.search(it.get("description") or ""): why = "not a library/application repo"
                    if not why:
                        langs = get(f"https://api.github.com/repos/{name}/languages")
                        b = langs.get(L, 0); tot = sum(langs.values()) or 1
                        if not (MIN_BYTES <= b <= MAX_BYTES): why = f"{L} code {b / 1e6:.2f} MB outside 0.3-30 MB"
                        elif b / tot < MIN_SHARE: why = f"{L} only {b / tot:.0%} of code"
                    if why:
                        skipped.append({"repo": name, "reason": why}); continue
                    owners.add(owner)
                    picked.append({"repo": name, "stars": it["stargazers_count"], "licence": lic, "created": it["created_at"][:10],
                                   "code_mb": round(b / 1e6, 2), "code_share": round(b / tot, 2),
                                   "description": (it.get("description") or "")[:90]})
                    print(f"  {lang} {kind} {len(picked):2d}: {name} ({b / 1e6:.2f} MB {L}, {b / tot:.0%})", flush=True)
            res[kind] = {"picked": picked, "skipped": skipped}
            print(f"{lang} {kind}: picked {len(picked)}, skipped {len(skipped)}", flush=True)
        out["languages"][lang] = res
        (ROOT / "checks/multilang/proposed_repos_v3.json").write_text(json.dumps(out, indent=1) + "\n")
    lines = ["# Proposed repos per language, v3 rule (metadata only; nothing cloned)", "",
             "Rule: see propose_repos_v3.py. One repo per owner across all lists; same-name benchmark repos excluded; "
             "target-language code 0.3-30 MB and >= 50% of the repo's code.", ""]
    for lang in ORDER:
        res = out["languages"][lang]
        for kind, title in (("train", "training pool (10 train / 2 validation, split later by seeded shuffle)"), ("heldout", "fresh held-out (locked, scored once)")):
            lines += [f"## {lang}: {title}", "", "| # | owner/repo | stars | licence | created | code MB (share) | description |", "|---|---|---|---|---|---|---|"]
            for i, p in enumerate(res[kind]["picked"], 1):
                lines.append(f"| {i} | {p['repo']} | {p.get('stars', '')} | {p.get('licence', '')} | {p.get('created', '')} | "
                             f"{p.get('code_mb', '')} ({p.get('code_share', '')}) | {p.get('description', p.get('note', ''))} |")
            reasons = {}
            for s in res[kind]["skipped"]:
                k = s["reason"].split(" (")[0].split(" only")[0][:40]; reasons[k] = reasons.get(k, 0) + 1
            lines += ["", f"Skipped {len(res[kind]['skipped'])}: {dict(sorted(reasons.items(), key=lambda x: -x[1]))}", ""]
    (ROOT / "checks/multilang/proposed_repos_v3.md").write_text("\n".join(lines) + "\n")
    print("written proposed_repos_v3.md")


if __name__ == "__main__":
    main()
