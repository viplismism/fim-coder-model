#!/usr/bin/env python3
"""PROPOSE repos for typescript, go, cpp with EXACTLY the v3 rule (propose_repos_v3.py: one repo per owner across all
lists; same-name Real-FIM-Eval repos excluded; target-language code 0.3-30 MB and >= 50% of the repo's code; MIT/Apache/BSD;
same search queries, star order). Owners already used are seeded from the lists actually in use: the final Python list
(python_list.json, incl. its replacements), the Rust v3 lists, and reth. Read-only GitHub metadata; nothing cloned.
Writes checks/multilang/proposed_repos_rest.json and <lang>_list.json ({"train": [12], "heldout": [10]}) per language.
"""
import gzip, json, sys, time, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "checks" / "multilang"))
from propose_repos import LANGS, LICENCES, NON_LIBRARY, QUERIES, TAKE  # noqa: E402
from propose_repos_v3 import get, MIN_BYTES, MAX_BYTES, MIN_SHARE  # noqa: E402

ORDER = tuple(sys.argv[1:]) or ("typescript", "go", "cpp")
OUT = ROOT / "checks/multilang/proposed_repos_rest.json"


def main():
    bench = set()
    for s in ("add", "edit"):
        for l in gzip.open(ROOT / f"sources/real_fim_eval/{s}/test.jsonl.gz", "rt"):
            bench.add(json.loads(l)["repo"])
    bench_full = {b.lower() for b in bench}
    bench_names = {b.split("/")[1].lower(): b for b in bench}
    heldout_v1 = {p["repo"].lower() for p in json.loads((ROOT / "checks/heldout/picked.json").read_text())["picked"]}
    used = list(json.loads((ROOT / "checks/multilang/python_list.json").read_text()).values())
    v3 = json.loads((ROOT / "checks/multilang/proposed_repos_v3_python_rust.json").read_text())["languages"]["rust"]
    used += [[p["repo"] for p in v3[k]["picked"]] for k in ("train", "heldout")]
    owners = {"paradigmxyz"} | {r.split("/")[0].lower() for lst in used for r in lst}
    out = json.loads(OUT.read_text()) if OUT.exists() else {"rule": __doc__, "searched": time.strftime("%Y-%m-%d"), "languages": {}}
    for lang, res in out["languages"].items():   # resume: owners of already finished languages stay used
        owners |= {p["repo"].split("/")[0].lower() for k in res.values() for p in k["picked"]}
    for lang in ORDER:
        if lang in out["languages"]:
            print(f"{lang}: already done", flush=True); continue
        L = LANGS[lang]; res = {}
        for kind in ("train", "heldout"):
            picked, skipped, page = [], [], 0
            while len(picked) < TAKE[kind] and page < 10:
                page += 1
                time.sleep(7)
                items = get("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                    {"q": QUERIES[kind].format(L=L), "sort": "stars", "order": "desc", "per_page": 100, "page": page}))["items"]
                if not items:
                    break
                for it in items:
                    if len(picked) >= TAKE[kind]:
                        break
                    name = it["full_name"]; owner = name.split("/")[0].lower(); rname = name.split("/")[1].lower()
                    lic = (it.get("license") or {}).get("spdx_id"); why = None
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
        OUT.write_text(json.dumps(out, indent=1) + "\n")
        lst = {k: [p["repo"] for p in res[k]["picked"]] for k in ("train", "heldout")}
        if len(lst["train"]) == TAKE["train"] and len(lst["heldout"]) == TAKE["heldout"]:
            (ROOT / f"checks/multilang/{lang}_list.json").write_text(json.dumps(lst, indent=1) + "\n")
            print(f"{lang}: list written", flush=True)
        else:
            print(f"{lang}: SHORT LIST, not written", flush=True)


if __name__ == "__main__":
    main()
