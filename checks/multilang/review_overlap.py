#!/usr/bin/env python3
"""Review of every verified MinHash match and every shared exact 5-line window for one language, before and after
removing ONLY standard licence headers: a file's leading comment block, removed only if it contains licence/copyright
wording. All other comments are kept. Each match gets a verdict; boilerplate exceptions are logged with the reason.
Usage: python checks/multilang/review_overlap.py <lang> <data_version>
Writes checks/multilang/<lang>_overlap_review.json and .md.
"""

import collections, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "neardup"))
from neardup import RustWordTokenizer  # noqa: E402
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupSignature  # noqa: E402
from datatrove.utils.hashing import HashConfig  # noqa: E402
from datatrove.utils.text import TextNormConfig  # noqa: E402
from fimexp import sources  # noqa: E402

COMMENT = {"python": "#"}
LICENCE_WORDS = re.compile(r"copyright|licensed under|\blicen[cs]e\b|spdx-license-identifier|permission is hereby granted|all rights reserved", re.I)
VERDICT_THRESHOLD = 0.3  # Jaccard after licence removal at or above this = remaining shared code, needs review


def strip_licence(text, comment):
    """Remove the leading comment block (plus blank and shebang lines) only if it contains licence wording."""
    lines = text.split("\n"); i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith(comment) or lines[i].startswith("#!")):
        i += 1
    head = "\n".join(lines[:i])
    return ("\n".join(lines[i:]), True) if LICENCE_WORDS.search(head) else (text, False)


def windows(text):
    rows = [l.strip() for l in text.split("\n") if l.strip()]
    return {hashlib.md5("\n".join(rows[i:i + 5]).encode()).hexdigest(): "\n".join(rows[i:i + 5])
            for i in range(len(rows) - 4) if sum(map(len, rows[i:i + 5])) >= 150}


def main():
    lang, version = sys.argv[1], sys.argv[2]
    comment = COMMENT.get(lang, "//")
    st = json.loads((ROOT / "checks/neardup/settings.json").read_text())
    mh = MinhashConfig(n_grams=st["n_grams"], num_buckets=st["num_buckets"], hashes_per_bucket=st["hashes_per_bucket"],
                       hash_config=HashConfig(precision=64), norm_config=TextNormConfig(**st["norm"]))
    tok = RustWordTokenizer(); sig = MinhashDedupSignature(output_folder="/dev/null", config=mh, language=tok)
    sh = lambda t: set(sig.get_shingles(t).ravel().tolist()) if len(tok.word_tokenize(t)) >= st["n_grams"] else set()
    jac = lambda a, b: (len(sh(a) & sh(b)) / len(sh(a) | sh(b))) if sh(a) and sh(b) else 0.0
    pins = json.loads((ROOT / "data" / version / "manifest.json").read_text())["pins"]["repos"]
    pool_src = lambda path: (ROOT / pins["/".join(path.split("/")[:2])]["path"] / "/".join(path.split("/")[2:])).read_text()
    bench = {}
    for r in sources.benchmark_rows(sources.load_config(), lang):
        bench.setdefault((r["repo"], r["path"]), []).append(r)
    rep = json.loads((ROOT / "checks/multilang" / f"{lang}_overlap_report.json").read_text())

    rows, seen = [], set()
    for key in ("pool_vs_benchmark", "pool_vs_heldout"):
        for it in rep["verified_items"].get(key, []):
            k = (key, it["a"]["path"], it["b"]["path"])
            if k in seen:
                continue
            seen.add(k)
            a = pool_src(it["a"]["path"])
            if key == "pool_vs_benchmark":
                r0 = bench[(it["b"]["repo"], it["b"]["path"])][0]
                b = r0["prompt"] + (r0.get("to_remove", "") if str(it["b"]["unit"]).startswith("pre") else r0["canonical_solution"]) + r0["suffix"]
                other = f"{it['b']['repo']}:{it['b']['path']}"
            else:
                b = pool_src(it["b"]["path"]); other = it["b"]["path"]
            sa, ha = strip_licence(a, comment); sb, hb = strip_licence(b, comment)
            j_after = round(jac(sa, sb), 3)
            shared = sorted({l.strip() for l in sa.split("\n") if l.strip()} & {l.strip() for l in sb.split("\n") if l.strip()})
            verdict = ("remaining shared code: REVIEW" if j_after >= VERDICT_THRESHOLD
                       else "boilerplate exception: matching text is the licence header" if (ha or hb)
                       else "low similarity without a licence header: REVIEW")
            rows.append({"check": key, "pool_file": it["a"]["path"], "other": other, "jaccard_before": it["jaccard"],
                         "jaccard_after_licence_removed": j_after, "licence_header_removed": {"pool": ha, "other": hb},
                         "pool_lines_after": len([l for l in sa.split("\n") if l.strip()]),
                         "shared_non_blank_lines_after": shared[:20], "verdict": verdict})

    split = json.loads((ROOT / "data" / version / "split.json").read_text())
    pool_w = {}
    for f in [x for s in ("train", "validation") for x in split["files_by_split"][s]]:
        pool_w.update(windows(pool_src(f)))
    bench_w = {}
    for rs in bench.values():
        bench_w.update(windows(rs[0]["prompt"] + rs[0]["canonical_solution"] + rs[0]["suffix"]))
    held_w = {}
    for repo, p in pins.items():
        if p["kind"] == "heldout":
            for f in (ROOT / p["path"]).rglob("*"):
                if f.is_file() and f.suffix == ".py":
                    try:
                        held_w.update(windows(f.read_text()))
                    except UnicodeDecodeError:
                        pass
    win = {}
    for name, other in (("benchmark", bench_w), ("heldout", held_w)):
        shared = [pool_w[h] for h in sorted(set(pool_w) & set(other))]
        lic = [w for w in shared if LICENCE_WORDS.search(w)]
        rest = [w for w in shared if not LICENCE_WORDS.search(w)]
        win[name] = {"shared_windows": len(shared), "licence_wording": len(lic), "other": len(rest), "other_windows": rest}

    summary = collections.Counter(f"{r['check']} | {r['verdict']}" for r in rows)
    out = {"rule": "only a leading licence/copyright comment block is removed; all other comments are kept",
           "verdict_threshold_jaccard_after": VERDICT_THRESHOLD, "minhash_summary": dict(summary),
           "max_jaccard_after_licence_removed": max((r["jaccard_after_licence_removed"] for r in rows), default=0.0),
           "minhash_matches": rows, "exact_windows": win}
    (ROOT / "checks/multilang" / f"{lang}_overlap_review.json").write_text(json.dumps(out, indent=1) + "\n")

    md = [f"# {lang} overlap review (licence headers only removed)", "", f"Rule: {out['rule']}.", "",
          "| check | pool file | other file | J before | J after | lines left | shared lines after | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['check']} | {r['pool_file']} | {r['other']} | {r['jaccard_before']} | {r['jaccard_after_licence_removed']} | "
                  f"{r['pool_lines_after']} | {'; '.join(r['shared_non_blank_lines_after'][:3]) or '(none)'} | {r['verdict']} |")
    md += ["", "## Exact 5-line windows shared with the training pool", ""]
    for name, w in win.items():
        md.append(f"- {name}: {w['shared_windows']} shared; {w['licence_wording']} contain licence wording; {w['other']} other (listed below)")
    for name, w in win.items():
        for x in w["other_windows"]:
            md += ["", f"**{name}, other window:**", "```", x, "```"]
    (ROOT / "checks/multilang" / f"{lang}_overlap_review.md").write_text("\n".join(md) + "\n")
    print(json.dumps({k: out[k] for k in ("minhash_summary", "max_jaccard_after_licence_removed")}, indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("shared_windows", "licence_wording", "other")} for k, v in win.items()}, indent=1))


if __name__ == "__main__":
    main()
