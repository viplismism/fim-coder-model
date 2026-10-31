#!/usr/bin/env python3
"""Check for overlap between a pinned reth snapshot and Real-FIM-Eval's Rust subset.

Standard library only. Inputs are local copies of the pinned snapshots (see run.sh).
A clean result means "no overlap detected by these checks", not proof that none exists.
"""

import argparse, collections, gzip, hashlib, json, os, re, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_lines(text):
    return [l.strip() for l in text.split("\n") if l.strip()]


def windows(lines, n, min_chars):
    for i in range(len(lines) - n + 1):
        win = lines[i:i + n]
        if sum(map(len, win)) >= min_chars:
            yield hashlib.md5("\n".join(win).encode()).hexdigest(), i


def github_repo(name):
    headers = {"Accept": "application/vnd.github+json"}
    if os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/repos/{name}", headers=headers), timeout=30) as r:
            d = json.load(r)  # urllib follows renames (301)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    return {"resolved": d["full_name"], "fork": d["fork"],
            "parent": (d.get("parent") or {}).get("full_name"), "source": (d.get("source") or {}).get("full_name")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", type=Path, default=Path(__file__).with_name("settings.json"))
    ap.add_argument("--reth", type=Path, required=True, help="reth checkout at the pinned commit")
    ap.add_argument("--bench", type=Path, required=True, help="real_fim_eval snapshot dir (add/, edit/)")
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--github_api", action="store_true", help="query GitHub for fork/rename status of source repos")
    args = ap.parse_args()
    cfg = json.loads(args.settings.read_text())

    head = subprocess.run(["git", "-C", str(args.reth), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != cfg["reth"]["commit"]:
        sys.exit(f"reth checkout is at {head}, settings pin {cfg['reth']['commit']}")
    bench_files = {s: args.bench / s / "test.jsonl.gz" for s in cfg["benchmark"]["splits"]}
    inputs = {"reth_commit": head, "benchmark_revision": cfg["benchmark"]["revision"],
              "benchmark_sha256": {s: sha256(p) for s, p in bench_files.items()}}

    rust = []
    for split, path in bench_files.items():
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            if r["lang"] == cfg["benchmark"]["lang"]:
                r["split"] = split
                rust.append(r)
    full = lambda r: r["prompt"] + r["canonical_solution"] + r["suffix"]
    repos = collections.Counter(r["repo"] for r in rust)
    report = {"inputs": inputs, "settings": cfg, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "benchmark_rust": {"examples": len(rust), "by_split": dict(collections.Counter(r["split"] for r in rust)),
                                 "repos": dict(repos.most_common())}}

    # 1. Source repo names
    rx = re.compile(cfg["name_patterns"]["repo"], re.I)
    report["repo_name_matches"] = sorted(r for r in repos if rx.search(r))

    # 2. Fork / rename status (network, optional)
    if args.github_api:
        report["github"] = {name: github_repo(name) for name in sorted(repos)}
        ok = {n: g for n, g in report["github"].items() if "error" not in g}
        report["github_checked"] = f"{len(ok)}/{len(repos)}"
        report["github_forks"] = {n: g for n, g in ok.items() if g["fork"]}
        report["github_renamed"] = {n: g["resolved"] for n, g in ok.items() if g["resolved"] != n}

    # 3. Mentions of reth and its dependencies in paths and code
    mentions = {}
    for name, pat in cfg["name_patterns"]["content"].items():
        crx, hits = re.compile(pat, re.I), collections.Counter()
        for r in rust:
            if crx.search(r["path"] + "\n" + full(r) + r.get("to_remove", "")):
                hits[r["repo"]] += 1
        mentions[name] = {"examples": sum(hits.values()), "by_repo": dict(hits)}
    report["content_mentions"] = mentions

    # 4. Verbatim multi-line windows shared with reth
    wcfg = cfg["window"]
    reth_files = sorted(args.reth.glob(cfg["reth"]["file_glob"]))
    reth_win, reth_long = {}, set()
    for p in reth_files:
        lines = norm_lines(p.read_text(errors="replace"))
        for h, _ in windows(lines, wcfg["lines"], wcfg["min_chars"]):
            reth_win.setdefault(h, str(p.relative_to(args.reth)))
        reth_long.update(l for l in lines if len(l) >= cfg["long_line"]["min_chars"])
    win_hits, mid_hits, examples = collections.Counter(), collections.Counter(), []
    for r in rust:
        pre, mid, suf = (norm_lines(r[k]) for k in ("prompt", "canonical_solution", "suffix"))
        lines = pre + mid + suf
        matched = [(h, i) for h, i in windows(lines, wcfg["lines"], wcfg["min_chars"]) if h in reth_win]
        if matched:
            win_hits[r["repo"]] += 1
            if any(i < len(pre) + len(mid) and i + wcfg["lines"] > len(pre) for _, i in matched):
                mid_hits[r["repo"]] += 1
            examples.append({"split": r["split"], "repo": r["repo"], "path": r["path"], "windows": len(matched),
                             "reth_file": reth_win[matched[0][0]]})
    report["window_overlap"] = {"reth_rs_files": len(reth_files), "reth_windows": len(reth_win),
                                "examples_with_shared_window": sum(win_hits.values()), "by_repo": dict(win_hits),
                                "examples_where_window_touches_middle": sum(mid_hits.values()),
                                "examples": sorted(examples, key=lambda e: -e["windows"])[:50]}

    # 5. Long single lines of the middle found verbatim in reth
    boiler = re.compile(cfg["long_line"]["boilerplate_regex"])
    line_hits, uniq = collections.Counter(), set()
    for r in rust:
        for l in norm_lines(r["canonical_solution"]):
            if len(l) >= cfg["long_line"]["min_chars"] and l in reth_long:
                line_hits[r["repo"]] += 1
                uniq.add(l)
    report["long_line_overlap"] = {"middle_line_hits": sum(line_hits.values()), "by_repo": dict(line_hits),
                                   "unique_lines": len(uniq), "non_boilerplate_lines": sorted(l for l in uniq if not boiler.match(l))}

    # 6. Positive control: the window detector must find same-repo reuse
    pc = cfg["positive_control"]
    ref = {h for r in rust if r["repo"] == pc["repo"] and r["path"].startswith(pc["reference_path_prefix"])
           for h, _ in windows(norm_lines(r["prompt"] + r["suffix"]), wcfg["lines"], wcfg["min_chars"])}
    hit = lambda r: any(h in ref for h, _ in windows(norm_lines(full(r)), wcfg["lines"], wcfg["min_chars"]))
    report["positive_control"] = {"same_repo_matches": sum(1 for r in rust if r["repo"] == pc["repo"] and hit(r)),
                                  "other_repo_matches": sum(1 for r in rust if r["repo"] != pc["repo"] and hit(r))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k not in ("settings", "benchmark_rust")}, indent=2)[:6000])


if __name__ == "__main__":
    main()
