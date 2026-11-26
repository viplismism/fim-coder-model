#!/usr/bin/env python3
"""Fetch (shallow, sparse: only the language's source files + licence files) and pin repos for one language.

Usage: python -m data.fetch_repos <lang> <list.json>
<list.json>: {"train": ["owner/repo", ...], "heldout": [...]}. Clones go to sources/multilang/<lang>/<owner>__<repo>;
pins (exact commit, file count, bytes) are written to results/multilang/leakage/<lang>_pins.json. Already-cloned repos are kept
and re-verified, never re-fetched at a different commit.
"""

import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.langs import get  # noqa: E402


def run(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, timeout=1800)


def main():
    lang, lists = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
    exts = get(lang)["extensions"]
    base = ROOT / "work" / "sources" / "multilang" / lang
    base.mkdir(parents=True, exist_ok=True)
    pins_path = ROOT / "configs" / "repos" / f"{lang}_pins.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {"lang": lang, "repos": {}}
    patterns = [f"*{e}" for e in exts] + ["LICENSE*", "LICENCE*", "COPYING*"]
    for kind in ("train", "heldout"):
        for repo in lists[kind]:
            d = base / repo.replace("/", "__")
            if not d.exists():
                # pinned repos are fetched at their pinned commit; unpinned ones at the default branch's HEAD
                ref = pins["repos"].get(repo, {}).get("commit")
                run("git", "init", "-q", str(d)); run("git", "-C", str(d), "remote", "add", "origin", f"https://github.com/{repo}.git")
                run("git", "-C", str(d), "sparse-checkout", "set", "--no-cone", *patterns)
                r = run("git", "-C", str(d), "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", ref or "HEAD")
                if r.returncode:
                    print(f"FAILED {repo}: {r.stderr.strip()[:200]}"); continue
                r = run("git", "-C", str(d), "checkout", "-q", "FETCH_HEAD")
                if r.returncode:
                    print(f"FAILED checkout {repo}: {r.stderr.strip()[:200]}"); continue
            sha = run("git", "-C", str(d), "rev-parse", "HEAD").stdout.strip()
            if repo in pins["repos"] and pins["repos"][repo]["commit"] != sha:
                sys.exit(f"{repo}: checkout {sha} differs from pinned {pins['repos'][repo]['commit']}")
            files = [p for p in d.rglob("*") if p.is_file() and p.suffix in exts and ".git" not in p.parts]
            lic = sorted(p.name for p in d.iterdir() if p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING")))
            pins["repos"][repo] = {"kind": kind, "commit": sha, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
                                   "licence_files": lic, "path": str(d.relative_to(ROOT))}
            print(f"{kind:8s} {repo:40s} {sha[:10]}  {len(files):5d} files  {pins['repos'][repo]['bytes'] / 1e6:6.2f} MB  licence {lic}", flush=True)
    pins_path.write_text(json.dumps(pins, indent=1) + "\n")


if __name__ == "__main__":
    main()
