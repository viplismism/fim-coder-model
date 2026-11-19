#!/usr/bin/env python3
"""Build the post-cutoff test candidates for one repo (CPU, git only; no models).

An example is one contiguous block of ADDED lines in a non-generated .go file, in a first-parent commit after the
cutoff. The file is taken at that commit; the hole is the added block; the prefix and suffix are the rest of that
file (Real-FIM-Eval "Add" construction). Steps:
  1. Commits: first-parent commits in (cutoff, head]. Leakage gate: author AND committer date >= cutoff, and the
     commit is not an ancestor of the cutoff commit.
  2. Files: .go only. Dropped as generated: zz_generated*, *.pb.go, *_pb2.go, mock dirs, mock_*.go, *_mock.go,
     *.mock.go, vendor/, and any file whose first 10 lines say "Code generated ... DO NOT EDIT". Files newly added
     in the commit are dropped (a hole would have no context of its own).
  3. Holes: 1-30 added lines with >= 1 non-blank line (rename detection on, so moved files are not "new" code).
  4. Content leakage: drop a hole whose whitespace-normalised text (>= 40 chars) or any 5-line window occurs in the
     cutoff snapshot's .go files.
Usage: python extract_test.py <repo_dir> <cutoff_iso> <out.jsonl>
"""
import collections, hashlib, json, re, subprocess, sys
from pathlib import Path

GEN_PATH = re.compile(r"(^|/)(zz_generated[^/]*|[^/]*\.pb\.go|[^/]*_pb2\.go|mock_[^/]*\.go|[^/]*_mock\.go|[^/]*\.mock\.go)$|(^|/)(mocks?|fakes?|vendor)/")
GEN_HEAD = re.compile(r"^// Code generated .* DO NOT EDIT\.?$", re.M)
MAX_LINES = 30


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, "-c", "core.quotepath=off", *a], capture_output=True, text=True, check=True).stdout


def norm(s):
    return " ".join(s.split())


def windows(lines):
    rows = [l.strip() for l in lines if l.strip()]
    return {hashlib.md5("\n".join(rows[i:i + 5]).encode()).hexdigest() for i in range(len(rows) - 4)}


def generated(path, text):
    return bool(GEN_PATH.search(path)) or bool(GEN_HEAD.search("\n".join(text.split("\n")[:10])))


def main():
    repo, cutoff, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    ccut = git(repo, "rev-list", "-1", "--first-parent", f"--before={cutoff}", "HEAD").strip()
    head = git(repo, "rev-parse", "HEAD").strip()
    stats = collections.Counter()
    # snapshot index for content leakage
    snap_norm, snap_w = [], set()
    for p in git(repo, "ls-tree", "-r", "--name-only", ccut).split("\n"):
        if p.endswith(".go"):
            t = git(repo, "show", f"{ccut}:{p}")
            snap_norm.append(norm(t)); snap_w |= windows(t.split("\n"))
    snap_blob = "\n".join(snap_norm)
    commits = git(repo, "rev-list", "--first-parent", "--reverse", f"{ccut}..{head}").split()
    examples = []
    for c in commits:
        ad, cd, subj = git(repo, "log", "-1", "--format=%aI%x00%cI%x00%s", c).strip().split("\x00")
        stats["commits"] += 1
        anc = subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", c, ccut]).returncode == 0
        if ad < cutoff or cd < cutoff or anc:
            stats["commits_dropped_date_or_ancestry"] += 1; continue
        parent = git(repo, "rev-parse", f"{c}^").strip()
        status = git(repo, "diff", "--name-status", "-M", parent, c, "--", "*.go")
        added_files = {l.split("\t")[-1] for l in status.split("\n") if l.startswith("A\t")}
        diff = git(repo, "diff", "-U0", "-M", "--no-color", parent, c, "--", "*.go")
        cur = None; blocks = []
        for line in diff.split("\n"):
            if line.startswith("+++ "):
                cur = None if line.startswith("+++ /dev/null") else line[6:].rstrip("\t")
            elif line.startswith("@@") and cur:
                m = re.match(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                start, n = int(m.group(2)), int(m.group(3) if m.group(3) is not None else 1)
                if n > 0:
                    blocks.append((cur, start, n, int(m.group(1) if m.group(1) is not None else 1)))
        files_cache = {}
        for path, start, n, ndel in blocks:
            stats["blocks"] += 1
            if path not in files_cache:
                files_cache[path] = git(repo, "show", f"{c}:{path}")
            text = files_cache[path]
            if path in added_files:
                stats["drop_new_file"] += 1; continue
            if generated(path, text):
                stats["drop_generated"] += 1; continue
            lines = text.split("\n")
            hole_lines = lines[start - 1:start - 1 + n]
            if n > MAX_LINES:
                stats["drop_over_30_lines"] += 1; continue
            if not any(l.strip() for l in hole_lines):
                stats["drop_blank"] += 1; continue
            hole = "\n".join(hole_lines) + "\n"
            prefix = "\n".join(lines[:start - 1]) + ("\n" if start > 1 else "")
            suffix = "\n".join(lines[start - 1 + n:])
            assert prefix + hole + suffix == text
            h = norm(hole)
            if (len(h) >= 40 and h in snap_blob) or (windows(hole_lines) & snap_w):
                stats["drop_in_cutoff_snapshot"] += 1; continue
            stats["kept"] += 1
            examples.append({"id": f"{c[:10]}:{path}:{start}", "commit": c, "parent": parent, "author_date": ad,
                             "commit_date": cd, "subject": subj[:100], "path": path, "module": module_of(repo, c, path),
                             "is_test_file": path.endswith("_test.go"), "hole_start_line": start, "hole_lines": n,
                             "replaced_lines": ndel, "prefix": prefix, "suffix": suffix, "middle": hole})
    out.write_text("".join(json.dumps(e) + "\n" for e in examples))
    per_commit = collections.Counter(e["commit"] for e in examples)
    summary = {"cutoff": cutoff, "cutoff_commit": ccut, "head": head, "stats": dict(stats),
               "examples": len(examples), "commits_with_examples": len(per_commit),
               "test_file_examples": sum(e["is_test_file"] for e in examples),
               "pure_additions": sum(e["replaced_lines"] == 0 for e in examples),
               "hole_lines": dict(sorted(collections.Counter(min(e["hole_lines"], 11) for e in examples).items())),
               "by_module": dict(collections.Counter(e["module"] for e in examples).most_common()),
               "max_examples_one_commit": max(per_commit.values(), default=0),
               "examples_if_capped_5_per_commit": sum(min(v, 5) for v in per_commit.values())}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


_mods = {}
def module_of(repo, c, path):
    if c not in _mods:
        _mods[c] = sorted((p[:-len("go.mod")].rstrip("/") for p in git(repo, "ls-tree", "-r", "--name-only", c).split("\n")
                           if p.endswith("go.mod")), key=len, reverse=True)
    for m in _mods[c]:
        if m == "" or path.startswith(m + "/"):
            return m or "."
    return "."


if __name__ == "__main__":
    main()
