#!/usr/bin/env python3
"""COUNT ONLY (CPU, git; writes a summary JSON, no training data): how many 'real edit' FIM holes the repo's
history BEFORE the cutoff would give, built exactly like the test set (extract_test.py) but on the training period:
  commits: first-parent commits up to and including the cutoff commit (all author+committer dates < cutoff);
  holes: contiguous added-line blocks of 1-30 lines with a non-blank line, in non-generated .rs files that are not
         new in that commit; the file at that commit gives prefix/suffix.
Also reported: blocks in the 5% validation files (would be excluded), blocks in crates absent from the cutoff
snapshot, the effect of a per-commit cap, size distribution vs the test set, and whether the hole text survives
unchanged into the cutoff snapshot (a measure of how much of it the AST/random runs already saw as plain code).
Usage: python count_edit_holes.py <repo_dir> <cutoff_iso> <split.json> <test_eval.jsonl> <out.json>
"""
import collections, json, statistics as st, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_test as X

X.LANG = "rust"
repo, cutoff, splitf, testf, out = sys.argv[1:6]
git = lambda *a: X.git(repo, *a)
ccut = git("rev-list", "-1", "--first-parent", f"--before={cutoff}", "HEAD").strip()
val = set(json.load(open(splitf))["files_by_split"]["validation"])
snap_files = set(git("ls-tree", "-r", "--name-only", ccut).split("\n"))
snap_crates = {X.module_of(repo, ccut, p) for p in snap_files if p.endswith(".rs")}
snap_blob = "\n".join(X.norm(git("show", f"{ccut}:{p}")) for p in snap_files if p.endswith(".rs"))
commits = git("rev-list", "--first-parent", "--reverse", ccut).split()
S = collections.Counter(); kept = []
for c in commits:
    S["commits"] += 1
    ps = git("rev-list", "--parents", "-n1", c).split()
    if len(ps) < 2:
        S["root_commit_skipped"] += 1; continue
    parent = ps[1]
    status = git("diff", "--name-status", "-M", parent, c, "--", "*.rs")
    added = {l.split("\t")[-1] for l in status.split("\n") if l.startswith("A\t")}
    diff = git("diff", "-U0", "-M", "--no-color", parent, c, "--", "*.rs")
    cur = None; blocks = []
    for line in diff.split("\n"):
        if line.startswith("+++ "):
            cur = None if line.startswith("+++ /dev/null") else line[6:].rstrip("\t")
        elif line.startswith("@@") and cur:
            import re
            m = re.match(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            n = int(m.group(3) if m.group(3) is not None else 1)
            if n > 0:
                blocks.append((cur, int(m.group(2)), n))
    cache = {}
    for path, start, n in blocks:
        S["blocks"] += 1
        if path in added:
            S["drop_new_file"] += 1; continue
        if path not in cache:
            try:
                cache[path] = git("show", f"{c}:{path}")
            except subprocess.CalledProcessError:
                cache[path] = None
        text = cache[path]
        if text is None or X.generated(path, text):
            S["drop_generated_or_unreadable"] += 1; continue
        if n > X.MAX_LINES:
            S["drop_over_30_lines"] += 1; continue
        hole = text.split("\n")[start - 1:start - 1 + n]
        if not any(l.strip() for l in hole):
            S["drop_blank"] += 1; continue
        crate = X.module_of(repo, c, path)
        kept.append({"commit": c, "path": path, "n": n, "in_val_file": path in val, "crate_at_cutoff": crate in snap_crates,
                     "survives": len(X.norm("\n".join(hole))) >= 40 and X.norm("\n".join(hole)) in snap_blob,
                     "is_test": X.is_test(path)})
usable = [k for k in kept if not k["in_val_file"] and k["crate_at_cutoff"]]
per_c = collections.Counter(k["commit"] for k in usable)
test_lines = [e["hole_lines"] for e in map(json.loads, open(testf)) if e.get("sensitive")]
bucket = lambda L: {k: round(v / len(L) * 100, 1) for k, v in sorted(collections.Counter(min(x, 11) for x in L).items())} if L else {}
res = {"cutoff": cutoff, "cutoff_commit": ccut, "stats": dict(S), "holes_before_filters": len(kept),
       "dropped_validation_file": sum(k["in_val_file"] for k in kept),
       "dropped_crate_not_in_snapshot": sum(not k["crate_at_cutoff"] and not k["in_val_file"] for k in kept),
       "usable_holes": len(usable), "commits_with_holes": len(per_c),
       "usable_if_cap_5_per_commit": sum(min(v, 5) for v in per_c.values()),
       "usable_if_cap_10_per_commit": sum(min(v, 10) for v in per_c.values()),
       "test_file_holes": sum(k["is_test"] for k in usable),
       "hole_text_survives_into_cutoff_snapshot_pct": round(100 * sum(k["survives"] for k in usable) / max(1, len(usable)), 1),
       "hole_lines": {"median": st.median([k["n"] for k in usable]) if usable else None,
                      "mean": round(st.mean([k["n"] for k in usable]), 1) if usable else None, "buckets_pct": bucket([k["n"] for k in usable])},
       "test_set_hole_lines": {"median": st.median(test_lines), "buckets_pct": bucket(test_lines)}}
Path(out).write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
