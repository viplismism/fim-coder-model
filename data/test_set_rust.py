#!/usr/bin/env python3
"""Rust post-filter for data.test_set output (packaged from the step used for facet; same logic):
  1. keep candidates whose crate (nearest Cargo.toml with [package]) already existed at the cutoff commit, so the
     crate can be built and tested with `cargo -p <name>` (the crate name is attached as crate_name);
  2. at most 5 examples per commit, seeded (20261120), so one big commit cannot dominate.
(For Go the per-commit cap is applied inside eval.exec_go instead.)
Usage: python -m data.test_set_rust <repo_dir> <raw.jsonl> <out.jsonl>
  reads <raw>.summary.json next to <raw.jsonl>; writes <out.jsonl> and <out>.summary.json
"""
import collections, json, random, re, subprocess, sys
from pathlib import Path

R, raw, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
s = json.load(open(raw.with_suffix(".summary.json"))); C = s['cutoff_commit']
names = subprocess.run(['git', '-C', R, 'ls-tree', '-r', '--name-only', C], capture_output=True, text=True).stdout.split('\n')
crates = {}
for p in names:
    if p.endswith('Cargo.toml') and '/fixtures/' not in '/' + p:
        d = p[:-10].rstrip('/'); t = subprocess.run(['git', '-C', R, 'show', f'{C}:{p}'], capture_output=True, text=True).stdout
        if '[package]' in t:
            m = re.search(r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', t, re.M | re.S); crates[d] = m.group(1) if m else None
ex = [json.loads(l) for l in open(raw)]
keep = [e for e in ex if e['module'] in crates]
by = collections.defaultdict(list)
for e in keep: by[e['commit']].append(e)
rng = random.Random(20261120); sel = []
for c in sorted(by): sel += sorted(rng.sample(by[c], min(5, len(by[c]))), key=lambda e: e['id'])
for e in sel: e['crate_name'] = crates[e['module']]
out.write_text(''.join(json.dumps(e) + '\n' for e in sel))
json.dump({"cutoff": s["cutoff"][:10], "cutoff_commit": C, "crates_at_cutoff": crates, "raw_candidates": len(ex), "in_cutoff_crates": len(keep),
           "commits": len(by), "capped_5_per_commit": len(sel), "test_files": sum(e['is_test_file'] for e in sel),
           "ai_tagged": sum(e['ai_tagged'] for e in sel), "bot_author": sum(e['bot_author'] for e in sel), "raw_stats": s['stats']},
          open(out.with_suffix(".summary.json"), 'w'), indent=1)
print(C, len(crates), 'crates;', len(ex), '->', len(keep), '->', len(sel), 'capped; commits', len(by))
