#!/usr/bin/env python3
"""Training data for the repo-specific run: the repo at the cutoff commit only, random holes (same sampler, hole
sizes and length rules as the earlier runs; the AST arm is built alongside for the record but not trained).
Files: non-generated .go files of the cutoff snapshot (same generated-file rule as the test set, plus the earlier
builder's own rules). Split: 5% of files (seeded) held out as validation; the rest train. Pairs capped at 7,549
train / 518 validation; fewer are reported, never padded.
Usage: python build_repo_data.py <config.yaml> <snapshot_dir> <cutoff_commit> <version> [repo] [lang] [train_cap]
Hole density comes from data.bytes_per_example in the config (examples per file = ceil(bytes / it)).
"""
import collections, json, random, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "checks" / "repo_specific"))
import yaml  # noqa: E402
import extract_test  # noqa: E402
from extract_test import generated  # noqa: E402
from fimexp import build_lang_data as B, sources  # noqa: E402
from fimexp.model import load_tokenizer  # noqa: E402
from fimexp.psm import PSM  # noqa: E402

SEED = 20261117


def main():
    cfgp, snap, commit, version = sys.argv[1:5]
    REPO = sys.argv[5] if len(sys.argv) > 5 else "telekom/controlplane"
    lang = sys.argv[6] if len(sys.argv) > 6 else "go"
    extract_test.LANG = lang
    train_cap = int(sys.argv[7]) if len(sys.argv) > 7 else 7549
    cfg = yaml.safe_load(open(cfgp)); dcfg = cfg["data"]
    out = ROOT / dcfg["output_dir"] / version
    if out.exists():
        sys.exit(f"{out} exists")
    assert sources.git_head(ROOT / snap) == commit, "snapshot is not at the cutoff commit"
    pins = {"repos": {REPO: {"path": snap, "commit": commit, "kind": "train"}}}
    files, excl = B.eligible_files(lang, pins, {"train"})
    gen = [f for f in files if generated(f[1], f[2])]
    files = [f for f in files if not generated(f[1], f[2])]
    excl["generated (test-set rule: zz_generated, pb, mocks, 'Code generated ... DO NOT EDIT')"] += len(gen)
    rels = sorted(f[1] for f in files)
    rng = random.Random(SEED)
    val = set(rng.sample(rels, max(1, round(0.05 * len(rels)))))
    psm = PSM(load_tokenizer(cfg), cfg["format"])
    t0 = time.time(); out.mkdir(parents=True)
    res = {}
    for name, subset, cap in (("train", [f for f in files if f[1] not in val], train_cap), ("validation", [f for f in files if f[1] in val], 518)):
        by_repo, drops, perr = B.build_pairs(lang, subset, psm, dcfg, lambda r, n=name: n)
        pairs = by_repo.get(REPO, [])
        chosen = rng.sample(pairs, min(cap, len(pairs)))
        B.write_split(out, name, chosen)
        res[name] = B.stats(chosen) | {"files": len(subset), "available_pairs": len(pairs), "cap": cap,
                                       "dropped": dict(drops), "parse_error_files": perr}
    (out / "split.json").write_text(json.dumps({"rule": "5% of snapshot files (seeded) are validation", "seed": SEED,
                                                "files_by_split": {"train": sorted(set(rels) - val), "validation": sorted(val)}}, indent=1) + "\n")
    man = {"version": version, "lang": lang, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "recipe": "paired AST holes + random holes of identical length (both arms written)", "repo": REPO, "cutoff_commit": commit,
           "tokenizer": cfg["model"], "file_exclusions": dict(excl), "splits": res, "pins": pins,
           "data_settings": {k: dcfg[k] for k in ("seed", "bytes_per_example", "max_seq_tokens")}, "seconds": round(time.time() - t0, 1)}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("files", "available_pairs", "pairs", "middle_tokens")} for k, v in res.items()}, indent=1))
    print("exclusions", dict(excl))


if __name__ == "__main__":
    main()
