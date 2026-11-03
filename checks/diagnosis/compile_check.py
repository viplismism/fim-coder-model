#!/usr/bin/env python3
"""cargo check of completions spliced into a writable copy of the pinned reth checkout (validation only).

For a seeded, package-stratified subset of validation pairs whose package is checkable (workspace member, no LLVM
JIT dependency), each model's completion is spliced into the full file, `cargo check --all-targets -p <package>`
is run, and the original file is restored. Exact matches inherit the baseline result (all packages pass
unmodified); identical completions are checked once.
Usage: python checks/diagnosis/compile_check.py <gen_dir> <n_pairs>
"""

import collections, json, os, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fimexp import sources  # noqa: E402

SCRATCH = ROOT / "scratch" / "reth-check"
ENV = {**os.environ, "BINDGEN_EXTRA_CLANG_ARGS": "-I/usr/lib/gcc/aarch64-linux-gnu/13/include"}
CARGO = str(Path.home() / ".cargo" / "bin" / "cargo")


def main():
    gen = Path(sys.argv[1]); n_pairs = int(sys.argv[2])
    cfg = sources.load_config(); reth = ROOT / cfg["sources"]["reth"]["path"]
    pkgs = {c: n for c, n, _ in json.load(open(ROOT / "scratch" / "val_packages.json"))}
    checkable = {c: n for c, n in pkgs.items() if n not in ("remote-exex", "tracking-state", "reth")}
    items = [json.loads(l) for l in open(gen / "items.jsonl")]
    outs = {m: {r["id"]: r for r in map(json.loads, open(gen / f"{m}.jsonl"))} for m in ("base", "ast", "random")}
    by_pkg = collections.defaultdict(set)
    for it in items:
        if it["crate"] in checkable:
            by_pkg[it["crate"]].add(it["pair_id"])
    rng = random.Random(20261103)
    order = {p: sorted(v) for p, v in sorted(by_pkg.items())}
    for v in order.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < n_pairs and any(order.values()):  # round-robin across packages
        for p in list(order):
            if order[p] and len(chosen) < n_pairs:
                chosen.append(order[p].pop())
    chosen = set(chosen)
    sel = [it for it in items if it["pair_id"] in chosen]
    out_path = gen / "compile.jsonl"
    done = {json.loads(l)["key"] for l in open(out_path)} if out_path.exists() else set()
    cache = {json.loads(l)["key"]: json.loads(l) for l in open(out_path)} if out_path.exists() else {}
    t0 = time.time(); ran = 0
    with open(out_path, "a") as f:
        for it in sel:
            original = sources.read_source(reth / it["path"])
            hs, he = it["hole_chars"]
            for m in ("base", "ast", "random"):
                text = outs[m][it["id"]]["text"]
                key = f"{it['id']}|{m}"
                if key in done:
                    continue
                if text == it["middle"]:
                    rec = {"key": key, "id": it["id"], "model": m, "result": "pass", "how": "exact match = baseline"}
                else:
                    twin = next((cache[k] for k in cache if cache[k]["id"] == it["id"] and cache[k].get("text") == text), None)
                    if twin:
                        rec = {"key": key, "id": it["id"], "model": m, "result": twin["result"], "how": "same completion as " + twin["model"]}
                    else:
                        target = SCRATCH / it["path"]
                        target.write_text(original[:hs] + text + original[he:])
                        try:
                            r = subprocess.run([CARGO, "check", "--locked", "--offline", "--all-targets", "-j", "8", "-p", checkable[it["crate"]],
                                                "--message-format", "short"], cwd=SCRATCH, env=ENV, capture_output=True, text=True, timeout=600)
                            result = "pass" if r.returncode == 0 else "fail"
                            err = next((l for l in r.stderr.splitlines() if l.startswith("error")), "")[:200]
                        except subprocess.TimeoutExpired:
                            result, err = "timeout", ""
                        finally:
                            target.write_text(original)
                        rec = {"key": key, "id": it["id"], "model": m, "result": result, "how": "cargo check", "first_error": err}
                        ran += 1
                rec["text"] = text
                cache[key] = rec
                f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{len(cache)}/{3 * len(sel)} done, {ran} cargo runs, {time.time() - t0:.0f}s", flush=True)
    summary = {"pairs_selected": len(chosen), "items": len(sel), "packages": sorted({it["crate"] for it in sel}),
               "eligible_pairs": sum(len(v) for v in by_pkg.values()), "all_pairs_in_generation": len({it["pair_id"] for it in items}),
               "not_checkable_packages": sorted(set(pkgs) - set(checkable)), "cargo_runs": ran}
    (gen / "compile_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
