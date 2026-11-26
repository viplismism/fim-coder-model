#!/usr/bin/env python3
"""Rust execution check of test candidates (CPU only, no models). Same logic as exec_check.py (Go):
  reference: the real file at its commit -> `cargo build -p <crate> --all-targets`, then `cargo test -p <crate>`;
             usable only if both pass.
  sensitivity: the hole replaced by nothing -> same commands; sensitive = build or tests fail.
Commits are processed in a seeded random order (a random sample if stopped early). Each worker keeps its own
CARGO_TARGET_DIR, so rebuilds across commits are incremental. Resumable (skips ids already in <out>).
Usage: python -m eval.exec_rust <repo_dir> <candidates.jsonl> <out.jsonl> [workers]
"""
import collections, json, os, random, shutil, subprocess, sys, tempfile, threading
from pathlib import Path

P = Path(__file__).resolve().parents[1]  # repo root
TMP = P / "work" / "tmp"; TMP.mkdir(parents=True, exist_ok=True)  # git worktrees for build+test runs
SEED = 20261120
LOCK = threading.Lock()


def env(worker):
    e = dict(os.environ)
    e.update(PATH=f"{Path.home()}/.cargo/bin:" + e["PATH"], CARGO_TARGET_DIR=str(P / "work/cache/cargo_target" / f"exec_w{worker}"),
             CARGO_TERM_COLOR="never", CARGO_BUILD_JOBS="5", CARGO_NET_RETRY="10")
    return e


def run(cmd, cwd, worker, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env(worker), capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def check(root, crate, worker):
    ok_b, log_b = run(["cargo", "build", "-p", crate, "--all-targets"], root, worker, 3600)
    if not ok_b:
        return {"build": False, "test": False, "log": log_b}
    ok_t, log_t = run(["cargo", "test", "-p", crate], root, worker, 1800)
    return {"build": True, "test": ok_t, "log": "" if ok_t else log_t}


def do_commit(repo, commit, exs, worker):
    wt = Path(tempfile.mkdtemp(prefix=f"rexec_{commit[:10]}_", dir=TMP)); shutil.rmtree(wt)
    with LOCK:
        subprocess.run(["git", "-C", repo, "worktree", "add", "-q", "--detach", str(wt), commit], check=True)
    res = []
    try:
        for e in exs:
            f = wt / e["path"]; orig = f.read_text()
            assert orig == e["prefix"] + e["middle"] + e["suffix"]
            ref = check(wt, e["crate_name"], worker)
            sens = None
            if ref["build"] and ref["test"]:
                f.write_text(e["prefix"] + e["suffix"])
                s = check(wt, e["crate_name"], worker)
                f.write_text(orig)
                sens = not (s["build"] and s["test"])
            res.append({"id": e["id"], "commit": commit, "crate": e["crate_name"], "path": e["path"], "is_test_file": e["is_test_file"],
                        "ai_tagged": e["ai_tagged"], "bot_author": e["bot_author"], "hole_lines": e["hole_lines"],
                        "ref_build": ref["build"], "ref_test": ref["test"], "ref_log": ref["log"][-600:], "sensitive": sens})
    finally:
        with LOCK:
            subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", str(wt)])
    return res


def main():
    repo, cand, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    by_c = collections.defaultdict(list)
    for e in map(json.loads, open(cand)):
        by_c[e["commit"]].append(e)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    order = sorted(by_c); random.Random(SEED).shuffle(order)
    queue = [(c, [e for e in by_c[c] if e["id"] not in done]) for c in order]
    queue = [q for q in queue if q[1]]
    qlock = threading.Lock(); olock = threading.Lock()

    def worker(w):
        while True:
            with qlock:
                if not queue:
                    return
                c, exs = queue.pop(0)
            for r in do_commit(repo, c, exs, w):
                with olock, open(out, "a") as fo:
                    fo.write(json.dumps(r) + "\n")
                print(json.dumps({k: r[k] for k in ("id", "ref_build", "ref_test", "sensitive")}), flush=True)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    [t.start() for t in ts]; [t.join() for t in ts]


if __name__ == "__main__":
    main()
