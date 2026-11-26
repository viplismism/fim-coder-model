#!/usr/bin/env python3
"""Execution check of the test candidates (CPU only, no models).

1. Seeded cap: at most 5 examples per commit (one large commit must not dominate).
2. For each kept example, at its commit (fresh detached worktree):
   - reference: the real file -> `go build ./...` in the module, then `go test -count=1 <package>`.
     The example is usable only if BOTH pass (otherwise build/test cannot judge a completion).
   - sensitivity: the hole replaced by nothing -> same commands. "sensitive" = build or tests fail, i.e. the
     execution check can tell a missing/wrong completion from the right one for this example.
Usage: python -m eval.exec_go <repo_dir> <candidates.jsonl> <out.jsonl> [workers]
"""
import collections, json, os, random, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

P = Path(__file__).resolve().parents[1]  # repo root
TMP = P / "work" / "tmp"; TMP.mkdir(parents=True, exist_ok=True)  # git worktrees for build+test runs
_ARCH = {"aarch64": "arm64", "x86_64": "amd64"}.get(__import__("platform").machine(), "amd64")
T = P / "work" / "toolchains"  # installed by scripts/fetch.sh toolchains
ENV = {**os.environ, "PATH": f"{T}/go/bin:" + os.environ["PATH"], "GOPATH": f"{T}/gopath",
       "GOCACHE": f"{P}/work/cache/go", "GOTOOLCHAIN": "local",
       "KUBEBUILDER_ASSETS": f"{T}/envtest/k8s/1.32.0-linux-{_ARCH}"}
CAP, SEED = 5, 20261117


def run(cmd, cwd, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, (r.stdout + r.stderr)[-1500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def check(mod_dir, pkg):
    ok_b, log_b = run(["go", "build", "./..."], mod_dir, 900)
    if not ok_b:
        return {"build": False, "test": False, "log": log_b}
    ok_t, log_t = run(["go", "test", "-count=1", pkg], mod_dir, 900)
    return {"build": True, "test": ok_t, "log": "" if ok_t else log_t}


def do_commit(repo, commit, exs):
    wt = Path(tempfile.mkdtemp(prefix=f"exec_{commit[:10]}_", dir=TMP))
    shutil.rmtree(wt)
    subprocess.run(["git", "-C", repo, "worktree", "add", "-q", "--detach", str(wt), commit], check=True)
    res = []
    try:
        for e in exs:
            mod_dir = wt / (e["module"] if e["module"] != "." else "")
            pkg = "./" + os.path.relpath(os.path.dirname(wt / e["path"]), mod_dir)
            f = wt / e["path"]; orig = f.read_text()
            assert orig == e["prefix"] + e["middle"] + e["suffix"]
            ref = check(mod_dir, pkg)
            sens = None
            if ref["build"] and ref["test"]:
                f.write_text(e["prefix"] + e["suffix"])
                s = check(mod_dir, pkg)
                f.write_text(orig)
                sens = not (s["build"] and s["test"])
            res.append({"id": e["id"], "commit": commit, "module": e["module"], "package": pkg, "path": e["path"],
                        "is_test_file": e["is_test_file"], "hole_lines": e["hole_lines"],
                        "ref_build": ref["build"], "ref_test": ref["test"], "ref_log": ref["log"][-600:],
                        "sensitive": sens})
            print(json.dumps({k: res[-1][k] for k in ("id", "ref_build", "ref_test", "sensitive")}), flush=True)
    finally:
        subprocess.run(["git", "-C", repo, "worktree", "remove", "--force", str(wt)])
    return res


def main():
    repo, cand, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    ex = [json.loads(l) for l in open(cand)]
    by_c = collections.defaultdict(list)
    for e in ex:
        by_c[e["commit"]].append(e)
    rng = random.Random(SEED)
    sel = {c: sorted(rng.sample(v, min(CAP, len(v))), key=lambda e: e["id"]) for c, v in sorted(by_c.items())}
    done = {}
    if out.exists():
        for l in open(out):
            r = json.loads(l); done[r["id"]] = r
    todo = {c: [e for e in v if e["id"] not in done] for c, v in sel.items()}
    todo = {c: v for c, v in todo.items() if v}
    with open(out, "a") as fo, ThreadPoolExecutor(workers) as pool:
        for rs in pool.map(lambda kv: do_commit(repo, *kv), todo.items()):
            for r in rs:
                fo.write(json.dumps(r) + "\n"); fo.flush()


if __name__ == "__main__":
    main()
