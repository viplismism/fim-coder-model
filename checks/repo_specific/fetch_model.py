#!/usr/bin/env python3
"""Resumable, verified model download to models/<name>/ (the HF client in this venv restarts files from zero).
Reuses partial files already in the HF cache (they are byte-prefixes of the same LFS objects, named by sha256),
downloads with `curl -C -` and retries, 4 files in parallel, then checks every file's sha256 (LFS) or size.
Token is read from ~/.cache/huggingface/token and passed to curl through a 0600 header file (never on argv).
Usage: python fetch_model.py <repo_id> <revision>
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
repo, rev = sys.argv[1], sys.argv[2]
dest = ROOT / "models" / repo.split("/")[1]; dest.mkdir(parents=True, exist_ok=True)
info = HfApi().model_info(repo, revision=rev, files_metadata=True)
want = [s for s in info.siblings if s.rfilename.endswith((".json", ".safetensors", ".txt")) and "/" not in s.rfilename]
blobs = Path.home() / ".cache/huggingface/hub" / f"models--{repo.replace('/', '--')}" / "blobs"
hdr = Path(tempfile.mkstemp(prefix="hfh_")[1]); os.chmod(hdr, 0o600)
hdr.write_text(f"Authorization: Bearer {(Path.home() / '.cache/huggingface/token').read_text().strip()}\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def get(s):
    out = dest / s.rfilename; sha = s.lfs.sha256 if s.lfs else None; size = s.size
    if out.exists() and out.stat().st_size == size and (sha is None or sha256(out) == sha):
        return s.rfilename, "already ok"
    if sha and blobs.exists() and not out.exists():
        parts = sorted(blobs.glob(f"{sha}*.incomplete"), key=lambda p: p.stat().st_size)
        if parts and parts[-1].stat().st_size < size:
            shutil.copyfile(parts[-1], out)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{s.rfilename}"
    for attempt in range(200):
        if out.exists() and out.stat().st_size == size:
            break
        subprocess.run(["curl", "-sSL", "-H", f"@{hdr}", "-C", "-", "--retry", "20", "--retry-all-errors",
                        "--speed-time", "120", "--speed-limit", "1000", "-o", str(out), url])
    ok = out.exists() and out.stat().st_size == size and (sha is None or sha256(out) == sha)
    return s.rfilename, "ok" if ok else "FAILED"


try:
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(get, want))
finally:
    hdr.unlink(missing_ok=True)
(dest / "DOWNLOAD.json").write_text(json.dumps({"repo": repo, "revision": rev, "files": dict(res)}, indent=1) + "\n")
print(json.dumps(dict(res), indent=1))
sys.exit(0 if all(v != "FAILED" for v in dict(res).values()) else 1)
