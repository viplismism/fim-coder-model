"""Config loading and pinned-source access (reth snapshot + Real-FIM-Eval)."""

import gzip, hashlib, json, subprocess, tomllib
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None) -> dict:
    with open(path or ROOT / "configs" / "experiment.yaml") as f:
        return yaml.safe_load(f)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def git_dirty(path) -> bool:
    return bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip())


def verify_pins(cfg: dict) -> dict:
    """Fail unless the local reth checkout and benchmark files match the pins. Returns what was verified."""
    reth, bench = cfg["sources"]["reth"], cfg["sources"]["benchmark"]
    reth_path = ROOT / reth["path"]
    head = git_head(reth_path)
    if head != reth["commit"]:
        raise RuntimeError(f"reth checkout at {head}, pinned {reth['commit']}")
    if git_dirty(reth_path):
        raise RuntimeError("reth checkout has local modifications")
    sums = {}
    for split, want in bench["sha256"].items():
        got = sha256_file(ROOT / bench["path"] / split / "test.jsonl.gz")
        if got != want:
            raise RuntimeError(f"benchmark {split} sha256 {got} != pinned {want}")
        sums[split] = got
    return {"reth_commit": head, "benchmark_revision": bench["revision"], "benchmark_sha256": sums}


def read_source(path: Path) -> str:
    """Exact file text: bytes decoded as strict UTF-8, no newline translation."""
    return path.read_bytes().decode("utf-8")


def reth_crates(cfg: dict) -> dict:
    """Map crate directory (relative) -> package name for every Cargo.toml with a [package]."""
    root = ROOT / cfg["sources"]["reth"]["path"]
    crates = {}
    for ct in sorted(root.rglob("Cargo.toml")):
        d = tomllib.loads(ct.read_text())
        if "package" in d:
            crates[ct.parent.relative_to(root).as_posix()] = d["package"]["name"]
    return crates


def reth_files(cfg: dict) -> list[dict]:
    """Every .rs file at the pinned commit with its crate (None if not inside a package)."""
    root = ROOT / cfg["sources"]["reth"]["path"]
    crates = reth_crates(cfg)
    out = []
    for p in sorted(root.rglob("*.rs")):
        rel = p.relative_to(root).as_posix()
        crate = next((parent.as_posix() for parent in Path(rel).parents if parent.as_posix() in crates), None)
        out.append({"path": rel, "crate": crate, "bytes": p.stat().st_size})
    return out


@lru_cache(maxsize=None)
def _load_benchmark(path: str, lang: str) -> tuple:
    rows = []
    for split in ("add", "edit"):
        for i, line in enumerate(gzip.open(Path(path) / split / "test.jsonl.gz", "rt", encoding="utf-8")):
            r = json.loads(line)
            if r["lang"] == lang:
                rows.append({**r, "split": split, "index": i})
    return tuple(rows)


def benchmark_rows(cfg: dict, lang: str | None = None) -> list[dict]:
    """Real-FIM-Eval rows of one language (both splits); `index` is the line number in the split file.
    `lang` defaults to the config's language (rust, as in the original experiment). For languages with a
    benchmark_extensions filter in fimexp.langs (C++: the .c files are excluded), only those files are kept."""
    b = cfg["sources"]["benchmark"]
    lang = lang or b["lang"]
    rows = list(_load_benchmark(str(ROOT / b["path"]), lang))
    if lang != b["lang"] or lang == "cpp":
        from .langs import LANGS
        exts = next((c["benchmark_extensions"] for c in LANGS.values() if c["real_fim_eval"] == lang), None)
        if exts:
            rows = [r for r in rows if r["path"].endswith(exts)]
    return rows
