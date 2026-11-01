#!/usr/bin/env python3
"""Offline W&B check: run a dummy offline run with the private settings and list every file it would upload."""
import json, shutil, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fimexp import sources, wandb_log  # noqa: E402

out = Path(tempfile.mkdtemp(prefix="wandb_offline_"))
try:
    cfg = sources.load_config()
    eff = {**cfg, "run": {"name": "offline-test", "optimizer_updates": 472}}
    run = wandb_log.init({"enabled": True, "mode": "offline", "project": "offline-test", "allowed_access": ["PRIVATE"]},
                         "offline-test", eff, out)
    print("CONSOLE-CANARY-7f3a: this line must not be captured")
    for i in range(3):
        run.log({"loss": 0.5 - i * 0.01, "grad_norm": 0.1}, step=i + 1)
    run.finish()
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    hits = {}
    for p in out.rglob("*"):
        if p.is_file():
            b = p.read_bytes()
            for needle in (b"CONSOLE-CANARY-7f3a", b"84caec5", b"github.com", b"def main", b"/home/vipul", b"api_key", b"WANDB_API_KEY"):
                if needle in b:
                    hits.setdefault(str(p.relative_to(out)), []).append(needle.decode())
    cfg_files = [p for p in out.rglob("config.yaml")]
    import collections
    from wandb.proto import wandb_internal_pb2 as pb

    def scan(path):
        """W&B transaction log: 7-byte file header, then LevelDB-style 32 KiB blocks of
        [crc32c(4) length(2) type(1)] chunks (1=full, 2=first, 3=middle, 4=last)."""
        data = Path(path).read_bytes()
        assert data[:4] == b":W&B", "unexpected file header"
        pos, block, buf = 7, 32768, b""
        while pos + 7 <= len(data):
            left = block - (pos % block)
            if left < 7:
                pos += left
                continue
            length = int.from_bytes(data[pos + 4:pos + 6], "little"); kind = data[pos + 6]
            if length == 0 and kind == 0:
                pos += left
                continue
            chunk = data[pos + 7:pos + 7 + length]; pos += 7 + length
            if kind == 1:
                yield chunk
            elif kind == 2:
                buf = chunk
            elif kind == 3:
                buf += chunk
            elif kind == 4:
                yield buf + chunk; buf = b""
    kinds, history_keys, config_keys = collections.Counter(), set(), set()
    report_tel = []
    for f in out.rglob("*.wandb"):
        for data in scan(f):
            rec = pb.Record(); rec.ParseFromString(data)
            kind = rec.WhichOneof("record_type"); kinds[kind] += 1
            if kind == "history":
                history_keys.update(i.key or "/".join(i.nested_key) for i in rec.history.item)
            if kind in ("config", "run"):
                items = rec.config.update if kind == "config" else rec.run.config.update
                config_keys.update(i.key for i in items)
            if kind == "telemetry":
                report_tel.append(str(rec.telemetry).replace("\n", " "))
            if kind == "files":
                kinds["files:" + ",".join(x.path for x in rec.files.files)] += 1
    report = {"files": files, "needle_hits": hits, "record_types": dict(kinds),
              "history_keys": sorted(history_keys), "config_keys": sorted(k for k in config_keys if not k.startswith("_")),
              "telemetry": report_tel}
    if cfg_files:
        import yaml
        report["config_top_keys"] = sorted(k for k in yaml.safe_load(cfg_files[0].read_text()) if not k.startswith("_"))
    report["settings"] = wandb_log.PRIVATE_SETTINGS
    (Path(__file__).parent / "wandb_offline.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
finally:
    shutil.rmtree(out, ignore_errors=True)
