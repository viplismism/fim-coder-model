"""Rust regression: the current hole-sampling code must reproduce every hole of the existing rust-fim-v2 data exactly.
Skipped if the local data/sources are not present (they are not committed)."""

import json
from pathlib import Path

import pytest

from fimexp import sources
from fimexp.holes import n_examples_for, parser, sample_file

CFG = sources.load_config()
DATA = sources.ROOT / "data" / "rust-fim-v2"
pytestmark = pytest.mark.skipif(not (DATA / "train_ast.jsonl").exists(), reason="rust-fim-v2 data not present locally")


@pytest.mark.parametrize("split", ["train", "validation"])
def test_every_rust_hole_is_reproduced(split):
    reth = sources.ROOT / CFG["sources"]["reth"]["path"]
    exempt = set(CFG["data"]["keyword_text_exempt_types"])
    want = {}
    for arm in ("ast", "random"):
        for line in open(DATA / f"{split}_{arm}.jsonl"):
            r = json.loads(line)
            want.setdefault(r["path"], {})[(arm, int(r["pair_id"].rsplit("#", 1)[1]))] = tuple(r["hole_chars"])
    p, checked = parser(), 0
    for path, holes in sorted(want.items()):
        text = sources.read_source(reth / path)
        tree = p.parse(text.encode())
        got = {q["k"]: q for q in sample_file(path, text, tree, n_examples_for(len(text.encode()), CFG["data"]["bytes_per_example"]),
                                              CFG["data"]["seed"], exempt)}
        for (arm, k), hc in holes.items():
            assert tuple(got[k][arm]) == hc, (path, arm, k)
            checked += 1
    assert checked == sum(len(v) for v in want.values()) > 0
