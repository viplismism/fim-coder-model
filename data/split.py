"""Reproducible crate-disjoint train/validation/test split of the reth snapshot."""

import random


def crate_split(files: list[dict], fractions: dict, seed: int, max_heldout_crate_fraction: float | None = None) -> dict:
    """Assign whole crates to splits, targeting `fractions` of source bytes.

    Crates are visited in a seeded random order; each goes to the split furthest below its
    byte target (relative deficit). Crates larger than `max_heldout_crate_fraction` of all bytes are
    placed in train first, so no single crate dominates validation or test. Files without a crate are
    excluded and returned separately.
    """
    sizes = {}
    for f in files:
        if f["crate"] is not None:
            sizes[f["crate"]] = sizes.get(f["crate"], 0) + f["bytes"]
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    total = sum(sizes.values())
    target = {s: frac * total for s, frac in fractions.items()}
    got = {s: 0 for s in fractions}
    crate_to_split = {}
    if max_heldout_crate_fraction is not None:
        for c in [c for c in order if sizes[c] > max_heldout_crate_fraction * total]:
            crate_to_split[c] = "train"
            got["train"] += sizes[c]
        order = [c for c in order if c not in crate_to_split]
    for c in order:
        s = max(fractions, key=lambda s: (target[s] - got[s]) / target[s])
        crate_to_split[c] = s
        got[s] += sizes[c]
    return {
        "crate_to_split": dict(sorted(crate_to_split.items())),
        "excluded_files": sorted(f["path"] for f in files if f["crate"] is None),
        "bytes": got,
        "fraction_of_bytes": {s: got[s] / total for s in got},
        "forced_to_train": sorted(c for c in sizes if max_heldout_crate_fraction is not None and sizes[c] > max_heldout_crate_fraction * total),
        "largest_crate_share": {s: max([sizes[c] / got[s] for c, v in crate_to_split.items() if v == s], default=0) for s in fractions},
        "crates": {s: sum(1 for v in crate_to_split.values() if v == s) for s in fractions},
        "files": {s: sum(1 for f in files if f["crate"] is not None and crate_to_split[f["crate"]] == s) for s in fractions},
    }


def package_count_split(files: list[dict], fractions: dict, seed: int) -> dict:
    """Shuffle all packages once with `seed`, then allocate by package COUNT:
    train = floor(f_train * N), validation = floor(f_validation * N), test = the rest. Bytes are not balanced."""
    sizes, nfiles = {}, {}
    for f in files:
        if f["crate"] is not None:
            sizes[f["crate"]] = sizes.get(f["crate"], 0) + f["bytes"]
            nfiles[f["crate"]] = nfiles.get(f["crate"], 0) + 1
    order = sorted(sizes)
    random.Random(seed).shuffle(order)
    n = len(order)
    n_train, n_val = int(fractions["train"] * n), int(fractions["validation"] * n)
    groups = {"train": order[:n_train], "validation": order[n_train:n_train + n_val], "test": order[n_train + n_val:]}
    crate_to_split = {c: s for s, cs in groups.items() for c in cs}
    total = sum(sizes.values())
    by = {s: sum(sizes[c] for c in cs) for s, cs in groups.items()}
    return {
        "rule": "shuffle sorted package dirs once with seed; train=floor(0.8N), validation=floor(0.1N), test=rest",
        "crate_to_split": dict(sorted(crate_to_split.items())),
        "shuffled_order": order,
        "excluded_files": sorted(f["path"] for f in files if f["crate"] is None),
        "bytes": by, "fraction_of_bytes": {s: by[s] / total for s in by},
        "crates": {s: len(cs) for s, cs in groups.items()},
        "files": {s: sum(nfiles[c] for c in cs) for s, cs in groups.items()},
        "largest_crate_share": {s: max(sizes[c] for c in cs) / by[s] for s, cs in groups.items()},
    }
