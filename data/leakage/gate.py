#!/usr/bin/env python3
"""Leakage gate for one language of the multi-language experiment: READY only if the built data has the planned pair
counts and every decontamination / overlap / train-vs-validation check is clean; otherwise STOP with the reasons.
Reads the reports in results/multilang/leakage/ and writes results/multilang/leakage/<lang>_gate.json.
Usage: python -m data.leakage.gate <lang> <data_version> <heldout_version>
"""
import json, sys
L, FINAL, HELD = sys.argv[1:]; M = "results/multilang/leakage"
P = []
man = json.load(open(f"work/datasets/{FINAL}/manifest.json")); hman = json.load(open(f"work/datasets/{HELD}/manifest.json"))
tr, va = man["splits"]["train"]["pairs"], man["splits"]["validation"]["pairs"]
if tr != 7549: P.append(f"train pairs {tr} != 7549 (shortfall, not padded)")
if va != 518: P.append(f"validation pairs {va} != 518 (shortfall, not padded)")
hp = hman["stats"]["pairs"]
if hp < 900: P.append(f"held-out pairs {hp} < 900")
full = json.load(open(f"{M}/{L}_decontamination.json"))
wf = full["planted_summary"]["whole file"]
if wf["caught"] < 18: P.append(f"planted whole-file copies caught {wf['caught']}/{wf['planted']} < 18")
chk = json.load(open(f"{M}/{L}_decontamination_check.json"))
if chk["removed_files"]: P.append(f"{len(chk['removed_files'])} files still match benchmark/held-out after rebuild")
tv = json.load(open(f"{M}/{L}_train_vs_validation.json"))
if tv["removed_validation_files"]: P.append(f"{len(tv['removed_validation_files'])} validation files still share code with training")
ov = json.load(open(f"{M}/{L}_overlap_report.json"))
# raw MinHash matches (no licence stripping) are reviewed with licence headers removed; any match that is not
# explained by a licence header stops the language (same standard as the Python gate).
rv0 = json.load(open(f"{M}/{L}_overlap_review.json"))
bad = [m for m in rv0["minhash_matches"] if "REVIEW" in m["verdict"]]
if bad: P.append(f"{len(bad)} MinHash matches not explained by licence headers (max J after removal {rv0['max_jaccard_after_licence_removed']})")
if ov["positive_control"]["found"] < 15: P.append(f"overlap report planted controls found {ov['positive_control']['found']}/20")
rv = json.load(open(f"{M}/{L}_overlap_review.json"))
for k, w in rv["exact_windows"].items():
    if w["other"]: P.append(f"{w['other']} non-licence exact windows shared with {k}")
gate = {"lang": L, "status": "STOP" if P else "READY", "problems": P, "data_version": FINAL, "heldout_version": HELD,
        "pairs": {"train": tr, "validation": va, "heldout": hp}, "decontamination_removed_files": len(full["removed_files"]),
        "licence_header_exceptions": len(full["licence_header_exceptions"]), "planted": full["planted_summary"],
        "raw_minhash_matches": ov["minhash_verified"], "train_repos": json.load(open(f"work/datasets/{FINAL}/split.json"))["train_repos"],
        "validation_repos": json.load(open(f"work/datasets/{FINAL}/split.json"))["validation_repos"]}
json.dump(gate, open(f"{M}/{L}_gate.json", "w"), indent=1)
print(json.dumps(gate, indent=1))
