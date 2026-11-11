#!/usr/bin/env bash
# Python pilot chain: replacements -> fetch -> build data -> overlap checks (gate) -> pilot (train AST, train random,
# validation scoring). Stops for review if a check fails. Final evaluation is NOT part of this script.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_python_prep_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
trap 'log "FAILED at: ${STEP:-startup} (exit $?)"' ERR
STEP="replacements"; log "$STEP"
python3 checks/multilang/python_replacements.py | tee -a "$LOG"
python3 - <<'PYX'
import json
l = json.load(open("checks/multilang/python_list.json"))
r = json.load(open("checks/multilang/python_replacements.json"))["picked"]
assert len(r) == 2, "could not find 2 replacements"
l["train"] = l["train"] + [p["repo"] for p in r if p["repo"] not in l["train"]]
assert len(l["train"]) == 12 and len(l["heldout"]) == 10
json.dump(l, open("checks/multilang/python_list.json", "w"), indent=1)
print("final python list:", l)
PYX
STEP="fetch"; log "$STEP"
$PY checks/multilang/fetch_repos.py python checks/multilang/python_list.json | tee -a "$LOG"
STEP="build data"; log "$STEP"
HF_HUB_OFFLINE=1 $PY -m fimexp.build_lang_data --lang python --pins checks/multilang/python_pins.json --version python-fim-v1 2>&1 | grep -v "longer than" | tee -a "$LOG"
STEP="overlap check"; log "$STEP"
$PY checks/multilang/overlap_lang.py python python-fim-v1 2>/dev/null | tee -a "$LOG"
STEP="overlap gate"
$PY - <<'PYX' | tee -a "$LOG"
import json, sys
r = json.load(open("checks/multilang/python_overlap_report.json"))
v, w, pc = r["minhash_verified"], r["exact_5line_windows"], r["positive_control"]
problems = []
if v["pool_vs_benchmark"]: problems.append(f"{v['pool_vs_benchmark']} verified near-duplicates of benchmark code")
if v["pool_vs_heldout"]: problems.append(f"{v['pool_vs_heldout']} verified near-duplicates of held-out code")
if pc["found"] < 15: problems.append(f"positive control found only {pc['found']}/{pc['planted']}")
if w["shared_with_benchmark"] > 0.005 * w["pool_windows"]: problems.append(f"{w['shared_with_benchmark']} exact windows shared with benchmark")
if problems:
    print("OVERLAP GATE: STOP FOR REVIEW:", "; ".join(problems)); sys.exit(3)
print("OVERLAP GATE: clean (train-vs-validation near-duplicates:", v["train_vs_validation"], "reported, not gating)")
PYX
STEP="pilot"; log "overlap clean; starting the pilot (train AST, then random, then validation scoring)"
bash scripts/run_lang_pilot.sh python python-fim-v1 configs/lang_python.yaml
log "chain finished"
