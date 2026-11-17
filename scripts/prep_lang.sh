#!/usr/bin/env bash
# CPU-only data preparation for one language, same rules as Python (checks/multilang/DECONTAMINATION.md incl. the
# train-vs-validation addendum), ending in an automatic gate. Never touches the GPU, never modifies test sets.
#   repo list -> fetch + pin -> file-level decontamination vs Real-FIM-Eval + held-out -> build <lang>-fim-v1
#   -> train-vs-validation check (if any: remove those validation files, rebuild <lang>-fim-v2 reusing the held-out)
#   -> rerun every check on the final version -> gate: checks/multilang/<lang>_gate.json (READY or STOP + reasons)
# Usage: bash scripts/prep_lang.sh <lang>
set -uo pipefail
cd "$(dirname "$0")/.."
L=$1; PY=.venv/bin/python; M=checks/multilang
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LOG=logs/$(date +%Y%m%dT%H%M%S%z)_${L}_prep_pipeline.log
log() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$LOG"; }
stop() { log "STOP: $*"; $PY -c "import json,sys; json.dump({'lang':'$L','status':'STOP','problems':[sys.argv[1]]}, open('$M/${L}_gate.json','w'), indent=1)" "$*"; exit 3; }
log "waiting for $M/${L}_list.json"
until [ -f $M/${L}_list.json ]; do grep -q "$L: SHORT LIST" logs/propose_repos_rest.log 2>/dev/null && stop "repo search found too few repos"; sleep 60; done
log "list: $(tr -d '\n ' < $M/${L}_list.json)"
log "fetch + pin"; $PY $M/fetch_repos.py $L $M/${L}_list.json >> "$LOG" 2>&1 || stop "fetch failed"
log "decontamination (pool vs benchmark + held-out)"
$PY $M/decontaminate_lang.py $L $M/${L}_pins.json 2>/dev/null | tee -a "$LOG" || stop "decontamination failed"
log "build $L-fim-v1"
$PY -m fimexp.build_lang_data --lang $L --pins $M/${L}_pins.json --version $L-fim-v1 --exclude_files $M/${L}_exclude_files.json 2>&1 | grep -v "longer than" >> "$LOG"
[ -f data/$L-fim-v1/manifest.json ] || stop "build v1 failed"
FINAL=$L-fim-v1; HELD=$L-fim-v1-heldout
log "train vs validation check on $FINAL"
N=$($PY $M/trainval_lang.py $L $FINAL 2>/dev/null | tee -a "$LOG" | $PY -c "import json,sys; print(json.loads(sys.stdin.read().strip().splitlines()[-1])['remove'])") || stop "train-vs-validation check failed"
if [ "$N" != 0 ]; then
  log "$N validation files share code with training: removing them (addendum rule), rebuilding $L-fim-v2 with the same held-out"
  $PY -c "import json; a=json.load(open('$M/${L}_exclude_files.json')); b=list(json.load(open('$M/${L}_train_vs_validation_$FINAL.json'))['removed_validation_files']); json.dump(sorted(set(a)|set(b)), open('$M/${L}_exclude_files_v2.json','w'), indent=1)"
  $PY -m fimexp.build_lang_data --lang $L --pins $M/${L}_pins.json --version $L-fim-v2 --exclude_files $M/${L}_exclude_files_v2.json --reuse_heldout $HELD 2>&1 | grep -v "longer than" >> "$LOG"
  [ -f data/$L-fim-v2/manifest.json ] || stop "build v2 failed"
  FINAL=$L-fim-v2
  $PY $M/trainval_lang.py $L $FINAL 2>/dev/null | tee -a "$LOG"
fi
log "rerun all checks on $FINAL"
$PY $M/decontaminate_lang.py $L $M/${L}_pins.json --pool_only $FINAL 2>/dev/null | tail -2 | tee -a "$LOG"
$PY $M/overlap_lang.py $L $FINAL 2>/dev/null | tee -a "$LOG"
$PY $M/review_overlap.py $L $FINAL 2>/dev/null | tee -a "$LOG"
log "gate"
$PY - "$L" "$FINAL" "$HELD" <<'PYX' 2>&1 | tee -a "$LOG"
import json, sys
L, FINAL, HELD = sys.argv[1:]; M = "checks/multilang"
P = []
man = json.load(open(f"data/{FINAL}/manifest.json")); hman = json.load(open(f"data/{HELD}/manifest.json"))
tr, va = man["splits"]["train"]["pairs"], man["splits"]["validation"]["pairs"]
if tr != 7549: P.append(f"train pairs {tr} != 7549 (shortfall, not padded)")
if va != 518: P.append(f"validation pairs {va} != 518 (shortfall, not padded)")
hp = hman["stats"]["pairs"]
if hp < 900: P.append(f"held-out pairs {hp} < 900")
full = json.load(open(f"{M}/{L}_decontamination.json"))
wf = full["planted_summary"]["whole file"]
if wf["caught"] < 18: P.append(f"planted whole-file copies caught {wf['caught']}/{wf['planted']} < 18")
chk = json.load(open(f"{M}/{L}_decontamination_check_{FINAL}.json"))
if chk["removed_files"]: P.append(f"{len(chk['removed_files'])} files still match benchmark/held-out after rebuild")
tv = json.load(open(f"{M}/{L}_train_vs_validation_{FINAL}.json"))
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
        "raw_minhash_matches": ov["minhash_verified"], "train_repos": json.load(open(f"data/{FINAL}/split.json"))["train_repos"],
        "validation_repos": json.load(open(f"data/{FINAL}/split.json"))["validation_repos"]}
json.dump(gate, open(f"{M}/{L}_gate.json", "w"), indent=1)
print(json.dumps(gate, indent=1))
PYX
log "prep finished: $(python3 -c "import json; print(json.load(open('$M/${L}_gate.json'))['status'])")"
