#!/usr/bin/env bash
# Fetch the pinned snapshots from settings.json into a temp dir and run the overlap check.
# Usage: bash checks/overlap/run.sh [--github_api]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SETTINGS="$HERE/settings.json"
get() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]][sys.argv[3]])' "$SETTINGS" "$1" "$2"; }

RETH_URL=$(get reth url); RETH_SHA=$(get reth commit)
DS=$(get benchmark hf_dataset); REV=$(get benchmark revision)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git init -q "$WORK/reth"
git -C "$WORK/reth" fetch -q --depth 1 "$RETH_URL" "$RETH_SHA"
git -C "$WORK/reth" checkout -q FETCH_HEAD

for split in add edit; do
  mkdir -p "$WORK/bench/$split"
  curl -sfL -o "$WORK/bench/$split/test.jsonl.gz" "https://huggingface.co/datasets/$DS/resolve/$REV/$split/test.jsonl.gz"
done

python3 -I "$HERE/overlap_check.py" --settings "$SETTINGS" --reth "$WORK/reth" --bench "$WORK/bench" --out_dir "$HERE" "$@"
