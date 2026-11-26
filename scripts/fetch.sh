#!/usr/bin/env bash
# Fetch external inputs at the pinned commits / revisions into work/ (nothing third-party is committed here).
#   bash scripts/fetch.sh facet          facet-rs/facet history + cutoff worktree, Qwen2.5-Coder-14B and -32B
#   bash scripts/fetch.sh controlplane   telekom/controlplane history + cutoff worktree, Qwen 14B/32B, DeepSeek-Coder-6.7B
#   bash scripts/fetch.sh rust           reth, Real-FIM-Eval, DeepSeek-Coder-6.7B, 15 Rust held-out repos
#   bash scripts/fetch.sh multilang      Real-FIM-Eval, DeepSeek-Coder-6.7B, 22 pinned repos per language (python ts go cpp)
#   bash scripts/fetch.sh toolchains     Go 1.26.11 + Kubernetes envtest 1.32.0 (controlplane tests); checks rustup (facet)
# Public models and repos need no token; ~/.cache/huggingface/token is used for downloads if present.
source "$(dirname "$0")/lib.sh"
S=work/sources; mkdir -p $S work/models

pin_clone() {  # url dir commit: one commit, shallow
  [ -d "$2/.git" ] && return
  git init -q "$2"; git -C "$2" fetch -q --depth 1 "$1" "$3"; git -C "$2" checkout -q FETCH_HEAD
}
full_clone() {  # url dir head cutoff cutoff_worktree: full history (the post-cutoff test set is built from it)
  [ -d "$2/.git" ] || git clone -q "$1" "$2"
  git -C "$2" checkout -q "$3"
  [ -d "$5" ] || git -C "$2" worktree add -q --detach "$(realpath -m "$5")" "$4"
}
qwen() {
  $PY -m data.fetch_model Qwen/Qwen2.5-Coder-14B f2ad5164aade432d6d56c24bb71589184d5d613d
  $PY -m data.fetch_model Qwen/Qwen2.5-Coder-32B 2e12b5f7bc878d424d222e224ed40aee564ec45f
}
deepseek() { HF_HUB_OFFLINE=0 $PY -c "import core; from huggingface_hub import snapshot_download as s; \
  s('deepseek-ai/deepseek-coder-6.7b-base', revision='ce2207a8bfef3ee92bd7dd4cc31c52cfa0046912')"; }
benchmark() {
  HF_HUB_OFFLINE=0 $PY - <<'E'
import shutil, pathlib, core
from huggingface_hub import hf_hub_download
for split in ("add", "edit"):
    p = hf_hub_download("gonglinyuan/real_fim_eval", f"{split}/test.jsonl.gz", repo_type="dataset",
                        revision="a36062544c7ed6c4e5ffb8dad8536fc7777e1f36")
    d = pathlib.Path("work/sources/real_fim_eval") / split; d.mkdir(parents=True, exist_ok=True)
    shutil.copy(p, d / "test.jsonl.gz")
E
  $PY -c "from core import config as c; print('benchmark pins:', c.verify_pins(c.load_config()))"
}

case "${1:-}" in
facet)
  full_clone https://github.com/facet-rs/facet $S/facet-rs__facet 65bae5c31a7ce401bc44630fb96250ea884cfd3e \
    d3cd33f75da7b80fc0134ec7346c26512d3ada72 $S/facet-cutoff
  qwen ;;
controlplane)
  full_clone https://github.com/telekom/controlplane $S/telekom__controlplane cf866975bcaec72810622aa1f8869ead03c6fced \
    9164dcafe005913bf2c6e4870db776548c66662f $S/controlplane-cutoff
  qwen; deepseek ;;
rust)
  pin_clone https://github.com/paradigmxyz/reth $S/reth 84caec584e79ddef33572394f787d2d1203a23c7
  benchmark; deepseek
  $PY - <<'E'
import json, subprocess, pathlib
for r in json.load(open("configs/repos/rust_heldout_repos.json"))["picked"]:
    d = pathlib.Path("work/sources/heldout") / r["repo"].replace("/", "__")
    if not d.exists():
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "fetch", "-q", "--depth", "1", f"https://github.com/{r['repo']}", r["commit"]], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "-q", "FETCH_HEAD"], check=True)
E
  ;;
multilang)
  benchmark; deepseek
  for L in python typescript go cpp; do $PY -m data.fetch_repos $L configs/repos/${L}_repos.json; done ;;
toolchains)
  T=work/toolchains; mkdir -p $T
  ARCH=$(uname -m | sed 's/aarch64/arm64/; s/x86_64/amd64/')
  [ -x $T/go/bin/go ] || curl -fsSL https://go.dev/dl/go1.26.11.linux-$ARCH.tar.gz | tar -xz -C $T
  GOPATH=$PWD/$T/gopath GOCACHE=$PWD/work/cache/go $T/go/bin/go install \
    sigs.k8s.io/controller-runtime/tools/setup-envtest@v0.0.0-20250308055145-5fe7bb3edc86
  $T/gopath/bin/setup-envtest use 1.32.0 --bin-dir $PWD/$T/envtest -p path
  command -v rustup >/dev/null || echo "rustup not found: install it from https://rustup.rs (facet pins its own toolchain)" ;;
*) echo "usage: $0 facet|controlplane|rust|multilang|toolchains"; exit 2 ;;
esac
log "fetch ${1}: done"
