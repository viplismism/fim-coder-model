#!/usr/bin/env bash
# go build + go test for every Go module of a checkout (CPU only). Usage: build_test.sh <worktree> <out.tsv>
# Uses the project-local Go toolchain and envtest assets (Kubernetes 1.32.0 control plane for controller tests).
set -uo pipefail
P=/home/vipul/Nova/projects/fim-coder-model
export PATH=$P/tools/go/bin:$PATH GOPATH=$P/tools/gopath GOCACHE=$P/tools/gocache GOTOOLCHAIN=local
export KUBEBUILDER_ASSETS=$P/tools/envtest/k8s/1.32.0-linux-arm64
WT=$(realpath "$1"); OUT=$(realpath -m "$2"); LOGD=${OUT%.tsv}_logs; mkdir -p "$LOGD"
echo -e "module\tbuild\ttest\ttest_seconds\tpkgs_ok\tpkgs_fail\tpkgs_notest" > "$OUT"
cd "$WT"
for m in $(find . -name go.mod -not -path './.git/*' | sed 's#/go.mod##; s#^\./##' | sort); do
  n=$(echo "$m" | tr / _)
  (cd "$m" && timeout 1200 go build ./... > "$LOGD/$n.build.log" 2>&1); b=$?
  s=$(date +%s); (cd "$m" && timeout 2400 go test -count=1 ./... > "$LOGD/$n.test.log" 2>&1); t=$?; e=$(( $(date +%s) - s ))
  ok=$(grep -c '^ok ' "$LOGD/$n.test.log"); fl=$(grep -c '^FAIL\s' "$LOGD/$n.test.log"); nt=$(grep -c 'no test files' "$LOGD/$n.test.log")
  echo -e "$m\t$([ $b = 0 ] && echo pass || echo FAIL)\t$([ $t = 0 ] && echo pass || echo FAIL)\t$e\t$ok\t$fl\t$nt" | tee -a "$OUT"
done
