#!/usr/bin/env bash
# Reproduces all Evaluation results (RQ1-RQ4). Each RQ rewrites its results/ folder.
set -euo pipefail
cd "$(dirname "$0")"
for rq in rq1 rq2 rq3 rq4; do
  echo "================ $rq"
  (cd "$rq" && ./run.sh > /dev/null)
  echo "wrote $rq/results/"
done
