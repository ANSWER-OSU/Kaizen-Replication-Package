#!/usr/bin/env bash
# Reproduces the RQ1 results. Run from this directory.
set -euo pipefail
python3 scripts/rq1_analysis.py
Rscript scripts/rq1_glmm.R
