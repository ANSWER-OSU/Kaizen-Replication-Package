#!/usr/bin/env bash
# Reproduces the RQ3 results and Fig. 4. Run from this directory.
set -euo pipefail
python3 scripts/rq3_analysis.py
python3 scripts/make_fig4.py
