#!/bin/bash

python3 compute_compilability.py make-benchmark-summary-translated-outputs-codellama-20260427-124959.csv | grep $1
python3 compute_compilability.py make-benchmark-summary-translated-outputs-starcoder-20260427-125221.csv | grep $1
python3 compute_compilability.py make-benchmark-summary-translated-outputs-hpccoder-20260427-125232.csv | grep $1

