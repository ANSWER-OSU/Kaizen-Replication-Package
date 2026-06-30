#!/usr/bin/env python3
"""
generate_table.py  —  Kaizen RQ1 full LaTeX table generator
============================================================
Computes all columns and outputs the full LaTeX table.

Requires:
  1. kaizen-output-clean dir (for #CUDA variants, #OMP translated, compilability)
  2. offloading-differential-results dir (for unique.out mapping files)
  3. Optional combined correctness CSV (app, llm, seed_dir, test_folder, variant_num, status)

Usage:
  python3 generate_table.py \
    <KAIZEN_OUTPUT_CLEAN_DIR> \
    <OFFLOADING_DIFF_RESULTS_DIR> \
    [--results <per_test_correctness_all.csv>]

Example:
  python3 generate_table.py \
    /scratch/projects/anklesan/LLM-Code-Porting-Testing/kaizen-output-clean \
    /scratch/projects/anklesan/LLM-Code-Porting-Testing/output/offloading-differential-results \
    --results /scratch/projects/anklesan/LLM-Code-Porting-Testing/per_test_correctness_all.csv \
    --chi2-dir /scratch/projects/anklesan/LLM-Code-Porting-Testing/chi2_only
"""

import sys
import csv
import re
from pathlib import Path

APPS = [
    "ace", "burger", "ising", "aidw", "assert",
    "background-subtract", "p4", "winograd", "attention", "channelSum",
    "dense-embedding", "flip", "glu",
    # "permute", "chi2",  # commented out — excluded from table
    "cross", "nlll", "lif",
]
APP_SEED_OVERRIDES = {
    ("attention", "codellama"): "seed11687",
}

LLMS = ["codellama", "starcoder", "hpccoder"]

LLM_LATEX = {
    "codellama": r"$ChatPORT_{CL\_13B}$",
    "starcoder": r"$ChatPORT_{SCB\_7B}$",
    "hpccoder":  r"$ChatPORT_{HPC\_C\_6.7B}$",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_all_seed_dirs(base, app, n=5):
    candidates = sorted(base.glob("%s-seed*-*" % app))
    def seed_key(p):
        for part in p.name.split("-"):
            if part.startswith("seed"):
                try: return int(part[4:])
                except ValueError: pass
        return 0
    candidates.sort(key=seed_key)
    return candidates[:n]

def count_cuda_variants(seed_dir):
    sq = seed_dir / "source-queue"
    return len(list(sq.glob("*.cu"))) if sq.exists() else 0

def count_omp_translated(seed_dir, llm):
    td = seed_dir / ("translated-outputs-%s" % llm)
    return len(list(td.glob("*.cpp"))) if td.exists() else 0

def find_compilability_csv(base, seed_dir, llm):
    dataset_name = seed_dir.name
    for paths in [
        list(seed_dir.glob("make-benchmark-summary-translated-outputs-%s-*.csv" % llm)),
        list(base.glob("make-benchmark-summary-translated-outputs-%s-*.csv" % llm)),
        list(seed_dir.glob("make-benchmark-summary-*%s*.csv" % llm)),
        list(base.glob("make-benchmark-summary-*%s*.csv" % llm)),
    ]:
        if paths:
            csv_file = sorted(paths)[-1]
            try:
                with open(str(csv_file)) as f:
                    for row in csv.DictReader(f):
                        if row.get("dataset", "").strip() == dataset_name:
                            total = int(row.get("total_variants", 0) or 0)
                            passed = int(row.get("pass", 0) or 0)
                            return passed, total
            except Exception as e:
                print("  [WARN] %s: %s" % (csv_file, e), file=sys.stderr)
    return None, None

def avg_or_none(values):
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None

def fmt_count(val):
    return "--" if val is None else str(round(val))

def fmt_frac(num, denom):
    """Format as $num/denom=fraction$"""
    if num is None or denom is None or denom == 0:
        return "--"
    return "$%d/%d=%.2f$" % (num, denom, num / denom)

def get_seed_out_file(offload_dir, app, llm):
    seed = APP_SEED_OVERRIDES.get((app, llm), "seed10292")
    candidates = list(offload_dir.glob("%s-%s-*-%s-unique.out" % (app, seed, llm)))
    return candidates[0] if candidates else None

def parse_unique_out(out_file):
    """
    Parse unique.out file.
    Returns list of (representative_variant, hash, main_line_count) tuples.
    count = number of main-XXX.cpp lines in the group (unique + duplicates).
    """
    groups = []
    hash_val = None
    representative = None
    count = 0
    with open(str(out_file)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Hash:"):
                if representative is not None:
                    groups.append((representative, hash_val, count))
                hash_val = line.split(":", 1)[1].strip()
                representative = None
                count = 0
                continue
            m = re.search(r'main-(\d+)\.cpp', line)
            if m:
                count += 1
                if representative is None:
                    representative = m.group(1)
    if representative is not None:
        groups.append((representative, hash_val, count))
    return groups

def load_test_results(csv_path):
    """
    Load combined per-test correctness CSV.
    Returns dict keyed by (app, llm, variant_num) -> status.
    """
    results = {}
    with open(str(csv_path)) as f:
        for row in csv.DictReader(f):
            key = (row["app"], row["llm"], row["variant_num"])
            results[key] = row["status"]
    print("Loaded %d results from %s" % (len(results), csv_path), file=sys.stderr)
    return results

def compute_correctness(app, llm, offload_dir, test_results_path):
    out_file = get_seed_out_file(offload_dir, app, llm)
    if out_file is None:
        return None, None

    # Parse all members of each group, not just the representative
    groups_full = []
    hash_val = None
    members = []
    with open(str(out_file)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Hash:"):
                if members:
                    groups_full.append((hash_val, members, len(members)))
                hash_val = line.split(":", 1)[1].strip()
                members = []
                continue
            m = re.search(r'main-(\d+)\.cpp', line)
            if m:
                members.append(m.group(1))
    if members:
        groups_full.append((hash_val, members, len(members)))

    if not groups_full:
        return None, None

    total = sum(g[2] for g in groups_full)

    # Build lookup: variant_num -> (hash, group_size)
    variant_to_group = {}
    for hash_val, members, size in groups_full:
        for v in members:
            variant_to_group[v] = (hash_val, size)

    # Load correctness status for this app+llm
    correct = 0
    seen_hashes = set()
    with open(str(test_results_path)) as f:
        for row in csv.DictReader(f):
            if row["app"] != app or row["llm"] != llm:
                continue
            variant_num = row["variant_num"]
            status = row["status"]
            if variant_num not in variant_to_group:
                continue
            h, size = variant_to_group[variant_num]
            if h in seen_hashes:
                continue  # already counted this group
            seen_hashes.add(h)
            print("FOUND MATCH", app, llm, variant_num, status, "group_size=%d" % size)
            if status == "correct":
                correct += size

    print("NUM, DENOM (%s, %s): %s / %s" % (app, llm, correct, total), file=sys.stderr)
    return correct, total

# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    base        = Path(args[0])
    offload_dir = Path(args[1])
    results_csv = None
    chi2_dir    = None

    i = 2
    while i < len(args):
        if args[i] == "--results":
            if i + 1 < len(args):
                results_csv = Path(args[i + 1])
                i += 2
            else:
                print("ERROR: --results requires a CSV path")
                sys.exit(1)
        elif args[i] == "--chi2-dir":
            if i + 1 < len(args):
                chi2_dir = Path(args[i + 1])
                i += 2
            else:
                print("ERROR: --chi2-dir requires a path")
                sys.exit(1)
        else:
            i += 1

    return base, offload_dir, results_csv, chi2_dir

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    base, offload_dir, results_csv, chi2_dir = parse_args()

    for p in [base, offload_dir]:
        if not p.exists():
            print("ERROR: %s not found" % p, file=sys.stderr)
            sys.exit(1)

    # Load combined correctness CSV (or None if not provided)
    if results_csv and results_csv.exists():
        test_results = load_test_results(results_csv)
    else:
        test_results = None
        if results_csv:
            print("  [WARN] CSV not found: %s" % results_csv, file=sys.stderr)
        else:
            print("  [INFO] No --results CSV provided — correctness will be --", file=sys.stderr)

    if chi2_dir:
        if chi2_dir.exists():
            print("  [INFO] Using chi2-dir for chi2: %s" % chi2_dir, file=sys.stderr)
        else:
            print("  [WARN] chi2-dir not found: %s" % chi2_dir, file=sys.stderr)

    print(file=sys.stderr)

    # Collect results
    results = {}
    for app in APPS:
        seed_dirs = get_all_seed_dirs(base, app, n=5)
        if not seed_dirs:
            print("  [WARN] No seed dirs for '%s'" % app, file=sys.stderr)
            for llm in LLMS:
                results[(app, llm)] = dict(cuda=None, omp=None,
                                           comp_pass=None, comp_total=None,
                                           correct=None, correct_total=None)
            continue

        print("  %-25s  %d seeds" % (app, len(seed_dirs)), file=sys.stderr)
        avg_cuda = avg_or_none([count_cuda_variants(sd) for sd in seed_dirs])

        for llm in LLMS:
            avg_omp = avg_or_none([count_omp_translated(sd, llm) for sd in seed_dirs])

            comp_passes = []
            comp_totals = []
            for sd in seed_dirs:
                p, t = find_compilability_csv(base, sd, llm)
                if p is not None and t and t > 0:
                    comp_passes.append(p)
                    comp_totals.append(t)
            avg_comp_pass  = round(avg_or_none(comp_passes)) if comp_passes else None
            avg_comp_total = round(avg_or_none(comp_totals)) if comp_totals else None

            app_offload_dir = chi2_dir if (app == "chi2" and chi2_dir is not None) else offload_dir
            print("CALLING compute_correctness for app=%s, llm=%s" % (app, llm), file=sys.stderr)

            correct_num, correct_total = compute_correctness(app, llm, app_offload_dir, results_csv)

            results[(app, llm)] = dict(
                cuda=avg_cuda,
                omp=avg_omp,
                comp_pass=avg_comp_pass,
                comp_total=avg_comp_total,
                correct=correct_num,
                correct_total=correct_total,
            )

    # ── Emit LaTeX table ───────────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Relationship Between Syntactic and Semantic Success in LLM-based Automated Kernel Translation of HPC scientific applications. \textbf{application}: application name; \textbf{Compilability}: avg fraction of OpenMP programs that compile successfully; \textbf{Correctness}: fraction of OpenMP programs that match the outputs on all the tests when compared against the CUDA variants for a single seed(10292).}")
    lines.append(r"\label{tab:rq1_syntactic_semantic}")
    lines.append(r"% \resizebox{\cwidth}{!}{")
    lines.append(r"\begin{tabular}{llrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{application} & \textbf{LLM} & \textbf{Compilability} & \textbf{Correctness} \\")
    lines.append(r"\toprule")

    for i, app in enumerate(APPS):
        for j, llm in enumerate(LLMS):
            d = results[(app, llm)]

            comp_str = fmt_frac(d['comp_pass'], d['comp_total'])

            comp_is_zero = (d['comp_pass'] is not None and d['comp_pass'] == 0)
            if comp_is_zero:
                corr_str = "N/A"
            else:
                corr_str = fmt_frac(d['correct'], d['correct_total'])

            llm_label = LLM_LATEX[llm]

            if j == 0:
                lines.append(r"\multirow{3}{*}{%s} &  %s & %s & %s  \\" % (
                    app, llm_label, comp_str, corr_str))
            else:
                lines.append(r"&  %s & %s & %s \\" % (
                    llm_label, comp_str, corr_str))

        if i < len(APPS) - 1:
            lines.append(r"\hline ")

    lines.append(r"\hline")
    lines.append(r"\multicolumn{4}{l}{*: \toolname-generated tests inputs found bugs in developer-written CUDA program.}")
    lines.append(r"\end{tabular}")
    lines.append(r"% }")
    lines.append(r"\end{table}")

    print("\n".join(lines))

if __name__ == "__main__":
    main()