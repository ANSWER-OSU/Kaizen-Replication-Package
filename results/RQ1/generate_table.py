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
    "dense-embedding", "flip", "glu", "permute", "chi2",
    "cross", "nlll", "lif",
]

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

def fmt_frac(val):
    return "--" if val is None else "%.2f" % val

def get_seed10292_out_file(offload_dir, app, llm):
    candidates = list(offload_dir.glob("%s-seed10292-*-%s-unique.out" % (app, llm)))
    return candidates[0] if candidates else None

def parse_unique_out(out_file):
    groups = []
    current_group = False
    count = 0
    with open(str(out_file)) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Hash:"):
                if len(groups) > 0 and count >0:
                    groups[-1] = (groups[-1][0], groups[-1][1], count)

                count = 0
                current_group = True
                hash_val = line.split(":", 1)[1].strip()
                continue    
            elif "main-" in line and current_group == True:
                m = re.search(r'main-(\d+)\.cpp', line)
                if m:
                    groups.append((m.group(1), hash_val))
                    current_group = False
            count = count + 1    

    groups[-1] = (groups[-1][0], groups[-1][1], count)

    #print("Computed Groups" , groups)
                          
    # if current_group:
    #     groups.append((current_group, hash_val))
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
    print("APPLLM",app,llm)
    """
    Extrapolate correctness for app+llm using unique.out mapping + HPC results.
    Numerator:   total variants (incl. duplicates) whose representative was 'correct'
    Denominator: total variants across all groups (incl. 'not_run')
    Returns fraction or None if data not available.
    """
    # if app != "aidw":
    #     return None

    # if llm != "codellama":
    #     return None

    out_file = get_seed10292_out_file(offload_dir, app, llm)
    if out_file is None:
        return None

    #print(out_file)
    groups = parse_unique_out(out_file)
    if not groups:
        return None

    print(len(groups))
    total = 0
    correct = 0




    #with open(str()) as f:

 
    variant_hash = dict() #dict to store variant, hash for variants in correctness csv file 


    with open(str(test_results_path)) as f:
        for row in csv.DictReader(f):
            status = row["status"]
            variant_num = row["variant_num"]
            app_csv = row["app"]
            llm_csv = row["llm"]
    
            for group in groups:
                #print("GROUp",group)
                unique_var = group[0]
                unique_var_hash = group[1]
                tota_var_count = group[2]
                #print(">>>",unique_var,">>>",variant_num)
                if unique_var == variant_num and app_csv == app and llm_csv == llm:
                    print("FOUND MATCH",app,llm,variant_num,status)
                    variant_hash[(app, llm, variant_num)] = (unique_var_hash, tota_var_count, status)
                    print("length group " ,tota_var_count)

    #print(variant_hash)

    for key in variant_hash:
        if variant_hash[key][2] == "correct":
            correct += variant_hash[key][1]
        total +=  variant_hash[key][1]
    print("NUM, DENOM :",correct, total)
    correctness = (correct / total) if total > 0 else None
        

    # for group in groups:
    #     unique_var = group[0]
    #     print(app, llm, unique_var)
    #     status = test_results.get((app, llm, unique_var), "not found")
    #     if status == "not found":
    #         print("  [WARN] No test result for %s, %s, variant %s" % (app, llm, unique_var), file=sys.stderr)
    #         continue
    #     total += len(group)
    #     print(">>>",len(group))
    #     for i,_ in enumerate(group):
    #         print("hello",len(group),i,status)
    #         if status == "correct":
    #             correct += 1
    #     print(correct,total)        



    # print("NUM, DENOM :",correct, total)
    
    # return (correct / total) if total > 0 else None

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
                results[(app, llm)] = dict(cuda=None, omp=None, comp=None, correct=None)
            continue

        print("  %-25s  %d seeds" % (app, len(seed_dirs)), file=sys.stderr)
        avg_cuda = avg_or_none([count_cuda_variants(sd) for sd in seed_dirs])
        
        for llm in LLMS:

            avg_omp = avg_or_none([count_omp_translated(sd, llm) for sd in seed_dirs])

            comp_fracs = []
            for sd in seed_dirs:
                p, t = find_compilability_csv(base, sd, llm)
                if p is not None and t and t > 0:
                    comp_fracs.append(p / t)
            avg_comp = avg_or_none(comp_fracs)

            app_offload_dir = chi2_dir if (app == "chi2" and chi2_dir is not None) else offload_dir
            print("CALLING compute_correctness for app=%s, llm=%s, offload_dir=%s, results_csv=%s" % (
                app, llm, app_offload_dir, results_csv), file=sys.stderr)

            correctness = compute_correctness(app, llm, app_offload_dir, results_csv)
            #break
            results[(app, llm)] = dict(
                cuda=avg_cuda,
                omp=avg_omp,
                comp=avg_comp,
                correct=correctness,
            )
        #break

    # ── Emit LaTeX table ───────────────────────────────────────────────────────
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Relationship Between Syntactic and Semantic Success in LLM-based Automated Kernel Translation of HPC scientific applications. \textbf{application (\#CUDA variants)}: application name with avg number of CUDA variants generated by \toolname across 5 runs; \textbf{\#OpenMP translated}: avg number of OpenMP programs translated by LLM; \textbf{Compilability}: avg fraction of OpenMP programs that compile successfully; \textbf{Correctness}: avg fraction of OpenMP programs that match the outputs on all the tests when compared against the CUDA variants.}")
    lines.append(r"\label{tab:rq1_syntactic_semantic}")
    lines.append(r"% \resizebox{\cwidth}{!}{")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{application (\#CUDA variants)} & \textbf{LLM} & \textbf{\#OpenMP translated} & \textbf{Compilability} & \textbf{Correctness} \\")
    lines.append(r"\toprule")

    for i, app in enumerate(APPS):
        cuda_val  = results[(app, LLMS[0])]['cuda']
        app_label = "%s (%s)" % (app, fmt_count(cuda_val))

        for j, llm in enumerate(LLMS):
            d         = results[(app, llm)]
            omp_str   = fmt_count(d['omp'])
            comp_str  = fmt_frac(d['comp'])
            corr_str  = fmt_frac(d['correct'])
            if d['comp'] is not None and d['comp'] == 0.0:
                corr_str = "N/A"
            llm_label = LLM_LATEX[llm]

            if j == 0:
                lines.append(r"\multirow{3}{*}{%s} &  %s & %s & %s & %s  \\" % (
                    app_label, llm_label, omp_str, comp_str, corr_str))
            else:
                lines.append(r"&  %s & %s & %s & %s \\" % (
                    llm_label, omp_str, comp_str, corr_str))

        if i < len(APPS) - 1:
            lines.append(r"\hline ")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"% }")
    lines.append(r"\end{table*}")

    print("\n".join(lines))

if __name__ == "__main__":
    main()