#!/usr/bin/env python3
"""RQ3: What compile-time failures do the translators produce when translating
CUDA kernels to OpenMP?

Input (data/):
  compile_error_taxonomy.tsv  one row per error category: description, the
                              error instances (unique compilation errors across
                              all five mutation runs) (subcategory: application(translator)
                              list), and the number of error instances per
                              translator (CL, SC, HC) and in total.
Outputs (results/):
  rq3_results.json            counts, shares, ranks, and affected applications
  rq3_summary.md              human-readable summary
  rq3_categories.csv          one row per category and translator
  rq3_table.tex               LaTeX table

Usage: python3 scripts/rq3_analysis.py   (run from the rq3/ directory)
"""
import csv, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)
TR = {"CL": "FT-CodeLlama-13B", "SC": "FT-StarCoderBase-7B", "HC": "FT-HPC-Coder-6.7B"}
# Categories that arise from the gap between CUDA's execution model and OpenMP's
# directive-based model (the remaining categories also occur when translating
# between conventional languages).
MODEL_SPECIFIC = {"Incorrect loop construct", "Unsupported construct usage", "Invalid construct usage"}
COLORS = {"CL": "#2a78d6", "SC": "#eb6834", "HC": "#1baf7a"}   # validated categorical slots 1-3

rows = list(csv.reader(open(DATA / "compile_error_taxonomy.tsv"), delimiter="\t"))
header, body = rows[0], rows[1:]
cats, total_row = [], None
for r in body:
    if r[0].strip().lower() == "total":
        total_row = r
        continue
    counts = {k: int(r[header.index(k)]) for k in TR}
    total = int(r[header.index("Total")])
    assert sum(counts.values()) == total, (r[0], counts, total)
    apps = {k: set() for k in TR}
    for app, t in re.findall(r"([A-Za-z0-9\-]+)\((CL|SC|HC)\)", r[2]):
        apps[t].add(app)
    cats.append(dict(category=r[0].strip(), description=r[1].strip(), counts=counts, total=total,
                     apps={k: sorted(v) for k, v in apps.items()},
                     apps_any=sorted(set().union(*apps.values())),
                     model_specific=r[0].strip() in MODEL_SPECIFIC))
for k in TR:   # column totals must match the Total row
    assert sum(c["counts"][k] for c in cats) == int(total_row[header.index(k)]), k
assert sum(c["total"] for c in cats) == int(total_row[header.index("Total")])
assert {c["category"] for c in cats} >= MODEL_SPECIFIC

T = {k: sum(c["counts"][k] for c in cats) for k in TR}
GT = sum(T.values())
for c in cats:
    c["share"] = {k: c["counts"][k] / T[k] for k in TR}
    c["share_total"] = c["total"] / GT
cats.sort(key=lambda c: -c["total"])
for k in TR:
    for rank, c in enumerate(sorted(cats, key=lambda c: -c["counts"][k]), 1):
        c.setdefault("rank", {})[k] = rank

R = {"total_instances": GT, "instances_per_translator": {TR[k]: v for k, v in T.items()},
     "categories": [{"category": c["category"], "model_specific": c["model_specific"],
                     "instances": {TR[k]: c["counts"][k] for k in TR}, "total": c["total"],
                     "share_of_translator_errors": {TR[k]: round(c["share"][k], 4) for k in TR},
                     "share_of_all_errors": round(c["share_total"], 4),
                     "rank_within_translator": {TR[k]: c["rank"][k] for k in TR},
                     "affected_applications": {TR[k]: c["apps"][k] for k in TR},
                     "affected_applications_any": c["apps_any"]} for c in cats]}
ms = [c for c in cats if c["model_specific"]]
R["model_specific"] = {"instances": {TR[k]: sum(c["counts"][k] for c in ms) for k in TR},
                       "share": {TR[k]: round(sum(c["counts"][k] for c in ms) / T[k], 4) for k in TR},
                       "share_all": round(sum(c["total"] for c in ms) / GT, 4)}
json.dump(R, open(RES / "rq3_results.json", "w"), indent=2)

with open(RES / "rq3_categories.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "model_specific", "translator", "instances", "share_of_translator_errors",
                "rank_within_translator", "affected_applications", "n_affected_applications"])
    for c in cats:
        for k in TR:
            w.writerow([c["category"], int(c["model_specific"]), TR[k], c["counts"][k], round(c["share"][k], 4),
                        c["rank"][k], " ".join(c["apps"][k]), len(c["apps"][k])])

# Fig. 11 is drawn by scripts/make_fig4.py.

# ------------------------------------------------------------------- table
desc = {r["category"].lower(): r for r in csv.DictReader(open(DATA / "category_descriptions.tsv"), delimiter="\t")}
assert set(desc) == {c["category"].lower() for c in cats}, "category_descriptions.tsv must list every category"
fmt = lambda n: f"{n:,}".replace(",", "{,}")
lines = []
for c in cats:
    d = desc[c["category"].lower()]
    name = c["category"] + ("$^\\dagger$" if c["model_specific"] else "")
    counts = [(f"{fmt(c['counts'][k])} (" + ("$<$1" if 100 * c["share"][k] < 0.5 else f"{100 * c['share'][k]:.0f}") + "\\%)") if c["counts"][k] else "0" for k in TR]
    lines.append(" & ".join([name, d["description"], d["examples_latex"], d["likely_cause"]] + counts + [fmt(c["total"])]) + r" \\" + "\n\\midrule")
lines[-1] = lines[-1].replace("\n\\midrule", "")
lines.append("\\midrule\n\\textbf{Total} & & & & " + " & ".join(fmt(T[k]) for k in TR) + f" & {fmt(GT)}" + r" \\")
(RES / "rq3_table.tex").write_text(r"""\begin{table*}[!t]
\centering
\scriptsize
\caption{RQ3: Compile-time errors in CUDA-to-OpenMP kernel translations of \toolname-generated variants by FT-CodeLlama-13B (CL), FT-StarCoderBase-7B (SC), and FT-HPC-Coder-6.7B (HC), ordered by frequency. The count columns give the number of unique compilation errors across the five mutation runs and, in parentheses, their share of the translator's unique errors. Examples list affected applications with the translator in parentheses. $^\dagger$Categories that arise from the gap between CUDA's execution model and OpenMP's directive-based model.}
\label{tab:error_taxonomy_kernel_translation}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.9cm}>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{4.9cm}>{\raggedright\arraybackslash}p{3.1cm}rrrr@{}}
\toprule
\textbf{Error type} & \textbf{Description} & \textbf{Examples} & \textbf{Likely cause} & \textbf{CL} & \textbf{SC} & \textbf{HC} & \textbf{Total} \\
\midrule
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ----------------------------------------------------------------- summary
L = ["# RQ3 results", "", "Error instances are unique compilation errors across the five mutation runs.", "", f"Error instances: {GT:,} in total; " + ", ".join(f"{TR[k]} {T[k]:,}" for k in TR), "",
     "| Category | " + " | ".join(f"{TR[k]} (share, rank, apps)" for k in TR) + " | All (share, apps) |",
     "|---|" + "---|" * 4]
for c in cats:
    L.append(f"| {c['category']}{' (model-specific)' if c['model_specific'] else ''} | " +
             " | ".join(f"{c['counts'][k]:,} ({100 * c['share'][k]:.1f}%, #{c['rank'][k]}, {len(c['apps'][k])})" for k in TR) +
             f" | {c['total']:,} ({100 * c['share_total']:.1f}%, {len(c['apps_any'])}) |")
m = R["model_specific"]
L += ["", "Model-specific categories (incorrect loop construct, unsupported construct usage, invalid construct usage): " +
      ", ".join(f"{t} {v:,} ({100 * m['share'][t]:.1f}%)" for t, v in m["instances"].items()) +
      f"; all translators {100 * m['share_all']:.1f}% of instances."]
(RES / "rq3_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
