#!/usr/bin/env python3
"""RQ4: Which execution errors (translations that compile but do not reproduce
the CUDA variant's PASS/FAIL verdict) do the translators produce?

Input (data/):
  semantic_failure_taxonomy.tsv  one row per category: description, affected
                                 illustrative examples, and the number of distinct
                                 translation mistakes across all mutation runs and translators.
  category_descriptions.tsv      LaTeX description and examples per category
                                 (edited for the paper; see review_note).
  rq1_per_app.csv                copy of rq1/results/rq1_per_app.csv, used only
                                 to check that every example application has
                                 translations that compile but are incorrect.
Outputs (results/):
  rq4_results.json, rq4_summary.md, rq4_categories.csv, rq4_table.tex

Usage: python3 scripts/rq4_analysis.py   (run from the rq4/ directory)
"""
import csv, json, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)

# Application names as written in the taxonomy -> HeCBench names used in RQ1.
ALIAS = {"dense embed": "dense-embedding", "dense embedding": "dense-embedding",
         "dense_esuhm": "dense-embedding"}
# Authors' grouping (not part of the manual taxonomy): categories whose mistakes
# mistranslate CUDA's thread hierarchy or shared memory. Marked with a dagger in
# the table and reported as one share in the text; the Evaluation section must
# say that this grouping is the authors'.
THREAD_HIERARCHY_OR_SHARED_MEMORY = [
    "Execution Model Assumption Transfer", "Multi-dimensional Index Flattening",
    "Loop Bound Error", "Shared Memory Scope Mistranslation"]

def norm(a):
    a = a.strip()
    return ALIAS.get(a, a)

rows = list(csv.reader(open(DATA / "semantic_failure_taxonomy.tsv"), delimiter="\t"))
header, body = [h.strip() for h in rows[0]], rows[1:]
ci = header.index("Occurence across all LLM")
desc = {r["category"]: r for r in csv.DictReader(open(DATA / "category_descriptions.tsv"), delimiter="\t")}

cats = []
for r in body:
    name = r[0].strip()
    # Example groups are introduced as "(app[, app]): ..."
    apps = sorted({norm(a) for g in re.findall(r"\(([^()]+)\):", r[2]) for a in g.split(",")})
    cats.append(dict(category=name, count=int(r[ci]), apps=apps))
assert {c["category"] for c in cats} == set(desc), "category names differ between the two data files"
assert set(THREAD_HIERARCHY_OR_SHARED_MEMORY) <= {c["category"] for c in cats}

total = sum(c["count"] for c in cats)
cats.sort(key=lambda c: -c["count"])
for i, c in enumerate(cats, 1):
    c["rank"], c["share"] = i, c["count"] / total
thread_mem = sum(c["count"] for c in cats if c["category"] in THREAD_HIERARCHY_OR_SHARED_MEMORY)

# Consistency checks against RQ1 -------------------------------------------
# The examples are illustrative, not exhaustive: the applications they name show
# where a category occurs, not every application in which it occurs. The only
# check they support is that each example comes from an application with
# translations that compile but are incorrect in RQ1.
per_app = defaultdict(int)
for r in csv.DictReader(open(DATA / "rq1_per_app.csv")):
    per_app[r["application"]] += int(r["compiled"]) - int(r["correct"])
example_apps = sorted(set().union(*(c["apps"] for c in cats)))
unknown = [a for a in example_apps if a not in per_app]
assert not unknown, f"example applications not in RQ1: {unknown}"
no_failures = [a for a in example_apps if per_app[a] == 0]
assert not no_failures, f"examples from applications without compiled-but-incorrect translations: {no_failures}"

res = dict(
    total_occurrences=total, n_categories=len(cats),
    categories=[dict(rank=c["rank"], category=c["category"], count=c["count"], share=c["share"],
                     example_applications=c["apps"], apps=len(c["apps"])) for c in cats],
    apps_total=len(example_apps),
    example_applications=example_apps,
    unit="distinct translation mistake across all five mutation runs and all three translators",
    thread_hierarchy_or_shared_memory=dict(categories=THREAD_HIERARCHY_OR_SHARED_MEMORY, count=thread_mem,
                                           share=thread_mem / total,
                                           note="authors' grouping, not part of the manual taxonomy"),
)
json.dump(res, open(RES / "rq4_results.json", "w"), indent=2)

with open(RES / "rq4_categories.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["rank", "category", "occurrences", "share", "example_applications"])
    for c in cats:
        w.writerow([c["rank"], c["category"], c["count"], f"{c['share']:.6f}", " ".join(c["apps"])])

# Summary -------------------------------------------------------------------
pct = lambda x: f"{100 * x:.1f}%"
L = ["# RQ4 summary", "",
     f"{total:,} distinct translation mistakes in {len(cats)} categories. The same mistake made in several "
     "translations (in any mutation run, by any translator) is counted once. Apps: number of applications "
     "named in the examples, a lower bound on the applications in which the category occurs.", "",
     "| Rank | Category | Occurrences | Share | Apps | Example applications |", "|---|---|---|---|---|---|"]
for c in cats:
    L.append(f"| {c['rank']} | {c['category']} | {c['count']:,} | {pct(c['share'])} | {len(c['apps'])} | {', '.join(c['apps'])} |")
L.append(f"| | Total | {total:,} | | {len(example_apps)} | |")
top3 = sum(c["count"] for c in cats[:3])
L += ["", f"Top three categories: {top3:,} occurrences ({pct(top3 / total)}).", "",
      f"Thread hierarchy or shared memory (authors' grouping, not part of the manual taxonomy: "
      f"{', '.join(THREAD_HIERARCHY_OR_SHARED_MEMORY)}): {thread_mem:,} of {total:,} ({pct(thread_mem / total)}).", "",
      "Example applications are illustrative, not the full set of applications in which a category occurs.", "",
      "## Consistency with RQ1", "",
      f"All {len(example_apps)} example applications have translations that compile but are incorrect in RQ1."]
(RES / "rq4_summary.md").write_text("\n".join(L) + "\n")

# LaTeX table ----------------------------------------------------------------
def fmt(n):
    return f"{n:,}".replace(",", "{,}")
def share(x):
    return "$<$1\\%" if x < 0.005 else f"{round(100 * x)}\\%"
col = lambda w: rf">{{\raggedright\arraybackslash}}p{{{w}}}"
T = [r"\begin{table*}[!t]", r"\centering", r"\scriptsize",
     r"\caption{RQ4: Execution errors in CUDA-to-OpenMP kernel translations of \toolname-generated variants by "
     r"FT-CodeLlama-13B, FT-StarCoderBase-7B, and FT-HPC-Coder-6.7B, ordered by frequency. An execution error is a "
     r"translation that compiles but does not reproduce the CUDA variant's verdict (PASS or FAIL) from the "
     r"application's built-in verification: it prints the other verdict or terminates with an error message. "
     r"\emph{Occurrences} counts the distinct translation mistakes in each category, identified by manually "
     r"analyzing the failing OpenMP translations from the five mutation runs and three translators; the same "
     r"mistake made in several translations is counted once. Parentheses give the share of all distinct mistakes. "
     r"The examples illustrate each category and are not exhaustive. \emph{Apps}: number of applications named "
     r"in the examples, a lower bound on the applications in which the category occurs. $^\dagger$Categories that we group as "
     r"mistranslations of CUDA's thread hierarchy or shared memory.}",
     r"\label{tab:error_taxonomy_kernel_translation_runtime}",
     r"\setlength{\tabcolsep}{3pt}",
     rf"\begin{{tabular}}{{@{{}}{col('2.1cm')}{col('4.4cm')}{col('8.0cm')}rr@{{}}}}",
     r"\toprule",
     r"\textbf{Category} & \textbf{Description} & \textbf{Examples} & \textbf{Occurrences} & \textbf{Apps} \\",
     r"\midrule"]
for i, c in enumerate(cats):
    d = desc[c["category"]]
    dag = r"$^\dagger$" if c["category"] in THREAD_HIERARCHY_OR_SHARED_MEMORY else ""
    T.append(f"{c['category']}{dag} & {d['description_latex']} & {d['examples_latex']} & "
             f"{fmt(c['count'])} ({share(c['share'])}) & {len(c['apps'])} \\\\")
    T.append(r"\midrule")
T += [rf"\textbf{{Total}} & & & {fmt(total)} & {len(example_apps)} \\", r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
(RES / "rq4_table.tex").write_text("\n".join(T) + "\n")
print((RES / "rq4_summary.md").read_text())
