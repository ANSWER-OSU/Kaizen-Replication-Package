#!/usr/bin/env python3
"""RQ2: How much do source transformations and generated inputs each
contribute to the failures Kaizen exposes?

Reproduces Table~\\ref{tab:rq2} and the RQ2 numbers in the paper.

Inputs (data/):
  ablation_results.tsv          per translator and configuration: translations
                                that compile and translations that are correct,
                                on the original programs (out of 16) or on the
                                variants (average per mutation run, out of 2,900).
                                Used for the variant rows "compiled" and "source
                                transformations only" and for the translation-
                                level comparison.
  original_program_results.csv  copy of rq1/data/original_program_results.csv
  kaizen_variant_results.tsv    copy of rq1/data/kaizen_variant_results.tsv

Outputs (results/):
  rq2_summary.md     every RQ2 number, human-readable
  rq2_results.json   the same, machine-readable
  rq2_table.tex      Table rq2

Usage: python3 scripts/rq2_analysis.py   (run from the rq2/ directory)
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)
RUNS = 5
NAMES = {"ChatPORTCL_13B": "FT-CodeLlama-13B", "ChatPORTSCB_7B": "FT-StarCoderBase-7B",
         "ChatPORTHPC_C_6.7B": "FT-HPC-Coder-6.7B"}
TRANSLATORS = list(NAMES.values())

# ------------------------------------------------------------------ data
abl = {(r["configuration"], r["translator"]): {k: int(r[k]) for k in ("compiled", "correct", "total")}
       for r in csv.DictReader(open(DATA / "ablation_results.tsv"), delimiter="\t")}

orig = {}
for r in csv.DictReader(open(DATA / "original_program_results.csv")):
    f = {k: r[k] == "1" for k in r if k not in ("translator", "application")}
    orig[(r["translator"], r["application"])] = dict(
        compiles=f["omp_compiles"],
        manual=f["omp_compiles"] and f["omp_passes_developer_tests"],
        correct=f["omp_compiles"] and f["omp_passes_developer_and_generated_inputs"])
APPS = sorted({a for _, a in orig}, key=str.lower)

var, llm = {}, None
for r in list(csv.reader(open(DATA / "kaizen_variant_results.tsv"), delimiter="\t"))[2:]:
    if not any(c.strip() for c in r):
        continue
    if r[0].strip():
        llm = NAMES[r[0].strip()]
    cols = [r[s:s + RUNS] for s in (12, 19, 26)]
    if all(x.strip() not in ("N/A", "") for c in cols for x in c):
        var[(llm, r[1].strip())] = [[int(float(x)) for x in c] for c in cols]   # translated, compiled, correct

# ------------------------------------------------------------- the table
# Original programs (out of 16): compiled; conventional (manual test inputs);
# both inputs (manual and generated). Variants (average per mutation run):
# compiled and source transformations only from the ablation table; Kaizen from
# the RQ1 data (correct variant translations over five runs / 5).
R = {"table": {}, "consistency": {}}
for t in TRANSLATORS:
    kaizen_correct = sum(sum(var[(t, a)][2]) for a in APPS) / RUNS
    R["table"][t] = {
        "originals_compiled": sum(orig[(t, a)]["compiles"] for a in APPS),
        "conventional": sum(orig[(t, a)]["manual"] for a in APPS),
        "originals_both_inputs": sum(orig[(t, a)]["correct"] for a in APPS),
        "variants_compiled": abl[("mutation_only", t)]["compiled"],
        "source_transformations_only": abl[("mutation_only", t)]["correct"],
        "kaizen": round(kaizen_correct),
        "variants_per_run": abl[("mutation_only", t)]["total"],
    }
    # The ablation table's own Kaizen row and the RQ1 data differ slightly.
    R["consistency"][t] = {
        "kaizen_ablation_table": abl[("kaizen", t)]["correct"], "kaizen_rq1_data": kaizen_correct,
        "variants_compiled_ablation_table": abl[("mutation_only", t)]["compiled"],
        "variants_compiled_rq1_data": sum(sum(var[(t, a)][1]) for a in APPS) / RUNS,
        "conventional_ablation_table": abl[("conventional", t)]["correct"],
        "conventional_rq1_data": R["table"][t]["conventional"],
    }

# ---------------------------------------------------- application level
# Pairs that the conventional evaluation judges correct, and which component
# exposes their failures: generated inputs (the original's translation passes
# the manual test inputs but not all inputs) or source transformations (the
# original's translation is correct on all inputs, but some variant's is not).
gen, src, missed = [], [], []
for t in TRANSLATORS:
    for a in APPS:
        o = orig[(t, a)]
        if not o["manual"]:
            continue
        tr, _, ok = var[(t, a)]
        if not o["correct"]:
            gen.append(f"{t} on {a}")
        elif sum(ok) < sum(tr):
            src.append(f"{t} on {a}")
        else:
            missed.append(f"{t} on {a}")
R["application_level"] = {"conventional_pairs": len(gen) + len(src) + len(missed),
                          "exposed": len(gen) + len(src),
                          "exposed_by_generated_inputs": gen,
                          "exposed_only_by_source_transformations": src,
                          "not_exposed": missed}

# ---------------------------------------------------- translation level
# Variant translations that pass the manual test inputs but behave differently
# from the CUDA variant on a generated input, from the ablation table
# (source transformations only vs. Kaizen, per mutation run).
tl = {}
for t in ("FT-CodeLlama-13B", "FT-HPC-Coder-6.7B"):
    m, k = abl[("mutation_only", t)]["correct"], abl[("kaizen", t)]["correct"]
    tl[t] = {"pass_manual": m, "fail_generated": m - k, "share": (m - k) / m}
M = sum(v["pass_manual"] for v in tl.values())
F = sum(v["fail_generated"] for v in tl.values())
tl["both"] = {"pass_manual": M, "fail_generated": F, "share": F / M}
R["translation_level"] = tl
json.dump(R, open(RES / "rq2_results.json", "w"), indent=2)

# ------------------------------------------------------------ LaTeX figure
# Kaizen and three ablated configurations. Rows with program variants also include
# the 16 original programs. Compilability depends only on the programs; correctness
# also on the inputs. Kaizen-I correctness uses the ablation table's per-run count of
# variant translations that pass the manual test inputs (x5 runs).
from decimal import Decimal, ROUND_HALF_UP
T = R["table"]
num = lambda n: f"{n:,}".replace(",", "{,}")
pct = lambda x: str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
VARIANTS = sum(var[(TRANSLATORS[0], a)][0][i] for a in APPS for i in range(RUNS))   # 14,503 over five runs
N_ALL = VARIANTS + 16
def configs(t):
    co = sum(sum(var[(t, a)][1]) for a in APPS) + T[t]["originals_compiled"]
    ok = sum(sum(var[(t, a)][2]) for a in APPS) + T[t]["originals_both_inputs"]
    ok_manual = RUNS * T[t]["source_transformations_only"] + T[t]["conventional"]
    oc = T[t]["originals_compiled"]
    return {"full": (100 * co / N_ALL, 100 * ok / N_ALL),
            "kaizen_i": (100 * co / N_ALL, 100 * ok_manual / N_ALL),
            "kaizen_ii": (100 * oc / 16, 100 * T[t]["originals_both_inputs"] / 16),
            "baseline": (100 * oc / 16, 100 * T[t]["conventional"] / 16)}
C = {t: configs(t) for t in TRANSLATORS}
R["figure9"] = {t: {k: {"compilability": round(v[0], 2), "correctness": round(v[1], 2)} for k, v in c.items()} for t, c in C.items()}
ROWS = [("full", r"\toolname", "Original + variants", "Manual + generated", N_ALL),
        ("kaizen_i", r"\toolname-I", "Original + variants", "Manual", N_ALL),
        ("kaizen_ii", r"\toolname-II", "Original", "Manual + generated", 16),
        ("baseline", "Baseline", "Original", "Manual", 16)]
lines = []
for key, name, prog, inp, n in ROWS:
    cells = [name, prog, inp, num(n)] + [pct(x) for t in TRANSLATORS for x in C[t][key]]
    if key == "full":
        cells = [rf"\textbf{{{x}}}" for x in cells]
    lines.append(" & ".join(cells) + r" \\")
json.dump(R, open(RES / "rq2_results.json", "w"), indent=2)
(RES / "rq2_table.tex").write_text(r"""\begin{figure*}[!t]
\centering
\scriptsize
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lllr rr rr rr@{}}
\toprule
& & & & \multicolumn{2}{c}{\textbf{FT-CodeLlama-13B}} & \multicolumn{2}{c}{\textbf{FT-StarCoderBase-7B}} & \multicolumn{2}{c}{\textbf{FT-HPC-Coder-6.7B}} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}
\textbf{Configuration} & \textbf{Program Type} & \textbf{Test Input Type} & \textbf{Total Programs} & \textbf{Comp. (\%)} & \textbf{Corr. (\%)} & \textbf{Comp. (\%)} & \textbf{Corr. (\%)} & \textbf{Comp. (\%)} & \textbf{Corr. (\%)} \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{RQ2: Compilability and correctness of the translations under \toolname{} and three ablated configurations. \emph{Programs}: original programs, and program variants. \emph{Inputs}: manual test inputs, and generated inputs. \emph{Total}: programs translated by each translator (variants over the five mutation runs). \emph{Comp.} and \emph{Corr.}: percentage of translations that compile and that behave like the CUDA program on all inputs used.}
\label{tab:rq2}
\end{figure*}
""")

# ---------------------------------------------------------------- summary
a = R["application_level"]
L = ["# RQ2 results", "", "## Table (correct translations; variants: average per mutation run)", "",
     "| Row | " + " | ".join(TRANSLATORS) + " |", "|---|---|---|---|"]
for key, label in [("originals_compiled", "Original programs: compiled"), ("conventional", "Original programs, manual test inputs"),
                   ("originals_both_inputs", "Original programs, manual + generated inputs"), ("variants_compiled", "Variants: compiled"),
                   ("source_transformations_only", "Program variants, manual test inputs"),
                   ("kaizen", "Program variants, manual + generated inputs (Kaizen)")]:
    L.append(f"| {label} | " + " | ".join(f"{T[t][key]:,}" for t in TRANSLATORS) + " |")
L += ["", "## Application level", "",
      f"- Pairs judged correct by the conventional evaluation: {a['conventional_pairs']}; Kaizen exposes failures in {a['exposed']}",
      f"- Exposed by generated inputs (original's translation passes the manual test inputs but differs from the CUDA program "
      f"on a generated input): {len(a['exposed_by_generated_inputs'])}: " + "; ".join(a["exposed_by_generated_inputs"]),
      f"- Exposed only by source transformations (metamorphic relation violations): {len(a['exposed_only_by_source_transformations'])}",
      f"- Not exposed: " + ("; ".join(a["not_exposed"]) or "none"),
      "", "## Translation level", ""]
for t, v in tl.items():
    L.append(f"- {t}: {v['fail_generated']:,} of {v['pass_manual']:,} variant translations per run that pass the manual test "
             f"inputs fail a generated input ({100 * v['share']:.2f}%)")
L += ["", "## Consistency between the ablation table and the RQ1 data", "",
      "The table takes the Kaizen row from the RQ1 data and the variant rows 'compiled' and 'source transformations only' "
      "from the ablation table. The translation-level shares use the ablation table's own Kaizen row, so that both counts "
      "come from the same table.", ""]
for t, c in R["consistency"].items():
    L.append(f"- {t}: Kaizen per run {c['kaizen_ablation_table']} (ablation) vs {c['kaizen_rq1_data']:.1f} (RQ1); "
             f"compiled per run {c['variants_compiled_ablation_table']} (ablation) vs {c['variants_compiled_rq1_data']:.1f} (RQ1); "
             f"conventional {c['conventional_ablation_table']} (ablation) vs {c['conventional_rq1_data']} (RQ1)")
(RES / "rq2_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
