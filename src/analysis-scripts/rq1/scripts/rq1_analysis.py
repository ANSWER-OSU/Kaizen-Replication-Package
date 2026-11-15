#!/usr/bin/env python3
"""RQ1: How reliable are LLM-based HPC code translators on semantically
equivalent programs?

Reproduces Table~\\ref{tab:rq1} and every RQ1 number in the paper, and the
"excluding assert" numbers in the Threats to Validity.

Inputs (data/):
  kaizen_variant_results.tsv    per translator, application, and mutation run:
                                unique variants translated, translations that
                                compile, and translations that are correct
                                (same verdict as the CUDA variant on every
                                manual test input and generated input).
                                Variants only; the original program is not
                                included.
  original_program_results.csv  per translator and application: outcome of the
                                original program and its translation.
  cuda_failure_variants.tsv     per translator, application, and mutation run:
                                variant translations whose CUDA variant crashes
                                or fails its own verification on a generated
                                input. Used only to mark applications with * in
                                the table; no translation is excluded.

Outputs (results/):
  rq1_summary.md     every RQ1 number, human-readable
  rq1_results.json   the same, machine-readable
  rq1_table.tex      Table rq1 (per-application results)
  rq1_per_app.csv    per-application results
  mr_violations.csv  metamorphic relation violations per application-translator pair
  rq1_long.csv       per-run counts (input to rq1_glmm.R)

Usage: python3 scripts/rq1_analysis.py   (run from the rq1/ directory)
"""
import csv
import json
import random
import statistics as st
from pathlib import Path

from scipy.stats import binomtest, friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)

SEED = 1            # random seed for the cluster bootstrap
B = 10000           # bootstrap resamples
ALPHA = 0.05
RUNS = 5
NAMES = {"ChatPORTCL_13B": "FT-CodeLlama-13B",
         "ChatPORTSCB_7B": "FT-StarCoderBase-7B",
         "ChatPORTHPC_C_6.7B": "FT-HPC-Coder-6.7B"}
TRANSLATORS = list(NAMES.values())


# ---------------------------------------------------------------- load data
def load_variants():
    """kaizen_variant_results.tsv is the authors' spreadsheet export. Columns
    12-16, 19-23, and 26-30 hold, for mutation runs 1-5, the unique variants
    translated, the translations that compile, and the correct translations."""
    rows = list(csv.reader(open(DATA / "kaizen_variant_results.tsv"), delimiter="\t"))
    data, llm = {}, None
    for r in rows[2:]:
        if not any(c.strip() for c in r):
            continue
        if r[0].strip():
            llm = NAMES[r[0].strip()]

        def runs(start):
            v = r[start:start + RUNS]
            if any(x.strip() in ("N/A", "") for x in v):
                return None              # application excluded from the study
            return [int(float(x)) for x in v]

        tr, co, ok = runs(12), runs(19), runs(26)
        if None not in (tr, co, ok):
            data[(llm, r[1].strip())] = dict(translated=tr, compiled=co, correct=ok)
    return data


def load_originals():
    """A translation of the original program is correct if it compiles and
    reports the same verdict as the CUDA program on every manual test input and
    generated input (Eq. 3), the same rule as for the variants."""
    orig = {}
    for r in csv.DictReader(open(DATA / "original_program_results.csv")):
        f = {k: r[k] == "1" for k in r if k not in ("translator", "application")}
        assert f["cuda_passes_developer_tests"], r     # every CUDA program passes the manual test inputs
        orig[(r["translator"], r["application"])] = dict(
            compiles=f["omp_compiles"],
            manual=f["omp_compiles"] and f["omp_passes_developer_tests"],
            correct=f["omp_compiles"] and f["omp_passes_developer_and_generated_inputs"],
            cuda_fails_generated=not f["cuda_passes_developer_and_generated_inputs"])
    return orig


data, orig = load_variants(), load_originals()
APPS = sorted({a for (_, a) in orig}, key=str.lower)
assert len(APPS) == 16 and all((t, a) in data for t in TRANSLATORS for a in APPS)
for d in data.values():
    for tr, co, ok in zip(d["translated"], d["compiled"], d["correct"]):
        assert 0 <= ok <= co <= tr
# Applications in which at least one CUDA program (the original or a variant)
# crashes or fails its own verification on a generated input; marked * in the table.
CUDA_FAIL_APPS = sorted({a for (t, a), o in orig.items() if o["cuda_fails_generated"]} |
                        {r["application"] for r in csv.DictReader(open(DATA / "cuda_failure_variants.tsv"), delimiter="\t")
                         if any(int(r[f"run{i}"]) for i in range(1, RUNS + 1))}, key=str.lower)

S = lambda t, a, k: sum(data[(t, a)][k])          # variants only (used for violations and verdicts)
# Programs = the original program and its variants. Each mutation run tests the
# original and that run's variants; pooled over runs, the original counts once.
O = lambda t, a: {"translated": 1, "compiled": int(orig[(t, a)]["compiles"]), "correct": int(orig[(t, a)]["correct"])}
P = lambda t, a, k: S(t, a, k) + O(t, a)[k]
prog = {(t, a): {k: [x + O(t, a)[k] for x in data[(t, a)][k]] for k in ("translated", "compiled", "correct")}
        for t in TRANSLATORS for a in APPS}
rng = random.Random(SEED)
R = {"settings": {"bootstrap_seed": SEED, "bootstrap_resamples": B, "applications": APPS, "mutation_runs": RUNS},
     "applications_marked_with_asterisk": CUDA_FAIL_APPS}


def cluster_ci(units, num, den):
    """95% percentile CI of sum(num)/sum(den), resampling units with replacement."""
    vals = []
    for _ in range(B):
        smp = [rng.choice(units) for _ in units]
        d = sum(den(u) for u in smp)
        if d:
            vals.append(sum(num(u) for u in smp) / d)
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, ps[i] * (len(ps) - rank)))
        adj[i] = running
    return adj


def a12(x, y):
    gt = sum(1 for i in x for j in y if i > j)
    eq = sum(1 for i in x for j in y if i == j)
    return (gt + 0.5 * eq) / (len(x) * len(y))


# --------------------------------------------- 1. compile and correct rates
R["unique_variants_translated"] = sum(S(TRANSLATORS[0], a, "translated") for a in APPS)
R["programs_translated"] = R["unique_variants_translated"] + len(APPS)
R["rates"] = {}
for t in TRANSLATORS:
    T = sum(P(t, a, "translated") for a in APPS)
    C = sum(P(t, a, "compiled") for a in APPS)
    K = sum(P(t, a, "correct") for a in APPS)
    per_run = [sum(prog[(t, a)]["correct"][i] for a in APPS) / sum(prog[(t, a)]["translated"][i] for a in APPS)
               for i in range(RUNS)]
    R["rates"][t] = {
        "programs_translated": T, "compiled": C, "correct": K, "compiled_but_incorrect": C - K,
        "compile_rate": round(C / T, 6),
        "compile_rate_ci": cluster_ci(APPS, lambda a: P(t, a, "compiled"), lambda a: P(t, a, "translated")),
        "correct_rate": round(K / T, 6),
        "correct_rate_ci": cluster_ci(APPS, lambda a: P(t, a, "correct"), lambda a: P(t, a, "translated")),
        "incorrect_among_compiled": round((C - K) / C, 6) if C else None,
        "incorrect_among_compiled_ci": cluster_ci(APPS, lambda a: P(t, a, "compiled") - P(t, a, "correct"),
                                                  lambda a: P(t, a, "compiled")) if C else None,
        "correct_rate_per_run": [round(c, 4) for c in per_run],
        "correct_rate_run_mean": round(st.mean(per_run), 6),
        "correct_rate_run_sd": round(st.stdev(per_run), 4),
        "apps_with_no_correct_variant": [a for a in APPS if S(t, a, "correct") == 0],
    }

# ----------------------------------------------------------- 2. verdicts
# Conventional verdict: the original program's translation compiles and passes
# the manual test inputs. Kaizen verdict: the original's translation and every
# variant translation are correct on all inputs.
R["verdicts"] = {}
for t in TRANSLATORS:
    conv = {a: orig[(t, a)]["manual"] for a in APPS}
    kaiz = {a: orig[(t, a)]["correct"] and S(t, a, "correct") == S(t, a, "translated") for a in APPS}
    pf = [a for a in APPS if conv[a] and not kaiz[a]]
    fp = [a for a in APPS if not conv[a] and kaiz[a]]
    n = len(pf) + len(fp)
    R["verdicts"][t] = {
        "pass_conventional": sum(conv.values()), "pass_kaizen": sum(kaiz.values()),
        "pass_to_fail": pf, "fail_to_pass": fp,
        "conventional_pass_kaizen_pass": [a for a in APPS if conv[a] and kaiz[a]],
        "exact_mcnemar_p": round(binomtest(len(fp), n, 0.5).pvalue, 4) if n else None,
    }


# --------------------------------------- 3. metamorphic relation violations
def mr(apps, ci=True):
    rows, elig = [], []
    for t in TRANSLATORS:
        for a in apps:
            d = data[(t, a)]
            base = orig[(t, a)]["correct"]
            fail = [tr - ok for tr, ok in zip(d["translated"], d["correct"])]
            if base:
                elig.append((t, a))
            rows.append(dict(translator=t, application=a, original_correct=int(base),
                             variants=S(t, a, "translated"), variants_correct=S(t, a, "correct"),
                             violations=sum(fail) if base else 0,
                             reverse_cases=S(t, a, "correct") if not base else 0,
                             violations_per_run=" ".join(map(str, fail)) if base else ""))
    V = sum(r["violations"] for r in rows)
    N = sum(S(t, a, "translated") for t, a in elig)
    out = {
        "eligible_pairs": len(elig),
        "pairs_with_violation": sum(1 for r in rows if r["violations"] > 0),
        "pairs_with_violation_in_every_run": sum(
            1 for t, a in elig if all(tr > ok for tr, ok in zip(data[(t, a)]["translated"], data[(t, a)]["correct"]))),
        "violating_variants": V, "variants_in_eligible_pairs": N, "violation_rate": round(V / N, 6),
        "reverse_pairs": sum(1 for r in rows if r["reverse_cases"] > 0),
        "reverse_variants": sum(r["reverse_cases"] for r in rows),
    }
    if ci:
        out["violation_rate_ci"] = cluster_ci(elig, lambda u: S(u[0], u[1], "translated") - S(u[0], u[1], "correct"),
                                              lambda u: S(u[0], u[1], "translated"))
    return out, rows


R["metamorphic_relation"], mr_rows = mr(APPS)
R["metamorphic_relation_without_assert"], _ = mr([a for a in APPS if a != "assert"], ci=False)
R["metamorphic_relation"]["per_translator"] = {
    t: {"eligible_pairs": sum(r["original_correct"] for r in mr_rows if r["translator"] == t),
        "pairs_with_violation": sum(1 for r in mr_rows if r["translator"] == t and r["violations"] > 0),
        "violations": sum(r["violations"] for r in mr_rows if r["translator"] == t)} for t in TRANSLATORS}
with open(RES / "mr_violations.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(mr_rows[0]))
    w.writeheader()
    w.writerows(mr_rows)

# ----------------------------------------------- 4. translator differences
pa = {t: [st.mean([k / n for k, n in zip(prog[(t, a)]["correct"], prog[(t, a)]["translated"])]) for a in APPS]
      for t in TRANSLATORS}
fr = friedmanchisquare(*pa.values())
pairs = [(TRANSLATORS[0], TRANSLATORS[1]), (TRANSLATORS[0], TRANSLATORS[2]), (TRANSLATORS[1], TRANSLATORS[2])]
ps = [wilcoxon(pa[x], pa[y]).pvalue for x, y in pairs]
R["translator_comparison"] = {
    "mean_per_application_correctness": {t: round(st.mean(v), 4) for t, v in pa.items()},
    "friedman_chi2": round(fr.statistic, 3), "friedman_p": round(fr.pvalue, 4),
    "pairwise": [{"a": x, "b": y, "wilcoxon_p": round(p, 4), "holm_p": round(h, 4), "A12": round(a12(pa[x], pa[y]), 3)}
                 for (x, y), p, h in zip(pairs, ps, holm(ps))],
    "mixed_model": "results/rq1_glmm.txt (scripts/rq1_glmm.R)",
}

# ------------------------------------------ 5. variability across runs (descriptive)
# Per application-translator pair with both correct and incorrect translations:
# the share of correct translations in each mutation run and its range. No
# significance test is applied, because translations from one mutation run share
# mutation history and are not independent observations.
var = []
for t in TRANSLATORS:
    for a in APPS:
        d = prog[(t, a)]
        inc = [n - k for n, k in zip(d["translated"], d["correct"])]
        if sum(d["correct"]) and sum(inc):      # pairs with both correct and incorrect translations
            pr = [100 * k / n for k, n in zip(d["correct"], d["translated"])]
            var.append({"translator": t, "application": a,
                        "correct_pct_per_run": [round(x, 4) for x in pr],
                        "min_pct": round(min(pr), 4), "max_pct": round(max(pr), 4),
                        "range_pp": round(max(pr) - min(pr), 4)})
R["run_variability"] = {"pairs_with_correct_and_incorrect": len(var), "per_pair": var}

# ---------------------------------------------------------- write outputs
json.dump(R, open(RES / "rq1_results.json", "w"), indent=2)

with open(RES / "rq1_long.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["translator", "application", "run", "translated", "compiled", "correct"])
    for t in TRANSLATORS:
        for a in APPS:
            d = prog[(t, a)]
            for i in range(RUNS):
                w.writerow([t, a, i + 1, d["translated"][i], d["compiled"][i], d["correct"][i]])


def orig_code(t, a):
    """-- does not compile; C compiles but fails a manual test input; D passes the
    manual test inputs but differs from the CUDA program on a generated input;
    G behaves like the CUDA program on every input."""
    o = orig[(t, a)]
    return "G" if o["correct"] else "D" if o["manual"] else "C" if o["compiles"] else "-"


frac = lambda t, a, k: [x / n for x, n in zip(prog[(t, a)][k], prog[(t, a)]["translated"])]
with open(RES / "rq1_per_app.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["translator", "application", "marked_asterisk", "original_outcome", "programs", "compiled", "correct",
                "compile_mean", "compile_sd", "correct_mean", "correct_sd"])
    for t in TRANSLATORS:
        for a in APPS:
            cf, kf = frac(t, a, "compiled"), frac(t, a, "correct")
            w.writerow([t, a, int(a in CUDA_FAIL_APPS), orig_code(t, a), P(t, a, "translated"), P(t, a, "compiled"),
                        P(t, a, "correct"), round(st.mean(cf), 4), round(st.stdev(cf), 4),
                        round(st.mean(kf), 4), round(st.stdev(kf), 4)])

half_up = lambda x: int(x + 0.5)    # percentages rounded half up
pad = lambda x, w: "\\phantom{" + "0" * (w - len(x)) + "}" + x if len(x) < w else x   # align digits
ms = lambda v: f"{pad(str(half_up(100 * st.mean(v))), 3)}\\,$\\pm$\\,{pad(str(half_up(100 * st.stdev(v))), 2)}"
num = lambda n: f"{n:,}".replace(",", "{,}")
code = lambda c: "--" if c == "-" else f"\\textsf{{{c}}}"
# Columns: application | variants | per translator: Comp. | Corr. | Violation: T(Orig.), T(Var.)
lines = []
for a in APPS:
    cells = [a + ("$^{*}$" if a in CUDA_FAIL_APPS else ""), num(P(TRANSLATORS[0], a, "translated"))]
    for t in TRANSLATORS:
        c = orig_code(t, a)
        viol = num(S(t, a, "translated") - S(t, a, "correct")) if c == "G" else "--"
        cells += [ms(frac(t, a, "compiled")), ms(frac(t, a, "correct")), code(c), viol]
    lines.append(" & ".join(cells) + r" \\")
tot = [r"\textbf{All}", num(R["programs_translated"])]
for t in TRANSLATORS:
    r, v = R["rates"][t], R["metamorphic_relation"]["per_translator"][t]
    tot += [pad(f"{100 * r['compile_rate']:.1f}", 4), pad(f"{100 * r['correct_rate']:.1f}", 4), f"{R['verdicts'][t]['pass_conventional']}/16",
            num(v["violations"]) if v["eligible_pairs"] else "--"]
lines.append(r"\hline" + "\n" + " & ".join(tot) + r" \\")
TO, TV = r"\textbf{\boldmath$T$(Orig.)}", r"\textbf{\boldmath$T$(Var.)}"
CAPTION = r"""\caption{RQ1: Results per application. \emph{Programs}: original program and unique variants over five mutation runs. \emph{Comp.} and \emph{Corr.}: percentage of translations that compile and that behave like the CUDA program on all inputs (mean\,$\pm$\,SD over runs). \emph{Violation}: $T$(\emph{Orig.}) is the outcome of the translation of the original CUDA program, which itself compiles and passes the manual test inputs in every application (-- does not compile, \textsf{C} fails a manual test input, \textsf{D} passes the manual test inputs but differs from the CUDA program on a generated input, \textsf{G} behaves like the CUDA program on every input); $T$(\emph{Var.}) is the number of metamorphic relation violations, the variants over the five runs translated incorrectly although the original is translated correctly (\textsf{G}). $^{*}$: a CUDA program of the application fails on some generated input. Last row: pooled percentages, original translations that pass the manual test inputs, and total violations.}
\label{tab:rq1}"""
(RES / "rq1_table.tex").write_text(r"""\begin{figure*}[!t]
\centering
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\begin{tabular}{lr|c|c|cr|c|c|cr|c|c|cr}
\hline
\multirow{3}{*}{\textbf{Application}} & \multirow{3}{*}{\textbf{Programs}} & \multicolumn{4}{c|}{\textbf{FT-CodeLlama-13B}} & \multicolumn{4}{c|}{\textbf{FT-StarCoderBase-7B}} & \multicolumn{4}{c}{\textbf{FT-HPC-Coder-6.7B}} \\
\cline{3-14}
& & \multirow{2}{*}{\textbf{Comp. (\%)}} & \multirow{2}{*}{\textbf{Corr. (\%)}} & \multicolumn{2}{c|}{\textbf{Violation}} & \multirow{2}{*}{\textbf{Comp. (\%)}} & \multirow{2}{*}{\textbf{Corr. (\%)}} & \multicolumn{2}{c|}{\textbf{Violation}} & \multirow{2}{*}{\textbf{Comp. (\%)}} & \multirow{2}{*}{\textbf{Corr. (\%)}} & \multicolumn{2}{c}{\textbf{Violation}} \\
\cline{5-6}\cline{9-10}\cline{13-14}
& & & & """ + TO + " & " + TV + " & & & " + TO + " & " + TV + " & & & " + TO + " & " + TV + r""" \\
\hline
""" + "\n".join(lines) + "\n\\hline\n\\end{tabular}\n" + CAPTION + "\n\\end{figure*}\n")

# ---------------------------------------------------------------- summary
pct = lambda x: f"{100 * x:.1f}%"
ci = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
L = ["# RQ1 results", "",
     f"Programs translated by each translator: {R['programs_translated']:,} (the 16 original programs and {R['unique_variants_translated']:,} unique variants)", "",
     "Applications marked * (a CUDA program crashes or fails its verification on some generated input): " +
     ", ".join(CUDA_FAIL_APPS), "",
     "## Compile and correct rates (original programs and variants, pooled over the five mutation runs; 95% cluster-bootstrap CI)", "",
     "| Translator | Compile | Correct | Incorrect among compiled | Correct per run (mean +/- SD) | No correct variant |",
     "|---|---|---|---|---|---|"]
for t in TRANSLATORS:
    r = R["rates"][t]
    inc = f"{pct(r['incorrect_among_compiled'])} {ci(r['incorrect_among_compiled_ci'])}" if r["compiled"] else "n/a"
    L.append(f"| {t} | {pct(r['compile_rate'])} {ci(r['compile_rate_ci'])} | {pct(r['correct_rate'])} {ci(r['correct_rate_ci'])} "
             f"| {inc} | {pct(r['correct_rate_run_mean'])} +/- {pct(r['correct_rate_run_sd'])} "
             f"| {', '.join(r['apps_with_no_correct_variant'])} |")
L += ["", "## Verdicts (exact McNemar test)", "",
      "| Translator | Pass conventional | Pass Kaizen | Pass to fail | Fail to pass | Pass in both | p |",
      "|---|---|---|---|---|---|---|"]
for t in TRANSLATORS:
    v = R["verdicts"][t]
    L.append(f"| {t} | {v['pass_conventional']} | {v['pass_kaizen']} | {len(v['pass_to_fail'])} | {len(v['fail_to_pass'])} "
             f"| {', '.join(v['conventional_pass_kaizen_pass']) or '-'} | {v['exact_mcnemar_p']} |")
m, mw = R["metamorphic_relation"], R["metamorphic_relation_without_assert"]
top = sorted(mr_rows, key=lambda r: -r["violations"])[:4]
rev = sorted(mr_rows, key=lambda r: -r["reverse_cases"])[:4]
L += ["", "## Metamorphic relation violations", "",
      f"- Pairs whose original program is translated correctly: {m['eligible_pairs']}",
      f"- Pairs with at least one violation: {m['pairs_with_violation']}; in every mutation run: {m['pairs_with_violation_in_every_run']}",
      f"- Violating variants: {m['violating_variants']:,} of {m['variants_in_eligible_pairs']:,} "
      f"({pct(m['violation_rate'])}, 95% CI {ci(m['violation_rate_ci'])})",
      "- Most violations: " + "; ".join(f"{r['translator']} on {r['application']}: {r['violations']:,} "
                                        f"({r['variants_correct']:,} of {r['variants']:,} correct)" for r in top),
      f"- Reverse direction (original translated incorrectly, variant correctly): {m['reverse_variants']:,} variants "
      f"in {m['reverse_pairs']} pairs; most: " + "; ".join(f"{r['translator']} on {r['application']}: {r['reverse_cases']:,}" for r in rev),
      f"- Excluding assert (Threats to Validity): {mw['pairs_with_violation']} of {mw['eligible_pairs']} pairs with violations; "
      f"{mw['violating_variants']:,} of {mw['variants_in_eligible_pairs']:,} variants ({pct(mw['violation_rate'])})",
      "", "## Translator comparison (per-application correctness, mean over runs)", "",
      "- Mean per-application correctness: " + ", ".join(f"{t} {pct(v)}" for t, v in
                                                          R["translator_comparison"]["mean_per_application_correctness"].items()),
      f"- Friedman: chi2(2) = {R['translator_comparison']['friedman_chi2']}, p = {R['translator_comparison']['friedman_p']}"]
for p in R["translator_comparison"]["pairwise"]:
    L.append(f"- {p['a']} vs {p['b']}: Wilcoxon p = {p['wilcoxon_p']}, Holm p = {p['holm_p']}, A12 = {p['A12']}")
L.append("- Mixed-effects model: results/rq1_glmm.txt")
h = R["run_variability"]
L += ["", "## Variability across mutation runs (descriptive)", "",
      f"- Pairs with both correct and incorrect translations over the five runs: {h['pairs_with_correct_and_incorrect']}",
      "- Ranges are reported descriptively; no significance test is applied, because translations from one",
      "  mutation run share mutation history and are not independent observations.", "",
      "| Translator | Application | Correct per run (%) | Range (%) | Spread (percentage points) |",
      "|---|---|---|---|---|"]
for q in h["per_pair"]:
    pr = q["correct_pct_per_run"]
    L.append(f"| {q['translator']} | {q['application']} | {' '.join(str(half_up(x)) for x in pr)} | "
             f"{half_up(q['min_pct'])}-{half_up(q['max_pct'])} | {half_up(q['range_pp'])} |")
(RES / "rq1_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
