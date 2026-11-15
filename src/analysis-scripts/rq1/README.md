# RQ1: Reliability on semantically equivalent programs

Reproduces the RQ1 text and Fig. 8, and the "excluding assert" numbers in the Threats to Validity.

```bash
./run.sh      # python3 scripts/rq1_analysis.py, then Rscript scripts/rq1_glmm.R
```

## Data (`data/`)

| File | Contents |
|---|---|
| `kaizen_variant_results.tsv` | Spreadsheet export, one row per translator and application. For each of the five mutation runs (`Seed 1` to `Seed 5`): unique variants translated, translations that compile, and correct translations (same verdict as the CUDA variant on every manual test input and generated input). The first block of columns (CUDA variants per run, including duplicates) is not used. Variants only; the original program is in the next file. Rows marked N/A are applications excluded from the study. |
| `original_program_results.csv` | One row per translator and application (1 = yes): the CUDA program passes the manual test inputs (`cuda_passes_developer_tests`) and also the generated inputs (`cuda_passes_developer_and_generated_inputs`); the original program's translation compiles (`omp_compiles`), passes the manual test inputs (`omp_passes_developer_tests`), and is correct on all inputs (`omp_passes_developer_and_generated_inputs`). The last column (`omp_passes_inputs_where_cuda_passes`) is not used. |
| `cuda_failure_variants.tsv` | Per translator, application, and mutation run: variant translations whose CUDA variant crashes or fails its own verification on a generated input. Used only to mark applications with * in Fig. 8. No translation is excluded. |

## Scripts (`scripts/`)

| Script | Computes |
|---|---|
| `rq1_analysis.py` | Compile and correct rates with 95% cluster-bootstrap CIs; conventional and Kaizen verdicts (exact McNemar test); metamorphic relation violations, also without assert; translator comparison (Friedman, Holm-adjusted Wilcoxon signed-rank, Vargha-Delaney A12); variability across mutation runs (per-run correctness and ranges, descriptive); Fig. 8. |
| `rq1_glmm.R` | Binomial mixed-effects model comparing FT-CodeLlama-13B and FT-HPC-Coder-6.7B (random intercepts for application, application x run, and application x translator). Reads `results/rq1_long.csv`. |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `rq1_summary.md` | Every RQ1 number in the paper |
| `rq1_results.json` | The same, machine-readable |
| `rq1_table.tex` | Fig. 8 |
| `rq1_per_app.csv` | Per translator and application: original outcome (--, C, D, G); programs (the original and its variants), compiled, and correct; mean and SD of the compile and correct rates over runs |
| `mr_violations.csv` | Metamorphic relation violations and reverse cases per pair, with violations per run |
| `rq1_long.csv` | Counts per translator, application, and run (input to the mixed model) |
| `rq1_glmm.txt` | Mixed-effects model and odds ratio |

## Statistical choices

- Compile and correct rates cover the original program and its variants. Pooled over the five runs, the
  original counts once (14,519 programs per translator); each run's rate includes it. Violations and verdicts
  treat the original separately.

- Variants of one application are not independent, so CIs resample applications (or
  application-translator pairs for the violation rate): 10,000 resamples, seed 1.
- Verdicts are paired per application, so verdict changes use an exact McNemar test.
- Translators are compared on per-application correctness (mean over the five runs), with
  applications as blocks.
- Variability across runs is reported descriptively for each pair with both correct and
  incorrect translations (share of correct translations per run and its range). No
  significance test is applied, because translations from one mutation run share mutation
  history and are not independent observations.
