# Kaizen: replication package for the evaluation results

This package reproduces every table, figure, and statistic in the Evaluation (Section V) of
*Kaizen: Metamorphic and Differential Testing of LLM-Based HPC Code Translators*, and the
"excluding assert" numbers in the Threats to Validity (Section VIII). Each research question has
its own folder with its data, scripts, expected outputs, and README.

Kaizen's implementation (mutation engine, AFL++ custom mutators, differential-testing driver),
the variants, translations, generated inputs, and logs are in the main replication repository,
https://github.com/ANSWER-OSU/Kaizen-Replication-Package/src.

## Quick start

```bash
pip install -r requirements.txt                          # Python 3.10+
Rscript -e 'install.packages(c("lme4", "emmeans"))'      # R 4.x, RQ1 mixed-effects model only
./run_all.sh
```

`run_all.sh` runs each RQ's `run.sh`, which rewrites that RQ's `results/` folder. All scripts are
deterministic (the bootstrap uses a fixed seed), so the outputs match the files shipped in
`results/`. Tested with Python 3.13, SciPy 1.18, Matplotlib 3.11, R 4.3, lme4 1.1-35, and emmeans 1.10.

## Where each paper result comes from

| Paper | Output | Script |
|---|---|---|
| RQ1 text, Fig. 8 | `rq1/results/rq1_summary.md`, `rq1/results/rq1_table.tex` | `rq1/scripts/rq1_analysis.py` |
| RQ1 mixed-model odds ratio | `rq1/results/rq1_glmm.txt` | `rq1/scripts/rq1_glmm.R` |
| Threats: excluding assert | `rq1/results/rq1_summary.md` (Metamorphic relation violations) | `rq1/scripts/rq1_analysis.py` |
| RQ2 text, Fig. 9 | `rq2/results/rq2_summary.md`, `rq2/results/rq2_table.tex` | `rq2/scripts/rq2_analysis.py` |
| RQ3 text, Fig. 10 | `rq3/results/rq3_summary.md`, `rq3/results/rq3_table.tex` | `rq3/scripts/rq3_analysis.py` |
| Fig. 11 | `rq3/results/rq3_error_distribution.pdf` | `rq3/scripts/make_fig4.py` |
| RQ4 text, Fig. 12 | `rq4/results/rq4_summary.md`, `rq4/results/rq4_table.tex` | `rq4/scripts/rq4_analysis.py` |

## Terms

- **Translators.** FT-CodeLlama-13B, FT-StarCoderBase-7B, and FT-HPC-Coder-6.7B are the
  kernel-level CUDA-to-OpenMP translators of ChatPORT. Some data files use their ChatPORT names
  (`ChatPORTCL_13B`, `ChatPORTSCB_7B`, `ChatPORTHPC_C_6.7B`) or the abbreviations CL, SC, and HC.
- **Mutation run.** One of five independent runs of Kaizen's source transformations (5 minutes
  per application). Some data files call them `Seed 1` to `Seed 5`.
- **Manual test inputs.** The inputs of the conventional evaluation, derived from HeCBench's
  input and domain knowledge (data files: "developer tests" or "developer-written inputs").
- **Generated inputs.** The inputs from Kaizen's constraint-aware AFL++ campaigns.
- **Correct translation (Eq. 3).** The translation compiles and, on every manual test input and
  generated input, terminates and reports the same verdict (PASS or FAIL) of the application's
  built-in verification as the CUDA program it was translated from. No translation is excluded
  because its CUDA program fails.
- **Conventional verdict.** The original program's translation compiles and passes the manual
  test inputs. **Kaizen verdict.** The original program's translation and every variant
  translation are correct.
- **Metamorphic relation violation.** A variant translated incorrectly in an
  application-translator pair whose original program is translated correctly.
