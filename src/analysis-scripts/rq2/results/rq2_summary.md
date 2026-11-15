# RQ2 results

## Table (correct translations; variants: average per mutation run)

| Row | FT-CodeLlama-13B | FT-StarCoderBase-7B | FT-HPC-Coder-6.7B |
|---|---|---|---|
| Original programs: compiled | 9 | 1 | 13 |
| Original programs, manual test inputs | 8 | 0 | 10 |
| Original programs, manual + generated inputs | 6 | 0 | 8 |
| Variants: compiled | 1,863 | 51 | 1,924 |
| Program variants, manual test inputs | 1,713 | 7 | 1,461 |
| Program variants, manual + generated inputs (Kaizen) | 989 | 7 | 798 |

## Application level

- Pairs judged correct by the conventional evaluation: 18; Kaizen exposes failures in 17
- Exposed by generated inputs (original's translation passes the manual test inputs but differs from the CUDA program on a generated input): 4: FT-CodeLlama-13B on dense-embedding; FT-CodeLlama-13B on lif; FT-HPC-Coder-6.7B on burger; FT-HPC-Coder-6.7B on lif
- Exposed only by source transformations (metamorphic relation violations): 13
- Not exposed: FT-CodeLlama-13B on cross

## Translation level

- FT-CodeLlama-13B: 722 of 1,713 variant translations per run that pass the manual test inputs fail a generated input (42.15%)
- FT-HPC-Coder-6.7B: 664 of 1,461 variant translations per run that pass the manual test inputs fail a generated input (45.45%)
- both: 1,386 of 3,174 variant translations per run that pass the manual test inputs fail a generated input (43.67%)

## Consistency between the ablation table and the RQ1 data

The table takes the Kaizen row from the RQ1 data and the variant rows 'compiled' and 'source transformations only' from the ablation table. The translation-level shares use the ablation table's own Kaizen row, so that both counts come from the same table.

- FT-CodeLlama-13B: Kaizen per run 991 (ablation) vs 989.4 (RQ1); compiled per run 1863 (ablation) vs 1863.0 (RQ1); conventional 8 (ablation) vs 8 (RQ1)
- FT-StarCoderBase-7B: Kaizen per run 7 (ablation) vs 6.8 (RQ1); compiled per run 51 (ablation) vs 50.6 (RQ1); conventional 0 (ablation) vs 0 (RQ1)
- FT-HPC-Coder-6.7B: Kaizen per run 797 (ablation) vs 798.4 (RQ1); compiled per run 1924 (ablation) vs 1925.6 (RQ1); conventional 11 (ablation) vs 10 (RQ1)
