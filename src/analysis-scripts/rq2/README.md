# RQ2: Contribution of source transformations and generated inputs

Reproduces the RQ2 text and Fig. 9.

```bash
./run.sh      # python3 scripts/rq2_analysis.py
```

## Configurations

| Configuration | Programs | Inputs |
|---|---|---|
| Full Kaizen | Original programs + program variants | Manual test inputs + generated inputs |
| Kaizen-I | Original programs + program variants | Manual test inputs |
| Kaizen-II | Original programs | Manual test inputs + generated inputs |
| Baseline | Original programs | Manual test inputs |

All four use the same translations, inputs, and oracle (Eq. 3). Fig. 9 reports, for each, the percentage of
translations that compile and that are correct. Full Kaizen vs. Kaizen-I shows the contribution of the generated
inputs, Full Kaizen vs. Kaizen-II that of the program variants, and Full Kaizen vs. Baseline that of both.

## Data (`data/`)

| File | Contents |
|---|---|
| `ablation_results.tsv` | Per configuration and translator: translations that compile and translations that are correct, out of 16 original programs or, for variants, as an average per mutation run out of 2,900. Fig. 9 takes the variant rows "compiled" and "source transformations only" from this file. |
| `original_program_results.csv` | Copy of `rq1/data/original_program_results.csv` (original-program rows of Fig. 9). |
| `kaizen_variant_results.tsv` | Copy of `rq1/data/kaizen_variant_results.tsv` (Kaizen row of Fig. 9, and which pairs only source transformations expose). |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `rq2_summary.md` | Fig. 9 counts; application level (pairs judged correct by the conventional evaluation, and whether generated inputs or source transformations expose them); translation level (variant translations that pass the manual test inputs but fail a generated input); consistency between the ablation table and the RQ1 data |
| `rq2_results.json` | The same, machine-readable |
| `rq2_table.tex` | Fig. 9 |
