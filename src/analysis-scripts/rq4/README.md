# RQ4: Translation mistakes behind execution errors

Reproduces the RQ4 text and Fig. 12.

```bash
./run.sh      # python3 scripts/rq4_analysis.py
```

## Data (`data/`)

| File | Contents |
|---|---|
| `semantic_failure_taxonomy.tsv` | One row per category from the authors' manual analysis of translations that compile but report a different verdict from the CUDA program: description, illustrative examples per application, and the number of distinct translation mistakes. The same mistake in several translations (any mutation run, any translator) is counted once. |
| `category_descriptions.tsv` | LaTeX description and examples per category, as printed in Fig. 12. |
| `rq1_per_app.csv` | Copy of `rq1/results/rq1_per_app.csv`, used only to check that every example application has translations that compile but are incorrect. |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `rq4_summary.md` | Occurrences, shares, and example applications per category; the share of the four categories grouped as mistranslations of CUDA's thread hierarchy or shared memory (the authors' grouping, not part of the coded taxonomy) |
| `rq4_results.json` | The same, machine-readable |
| `rq4_categories.csv` | One row per category |
| `rq4_table.tex` | Fig. 12 |

*Apps* is the number of applications named in a category's examples, a lower bound on the
applications in which the category occurs.
