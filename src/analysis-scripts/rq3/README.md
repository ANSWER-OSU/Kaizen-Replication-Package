# RQ3: Compilation errors

Reproduces the RQ3 text, Fig. 10, and Fig. 11.

```bash
./run.sh      # python3 scripts/rq3_analysis.py, then python3 scripts/make_fig4.py
```

## Data (`data/`)

| File | Contents |
|---|---|
| `compile_error_taxonomy.tsv` | One row per category from the authors' open coding of compiler diagnostics: description, examples (application(translator) lists with the diagnostic), and the number of unique compilation errors per translator (CL, SC, HC) and in total. Two errors are the same if their compiler messages are identical; each unique error is counted once across the five mutation runs. |
| `category_descriptions.tsv` | LaTeX description, examples, and likely cause per category, as printed in Fig. 10. |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `rq3_summary.md` | Counts, shares, and ranks per category and translator |
| `rq3_results.json` | The same, machine-readable |
| `rq3_categories.csv` | One row per category and translator |
| `rq3_table.tex` | Fig. 10 |
| `rq3_error_distribution.pdf` | Fig. 11, vector, drawn at the IEEE column width (3.5 in) with embedded TrueType Times-compatible fonts; `.png` is a 600-dpi preview |

## Fig. 11 options

`scripts/make_fig4.py` draws the figure at its printed size, so LaTeX does not scale it.

```bash
python3 scripts/make_fig4.py                  # one column (3.5 in): \includegraphics[width=\columnwidth]
python3 scripts/make_fig4.py --width page     # full width (7.16 in) for figure*: \includegraphics[width=\textwidth]
python3 scripts/make_fig4.py --dpi 1200       # higher-resolution PNG
```

It uses Times New Roman or Liberation Serif (`apt install fonts-liberation`) and falls back to DejaVu Serif.
