# RQ4 summary

1,501 distinct translation mistakes in 9 categories. The same mistake made in several translations (in any mutation run, by any translator) is counted once. Apps: number of applications named in the examples, a lower bound on the applications in which the category occurs.

| Rank | Category | Occurrences | Share | Apps | Example applications |
|---|---|---|---|---|---|
| 1 | Floating-Point Precision Loss | 324 | 21.6% | 1 | background-subtract |
| 2 | Multi-dimensional Index Flattening | 283 | 18.9% | 1 | winograd |
| 3 | Execution Model Assumption Transfer | 204 | 13.6% | 4 | aidw, dense-embedding, glu, nlll |
| 4 | Intermediate Variable Elimination | 198 | 13.2% | 3 | aidw, glu, winograd |
| 5 | Incorrect API/Library Call | 126 | 8.4% | 1 | lif |
| 6 | Shared Memory Scope Mistranslation | 111 | 7.4% | 1 | nlll |
| 7 | Loop Bound Error | 101 | 6.7% | 3 | background-subtract, dense-embedding, nlll |
| 8 | Missing Statement Fault | 89 | 5.9% | 3 | flip, ising, lif |
| 9 | Incorrect Reduction | 65 | 4.3% | 1 | nlll |
| | Total | 1,501 | | 9 | |

Top three categories: 811 occurrences (54.0%).

Thread hierarchy or shared memory (authors' grouping, not part of the manual taxonomy: Execution Model Assumption Transfer, Multi-dimensional Index Flattening, Loop Bound Error, Shared Memory Scope Mistranslation): 699 of 1,501 (46.6%).

Example applications are illustrative, not the full set of applications in which a category occurs.

## Consistency with RQ1

All 9 example applications have translations that compile but are incorrect in RQ1.
