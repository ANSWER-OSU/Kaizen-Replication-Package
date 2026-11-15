# RQ1 results

Programs translated by each translator: 14,519 (the 16 original programs and 14,503 unique variants)

Applications marked * (a CUDA program crashes or fails its verification on some generated input): burger, dense-embedding, flip, lif, nlll, winograd

## Compile and correct rates (original programs and variants, pooled over the five mutation runs; 95% cluster-bootstrap CI)

| Translator | Compile | Correct | Incorrect among compiled | Correct per run (mean +/- SD) | No correct variant |
|---|---|---|---|---|---|
| FT-CodeLlama-13B | 64.2% [38.7, 85.8] | 34.1% [15.2, 56.8] | 46.9% [18.6, 72.9] | 34.2% +/- 3.9% | ace, assert, burger, channelSum, lif, nlll, p4 |
| FT-StarCoderBase-7B | 1.7% [0.0, 5.1] | 0.2% [0.0, 0.8] | 86.6% [0.0, 100.0] | 0.2% +/- 0.2% | ace, aidw, attention, background-subtract, burger, channelSum, cross, dense-embedding, flip, glu, ising, lif, nlll, p4, winograd |
| FT-HPC-Coder-6.7B | 66.4% [46.1, 83.0] | 27.6% [12.0, 47.1] | 58.5% [26.8, 81.2] | 27.5% +/- 2.6% | ace, burger, channelSum, ising, nlll, p4 |

## Verdicts (exact McNemar test)

| Translator | Pass conventional | Pass Kaizen | Pass to fail | Fail to pass | Pass in both | p |
|---|---|---|---|---|---|---|
| FT-CodeLlama-13B | 8 | 1 | 7 | 0 | cross | 0.0156 |
| FT-StarCoderBase-7B | 0 | 0 | 0 | 0 | - | None |
| FT-HPC-Coder-6.7B | 10 | 0 | 10 | 0 | - | 0.002 |

## Metamorphic relation violations

- Pairs whose original program is translated correctly: 14
- Pairs with at least one violation: 13; in every mutation run: 11
- Violating variants: 3,540 of 11,471 (30.9%, 95% CI [17.0, 46.3])
- Most violations: FT-HPC-Coder-6.7B on flip: 1,022 (116 of 1,138 correct); FT-HPC-Coder-6.7B on winograd: 535 (636 of 1,171 correct); FT-HPC-Coder-6.7B on assert: 509 (362 of 871 correct); FT-CodeLlama-13B on winograd: 442 (729 of 1,171 correct)
- Reverse direction (original translated incorrectly, variant correctly): 1,042 variants in 6 pairs; most: FT-CodeLlama-13B on background-subtract: 454; FT-HPC-Coder-6.7B on background-subtract: 257; FT-CodeLlama-13B on dense-embedding: 249; FT-CodeLlama-13B on attention: 38
- Excluding assert (Threats to Validity): 12 of 13 pairs with violations; 3,031 of 10,600 variants (28.6%)

## Translator comparison (per-application correctness, mean over runs)

- Mean per-application correctness: FT-CodeLlama-13B 36.8%, FT-StarCoderBase-7B 0.2%, FT-HPC-Coder-6.7B 34.5%
- Friedman: chi2(2) = 11.762, p = 0.0028
- FT-CodeLlama-13B vs FT-StarCoderBase-7B: Wilcoxon p = 0.0069, Holm p = 0.0152, A12 = 0.768
- FT-CodeLlama-13B vs FT-HPC-Coder-6.7B: Wilcoxon p = 0.9292, Holm p = 0.9292, A12 = 0.5
- FT-StarCoderBase-7B vs FT-HPC-Coder-6.7B: Wilcoxon p = 0.0051, Holm p = 0.0152, A12 = 0.203
- Mixed-effects model: results/rq1_glmm.txt

## Variability across mutation runs (descriptive)

- Pairs with both correct and incorrect translations over the five runs: 19
- Ranges are reported descriptively; no significance test is applied, because translations from one
  mutation run share mutation history and are not independent observations.

| Translator | Application | Correct per run (%) | Range (%) | Spread (percentage points) |
|---|---|---|---|---|
| FT-CodeLlama-13B | aidw | 19 97 89 44 77 | 19-97 | 78 |
| FT-CodeLlama-13B | attention | 0 47 7 1 1 | 0-47 | 47 |
| FT-CodeLlama-13B | background-subtract | 26 57 27 79 81 | 26-81 | 55 |
| FT-CodeLlama-13B | dense-embedding | 45 43 18 18 12 | 12-45 | 34 |
| FT-CodeLlama-13B | flip | 87 91 83 87 88 | 83-91 | 8 |
| FT-CodeLlama-13B | glu | 92 91 87 69 93 | 69-93 | 24 |
| FT-CodeLlama-13B | ising | 97 99 98 89 95 | 89-99 | 10 |
| FT-CodeLlama-13B | winograd | 80 92 38 43 56 | 38-92 | 54 |
| FT-StarCoderBase-7B | assert | 4 3 10 2 1 | 1-10 | 9 |
| FT-HPC-Coder-6.7B | aidw | 92 100 97 73 95 | 73-100 | 27 |
| FT-HPC-Coder-6.7B | assert | 75 73 12 25 25 | 12-75 | 63 |
| FT-HPC-Coder-6.7B | attention | 20 27 95 80 94 | 20-95 | 74 |
| FT-HPC-Coder-6.7B | background-subtract | 17 22 25 81 5 | 5-81 | 76 |
| FT-HPC-Coder-6.7B | cross | 92 94 97 100 94 | 92-100 | 8 |
| FT-HPC-Coder-6.7B | dense-embedding | 83 71 41 75 87 | 41-87 | 45 |
| FT-HPC-Coder-6.7B | flip | 5 7 8 22 8 | 5-22 | 17 |
| FT-HPC-Coder-6.7B | glu | 94 91 87 94 95 | 87-95 | 8 |
| FT-HPC-Coder-6.7B | lif | 0 0 2 0 0 | 0-2 | 2 |
| FT-HPC-Coder-6.7B | winograd | 44 44 55 72 61 | 44-72 | 28 |
