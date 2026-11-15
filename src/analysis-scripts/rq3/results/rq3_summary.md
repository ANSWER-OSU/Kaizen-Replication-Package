# RQ3 results

Error instances are unique compilation errors across the five mutation runs.

Error instances: 10,525 in total; FT-CodeLlama-13B 2,025, FT-StarCoderBase-7B 6,429, FT-HPC-Coder-6.7B 2,071

| Category | FT-CodeLlama-13B (share, rank, apps) | FT-StarCoderBase-7B (share, rank, apps) | FT-HPC-Coder-6.7B (share, rank, apps) | All (share, apps) |
|---|---|---|---|---|
| Undeclared identifier | 166 (8.2%, #4, 10) | 2,584 (40.2%, #2, 14) | 833 (40.2%, #1, 12) | 3,583 (34.0%, 15) |
| Syntax error | 85 (4.2%, #6, 7) | 2,810 (43.7%, #1, 15) | 227 (11.0%, #3, 7) | 3,122 (29.7%, 15) |
| Function signature mismatch | 1,090 (53.8%, #1, 6) | 725 (11.3%, #3, 6) | 538 (26.0%, #2, 7) | 2,353 (22.4%, 10) |
| Incorrect loop construct (model-specific) | 145 (7.2%, #5, 6) | 79 (1.2%, #5, 9) | 213 (10.3%, #4, 7) | 437 (4.2%, 12) |
| Incorrect API/runtime name | 256 (12.6%, #2, 4) | 32 (0.5%, #7, 7) | 119 (5.7%, #5, 4) | 407 (3.9%, 9) |
| Unsupported construct usage (model-specific) | 215 (10.6%, #3, 4) | 11 (0.2%, #8, 5) | 0 (0.0%, #9, 0) | 226 (2.1%, 6) |
| Invalid construct usage (model-specific) | 5 (0.2%, #9, 3) | 142 (2.2%, #4, 12) | 26 (1.3%, #8, 6) | 173 (1.6%, 12) |
| Type mismatch | 42 (2.1%, #7, 5) | 36 (0.6%, #6, 6) | 35 (1.7%, #7, 4) | 113 (1.1%, 10) |
| Invalid control flow | 21 (1.0%, #8, 5) | 10 (0.2%, #9, 5) | 80 (3.9%, #6, 5) | 111 (1.1%, 7) |

Model-specific categories (incorrect loop construct, unsupported construct usage, invalid construct usage): FT-CodeLlama-13B 365 (18.0%), FT-StarCoderBase-7B 232 (3.6%), FT-HPC-Coder-6.7B 239 (11.5%); all translators 7.9% of instances.
