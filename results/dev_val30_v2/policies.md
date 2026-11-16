# Policy analysis: `dev_val30_v2` (30 problems)

CodeBERT flags 13% of first drafts (threshold 0.795).

| policy | pass@1 (start code) | developer calls / problem |
|---|---|---|
| greedy | 40.0% | 1.00 |
| always | 43.3% | 4.00 |
| gate:codebert | 40.0% | 1.40 |
| gate:random | 40.4% | 1.40 |
| gate:example | 43.3% | 2.60 |
| gate:both | 43.3% | 3.00 |

Upper bound (any candidate passes): 43.3%

Tie-breaking among equally-tested candidates (27 problems had a real tie):

- first: 43.3%
- codebert: 40.0%
- random: 42.3%
