# Policy analysis: `mbpp_test60_v2` (60 problems)

CodeBERT flags 15% of first drafts (threshold 0.795).

| policy | pass@1 (start code) | developer calls / problem |
|---|---|---|
| greedy | 51.7% | 1.00 |
| always | 53.3% | 4.00 |
| gate:codebert | 51.7% | 1.45 |
| gate:random | 51.9% | 1.44 |
| gate:example | 53.3% | 2.35 |
| gate:both | 53.3% | 2.55 |

Upper bound (any candidate passes): 53.3%

Tie-breaking among equally-tested candidates (51 problems had a real tie):

- first: 53.3%
- codebert: 53.3%
- random: 52.9%
