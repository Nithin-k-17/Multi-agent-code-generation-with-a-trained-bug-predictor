# Experiment Tracker (v3 Project Cycle)

This table tracks all training runs in v3. All metrics are recorded on the validation set during development; held-out test evaluations occur strictly once at the end.

| Run ID | Model Architecture | Training Dataset | Hyperparameters | Seed | Val ROC-AUC | Val F1 | Val Accuracy | Training Time | Status / Decision | Notes |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---|---|
| B1 | Baseline: Code Length | v3 LLM train (full 3,831) | Logistic Regression | 42 | 0.5987 | 0.6950 | 0.6009 | <1s | Discarded | Pairwise: 0.5405. Length shortcut |
| B1-bal | Baseline: Code Length | v3 LLM train (bal 1,018) | Logistic Regression | 42 | 0.5374 | 0.5327 | 0.5255 | <1s | Discarded | Balanced pairwise: 0.5302 |
| B2 | Baseline: Line Count | v3 LLM train (full 3,831) | Logistic Regression | 42 | 0.5794 | 0.6674 | 0.5658 | <1s | Discarded | Pairwise: 0.5531. Line count shortcut |
| B2-bal | Baseline: Line Count | v3 LLM train (bal 1,018) | Logistic Regression | 42 | 0.5367 | 0.5411 | 0.5153 | <1s | Discarded | Balanced pairwise: 0.5603 |
| B3 | Baseline: Char + Line | v3 LLM train (full 3,831) | Logistic Regression | 42 | 0.5988 | 0.6950 | 0.6009 | <1s | Discarded | Pairwise: 0.5405 |
| B4 | Baseline: TF-IDF (1-2 gram) | v3 LLM train (full 3,831) | TF-IDF + Logistic Reg | 42 | 0.5694 | 0.6767 | 0.5906 | ~3s | Discarded | Pairwise: 0.5618. Lexical bag-of-words |
| B4-bal | Baseline: TF-IDF (1-2 gram) | v3 LLM train (bal 1,018) | TF-IDF + Logistic Reg | 42 | 0.5142 | 0.4186 | 0.4898 | ~1s | Discarded | Balanced pairwise: 0.4871 |
| B5 | Baseline: Majority Class | v3 LLM train (full 3,831) | Constant label 1 | 42 | 0.5000 | 0.7239 | 0.5673 | <1s | Discarded | Chance level AUC 0.5000 |
| C1 | CodeBERT Run 1 (3 seeds) | v3 LLM train (3,831) | lr=2e-5, max_len=256, fp16 | 42, 7, 123 | 0.7033 ± 0.0072 | 0.7484 ± 0.0091 | 0.6545 ± 0.0148 | ~11.6m (3 seeds) | Kept (Candidate) | Seed 123 peak AUC 0.7132, Acc 67.54% |
| C2 | CodeBERT Run 2 (3 seeds) | Combined train (8,030) | lr=2e-5, max_len=256, fp16 | 42, 7, 123 | 0.6755 ± 0.0067 | 0.7248 ± 0.0030 | 0.6335 ± 0.0108 | ~23.4m (3 seeds) | Discarded | Combined data causes negative transfer (AUC -0.0278 vs C1) |
| C3 | CodeBERT Run 3 (3 seeds) | v3 LLM train (3,831) | lr=2e-5, max_len=512, fp16 | 42, 7, 123 | 0.7143 ± 0.0169 | 0.7403 ± 0.0127 | 0.6637 ± 0.0155 | ~35.4m (3 seeds) | Champion Selected | Peak AUC 0.7375 (Seed 123), Peak Acc 68.57% (Seed 42). Top performer across all runs. |
| C4 | UniXcoder Run 4 (3 seeds) | v3 LLM train (3,831) | lr=2e-5, max_len=256, fp16 | 42, 7, 123 | 0.6432 ± 0.0058 | 0.7314 ± 0.0063 | 0.6014 ± 0.0079 | ~10.1m (3 seeds) | Discarded | Trails CodeBERT by -0.0711 AUC and -6.23% accuracy |


## Multi-agent pipeline runs

| Run | Split / problems | Pipeline | LLM backend | Purpose | pass@1 (base / prop / rand) | Test accuracy used | Notes |
|---|---|---|---|---|:---:|:---:|---|
| `mbpp_test_ollama_30` | test 601–630 | v1 | Ollama | first experiment | 50.0 / 50.0 / 50.0 | 47.8% | Fixer repaired 0%; 45/64 draft failures caused by wrong tests |
| `dev_val30_v2` | val 511–540 | v2 (k=4) | llama.cpp `--fast` | design decisions | 43.3 / 40.0 / 43.3 | 64.4% (raw 45.7%) | CodeBERT ROC-AUC on drafts 0.505; example-gated sampling = always-sample at 65% of the calls |
| `mbpp_test60_v2` | test 601–660 | v2 (k=4) | llama.cpp `--fast` | final, run once | 53.3 / 53.3 / 53.3 | 70.6% (raw 40.0%) | Selection = upper bound 53.3%; fix iterations 2.3 → 0.4; 60 s per problem |
