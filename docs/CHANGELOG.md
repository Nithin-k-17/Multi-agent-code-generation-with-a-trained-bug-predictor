# Changelog (v3 Project Cycle)

All changes to code, datasets, configurations, and models are tracked here with semantic versioning.

---

## [v3.0.0] - 2026-10-13 01:30:00

### Initial Setup & Architectural Pivot
- **Motivation**: Pointwise classification accuracy in v1 and v2 remained near chance (~54%) because training data consisted of either real-world repository diff hunks (BugsInPy) or synthetic AST mutations (MBPP mutations), creating a major distribution shift when predicting bugs on realistic single-function LLM completions. Phase 3 focuses entirely on pointwise defect prediction for real LLM-generated Python code.
- **Backups Created**:
  - `backup_v1_v2/` containing all historical checkpoints, data splits, and reports.
- **Documentation System**:
  - Initialized `docs/DEVLOG.md`, `docs/CHANGELOG.md`, `docs/EXPERIMENTS.md`, and `docs/DATA_CARD.md`.
- **Infrastructure**:
  - Created `v3/` root folder with `v3/data/`, `v3/models/`, `v3/baselines/`, `v3/eval/`.
## [v3.1.0] - 2026-10-13 01:40:00
### Phase A Data Generation Pipeline
- **Added**: `v3_generate_data.py` - full automated generator and sandboxed test execution harness.
- **Data Partitions**:
  - Training: MBPP task IDs 11-510 (500 problems).
  - Validation: MBPP task IDs 511-600 (90 problems).
  - Test: MBPP task IDs 601-974 (374 problems, held-out).
- **Sampling**: 8 candidates per problem with Qwen2.5-Coder-1.5B-Instruct ($T=0.8, \text{top\_p}=0.95$).
- **Verification**: Subprocess sandboxing with timeout 5.0s, labeling pass=0, fail=1, discarding syntax errors and timeouts.
- **Output Target**: `v3/data/train.jsonl`, `v3/data/val.jsonl`, `v3/data/test.jsonl` plus problem-balanced counterparts.

## [v3.2.0] - 2026-10-13 03:45:00
### Phase A Completion & Phase B Trivial Baselines
- **Completed**: End-to-end dataset generation and sandboxed verification:
  - Validation: 684 samples (90 problems)
  - Train: 3,831 samples (500 problems)
  - Test: 2,869 samples (374 problems, held out)
- **Resolved**: GPU memory fragmentation via sequence chunking (2x4) and 250-char assertion truncation, eliminating prompt KV cache spikes.
- **Added**: Phase B evaluation in `v3_trivial_baselines.py`:
  - Code character length: Val AUC 0.5987 (balanced: 0.5374)
  - Line count: Val AUC 0.5794 (balanced: 0.5367)
  - TF-IDF bag-of-words: Val AUC 0.5694 (balanced: 0.5142)
  - Majority class: Val AUC 0.5000
## [v3.3.0] - 2026-10-13 09:15:00
### Phase C Model Training & Champion Architecture Selection
- **Runs Completed**:
  - Run 1 (CodeBERT, LLM data, `max_len=256`, 3 seeds): Val ROC-AUC 0.7033 ± 0.0072 (Seed 123 peak: 0.7132, Acc: 67.54%).
  - Run 2 (CodeBERT, Combined data [LLM + Mutations + BugsInPy], `max_len=256`, 3 seeds): Val ROC-AUC 0.6755 ± 0.0067. Proved out-of-domain negative transfer; discarded.
  - Run 3 (CodeBERT, LLM data, `max_len=512`, 3 seeds): Val ROC-AUC 0.7143 ± 0.0169 (Seed 123 peak: **0.7375**, Seed 42 Acc: **68.57%**, Prec: **70.64%**, Seed 7 BoN win rate: **49.44%**).
  - Run 4 (UniXcoder, LLM data, `max_len=256`, 3 seeds): Val ROC-AUC 0.6432 ± 0.0058 (Acc: 60.14%). Trails CodeBERT by -0.0711 AUC; discarded.
## [v3.4.0] - 2026-10-13 10:55:00
### Phase D Final Single-Shot Evaluation & Packaging
- **Evaluated**: Champion model `v3/runs/run3_codebert_maxlen512_seed_123` evaluated once across all 3 held-out benchmarks with 2,000 bootstrap resamples:
  - MBPP Held-Out Test (2,869 samples): ROC-AUC 0.7007 [95% CI: 0.6812 – 0.7189], Accuracy 60.06%, Recall 90.02%, Pairwise 58.11%, Best-of-N 46.52% vs Random 45.61%.
  - HumanEval Held-Out Benchmark (1,081 samples, zero-shot): ROC-AUC 0.6347 [95% CI: 0.5997 – 0.6694], Recall 96.08%, Pairwise 53.97%, Best-of-N 61.11% vs Random 60.22%.
  - BugsInPy Test Split (248 samples): ROC-AUC 0.4433, Accuracy 48.79%, Recall 83.87%, Pairwise 37.10%. Confirmed extreme domain divergence between full LLM functions and repository diff hunks.
- **Agent Integration**: Updated `bug_agent.py` defaults to champion model (`max_len=512`, $\tau=0.34$) and verified inference on CPU with AST decomposition.
- **Deliverables**: Generated submission packages via `v3_pack.py` (`bug_predictor_model_v3.zip`, `dataset_v3.zip`, `extras_v3.zip`).




