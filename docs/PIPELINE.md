# Multi-Agent Pipeline with a Trained Bug-Prediction Agent

Implements proposal Section 3 (Steps 1–6) around the trained CodeBERT model (`bug_predictor/`, agents in `pipeline/agents.py`).

```
               shared by all variants (paired design)          variant-specific
 task ──► [1] Requirement ──► [2] Developer ──► draft ──┬──► [3] Bug-Prediction (proposed / random only)
                    │                                    │        score, risky functions, hot lines
                    └──► [4a] Tester: base tests ────────┤
                                                         ├──► [4b] Tester: targeted tests on flagged code
                                                         └──► run tests ─► fail? ─► [5] Fixer (+ risk info) ─► re-score ─► loop (≤ max_fix_iters)
                                                                          final code ─► official hidden tests ─► pass@1
```

| Variant | Steps | Purpose |
|---|---|---|
| `baseline` | 1, 2, 4, 5 | FlowGen / AgentCoder-style reactive pipeline |
| `proposed` | 1, 2, **3**, 4, 5 | CodeBERT risk score guides the Tester (targeted tests, run first) and the Fixer (flagged lines in prompt) |
| `random`   | 1, 2, 3*, 4, 5 | Ablation: same mechanism, but flags are random at the same flag rate. Separates the *learned signal* from simply *spending more tests* |

**v2 (default since the Phase F diagnosis).** Only ~48% of agent-written tests were correct, so the
Fixer kept "repairing" correct code. v2 adds, for every variant: k candidate solutions (`--candidates`,
default 4), test voting over the candidates that pass the example test, selection of the best-tested
candidate (proposed breaks ties by lowest CodeBERT risk, random at random), and a fix guard (keep a fix
only if it passes more trusted tests). `--legacy` reproduces v1 exactly. See `docs/FINAL_REPORT.md` §6–7.

**Fairness.** Steps 1–2 and the black-box base tests are produced once per problem and reused by every variant. Decoding is greedy. So any difference between variants comes from Step 3 and what Steps 4–5 do with it.

**No leakage.** Agents only see the public example tests (MBPP's first assert, HumanEval's docstring examples). The official tests are used only to score pass@1. The main experiment uses MBPP tasks **601–974** (the bug predictor never saw them) and HumanEval. The threshold is tuned on the **validation** split (MBPP 511–600).

**Localisation.** "Hot lines" come from gradient × input saliency of CodeBERT's *buggy* logit, summed per line. Occlusion (replacing a statement with `pass`) was tried first. It doesn't work: CodeBERT rates the edited code as *more* buggy.

## Files

| File | Role |
|---|---|
| `benchmarks.py` | Loads MBPP/HumanEval (downloads to `data/benchmarks/`), splits, public and hidden tests |
| `executor.py` | Sandboxed subprocess runner (`python -I`, temp dir, timeout), per-test results |
| `llm.py` | LLM backends: local HF (default Qwen2.5-Coder-1.5B-Instruct) or any OpenAI-compatible server |
| `agents.py` | The five agents |
| `pipeline.py` | Shared stage plus per-variant loop and metrics |
| `run_experiment.py` | CLI. Resumable; writes `results/<run>/{shared,results}.jsonl` |
| `analyze.py` | pass@1, fix iterations, test/LLM cost, McNemar and paired bootstrap, test accuracy vs. reference solutions; writes `summary.md` |
| `policies.py` | Offline replay of recorded candidates: compute-gating and tie-breaking policies (`python main.py policies RUN_DIR`) |
| `calibrate_threshold.py` | Chooses τ on `data/val.jsonl` for a target flag rate |
| `selftest.py` | Offline end-to-end test with a scripted fake LLM |

## How to run (from the `ZIPFORLAP/` folder)

```bash
pip install torch transformers accelerate scikit-learn numpy openai
python -m pipeline.selftest --model
python -m pipeline.calibrate_threshold --target-flag-rate 0.4
python -m pipeline.run_experiment --split mbpp_val --limit 30 --variants baseline,proposed,random --threshold-file pipeline/threshold_calibrated.json
python -m pipeline.run_experiment --split mbpp_test --variants baseline,proposed,random --threshold-file pipeline/threshold_calibrated.json
python -m pipeline.run_experiment --split humaneval --variants baseline,proposed,random --threshold-file pipeline/threshold_calibrated.json
python -m pipeline.analyze results/mbpp_test_qwen1.5b
```

A GPU is strongly recommended for the LLM. CodeBERT runs on CPU by default (`--bug-device`). Re-running the same command resumes where it stopped. Use `--run-name` to keep different configurations apart.

Useful knobs: `--max-fix-iters` (default 3), `--n-base-tests` (5), `--n-targeted-tests` (5), `--top-k-lines` (3), `--threshold`, `--llm openai --base-url http://localhost:11434/v1 --model qwen2.5-coder:1.5b` (Ollama).

## Speed

Measured on the i5-1235U laptop CPU (Qwen2.5-Coder-1.5B, greedy):

| Setting | Decode speed |
|---|---|
| fp32, 10 threads (old default) | 1.96 tok/s |
| fp32, 4 threads | 2.22 tok/s |
| int8, 10 threads | 3.24 tok/s |
| **int8, 4 threads (new CPU default)** | **4.54 tok/s** |

Other speed-ups:
- **Stop at code end.** Developer, Tester and Fixer stop generating as soon as their code block closes, instead of writing explanations up to the token limit.
- **Tighter token budgets.** The spec is limited to 320 tokens; tests get about 40 tokens per assert.
- **Response cache.** Identical greedy prompts are answered from `.cache/llm_cache.jsonl`, so re-runs and resumed experiments are instant.
- **Per-call time cap.** `PIPELINE_MAX_SECONDS_PER_CALL` (default 240) bounds the worst case.
- **Warm-up.** The web server loads both models at startup.

Environment variables:
- `PIPELINE_THREADS` (default 4)
- `PIPELINE_QUANTIZE` (`int8` | `none`; default int8 on CPU)
- `PIPELINE_MAX_SECONDS_PER_CALL`
- `PIPELINE_WARMUP=0` to skip warm-up

For the full experiment, a GPU (fp16) or an OpenAI-compatible server running a quantized model (e.g. Ollama `qwen2.5-coder:1.5b`, 4-bit llama.cpp kernels; not yet measured on this laptop) is typically several times faster again: `--llm openai --base-url http://localhost:11434/v1 --model qwen2.5-coder:1.5b`.

## Metrics reported (`summary.md`)
- **pass@1**: final code passes the official tests. Also shows the first draft's pass@1, shared by all variants.
- **avg fix iterations**: over all problems and over the problems that needed fixing. Also the iterations until the hidden tests first pass.
- **repaired initially failing / broke initially passing**: where the fix loop helps and where it hurts.
- **test executions, targeted tests, LLM calls/tokens, seconds**: cost of each variant.
- **Paired tests**: McNemar exact p-value on pass@1, and 95% bootstrap CIs for the pass@1 and fix-iteration differences.
- **Bug-agent quality on first drafts**: ROC-AUC, bug rate among flagged vs. unflagged drafts, recall.

## Known benchmark quirks
- HumanEval/47 and HumanEval/148 have wrong docstring examples. Both variants see them as public tests.
- 60 of the 164 HumanEval problems have no literal-argument docstring examples, so their only internal tests are the LLM-generated ones.
