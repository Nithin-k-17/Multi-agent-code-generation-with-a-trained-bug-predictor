# Multi-Agent Code Generation with a Trained Bug-Prediction Agent

### Full Project Report: an extension of FlowGen (SOEN-101, ICSE 2025)

| | |
|---|---|
| **Student** | _Name, Roll No._ |
| **Guide** | _Name_ |
| **Department / Institution** | _Department, University_ |
| **Base paper** | F. Lin, D. J. Kim, T.-H. Chen, *SOEN-101: Code Generation by Emulating Software Process Models Using Large Language Model Agents*, IEEE/ACM ICSE 2025, DOI 10.1109/ICSE55347.2025.00140 |
| **Area** | Agentic AI (multi-agent LLM systems) + Deep Learning (fine-tuned code transformer) |
| **Live demo** | https://huggingface.co/spaces/nithin1707/Bug_Predicting_Agent |

---

## Abstract

Multi-agent LLM systems such as **FlowGen** (the base paper) write code by giving LLM agents
software-engineering roles (requirement engineer, developer, tester, …) and letting them test and refine
each other's work. FlowGen showed that the **testing activity matters most**, but every agent in it is a
prompted LLM: nothing in the loop is *trained* to recognise buggy code, and nobody checks whether the
tests the agents write are themselves correct.

This project extends FlowGen in both directions. **(1) Deep learning:** we built an
execution-verified dataset of 8,465 LLM-written Python functions and fine-tuned **CodeBERT** to predict
whether a single function is buggy. On 2,869 held-out functions it reaches **ROC-AUC 0.701
[95% CI 0.681–0.719]** and **90.0% recall**, clearly beating all surface baselines (AUC 0.51–0.60), and it
transfers to HumanEval without any tuning (ROC-AUC 0.635, 96.1% recall). **(2) Agentic AI:** we built a
five-agent pipeline (Requirement → Developer → **Bug Predictor** → Tester → Fixer) that runs fully
offline on a laptop CPU, and evaluated it with a paired design, a **random-flag control**, McNemar tests
and bootstrap confidence intervals.

The key finding is that **wrong agent-written tests are the real bottleneck**: only 40–48% of the tests
written by the agents were correct, so the Fixer kept "repairing" correct code. Our redesign (candidate
voting, best-candidate selection and a fix guard) raised the accuracy of the tests actually used to
**70.6%**, cut fix iterations from 2.3 to 0.4 per problem and halved the run time, reaching
**53.3% pass@1** on 60 held-out MBPP problems, which equals the upper bound set by the 1.5B generator.
The random control shows honestly that, at this model size, the trained bug predictor ranks bad drafts
above chance but does **not** change pipeline outcomes. The whole system is delivered as a local web
application.

---

## Contents

1. [Introduction](#1-introduction)
2. [Base paper: FlowGen (SOEN-101)](#2-base-paper-flowgen-soen-101)
3. [What this project adds beyond the base paper](#3-what-this-project-adds-beyond-the-base-paper)
4. [System overview](#4-system-overview)
5. [Deep-learning component: the bug predictor](#5-deep-learning-component-the-bug-predictor)
6. [Agentic component: the multi-agent pipeline](#6-agentic-component-the-multi-agent-pipeline)
7. [Pipeline experiments and results](#7-pipeline-experiments-and-results)
8. [Engineering, web platform and deployment](#8-engineering-web-platform-and-deployment)
9. [Changes made during the project](#9-changes-made-during-the-project)
10. [Discussion: comparison with the base paper](#10-discussion-comparison-with-the-base-paper)
11. [Limitations and threats to validity](#11-limitations-and-threats-to-validity)
12. [Conclusions](#12-conclusions)
13. [Future work](#13-future-work)
14. [Reproducing the results](#14-reproducing-the-results)
15. [References](#15-references)

---

## 1. Introduction

### 1.1 Motivation
LLMs can write code, but small models make many mistakes, and even large ones fail silently. Multi-agent
systems try to fix this the way a software team would: one agent writes the code, another tests it,
another repairs it. Two questions remain open:

- Can a **trained deep-learning model** act as a reviewer agent and spot likely-buggy code *before* it is
  tested, so that testing and repair effort goes where it is needed?
- Are the **tests the agents write** actually correct? A tester agent that writes wrong expected values
  will make the fixer break correct code.

### 1.2 Objectives
1. Build a labelled, execution-verified dataset of LLM-generated Python functions.
2. Train and rigorously evaluate a deep-learning bug predictor for single functions.
3. Integrate it as an agent in a FlowGen-style multi-agent pipeline.
4. Measure, with proper controls and statistics, whether the bug predictor improves the pipeline.
5. Diagnose and fix whatever limits the pipeline.
6. Deliver everything as a free, offline, easy-to-use application.

### 1.3 Contributions
1. **Dataset:** 8,465 execution-labelled LLM-generated functions (MBPP + HumanEval), split by problem ID
   with no leakage.
2. **Trained bug predictor:** fine-tuned CodeBERT (125M) with a 4-configuration × 3-seed study, held-out
   evaluation with 95% bootstrap CIs and an out-of-distribution test.
3. **Bug-aware multi-agent pipeline** with a paired design and a random-flag ablation.
4. **Diagnosis of a failure mode not studied in FlowGen:** most agent-written tests are wrong. We
   measured it against the official reference solutions and fixed it with test voting (40% → 71%
   correct).
5. **Honest negative result:** the learned bug signal does not beat a random control inside the pipeline
   at the 1.5B scale, while a cheap execution signal (the task's own example test) gives the same gain
   with 41% fewer generation calls.
6. **Working product:** a web app (Bug Scanner + live Agent Pipeline) that runs on a laptop CPU, with a
   choice of code model (Qwen2.5-Coder 1.5B or 7B), and is **deployed publicly** on Hugging Face Spaces.

---

## 2. Base paper: FlowGen (SOEN-101)

### 2.1 What FlowGen does
FlowGen is a code-generation framework that **emulates software process models** with multiple LLM
agents:

| Aspect | FlowGen |
|---|---|
| Process models | **Waterfall**, **TDD** (test-driven development), **Scrum** |
| Agent roles | Requirement Engineer, Architect, Developer, Tester, Scrum Master (Scrum only) |
| Techniques | Role prompts, chain-of-thought, prompt composition, self-refinement; testers design, write and run tests and send failure reports to the developer |
| LLM | GPT-3.5 (cloud API), zero-shot |
| Benchmarks | HumanEval, HumanEval-ET, MBPP, MBPP-ET |
| Baselines | RawGPT (single prompt), CodeT, Reflexion |
| Other analyses | Code smells and exception handling (static checkers), ablation of each activity, stability across GPT-3.5 versions and temperatures, 5 repeated runs |

### 2.2 Main results of FlowGen
- FlowGen-Scrum was the best process model: **Pass@1 75.2 (HumanEval), 65.5 (HumanEval-ET), 82.5 (MBPP),
  56.7 (MBPP-ET)**, about 15% better on average than RawGPT.
- **Removing the testing activity hurt most** (Pass@1 drops of 17.0% to 56.1%).
- Design and code review reduced code smells and added exception handling.
- Combining CodeT with FlowGen-Scrum gave the highest Pass@1.

### 2.3 Gaps this project addresses
| Gap in FlowGen | How this project addresses it |
|---|---|
| No trained component: every agent is a prompted LLM | A fine-tuned **CodeBERT bug-prediction agent** |
| Tests written by the agents are assumed correct | We **measure test correctness** against the official reference solutions and **fix it** |
| Fixes are always accepted | **Fix guard:** a fix is kept only if it passes more trusted tests |
| Gains compared against other methods, not against a placebo | A **random-flag control** at the same flag rate isolates the learned signal |
| Requires a paid cloud LLM | Runs **offline on a laptop CPU** with open models |
| Research code only | A usable **web application** |

---

## 3. What this project adds beyond the base paper

| | FlowGen (base paper) | This project |
|---|---|---|
| Deep learning | None | Fine-tuned CodeBERT bug predictor, 4 configurations × 3 seeds, held-out + OOD evaluation |
| Own dataset | No (uses benchmarks only) | 8,465 execution-verified LLM functions, problem-level splits |
| Test-quality check | Not measured | Measured: only 40–48% of agent tests correct; raised to 70.6% |
| Candidate generation | Single solution | k = 4 candidates (greedy + 3 seeded samples), test voting, best-candidate selection |
| Fixer safety | Always accepts the fix | Fix guard (keeps only strictly better fixes) |
| Experimental control | Comparison with other methods | Paired design + random-flag ablation + McNemar + bootstrap CIs |
| Leakage control | Oracles hidden from agents | Same, plus all design decisions made on a separate validation split |
| Cost / deployment | GPT-3.5 cloud API | Free, offline, laptop CPU (llama.cpp / Ollama), 1.5B or 7B model |
| Speed engineering | Not addressed | Prompt-lookup decoding (≈2× faster), answer cache, streaming |
| Interface | None | Web app: Bug Scanner, live Agent Pipeline, model picker |

**What FlowGen covers and this project does not** (see §11): three process models, Architect and Scrum
Master roles, the -ET benchmarks, code-smell analysis, five repeated runs and temperature/version
stability.

---

## 4. System overview

```
                                   ┌──────────────────────────────────────────────┐
  Task (text + function name) ───► │ 1. Requirement agent   → structured spec       │
                                   │ 2. Developer agent     → k candidate solutions │
                                   │ 3. Bug-Prediction agent (CodeBERT)             │
                                   │      risk score, risky functions, hot lines    │
                                   │ 4. Tester agent        → tests, voting, sandbox│
                                   │ 5. Fixer agent         → guarded repair loop   │
                                   └───────────────┬──────────────────────────────┘
                                                   ▼
                                       final code + test report + bug report
```

| Layer | Technology |
|---|---|
| Bug predictor | `microsoft/codebert-base` (125M), PyTorch + Hugging Face Transformers, max length 512 |
| Code generator | Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) via llama.cpp or Ollama; optional Qwen2.5-Coder-7B |
| Sandbox | Isolated Python subprocess (`python -I`, temp dir, 5 s timeout) |
| Backend | FastAPI, Server-Sent Events for live streaming |
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Statistics | scikit-learn metrics, bootstrap CIs (2,000 resamples), exact McNemar test |

---

## 5. Deep-learning component: the bug predictor

### 5.1 Task
Given **one** Python function written by an LLM, predict whether it is buggy (fails its tests) or clean.

### 5.2 Why earlier attempts failed
Two earlier versions of the predictor stayed near chance:

| Version | Training data | Val ROC-AUC | Pointwise accuracy | Problem |
|---|---|:---:|:---:|---|
| v1 | BugsInPy git diff hunks | 0.608 | 54.8% | Repository fragments look nothing like whole LLM functions |
| v2 | Synthetic AST mutations (`+`→`-`, `<`→`<=`) | 0.587 | 54.4% | LLM bugs are logic and edge-case errors, not single-token typos |
| **v3 (final)** | **Real LLM-generated functions, labelled by execution** | **0.738** | 65–69% (val) | Matches the real bug distribution |

**Lesson:** the training data must look like the code the model will judge.

### 5.3 Dataset construction
1. **Problem-level splits (no leakage):** MBPP 1–10 prompt examples · **11–510 train** · **511–600
   validation** · **601–974 held-out test** · **HumanEval 0–163** out-of-distribution test, never used for
   training or tuning.
2. **Generation:** Qwen2.5-Coder-1.5B-Instruct, T = 0.8, top-p = 0.95, about 8 samples per problem.
3. **Filtering:** drop code that does not parse; drop exact duplicates per problem.
4. **Labelling by execution:** run the official tests in a sandbox (5 s timeout). Passes all → **clean
   (0)**; assertion failure or exception → **buggy (1)**; timeouts dropped.

| Split | Problems | Samples | Clean | Buggy | Pass rate |
|---|:---:|:---:|:---:|:---:|:---:|
| Train (MBPP 11–510) | 500 | 3,831 | 1,658 | 2,173 | 43.3% |
| Validation (MBPP 511–600) | 90 | 684 | 296 | 388 | 43.3% |
| Test (MBPP 601–974) | 374 | 2,869 | 1,336 | 1,533 | 46.6% |
| HumanEval (0–163) | 164 | 1,081 | 673 | 408 | 62.3% |

### 5.4 Baselines (must be beaten)
| Baseline | Val ROC-AUC | Val accuracy |
|---|:---:|:---:|
| Code length (characters) | 0.599 | 60.1% |
| Line count | 0.579 | 56.6% |
| TF-IDF 1–2 grams + logistic regression | 0.569 | 59.1% |
| Majority class (always "buggy") | 0.500 | 56.7% |

On problem-balanced data the length baselines drop to ≈ 0.537, so size and keywords carry almost no
signal. A useful model must clearly exceed AUC 0.65.

### 5.5 Training and model selection
AdamW, learning rate 2e-5, 10% warm-up, batch 16, fp16, early stopping on validation ROC-AUC (patience 2),
**3 seeds (42, 7, 123)** per configuration, trained on an NVIDIA RTX 4060 laptop GPU.

| Run | Model | Data | Max len | Val ROC-AUC (mean ± sd) | Best seed | Decision |
|---|---|---|:---:|:---:|:---:|---|
| 1 | CodeBERT | LLM functions (3,831) | 256 | 0.703 ± 0.007 | 0.713 | Baseline config |
| 2 | CodeBERT | LLM + mutations + BugsInPy (8,030) | 256 | 0.676 ± 0.007 | 0.684 | **Negative transfer**, discarded |
| **3** | **CodeBERT** | **LLM functions (3,831)** | **512** | **0.714 ± 0.017** | **0.738** | **Champion** |
| 4 | UniXcoder | LLM functions (3,831) | 256 | 0.643 ± 0.006 | 0.651 | Weaker, discarded |

**Findings:** adding out-of-domain data *hurts* (−0.028 AUC); longer context helps; CodeBERT beats
UniXcoder by 0.06–0.07 AUC.

**Champion:** Run 3, seed 123 (val ROC-AUC 0.738). Two thresholds were chosen **on validation only**:
- **τ = 0.34** (F1-optimal): used for standalone scanning and for all metrics below.
- **τ = 0.795** (40% flag rate): used inside the pipeline, where flags trigger extra work.

### 5.6 Held-out evaluation (run once, 95% bootstrap CIs, 2,000 resamples)

| Benchmark | n | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **MBPP test** (374 unseen problems) | 2,869 | **0.701** [0.681, 0.719] | 60.1% [58.4, 61.8] | 58.2% [56.2, 60.0] | **90.0%** [88.5, 91.5] | 0.707 [0.690, 0.723] |
| **HumanEval** (zero-shot, out of distribution) | 1,081 | **0.635** [0.600, 0.669] | 42.6% [39.6, 45.6] | 39.3% [36.4, 42.5] | **96.1%** [94.0, 97.9] | 0.558 [0.527, 0.591] |
| BugsInPy (repository diffs, stress test) | 248 | 0.443 [0.372, 0.515] | 48.8% | 49.3% | 83.9% | 0.621 |

**Reading the results**
- The model clearly beats every baseline (0.70 vs ≤ 0.60) and catches **9 in 10 buggy functions**.
- It ranks bugs on a benchmark it never saw (HumanEval, AUC 0.635).
- Accuracy is modest (60%) because τ = 0.34 is tuned for **high recall**: it flags 83% of MBPP test code.
  Missing a bug is costlier than re-checking correct code, so this trade-off is deliberate.
- It does **not** work on repository diff fragments (BugsInPy, below chance). It is a predictor for
  whole, self-contained functions only.

### 5.7 Bug localisation
For flagged code the agent also reports **risky functions** (each function scored separately via the
syntax tree) and **hot lines** (gradient × input saliency of the "buggy" logit, summed per line).
Occlusion (replacing a line with `pass`) was tried first and rejected: CodeBERT rates the edited code as
*more* buggy, so it gives misleading lines.

---

## 6. Agentic component: the multi-agent pipeline

### 6.1 Agents
| # | Agent | Input | Output |
|---|---|---|---|
| 1 | **Requirement** | task text, example tests | structured spec (inputs, outputs, edge cases) |
| 2 | **Developer** | spec | greedy draft + 3 seeded samples (T = 0.8) |
| 3 | **Bug Predictor** (CodeBERT) | code | risk score, flag, risky functions, hot lines |
| 4 | **Tester** | spec (and, if flagged, the risky lines) | base tests; targeted edge-case tests for flagged code; sandboxed execution |
| 5 | **Fixer** | code, failing tests, risk info | repaired code (accepted only if strictly better) |

### 6.2 Three variants (for a fair experiment)
| Variant | Steps | Purpose |
|---|---|---|
| `baseline` | 1, 2, 4, 5 | FlowGen-style pipeline without the bug predictor |
| `proposed` | 1, 2, **3**, 4, 5 | CodeBERT guides targeted tests, the Fixer and candidate tie-breaks |
| `random` | 1, 2, 3*, 4, 5 | **Control:** flags drawn at random at the same flag rate |

**Paired design:** steps 1–2 and the base tests are produced once per problem and shared by all three
variants, and decoding is deterministic, so any difference between variants comes only from step 3.

**No leakage:** agents see only the task's public example test. The official hidden tests are used only
to score pass@1. All design choices were made on validation problems (MBPP 511–540); the test problems
(MBPP 601–660) were run once with the frozen design.

### 6.3 Pipeline v1 and its diagnosis
v1 used one draft, accepted every agent test and every fix. On 30 test problems all three variants
scored **50.0% pass@1** and the Fixer repaired **0%** of failing drafts while spending 2.3 iterations per
problem.

To find out why, we checked every agent-written test against MBPP's **official reference solutions**
(analysis only, never shown to the agents):

- **Only 47.8% of the agent-written tests were correct.**
- Of 64 test failures on first drafts, **45 were caused by wrong tests** and only 19 by real bugs.
- The Fixer was mostly "repairing" correct code to match wrong expected values.
- The task's own example test caught 13 of the 15 truly failing drafts.

### 6.4 Pipeline v2 (applied equally to all variants)
1. **Multiple candidates:** greedy draft + 3 seeded samples (reproducible).
2. **Test voting:** a generated test is kept only if a majority of the *trusted* candidates (those passing
   the example test) pass it. If no example vouches for a lone candidate, all tests are kept.
3. **Best-candidate selection:** start from the candidate passing the most trusted tests (example tests
   first). Ties: lowest CodeBERT risk (`proposed`) or random (`random`).
4. **Fix guard:** a fix is kept only if it passes strictly more trusted tests; example tests are shown to
   the Fixer first and marked as certainly correct.

---

## 7. Pipeline experiments and results

**Setup:** Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) on an Intel i5-1235U laptop CPU, ≤ 3 fix iterations,
5 base + 5 targeted tests, pipeline threshold τ = 0.795. Confidence intervals for proportions are 95%
Wilson intervals; paired differences use 95% bootstrap CIs and the exact McNemar test.

### 7.1 Main result (held-out MBPP 601–660, n = 60, run once)

| Metric | v1 (MBPP 601–630) | **v2 (MBPP 601–660)** |
|---|:---:|:---:|
| Correct agent-written tests (raw) | 47.8% | 40.0% (102/255) [34.2, 46.1] |
| **Correct tests actually used** | 47.8% | **70.6%** (77/109) [61.5, 78.4] |
| Fix iterations per problem | 2.30 | **0.40** |
| Correct drafts broken by the Fixer | 0% | 0% |
| Failing drafts repaired by the Fixer | 0% | 3.4% (1/29) |
| Time per problem (3 variants) | 118 s | **60 s** |
| First-draft pass@1 | 50.0% | 51.7% [39.3, 63.8] |
| **Final pass@1 (all variants)** | 50.0% [33.2, 66.8] | **53.3%** [40.9, 65.4] |
| Upper bound (any of the 4 candidates correct) | n/a | 53.3% |
| Same 30 problems (601–630) | 15/30 | 15/30 |

**Interpretation:** v2 makes the agents use far better tests and stops wasted repair work. Selection
reaches the upper bound: whenever any candidate was correct, the pipeline picked it. The remaining
46.7% of problems had no correct candidate at all, which is a limit of the 1.5B generator.

### 7.2 Does the bug predictor help the pipeline?

| Comparison (n = 60, paired) | pass@1 difference | 95% CI | Discordant pairs | McNemar p |
|---|:---:|:---:|:---:|:---:|
| proposed vs baseline | 0.0% | [0.0, 0.0] | 0 / 0 | 1.00 |
| random vs baseline | 0.0% | [0.0, 0.0] | 0 / 0 | 1.00 |
| proposed vs random | 0.0% | [0.0, 0.0] | 0 / 0 | 1.00 |

On validation (MBPP 511–540, n = 30) the picture was the same: baseline 43.3%, proposed 40.0%,
random 43.3% (difference −3.3%, 95% CI [−10.0, 0.0], p = 1.0).

**Bug predictor on the first drafts (n = 60):** ROC-AUC **0.670**, so it ranks failing drafts above
chance, but at the pipeline threshold it flags only 10% of drafts with 10.3% recall.

### 7.3 When is extra work worth it? (offline policy analysis)
Replaying the recorded candidates (no new LLM calls) to compare ways of deciding when to generate the 3
extra candidates:

| Policy | pass@1 | Developer calls / problem |
|---|:---:|:---:|
| Never (greedy draft only) | 51.7% | 1.00 |
| Always | 53.3% | 4.00 |
| Only if CodeBERT flags the draft | 51.7% | 1.45 |
| At random, same rate (control) | 51.9% | 1.44 |
| **Only if the draft fails the task's example test** | **53.3%** | **2.35** |
| Example test failed **or** CodeBERT flag | 53.3% | 2.55 |

Tie-breaking among equally tested candidates (51 real ties): first 53.3%, CodeBERT 53.3%, random 52.9%.

**Result:** a cheap execution signal (failing the example test) keeps the full gain with **41% fewer
generation calls**; the learned signal is no better than random as a gate.

### 7.4 Speed of the local backends
Same 3 MBPP problems × 3 variants, answer cache off, laptop CPU:

| Backend | Time |
|---|:---:|
| Ollama | 443 s |
| llama.cpp | 375 s |
| **llama.cpp + prompt-lookup decoding** | **185 s** (≈ 2× faster) |

---

## 8. Engineering, web platform and deployment

### 8.1 Running locally
- **Backends:** llama.cpp (fastest), Ollama, or PyTorch, chosen automatically.
- **Prompt-lookup decoding** (implemented in this project): the Fixer mostly copies code from its prompt,
  so copied tokens are verified in one step; ≈ 2× faster repairs.
- **Answer cache:** identical prompts are answered instantly, so experiments can be resumed.
- **Model choice:** the website offers Qwen2.5-Coder **1.5B** (default, fastest) or **7B** (better code,
  about 4× slower on CPU); the unused model is unloaded to save memory.

### 8.2 Web application
| Page | What it does |
|---|---|
| **Bug Scanner** | Paste Python code and get the CodeBERT risk score per function, highlighted risky lines, exact static findings (syntax, runtime checks) with line numbers and severity, and an AI review with a suggested fix that is re-checked |
| **Agent Pipeline** | Describe a task (or pick a benchmark problem) and watch the five agents work live: spec, code, bug check, tests, fixes, with a token-speed meter, speed presets, an optional baseline/random comparison and a model picker |

### 8.3 Quality checks
An offline self-test (`python main.py selftest`) runs both pipeline versions end to end with a scripted
LLM, and `python main.py check` verifies that every required part is installed.

---

### 8.4 Public deployment (Hugging Face Spaces)

The complete system is deployed and publicly usable, with no installation:

- **Space page:** https://huggingface.co/spaces/nithin1707/Bug_Predicting_Agent
- **Direct app link:** https://nithin1707-bug-predicting-agent.hf.space

| Aspect | Details |
|---|---|
| Platform | Hugging Face Spaces, free tier (Gradio SDK, ZeroGPU hardware) |
| What runs | The same FastAPI server and React website as on the laptop, started by `app.py` on port 7860 |
| Models | Trained CodeBERT bug predictor (uploaded with the app) and Qwen2.5-Coder-1.5B, 4-bit, llama.cpp (downloaded on first start) |
| Compute | CPU only. ZeroGPU requires one declared GPU function, so the app registers a placeholder that is never called |
| Safety | Visitor code (test cases, scanned code) runs in an isolated subprocess with a timeout and Linux limits on CPU time, memory (1 GB), file size and open files |
| Verified live | Website loads; models ready; Bug Scanner returns score and findings; a full agent run solved MBPP 601 in 74 s |

**Limits of the free tier:** runs are slower than on a laptop (about 1–2 minutes per task on the Fast
preset), one agent run can be active at a time (others wait), only the 1.5B model is available, and the
Space sleeps after 48 hours without visitors (the next visit wakes it in about a minute).

**Redeploying after a change:** `python deploy/huggingface/deploy.py nithin1707/Bug_Predicting_Agent`
rebuilds the upload folder (only the files the website needs, about 504 MB) and uploads it; the Space
then restarts automatically.

**Problems solved during deployment:** (1) a library conflict between the newest Gradio and
`transformers` 4.x, fixed by pinning Gradio 5.50; (2) ZeroGPU stops apps without a GPU function, fixed
with the placeholder above (the free CPU-only tier is not offered for this Space without a paid plan).

---

## 9. Changes made during the project

| Stage | Change | Why | Effect |
|---|---|---|---|
| Bug predictor v1 → v2 | BugsInPy diffs → synthetic AST mutations | v1 near chance on whole functions | Still near chance (AUC 0.59) |
| Bug predictor v2 → v3 | Real LLM functions labelled by execution | Training data must match real LLM bugs | AUC 0.59 → **0.74** (val), **0.70** (test) |
| Model study | 4 configurations × 3 seeds, 256 → 512 tokens, CodeBERT vs UniXcoder | Choose the best model fairly | Champion: CodeBERT, 512 tokens |
| Thresholds | τ = 0.34 (scanner) and τ = 0.795 (pipeline), both on validation | Different jobs need different trade-offs | No test-set tuning |
| Pipeline v1 → v2 | Candidates, test voting, selection, fix guard | 52% of agent tests were wrong | Tests used 48% → 71% correct; 2× faster |
| Localisation | Occlusion → gradient × input saliency | Occlusion pointed at wrong lines | Meaningful hot lines |
| Speed | PyTorch → llama.cpp/Ollama, prompt-lookup decoding, cache | CPU-only laptop | Interactive runs in about 1 minute |
| Test voting fix | Keep all tests when a lone candidate has no example to vouch for it | Fast mode deleted every test | Tests now always reach the Fixer |
| Product | Web app, live streaming, model picker (1.5B / 7B) | Make the system usable | One command: `python main.py serve` |
| Deployment | Hugging Face Space, sandbox resource limits, Gradio 5.50 pin, ZeroGPU placeholder | Let anyone use it without installing | Public link, verified working |

---

## 10. Discussion: comparison with the base paper

| | FlowGen | This project |
|---|---|---|
| MBPP Pass@1 | **82.5%** (GPT-3.5, best process model) | 53.3% (1.5B open model on a CPU) |
| Trained component | none | CodeBERT bug predictor (AUC 0.70, recall 90%) |
| Test correctness | assumed | measured and improved (40% → 71%) |
| Breadth | 3 process models, 4 benchmarks, code-smell analysis | 1 pipeline, MBPP pipeline study (bug model also on HumanEval) |
| Cost | paid API | free, offline |

The Pass@1 numbers are **not directly comparable**: GPT-3.5 is a far larger model, and our pipeline
already reaches the upper bound set by its 1.5B generator. FlowGen is broader and more accurate; this
project goes deeper in a direction FlowGen does not cover. It adds a trained deep-learning agent, checks
the reliability of the agents' own tests (FlowGen's most important activity), and uses a control
experiment that separates real effects from chance.

FlowGen found that testing matters most. This project adds that **the quality of the tests matters just
as much**: with a small model, most agent-written tests are wrong, and fixing that is what made the
pipeline both faster and more reliable.

---

## 11. Limitations and threats to validity

- **Small pipeline sample:** 60 test problems, so pass@1 has wide intervals (±12 points). Small effects
  cannot be ruled out, although the paired variants agreed on every problem.
- **Single generator family:** all pipeline results use Qwen2.5-Coder-1.5B. With a stronger generator,
  more problems have several correct candidates, and the bug predictor's tie-breaking may matter more.
- **One run, greedy decoding:** FlowGen averaged 5 runs. Our design is deterministic and paired, but
  sensitivity to temperature was not studied.
- **Benchmarks:** MBPP and HumanEval are short, self-contained functions. Results may not transfer to
  real multi-file projects, and the predictor fails on repository diffs (BugsInPy AUC 0.44).
- **Threshold trade-off:** the high-recall threshold flags most code; the pipeline threshold flags few.
  Neither makes the predictor useful as a pipeline gate at this scale.
- **Not covered from FlowGen:** process-model comparison (Waterfall/TDD/Scrum), Architect and Scrum Master
  roles, the -ET benchmarks and code-smell analysis.

---

## 12. Conclusions

1. **A trained deep-learning bug predictor works for single LLM-written functions:** ROC-AUC 0.701
   [0.681, 0.719] and 90% recall on unseen problems, well above all baselines, and it transfers to
   HumanEval (AUC 0.635).
2. **The training data must match the target code:** real, execution-labelled LLM functions beat
   repository diffs and synthetic mutations by a wide margin, and mixing them in causes negative transfer.
3. **In small-model agent loops, wrong tests are the bottleneck:** 52–60% of agent-written tests were
   wrong. Candidate voting raised the accuracy of the tests used to 70.6%, cut repair iterations from 2.3
   to 0.4 and halved the run time.
4. **The bug predictor does not change pipeline outcomes at the 1.5B scale.** The random-flag control makes
   this a clean, honest negative result: identical pass@1 (53.3%), McNemar p = 1.0.
5. **Cheap execution signals beat the learned signal as a gate:** using the example test to decide when
   to generate more candidates keeps the full gain with 41% fewer calls.
6. **The remaining ceiling is the generator:** pass@1 already equals the best any candidate achieved;
   larger models or more diverse candidates are needed for further gains.

---

## 13. Future work

- Run the pipeline with the 7B model (already available in the app) and on HumanEval.
- Give the bug predictor the **task description together with the code**, since a bug often depends on
  what was asked.
- Train on a larger and more diverse set of generated functions.
- Add FlowGen's process models (TDD, Scrum) and an Architect/review agent, and compare them with the same
  controls.
- Add code-smell and reliability analysis as in FlowGen.

---

## 14. Reproducing the results

```bash
python main.py check                                  # verify the installation
python main.py selftest                               # offline end-to-end test

python main.py evaluate --help                        # held-out bug-predictor evaluation (§5.6)
python main.py experiment --split mbpp_test --limit 60 \
       --variants baseline,proposed,random --run-name mbpp_test60_v2   # pipeline (§7.1)
python main.py analyze  results/mbpp_test60_v2        # tables, CIs, McNemar
python main.py policies results/mbpp_test60_v2        # policy analysis (§7.3)
python main.py experiment ... --legacy                # reproduce pipeline v1

python main.py serve                                  # web app at http://localhost:8000
python deploy/huggingface/deploy.py USER/SPACE       # deploy to a Hugging Face Space (§8.4)
```

Recorded results are in `results/` (`mbpp_test_ollama_30` = v1, `dev_val30_v2` = validation,
`mbpp_test60_v2` = final test). Training runs are listed in `docs/EXPERIMENTS.md`, the dataset in
`docs/DATA_CARD.md` and the full step-by-step history in `docs/DEVLOG.md`.

---

## 15. References

1. F. Lin, D. J. Kim, T.-H. Chen. *SOEN-101: Code Generation by Emulating Software Process Models Using
   Large Language Model Agents.* IEEE/ACM ICSE 2025. DOI 10.1109/ICSE55347.2025.00140.
2. Z. Feng et al. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages.* Findings of
   EMNLP 2020.
3. D. Guo et al. *UniXcoder: Unified Cross-Modal Pre-training for Code Representation.* ACL 2022.
4. B. Hui et al. *Qwen2.5-Coder Technical Report.* 2024.
5. J. Austin et al. *Program Synthesis with Large Language Models* (MBPP). 2021.
6. M. Chen et al. *Evaluating Large Language Models Trained on Code* (HumanEval). 2021.
7. B. Chen et al. *CodeT: Code Generation with Generated Tests.* ICLR 2023.
8. N. Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning.* NeurIPS 2023.
9. R. Widyasari et al. *BugsInPy: A Database of Existing Bugs in Python Programs.* ESEC/FSE 2020.
10. Q. McNemar. *Note on the sampling error of the difference between correlated proportions.*
    Psychometrika, 1947.
