# Multi-Agent Code Generation with a Trained Bug-Prediction Agent

**Live demo:** https://huggingface.co/spaces/nithin1707/Bug_Predicting_Agent (free hosting: the first visit after a quiet period takes about a minute to wake up)

LLM multi-agent coding systems (FlowGen / AgentCoder style: Developer → Tester → Fixer) only react
**after** a test fails. This project adds a fifth agent, a **trained Bug-Prediction Agent**
(fine-tuned CodeBERT), between the Developer and the Tester. It scores the draft code for bug risk,
points to the risky functions and lines, and that information is used to:

1. generate **targeted tests** on the flagged code, and
2. give the **Fixer** the flagged lines in its prompt.

```
task ─► [1] Requirement ─► [2] Developer ─► draft ─┬─► [3] Bug Predictor (CodeBERT): risk score, risky lines
                 │                                 ├─► [4] Tester: base tests + targeted tests on flagged code
                 └─────────────────────────────────┴─► run tests ─► fail? ─► [5] Fixer (+ risk info) ─► repeat (max 3)
                                                     final code ─► official hidden tests ─► pass@1
```

| Component | Details |
|---|---|
| Bug predictor | `microsoft/codebert-base` (125M), fine-tuned, max length 512 |
| Code generator | `Qwen/Qwen2.5-Coder-1.5B-Instruct` (via Ollama, llama.cpp or Hugging Face) |
| Benchmarks | MBPP (train 11–510, val 511–600, test 601–974) and HumanEval (164, zero-shot) |
| Website | FastAPI back end + React/Vite front end |

**Main results of the bug predictor** (held-out, 95% bootstrap CIs; full details in [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)):

| Benchmark | ROC-AUC | Accuracy | Recall |
|---|---|---|---|
| MBPP test (2,869 functions, unseen problems) | 0.701 [0.681, 0.719] | 60.1% | 90.0% |
| HumanEval (1,081 functions, zero-shot) | 0.635 [0.600, 0.669] | 42.6% | 96.1% |
| Best trivial baseline (code length, validation) | 0.599 | 60.1% | – |

**Main results of the multi-agent pipeline** (MBPP test 601–660, run once with the design frozen on validation;
details in [docs/FINAL_REPORT.md §7](docs/FINAL_REPORT.md#7-pipeline-experiments-and-results)):

| | v1 pipeline | v2 pipeline |
|---|:---:|:---:|
| Agent-written tests that are actually correct | 47.8% | **70.6%** (candidate voting) |
| Fix iterations per problem | 2.3 | **0.4** (no more "fixing" correct code) |
| Time per problem (3 variants, laptop CPU) | 118 s | **60 s** |
| pass@1 | 50.0% | **53.3%** (= best of the generated candidates) |
| Bug agent vs. baseline vs. random control | no difference | no difference (p = 1.0) |

Honest takeaway: with a 1.5B generator, **wrong LLM-written tests** are the bottleneck; fixing them helps far more
than bug prediction. The trained predictor ranks failing drafts above chance (ROC-AUC 0.67) but does not change
outcomes, and the task's own example test is a better signal for when to spend extra compute (same pass@1 with
41% fewer generation calls).

---

## 1. Requirements

| | |
|---|---|
| **Python** | **3.10, 3.11, 3.12 or 3.13** (developed and tested with Python 3.13.1) |
| **OS** | Windows 10/11, macOS or Linux |
| **RAM / disk** | 8 GB RAM minimum (16 GB recommended); about 2 GB free disk (+3 GB if the Hugging Face LLM is downloaded) |
| **GPU** | Not needed to run the demo, the website or the evaluation. Recommended (NVIDIA, ≥ 6 GB) for **training** and **dataset generation** |
| **Node.js** | Not needed (the website is already built in `web/dist/`). Only needed to edit the front end |
| **Ollama** | Optional: an alternative way to run the code-generator LLM (same speed as the built-in llama.cpp backend) |

Python libraries (all listed in [requirements.txt](requirements.txt)):
`torch`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `numpy`, `scikit-learn`,
`openai` (client for Ollama), `fastapi`, `uvicorn`, `pydantic`.

---

## 2. Installation

Unzip the project, then open a terminal **inside the project folder** (the folder containing `main.py`).

### Option A: one-step setup script

Windows:
```bat
setup_windows.bat
```
macOS / Linux:
```bash
bash setup_unix.sh
```
The script creates a virtual environment in `.venv`, installs `requirements.txt` and runs `python main.py check`.

### Option B: manual setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux

# 2. Install the libraries
pip install --upgrade pip
pip install -r requirements.txt

# 3. Check that everything is in place
python main.py check
```

**NVIDIA GPU (optional, for training):** install the CUDA build of PyTorch **before** step 2, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu121`
(see https://pytorch.org/get-started/locally/ for your CUDA version).

### Optional: Ollama (faster code generation)

The multi-agent pipeline needs the Qwen2.5-Coder-1.5B LLM. The fastest way to run it on a normal laptop is Ollama:

1. Install Ollama from https://ollama.com/download
2. Download the model (≈1 GB):
   ```bash
   ollama pull qwen2.5-coder:1.5b
   ```
3. Keep Ollama running (it starts automatically after installation).

**Bigger model (optional).** `ollama pull qwen2.5-coder:7b` (≈4.7 GB, needs ~5 GB free RAM). On the
Agent Pipeline page, the model dropdown next to **Run pipeline** then offers *Qwen2.5-Coder 7B* (better
code, about 4× slower on CPU). The 1.5B model stays the default; every model pulled in Ollama appears in
that list, and the server unloads the model you are not using to free memory.

### Speed: which LLM backend is used

`auto` (the default) picks the first available of: **llama.cpp** (`llama-cpp-python` installed), **Ollama**
(running, with the model pulled), **PyTorch** (Hugging Face, slowest on CPU).

Measured on an i5 laptop CPU, same 3 MBPP problems × 3 variants, answer cache off:

| Backend | Time |
|---|---|
| Ollama | 443 s |
| llama.cpp | 375 s |
| llama.cpp + prompt-lookup decoding (`--fast`, website default) | **185 s** (≈2× faster, same pass/fail on every problem) |

llama.cpp and Ollama read prompts and write tokens at the same speed (~95 and ~27 tokens/s);
the gain comes from **prompt-lookup decoding**: the Fixer mostly re-writes code that is already in its
prompt, so the model checks several copied tokens in one step instead of one at a time.
It can change the wording of an answer in rare near-tie cases, so experiments use plain greedy decoding
unless you pass `--fast`; the website always uses it. Set `PIPELINE_LLM=ollama` to use Ollama instead.

---

## 3. Model and dataset

### Get the trained model

The trained weights file `model.safetensors` (≈500 MB) is too large for a normal GitHub file (100 MB limit),
so it is attached to this repository's **Releases** page instead. Everything else (tokenizer, config,
thresholds, dataset, results) is in the repository.

1. Download [`model.safetensors`](https://github.com/Nithin-k-17/Multi-agent-code-generation-with-a-trained-bug-predictor/releases/download/v1.0/model.safetensors)
   (499 MB, from the [v1.0 release](https://github.com/Nithin-k-17/Multi-agent-code-generation-with-a-trained-bug-predictor/releases/tag/v1.0)).
2. Put it in `models/bug_predictor/` (next to `config.json`).
3. Run `python main.py check`. It should show `[ok] bug-predictor model`.

Alternatively, retrain it yourself with `python main.py train ...` (GPU, see section 4).

| What | Where | Notes |
|---|---|---|
| Trained bug predictor (CodeBERT, ≈480 MB) | `models/bug_predictor/` | `model.safetensors` + tokenizer, `threshold.json` (τ = 0.34), `pipeline_threshold.json` (τ = 0.795, 40% flag rate) |
| Labelled dataset (7,384 MBPP + 1,081 HumanEval functions) | `data/processed/*.jsonl` | Built by `generate-data`; one JSON object per line: `task_id`, `prompt`, `code`, `label` (0 = clean, 1 = buggy) |
| Raw benchmarks | `data/benchmarks/` | `mbpp.jsonl`, `HumanEval.jsonl.gz` |

If `data/benchmarks/` is ever missing, the files are downloaded automatically on first use from:
- MBPP: https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
- HumanEval: https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz

(or download them manually and place them in `data/benchmarks/` with those file names).

Dataset files:

| File | Split | Samples |
|---|---|---|
| `train.jsonl` / `train_balanced.jsonl` | MBPP 11–510 | 3,831 / 1,018 |
| `val.jsonl` / `val_balanced.jsonl` | MBPP 511–600 | 684 / 196 |
| `test.jsonl` / `test_balanced.jsonl` | MBPP 601–974 (held out) | 2,869 / 806 |
| `humaneval_test.jsonl` / `humaneval_balanced.jsonl` | HumanEval 0–163 (held out) | 1,081 / 328 |
| `train_combined.jsonl` | train + AST mutations + BugsInPy (ablation, Run 2) | 8,030 |

---

## 4. How to run

**Main file: [`main.py`](main.py).** Every task is a command: `python main.py <command>`.
Run `python main.py --help` for the list, or `python main.py <command> --help` for a command's options.

### Quick start

```bash
python main.py check        # verify installation
python main.py demo         # score sample buggy / clean functions with the trained model
python main.py serve        # start the website, then open http://localhost:8000
```

The website has four pages: **Home** (overview with live numbers), **Bug Scanner** (paste code → bugs,
lines, severity and a fixed version), **Agent Pipeline** (run all 5 agents live on your own task or a benchmark
problem, with a live tokens/second meter) and **Results** (bug-model accuracy and the experiment dashboard:
pass@1 per variant, significance tests and every problem; it lists the runs in `results/`).

Score your own file from the terminal:
```bash
python main.py scan path/to/your_code.py
```

### Train

```bash
python main.py baselines                    # Phase B: trivial baselines (seconds, CPU)
python main.py train                        # Phase C: fine-tune CodeBERT with the champion settings (GPU recommended)
python main.py train --seed 42 --max-len 256                 # other runs from the report
python main.py train --model microsoft/unixcoder-base --max-len 256
python main.py train --train-file train_combined --max-len 256
```
Training writes the best checkpoint to `runs/<run-name>/`. To use your own model everywhere,
copy the files from that folder into `models/bug_predictor/` (or pass `--model runs/<run-name>` to `evaluate`).

To rebuild the dataset from scratch (GPU strongly recommended; output goes to `data/generated/`):
```bash
python main.py generate-data --split train
python main.py generate-data --split val
python main.py generate-data --split test
python main.py generate-data --split humaneval
```

### Test / evaluate

```bash
python main.py evaluate                     # Phase D: MBPP test + HumanEval, with 95% bootstrap CIs (CPU ok, ~10-15 min)
python main.py evaluate --splits val        # validation split only
python main.py calibrate                    # re-choose the pipeline threshold on validation (40% flag rate)
python main.py selftest                     # offline test of the whole pipeline with a fake LLM (seconds)
python main.py selftest --model             # same, also loading the real CodeBERT model
```

### Multi-agent experiment (baseline vs. proposed vs. random)

```bash
# 30 held-out MBPP problems, about 1 hour on a laptop CPU (add --fast for about half the time)
python main.py experiment --split mbpp_test --limit 30 --variants baseline,proposed,random --run-name mbpp_test_30
python main.py analyze results/mbpp_test_30

# full experiment (GPU or Ollama recommended)
python main.py experiment --split mbpp_test --variants baseline,proposed,random
python main.py experiment --split humaneval --variants baseline,proposed,random
```

The **v2 pipeline** (default) gives every variant: 4 candidate solutions (greedy draft + 3 seeded samples),
**test voting** (a generated test is kept only if most candidates that pass the task's example agree with it),
**selection** of the best-tested candidate, and a **fix guard** (a fix is kept only if it passes more trusted
tests). `--legacy` runs the original v1 pipeline; `--candidates N` changes the number of candidates.

```bash
python main.py policies results/mbpp_test60_v2   # offline: when is extra compute / the bug model worth it?
```

| Variant | Agents | Purpose |
|---|---|---|
| `baseline` | 1, 2, 4, 5 | Reactive FlowGen / AgentCoder-style pipeline |
| `proposed` | 1, 2, **3**, 4, 5 | CodeBERT risk guides the Tester and the Fixer |
| `random` | 1, 2, 3*, 4, 5 | Ablation: random flags at the same rate (separates the learned signal from simply running more tests) |

The experiment is resumable: re-run the same command to continue where it stopped.
`analyze` writes `summary.md` (pass@1, fix iterations, cost, McNemar test, bootstrap CIs) into the run folder.
Choose the LLM backend with `--llm auto|ollama|hf|gguf|openai` (default `auto`). Add `--fast` for
prompt-lookup decoding (about 2× faster; results are not bit-comparable with runs made without it).

---

## 5. Folder structure

```
Final_Code_Generation_Project/
├── main.py                  ← MAIN FILE: entry point for every command
├── config.py                ← all paths and settings in one place
├── requirements.txt         ← Python libraries
├── setup_windows.bat        ← one-step setup (Windows)
├── setup_unix.sh            ← one-step setup (macOS / Linux)
├── README.md
│
├── bug_predictor/           ← Bug-Prediction Agent (inference)
│   ├── predictor.py         ·  BugPredictor: score(), is_risky(), score_functions()
│   └── demo.py              ·  `python main.py demo`
│
├── training/                ← building and evaluating the bug predictor
│   ├── generate_data.py     ·  Phase A: sample LLM solutions, label them by running the tests
│   ├── baselines.py         ·  Phase B: length / line-count / TF-IDF / majority baselines
│   ├── train.py             ·  Phase C: fine-tune CodeBERT with early stopping
│   ├── evaluate.py          ·  Phase D: held-out metrics with bootstrap confidence intervals
│   └── common.py            ·  data loading and metrics (ROC-AUC, F1, pairwise, best-of-N)
│
├── pipeline/                ← the 5-agent code-generation system
│   ├── agents.py            ·  Requirement, Developer, Bug-Prediction, Tester, Fixer agents
│   ├── pipeline.py          ·  runs the agents for each variant
│   ├── llm.py               ·  LLM backends (Ollama, Hugging Face, llama.cpp, OpenAI-compatible)
│   ├── executor.py          ·  sandboxed test execution (isolated subprocess + timeout)
│   ├── benchmarks.py        ·  MBPP / HumanEval loaders and splits
│   ├── run_experiment.py    ·  `python main.py experiment`
│   ├── analyze.py           ·  `python main.py analyze` (pass@1, McNemar, bootstrap)
│   ├── calibrate_threshold.py · `python main.py calibrate`
│   ├── static_check.py      ·  exact syntax / runtime findings for the Scanner
│   ├── review.py            ·  AI code review used by the website
│   └── selftest.py          ·  offline end-to-end test
│
├── server/app.py            ← FastAPI back end (`python main.py serve`)
├── web/                     ← React front end (source in web/src, built site in web/dist)
│
├── models/bug_predictor/    ← trained CodeBERT checkpoint + thresholds
├── data/
│   ├── benchmarks/          ·  raw MBPP and HumanEval
│   └── processed/           ·  labelled train / val / test datasets
├── results/                 ← experiment outputs and baseline results
└── docs/                    ← final report, review slides, data card, experiment log
```

Folders created when you run things (not part of the submission): `.venv/`, `.cache/` (LLM response cache),
`runs/` (new training checkpoints), `data/generated/` (newly generated data).

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment (`.venv\Scripts\activate` or `source .venv/bin/activate`) and run `pip install -r requirements.txt` |
| `python` not found (macOS/Linux) | Use `python3` instead of `python` |
| Agent Pipeline is very slow | Check that `python main.py check` finds the llama.cpp or Ollama backend; without them the slow PyTorch backend is used |
| Port 8000 already in use | `python main.py serve --port 8080` and open http://localhost:8080 |
| Training is extremely slow | Training needs a GPU; for a quick check use `python main.py train --limit 64 --epochs 1 --max-len 128` |
| Bug-predictor model not found | Download [`model.safetensors`](https://github.com/Nithin-k-17/Multi-agent-code-generation-with-a-trained-bug-predictor/releases/download/v1.0/model.safetensors) into `models/bug_predictor/` (see section 3) |

---

## 7. Notes

- The data split is by problem ID, so no problem appears in more than one split. HumanEval is never used for training or tuning.
- The pipeline uses a stricter threshold (τ = 0.795, flags 40% of validation code) than the standalone
  scanner (τ = 0.34, best F1) so that it does not flag almost everything.
- The scripts in `training/` follow the exact settings documented in [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md)
  and [docs/DEVLOG.md](docs/DEVLOG.md). The shipped checkpoint in `models/bug_predictor/` is the champion model
  from that report (Run 3, CodeBERT, max length 512, seed 123). Re-training gives similar but not bit-identical
  numbers, because GPU training is not fully deterministic.
