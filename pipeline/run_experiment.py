"""Runs the with / without bug-prediction-agent experiment (proposal Step 6).

Examples (run from the project folder):
  # 30 held-out MBPP problems, all three variants (uses Ollama automatically if it is running)
  python main.py experiment --split mbpp_test --limit 30 --variants baseline,proposed,random

  # full experiment
  python main.py experiment --split mbpp_test --variants baseline,proposed,random
  python main.py experiment --split humaneval --variants baseline,proposed,random

  # force a backend
  python main.py experiment --llm ollama ...      (local Ollama server)
  python main.py experiment --llm hf ...          (PyTorch / Hugging Face, GPU recommended)

Outputs go to results/<run-name>/: shared.jsonl (Steps 1-2 per problem), results.jsonl
(one row per problem x variant), config.json. Re-running the same command resumes.
Then: python main.py analyze results/<run-name>
"""
import os
import json
import time
import argparse


from pipeline.benchmarks import load_split, SPLITS  # noqa: E402
from pipeline.agents import BugPredictionAgent  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402
from pipeline.llm import make_llm  # noqa: E402
from config import BUG_MODEL_DIR, PIPELINE_THRESHOLD_FILE, RESULTS_DIR  # noqa: E402

VARIANTS = ("baseline", "proposed", "random")


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="mbpp_test", choices=list(SPLITS))
    p.add_argument("--limit", type=int, default=None, help="only the first N problems")
    p.add_argument("--variants", default="baseline,proposed", help=f"comma list from {VARIANTS}")
    p.add_argument("--llm", default="auto", choices=["auto", "ollama", "hf", "gguf", "openai"],
                   help="code-generator backend (auto: llama.cpp if installed, else Ollama if running, else Hugging Face)")
    p.add_argument("--model", default=None, help="HF model id or served model name")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL (openai backend)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--bug-model", default=None, help="bug predictor checkpoint dir (default: models/bug_predictor)")
    p.add_argument("--threshold", type=float, default=None, help="risk threshold (overrides --threshold-file)")
    p.add_argument("--threshold-file", default=PIPELINE_THRESHOLD_FILE,
                   help="JSON with a 'threshold' key (default: the calibrated 40%% flag-rate threshold)")
    p.add_argument("--random-flag-rate", type=float, default=None,
                   help="flag rate for the random ablation (default: threshold file's val flag rate, else 0.5)")
    p.add_argument("--top-k-lines", type=int, default=3, help="hot lines reported by occlusion localisation (0 = off)")
    p.add_argument("--n-base-tests", type=int, default=5)
    p.add_argument("--n-targeted-tests", type=int, default=5)
    p.add_argument("--max-fix-iters", type=int, default=3)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--bug-device", default="cpu", help="device for CodeBERT (cpu keeps GPU free for the LLM)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None)
    p.add_argument("--fast", action="store_true",
                   help="prompt-lookup decoding (llama.cpp backend): faster fixes, but answers may differ "
                        "slightly from plain greedy decoding, so results are not bit-comparable with older runs")
    p.add_argument("--no-download", action="store_true", help="fail instead of downloading MBPP/HumanEval")
    p.add_argument("--candidates", type=int, default=4,
                   help="v2: candidate solutions per problem (greedy draft + k-1 seeded samples) used for test voting")
    p.add_argument("--sample-temperature", type=float, default=0.8, help="v2: temperature of the extra candidates")
    p.add_argument("--legacy", action="store_true",
                   help="v1 pipeline (no candidates, no test voting, no fix guard) to reproduce older runs")
    return p.parse_args(argv)


def build_bug_agents(args, variants):
    bug_agents = {"baseline": None}
    need = [v for v in variants if v in ("proposed", "random")]
    if not need:
        return bug_agents, None
    threshold, flag_rate = args.threshold, args.random_flag_rate
    if args.threshold_file and os.path.exists(args.threshold_file):
        with open(args.threshold_file, encoding="utf-8") as f:
            meta = json.load(f)
        threshold = threshold if threshold is not None else meta["threshold"]
        flag_rate = flag_rate if flag_rate is not None else meta.get("val_flag_rate")
    predictor = None
    if "proposed" in need:
        from bug_predictor import BugPredictor
        predictor = BugPredictor(path=args.bug_model or BUG_MODEL_DIR, device=args.bug_device)
        if threshold is None:
            print(f"WARNING: using the F1-optimal threshold {predictor.threshold} from threshold.json, which flags "
                  "~80-90% of code. Run `python main.py calibrate` first.")
    if "proposed" in need:
        bug_agents["proposed"] = BugPredictionAgent(predictor, mode="model", threshold=threshold, top_k_lines=args.top_k_lines)
    if "random" in need:
        bug_agents["random"] = BugPredictionAgent(predictor, mode="random", top_k_lines=args.top_k_lines,
                                                  random_flag_rate=flag_rate if flag_rate is not None else 0.5, seed=args.seed)
    used_threshold = bug_agents["proposed"].threshold if "proposed" in bug_agents else threshold
    return bug_agents, {"threshold": used_threshold, "random_flag_rate": bug_agents["random"].random_flag_rate if "random" in bug_agents else None}


def main(argv=None, llm=None):
    args = parse_args(argv)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"Unknown variants {bad}; choose from {VARIANTS}")

    problems = load_split(args.split, limit=args.limit, allow_download=not args.no_download)
    run_name = args.run_name or f"{args.split}_{(args.model or 'qwen1.5b').replace('/', '_').replace(':', '_')}"
    out_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)
    shared_path, results_path = os.path.join(out_dir, "shared.jsonl"), os.path.join(out_dir, "results.jsonl")

    bug_agents, bug_cfg = build_bug_agents(args, variants)
    if getattr(args, "fast", False):
        os.environ["PIPELINE_LOOKUP"] = "10"
    llm = llm or make_llm(args.llm, args.model, args.base_url, args.temperature)
    pipe = Pipeline(llm, bug_agents, n_base_tests=args.n_base_tests, n_targeted_tests=args.n_targeted_tests,
                    max_fix_iters=args.max_fix_iters, timeout=args.timeout,
                    n_candidates=args.candidates, legacy=args.legacy, sample_temperature=args.sample_temperature)

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "variants": variants, "llm_name": getattr(llm, "name", str(type(llm))),
                   "bug": bug_cfg, "n_problems": len(problems)}, f, indent=2)

    shared_cache = {r["task_id"]: r for r in _read_jsonl(shared_path)}
    done = {(r["task_id"], r["variant"]) for r in _read_jsonl(results_path)}
    print(f"Run dir: {out_dir}\nProblems: {len(problems)} | variants: {variants} | bug config: {bug_cfg}")

    t_start = time.time()
    for i, prob in enumerate(problems, 1):
        todo = [v for v in variants if (prob.task_id, v) not in done]
        if not todo:
            continue
        shared = shared_cache.get(prob.task_id)
        if shared is None:
            shared = pipe.shared_stage(prob)
            _append(shared_path, shared)
            shared_cache[prob.task_id] = shared
        line = [f"[{i}/{len(problems)}] {prob.task_id} init={'P' if shared['initial_hidden_pass'] else 'F'}"]
        for v in todo:
            row = pipe.run_variant(prob, shared, v)
            _append(results_path, row)
            flag = "" if row["flagged_initial"] is None else (" flag" if row["flagged_initial"] else " ok")
            line.append(f"{v}={'P' if row['final_hidden_pass'] else 'F'}(fix={row['fix_iterations']}{flag})")
        print("  ".join(line) + f"  [{time.time() - t_start:.0f}s]", flush=True)

    print(f"\nDone. Analyse with:  python main.py analyze \"{out_dir}\"")
    return out_dir


if __name__ == "__main__":
    main()
