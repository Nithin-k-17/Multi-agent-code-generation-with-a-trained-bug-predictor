"""Phase A: builds the execution-labelled dataset used to train the bug predictor.

For every problem, Qwen2.5-Coder-1.5B-Instruct samples N candidate solutions (T=0.8, top_p=0.95).
Each candidate is run against the official tests in an isolated subprocess (5 s timeout):
    passes all tests          -> label 0 (clean)
    assertion error/exception -> label 1 (buggy)
    syntax error or timeout   -> dropped
Exact duplicates within a problem are removed, then a per-problem balanced subset is written.

    python main.py generate-data --split train          (MBPP 11-510)
    python main.py generate-data --split val            (MBPP 511-600)
    python main.py generate-data --split test           (MBPP 601-974)
    python main.py generate-data --split humaneval      (all 164 HumanEval problems)

A GPU is strongly recommended (about 4 GB VRAM). Output goes to data/generated/ so the shipped
dataset in data/processed/ is never overwritten. The run is resumable: finished problems are skipped.
"""
import argparse
import ast
import os

from config import DATA_DIR, GENERATOR_MODEL, SPLIT_RANGES
from pipeline.benchmarks import _path, _read_jsonl, download
from pipeline.executor import run_program
from pipeline.llm import extract_code
from training.common import balance_by_task, read_jsonl, write_jsonl

SYSTEM = "You are an expert Python programmer. Answer with a single ```python code block."
MAX_TEST_CHARS = 250


def load_problems(split):
    """Returns a list of dicts: task_id, prompt, program_suffix (official tests appended to the solution)."""
    if split == "humaneval":
        download("humaneval")
        return [{"task_id": r["task_id"], "prompt": r["prompt"],
                 "user": f"Complete this Python function:\n```python\n{r['prompt']}```",
                 "tests": r["test"] + f"\ncheck({r['entry_point']})\n"}
                for r in _read_jsonl(_path("humaneval"))]
    lo, hi = SPLIT_RANGES[split]
    download("mbpp")
    problems = []
    for r in _read_jsonl(_path("mbpp")):
        if lo <= int(r["task_id"]) <= hi:
            example = r["test_list"][0][:MAX_TEST_CHARS]
            setup = r.get("test_setup_code", "") or ""
            problems.append({"task_id": str(r["task_id"]), "prompt": r["text"],
                             "user": f"{r['text']}\nYour code should pass this test:\n{example}",
                             "tests": setup + "\n" + "\n".join(r["test_list"])})
    return problems


class Sampler:
    def __init__(self, model_name, temperature, top_p, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print("WARNING: no GPU found; generation on CPU is very slow.")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(self.device).eval()
        self.kw = dict(do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,
                       pad_token_id=self.tok.eos_token_id)

    def sample(self, user, n, chunk=4):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        outs = []
        while len(outs) < n:  # chunked sampling keeps peak GPU memory low
            k = min(chunk, n - len(outs))
            with self.torch.no_grad():
                gen = self.model.generate(**ids, num_return_sequences=k, **self.kw)
            outs += self.tok.batch_decode(gen[:, ids.input_ids.shape[1]:], skip_special_tokens=True)
            if self.device == "cuda":
                self.torch.cuda.empty_cache()
        return outs


def label_candidates(problem, completions, split, timeout):
    rows, seen = [], set()
    for idx, text in enumerate(completions):
        code = extract_code(text).strip()
        if not code or code in seen:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        passed, err = run_program(code + "\n" + problem["tests"], timeout=timeout)
        if not passed and err == "Timeout":
            continue
        seen.add(code)
        rows.append({"task_id": problem["task_id"], "split": split, "prompt": problem["prompt"], "code": code,
                     "label": 0 if passed else 1, "status": "passed" if passed else "failed",
                     "sample_idx": idx, "text": code})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the execution-labelled bug dataset.")
    p.add_argument("--split", required=True, choices=list(SPLIT_RANGES) + ["humaneval"])
    p.add_argument("--samples", type=int, default=8, help="candidates per problem")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--limit", type=int, default=None, help="only the first N problems (for a quick test)")
    p.add_argument("--model", default=GENERATOR_MODEL)
    p.add_argument("--out-dir", default=os.path.join(DATA_DIR, "generated"))
    args = p.parse_args(argv)

    name = "humaneval_test" if args.split == "humaneval" else args.split
    out_path = os.path.join(args.out_dir, f"{name}.jsonl")
    rows = read_jsonl(out_path) if os.path.exists(out_path) else []
    done = {r["task_id"] for r in rows} | set(_read_done_marker(out_path))

    problems = load_problems(args.split)[: args.limit]
    todo = [pr for pr in problems if pr["task_id"] not in done]
    print(f"{len(problems)} problems, {len(todo)} left -> {out_path}")
    sampler = Sampler(args.model, args.temperature, args.top_p, args.max_new_tokens) if todo else None

    for i, problem in enumerate(todo, 1):
        new = label_candidates(problem, sampler.sample(problem["user"], args.samples), name, args.timeout)
        rows += new
        write_jsonl(out_path, rows)
        _mark_done(out_path, problem["task_id"])
        n_fail = sum(r["label"] for r in new)
        print(f"[{i}/{len(todo)}] task {problem['task_id']}: {len(new)} kept ({len(new) - n_fail} pass / {n_fail} fail)",
              flush=True)

    balanced = balance_by_task(rows)
    write_jsonl(out_path.replace(".jsonl", "_balanced.jsonl"), balanced)
    n_fail = sum(int(r["label"]) for r in rows)
    print(f"\nTotal {len(rows)} samples ({len(rows) - n_fail} clean / {n_fail} buggy); balanced subset {len(balanced)}.")


def _marker(out_path):
    return out_path + ".done"


def _read_done_marker(out_path):
    if not os.path.exists(_marker(out_path)):
        return []
    with open(_marker(out_path), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _mark_done(out_path, task_id):
    with open(_marker(out_path), "a", encoding="utf-8") as f:
        f.write(f"{task_id}\n")


if __name__ == "__main__":
    main()
