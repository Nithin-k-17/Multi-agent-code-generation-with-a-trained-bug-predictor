"""Offline end-to-end check of the pipeline with a scripted fake LLM (no downloads, no GPU).

  python main.py selftest            # fake LLM + random bug agent (no CodeBERT load)
  python main.py selftest --model    # also loads the real CodeBERT bug predictor
"""
import os
import sys
import json
import shutil
import tempfile


from pipeline.benchmarks import Problem  # noqa: E402
from pipeline.agents import BugPredictionAgent, sanitize_code  # noqa: E402
from pipeline.executor import run_tests  # noqa: E402
from pipeline.llm import LLM, extract_code  # noqa: E402
from pipeline.pipeline import Pipeline, vote_tests  # noqa: E402
from pipeline import analyze  # noqa: E402

BUGGY = "def second_largest(xs):\n    xs = sorted(xs)\n    return xs[-1]\n\nprint(second_largest([1, 2]))\n"
FIXED = "def second_largest(xs):\n    ys = sorted(set(xs))\n    return ys[-2]\n"


class FakeLLM(LLM):
    """Answers each agent's prompt with canned output keyed on its system prompt."""
    name = "fake"

    def _chat(self, system, user, max_new_tokens, stop_at_code_end=False, prefix=""):
        if "requirements engineer" in system:
            out = json.dumps({"function_name": "second_largest", "edge_cases": ["duplicates"]})
        elif "Python developer" in system:
            out = f"Here you go:\n```python\n{BUGGY}```"
        elif "tester" in system:
            out = ("```python\nassert second_largest([3, 1, 2]) == 2\nassert second_largest([5, 5, 4]) == 4\n"
                   "assert second_largest([1, 2]) == 1\nnot an assert\n```")
        elif "debugger" in system:
            out = f"```python\n{FIXED}```"
        else:
            out = ""
        return out, len(user) // 4, len(out) // 4


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def main():
    use_model = "--model" in sys.argv
    check(extract_code("x\n```python\na=1\n```\n") == "a=1", "extract_code fenced block")
    check("print(" not in sanitize_code(BUGGY), "sanitize_code drops top-level example calls")

    rep = run_tests(FIXED, ["assert second_largest([3,1,2]) == 2", "assert second_largest([1]) == 1",
                            "assert second_largest([4,4,1]) == 1"])
    check([r.ok for r in rep.results] == [True, False, True], "executor per-test outcomes")
    check("IndexError" in rep.results[1].error, "executor captures error type")
    rep = run_tests("def f():\n    while True: pass\n", ["assert f() == 1"], timeout=2)
    check(rep.timed_out and not rep.results[0].ok, "executor timeout")
    rep = run_tests("def f(:\n", ["assert f() == 1"])
    check(bool(rep.load_error), "executor reports load (syntax) error")

    prob = Problem(task_id="toy/1", dataset="toy", description="Return the second largest distinct number in a list.",
                   entry_point="second_largest", starter_code="",
                   public_tests=["assert second_largest([1, 2, 3]) == 2"],
                   hidden_program="assert second_largest([1, 2, 3]) == 2\nassert second_largest([7, 7, 3]) == 3\n")

    predictor = None
    if use_model:
        from bug_predictor import BugPredictor
        predictor = BugPredictor(device="cpu")
    bug_agents = {"baseline": None,
                  "random": BugPredictionAgent(None, mode="random", random_flag_rate=1.0)}
    if predictor:
        bug_agents["proposed"] = BugPredictionAgent(predictor, mode="model", threshold=0.0)

    # test voting: tests that most example-passing candidates fail are dropped
    kept, info = vote_tests(["assert f(1) == 1"], ["assert f(2) == 2", "assert f(2) == 3"],
                            [[True, True, False], [True, True, False], [True, False, True], [False, False, True]])
    check(kept == ["assert f(2) == 2"] and info["pool"] == 3, "test voting keeps majority-agreed tests")

    for legacy in (True, False):
        tag = "v1" if legacy else "v2"
        pipe = Pipeline(FakeLLM(cache=False), bug_agents, n_base_tests=3, n_targeted_tests=2, max_fix_iters=2, legacy=legacy)
        shared = pipe.shared_stage(prob)
        check(not shared["initial_hidden_pass"], f"{tag}: buggy first draft fails hidden tests")
        check(len(shared["base_tests_raw"]) == 3, f"{tag}: tester parsed 3 base asserts")
        if not legacy:
            check(len(shared["candidates"]) == 4 and shared["vote"]["dropped"] == 3,
                  "v2: 4 candidates; tests every candidate fails are dropped")
        rows = []
        for v in bug_agents:
            row = pipe.run_variant(prob, shared, v)
            rows.append(row)
            check(row["final_hidden_pass"] and row["fix_iterations"] == 1, f"{tag} {v}: fixed in one iteration")
            if v != "baseline":
                check(row["flagged_initial"] is True and row["risk_initial"]["hot_lines"], f"{tag} {v}: flagged with hot lines")
    if predictor:
        r = next(r for r in rows if r["variant"] == "proposed")
        print("     proposed risk:", json.dumps(r["risk_initial"], indent=None)[:400])

    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "results.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with open(os.path.join(tmp, "shared.jsonl"), "w") as f:
            f.write(json.dumps(shared) + "\n")
        summary = analyze.main(tmp)
        check(summary["variants"]["baseline"]["pass@1"] == 1.0, "analyze computes pass@1")
    finally:
        shutil.rmtree(tmp)
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
