"""Offline analysis: where does the trained bug predictor add value on top of test-based agents?

  python main.py policies results/<run-name>

Uses only what a v2 run already recorded (every candidate solution, its agent-test results and its
hidden-test outcome), so no LLM calls are needed. Two questions:

1. Compute gating - extra candidates cost 3 more LLM calls per problem. Is it enough to generate
   them only when a cheap signal says the first draft is likely wrong?
     greedy         never generate extra candidates (1 developer call)
     always         always generate k candidates and pick the best by tests
     gate:codebert  extra candidates only if CodeBERT flags the first draft (calibrated threshold)
     gate:random    extra candidates at random, same rate as gate:codebert (control)
     gate:example   extra candidates only if the first draft fails the task's own example test
     gate:both      extra candidates if the draft fails the example test OR CodeBERT flags it
2. Tie-breaking - among candidates with the same test score, pick the first, a random one, or the
   one CodeBERT rates least bug-prone.
Pass rates are of the starting code chosen (before the Fixer), scored on the hidden tests.
"""
import json
import os
import random
import sys


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _scores(s, n_pub):
    kept = set(s["base_tests"])
    keep = [j for j, t in enumerate(s["base_tests_raw"]) if t in kept]
    out = []
    for row in s["candidate_matrix"]:
        out.append((-1, -1) if not row else (sum(row[:n_pub]), sum(row[:n_pub]) + sum(row[n_pub + j] for j in keep)))
    return out


def _pick(s, scores, how, risks, rng):
    best = max(scores)
    tied, seen = [], set()
    for i, sc in enumerate(scores):
        if sc == best and s["candidates"][i] not in seen:
            seen.add(s["candidates"][i])
            tied.append(i)
    if how == "first" or len(tied) == 1:
        return tied[0]
    if how == "random":
        return rng.choice(tied)
    return min(tied, key=lambda i: (risks[i], i))


def main(run_dir, n_boot=2000):
    from bug_predictor import BugPredictor
    from config import PIPELINE_THRESHOLD_FILE
    from pipeline.benchmarks import load_split

    shared = [s for s in _read(os.path.join(run_dir, "shared.jsonl")) if len(s.get("candidates") or []) > 1]
    if not shared:
        raise SystemExit("This run has no candidate data (run it with the v2 pipeline, --candidates 2 or more).")
    cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))
    probs = {p.task_id: p for p in load_split(cfg["split"], allow_download=False)}
    threshold = json.load(open(PIPELINE_THRESHOLD_FILE, encoding="utf-8"))["threshold"]
    pred = BugPredictor(device="cpu")

    rows = []
    for s in shared:
        n_pub = len(probs[s["task_id"]].public_tests)
        risks = [pred.score(c) for c in s["candidates"]]
        row0 = s["candidate_matrix"][0]
        rows.append({
            "task_id": s["task_id"], "hidden": s["candidates_hidden_pass"], "scores": _scores(s, n_pub), "risks": risks,
            "flag": risks[0] >= threshold,
            "fails_example": bool(n_pub) and (not row0 or not all(row0[:n_pub])), "s": s,
        })
    flag_rate = sum(r["flag"] for r in rows) / len(rows)

    def evaluate(gate, how="first", seed=0):
        rng = random.Random(seed)
        passed, calls = [], 0
        for r in rows:
            if gate(r, rng):
                i = _pick(r["s"], r["scores"], how, r["risks"], rng)
                calls += len(r["hidden"])
            else:
                i = 0
                calls += 1
            passed.append(int(bool(r["hidden"][i])))
        return passed, calls / len(rows)

    gates = {
        "greedy": lambda r, rng: False,
        "always": lambda r, rng: True,
        "gate:codebert": lambda r, rng: r["flag"],
        "gate:random": lambda r, rng: rng.random() < flag_rate,
        "gate:example": lambda r, rng: r["fails_example"],
        "gate:both": lambda r, rng: r["fails_example"] or r["flag"],
    }
    out = {"n": len(rows), "codebert_flag_rate": flag_rate, "threshold": threshold, "gating": {}, "tie_break": {}}
    for name, g in gates.items():
        if name == "gate:random":  # average over seeds: a single random draw is noisy
            runs = [evaluate(g, seed=k) for k in range(200)]
            out["gating"][name] = {"pass@1": sum(sum(p) for p, _ in runs) / (len(runs) * len(rows)),
                                   "dev_calls_per_problem": sum(c for _, c in runs) / len(runs)}
        else:
            p, c = evaluate(g)
            out["gating"][name] = {"pass@1": sum(p) / len(p), "dev_calls_per_problem": c}
    for how in ("first", "codebert", "random"):
        if how == "random":
            vals = [sum(evaluate(gates["always"], how, seed=k)[0]) / len(rows) for k in range(200)]
            out["tie_break"][how] = sum(vals) / len(vals)
        else:
            out["tie_break"][how] = sum(evaluate(gates["always"], how)[0]) / len(rows)
    out["oracle_any_candidate"] = sum(any(r["hidden"]) for r in rows) / len(rows)
    out["ties_with_distinct_code"] = sum(
        1 for r in rows if len({r["s"]["candidates"][i] for i, sc in enumerate(r["scores"]) if sc == max(r["scores"])}) > 1)

    lines = [f"# Policy analysis: `{os.path.basename(os.path.abspath(run_dir))}` ({len(rows)} problems)", "",
             f"CodeBERT flags {flag_rate:.0%} of first drafts (threshold {threshold:.3f}).", "",
             "| policy | pass@1 (start code) | developer calls / problem |", "|---|---|---|"]
    for name, v in out["gating"].items():
        lines.append(f"| {name} | {v['pass@1']:.1%} | {v['dev_calls_per_problem']:.2f} |")
    lines += ["", f"Upper bound (any candidate passes): {out['oracle_any_candidate']:.1%}", "",
              f"Tie-breaking among equally-tested candidates ({out['ties_with_distinct_code']} problems had a real tie):", ""]
    lines += [f"- {k}: {v:.1%}" for k, v in out["tie_break"].items()]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(run_dir, "policies.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(run_dir, "policies.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(text)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python main.py policies results/<run-name>")
    main(sys.argv[1])
