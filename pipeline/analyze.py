"""Summarises a run directory: pass@1, fix iterations, testing cost, paired significance tests.

  python main.py analyze results/<run-name>
Writes summary.md and summary.json into the run directory.

Metrics (per variant)
  pass@1               final code passes the official hidden tests (one greedy attempt per problem)
  initial pass@1       Developer's first draft, before any testing/fixing (identical for all variants)
  avg fix iterations   Fixer calls per problem (over all problems, and over problems that needed fixing)
  iters to hidden pass first iteration whose code passes hidden tests (analysis-only signal)
  test executions      total assert executions across all iterations (testing effort)
  LLM calls / tokens   cost of Steps 3-5 for that variant (Steps 1-2 are shared and reported once)
Paired comparisons use McNemar's exact test on pass@1 and a 2,000-resample paired bootstrap
for the pass@1 and fix-iteration differences.
"""
import os
import sys
import json
import math
import random
from collections import defaultdict


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value; b, c = discordant pair counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_bootstrap(diffs, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summarise_variant(rows):
    needed_fix = [r for r in rows if r["history"] and (r["history"][0]["n_failed"] > 0 or r["history"][0]["load_error"])]
    return {
        "n": len(rows),
        "pass@1": _mean([float(r["final_hidden_pass"]) for r in rows]),
        "initial_pass@1": _mean([float(r["initial_hidden_pass"]) for r in rows]),
        "selected_pass@1": _mean([float(r["selected_hidden_pass"]) for r in rows if r.get("selected_hidden_pass") is not None]),
        "internal_tests_passed": _mean([float(r["internal_tests_passed"]) for r in rows]),
        "avg_fix_iterations": _mean([r["fix_iterations"] for r in rows]),
        "avg_fixes_rejected": _mean([r.get("fixes_rejected") for r in rows]),
        "avg_fix_iterations_when_needed": _mean([r["fix_iterations"] for r in needed_fix]),
        "n_needed_fix": len(needed_fix),
        "repaired_initially_failing": _mean([float(r["final_hidden_pass"]) for r in rows if not r["initial_hidden_pass"]]),
        "broke_initially_passing": _mean([float(not r["final_hidden_pass"]) for r in rows if r["initial_hidden_pass"]]),
        "avg_iters_to_hidden_pass": _mean([r["fix_iters_to_hidden_pass"] for r in rows if not r["initial_hidden_pass"]]),
        "avg_test_executions": _mean([r["test_executions"] for r in rows]),
        "avg_targeted_tests": _mean([r["n_targeted_tests"] for r in rows]),
        "flag_rate": _mean([float(r["flagged_initial"]) for r in rows if r["flagged_initial"] is not None]),
        "avg_llm_calls": _mean([r["variant_llm"]["llm_calls"] for r in rows]),
        "avg_completion_tokens": _mean([r["variant_llm"]["completion_tokens"] for r in rows]),
        "avg_seconds": _mean([r["variant_seconds"] for r in rows]),
        "avg_bug_agent_seconds": _mean([r["bug_agent_seconds"] for r in rows]),
    }


def bug_agent_quality(rows):
    """How well the initial risk score predicts that the first draft fails the hidden tests."""
    pts = [(r["risk_score_initial"], int(not r["initial_hidden_pass"]), r["flagged_initial"]) for r in rows
           if r.get("risk_score_initial") is not None]
    if not pts:
        return {}
    out = {"n": len(pts)}
    try:
        from sklearn.metrics import roc_auc_score
        if len({y for _, y, _ in pts}) == 2:
            out["roc_auc"] = roc_auc_score([y for _, y, _ in pts], [s for s, _, _ in pts])
    except ImportError:
        pass
    flagged = [y for _, y, f in pts if f]
    unflagged = [y for _, y, f in pts if not f]
    out["bug_rate_flagged"] = _mean(flagged)
    out["bug_rate_unflagged"] = _mean(unflagged)
    out["recall"] = sum(1 for _, y, f in pts if f and y) / max(1, sum(y for _, y, _ in pts))
    return out


def test_quality(shared):
    """Analysis only: fraction of agent-written tests that the official MBPP reference solution passes,
    before (raw) and after test voting (kept). Agents never see the reference solutions."""
    from config import BENCHMARK_DIR
    from pipeline.executor import run_tests
    path = os.path.join(BENCHMARK_DIR, "mbpp.jsonl")
    if not os.path.exists(path) or not shared or not str(shared[0]["task_id"]).startswith("mbpp/"):
        return {}
    ref = {f"mbpp/{r['task_id']}": r for r in _read(path)}
    raw_ok = raw_n = kept_ok = kept_n = 0
    for s in shared:
        r = ref.get(s["task_id"])
        raw = s.get("base_tests_raw", s["base_tests"])
        if not r or not raw:
            continue
        rep = run_tests(r["code"], raw, setup=r.get("test_setup_code", "") or "", timeout=5)
        ok = {t: res.ok for t, res in zip(raw, rep.results)}
        raw_ok += sum(ok.values()); raw_n += len(raw)
        kept = [t for t in s["base_tests"] if t in ok]
        kept_ok += sum(ok[t] for t in kept); kept_n += len(kept)
    return {"raw_tests": raw_n, "raw_test_accuracy": raw_ok / raw_n if raw_n else float("nan"),
            "kept_tests": kept_n, "kept_test_accuracy": kept_ok / kept_n if kept_n else float("nan"),
            "tests_kept": kept_n / raw_n if raw_n else float("nan")}


def compare(a_rows, b_rows):
    """Paired comparison b vs a over shared task ids."""
    a, b = {r["task_id"]: r for r in a_rows}, {r["task_id"]: r for r in b_rows}
    ids = sorted(set(a) & set(b))
    pa = [int(a[i]["final_hidden_pass"]) for i in ids]
    pb = [int(b[i]["final_hidden_pass"]) for i in ids]
    b_only = sum(1 for x, y in zip(pa, pb) if y and not x)
    a_only = sum(1 for x, y in zip(pa, pb) if x and not y)
    d_pass = [y - x for x, y in zip(pa, pb)]
    d_fix = [b[i]["fix_iterations"] - a[i]["fix_iterations"] for i in ids]
    return {
        "n_paired": len(ids),
        "pass@1_diff": _mean(d_pass), "pass@1_diff_ci95": paired_bootstrap(d_pass),
        "solved_only_by_second": b_only, "solved_only_by_first": a_only,
        "mcnemar_p": mcnemar_exact(a_only, b_only),
        "fix_iter_diff": _mean(d_fix), "fix_iter_diff_ci95": paired_bootstrap(d_fix, seed=1),
    }


def _fmt(v, pct=False):
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{100 * v:.1f}%" if pct else f"{v:.3f}"
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_fmt(x, pct) for x in v) + "]"
    return str(v)


PCT_KEYS = {"pass@1", "initial_pass@1", "selected_pass@1", "any_candidate_pass", "raw_test_accuracy",
            "kept_test_accuracy", "tests_kept", "internal_tests_passed", "repaired_initially_failing",
            "broke_initially_passing", "flag_rate", "pass@1_diff", "pass@1_diff_ci95",
            "bug_rate_flagged", "bug_rate_unflagged", "recall"}


def main(run_dir):
    rows = _read(os.path.join(run_dir, "results.jsonl"))
    shared = _read(os.path.join(run_dir, "shared.jsonl"))
    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
    by_v = defaultdict(list)
    for r in rows:
        by_v[r["variant"]].append(r)

    summary = {"config": cfg, "variants": {v: summarise_variant(rs) for v, rs in by_v.items()},
               "shared_stage": {"n": len(shared),
                                "avg_llm_calls": _mean([s["shared_llm"]["llm_calls"] for s in shared]),
                                "avg_seconds": _mean([s["shared_seconds"] for s in shared]),
                                "any_candidate_pass": _mean([float(any(s.get("candidates_hidden_pass") or [s["initial_hidden_pass"]]))
                                                             for s in shared])},
               "test_quality": test_quality(shared),
               "comparisons": {}, "bug_agent": {}}
    for v in ("proposed", "random"):
        if v in by_v and "baseline" in by_v:
            summary["comparisons"][f"{v}_vs_baseline"] = compare(by_v["baseline"], by_v[v])
    if "proposed" in by_v and "random" in by_v:
        summary["comparisons"]["proposed_vs_random"] = compare(by_v["random"], by_v["proposed"])
    if "proposed" in by_v:
        summary["bug_agent"] = bug_agent_quality(by_v["proposed"])

    md = [f"# Pipeline results: `{os.path.basename(os.path.abspath(run_dir))}`", ""]
    if cfg:
        md.append(f"Split **{cfg.get('split')}**, LLM **{cfg.get('llm_name')}**, max fix iterations "
                  f"{cfg.get('max_fix_iters')}, base tests {cfg.get('n_base_tests')}, targeted tests "
                  f"{cfg.get('n_targeted_tests')}, bug config {cfg.get('bug')}.")
        md.append("")
    variants = [v for v in ("baseline", "proposed", "random") if v in summary["variants"]]
    keys = list(next(iter(summary["variants"].values())).keys()) if variants else []
    md.append("| metric | " + " | ".join(variants) + " |")
    md.append("|---|" + "---|" * len(variants))
    for k in keys:
        md.append(f"| {k} | " + " | ".join(_fmt(summary["variants"][v][k], k in PCT_KEYS) for v in variants) + " |")
    md += ["", "## Paired comparisons", ""]
    for name, c in summary["comparisons"].items():
        md.append(f"**{name}** (n={c['n_paired']}): pass@1 diff {_fmt(c['pass@1_diff'], True)} "
                  f"95% CI {_fmt(c['pass@1_diff_ci95'], True)}; solved only by second/first: "
                  f"{c['solved_only_by_second']}/{c['solved_only_by_first']}, McNemar p={c['mcnemar_p']:.4f}; "
                  f"fix-iteration diff {_fmt(c['fix_iter_diff'])} 95% CI {_fmt(c['fix_iter_diff_ci95'])}.")
        md.append("")
    if summary["test_quality"]:
        md += ["## Agent-written test accuracy (vs. official reference solutions, analysis only)", ""]
        md += [f"- {k}: {_fmt(v, k in PCT_KEYS)}" for k, v in summary["test_quality"].items()]
        md.append("")
    md.append(f"- candidates per problem: any candidate passes hidden tests in "
              f"{_fmt(summary['shared_stage']['any_candidate_pass'], True)} of problems")
    md.append("")
    if summary["bug_agent"]:
        md += ["## Bug-prediction agent on first drafts", ""]
        md += [f"- {k}: {_fmt(v, k in PCT_KEYS)}" for k, v in summary["bug_agent"].items()]
    text = "\n".join(md) + "\n"

    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    summary = _json_safe(summary)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(text)
    return summary


def _json_safe(obj):
    """NaN (e.g. a rate over zero problems) is not valid JSON; store it as null."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python main.py analyze results/<run-name>")
    main(sys.argv[1])
