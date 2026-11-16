"""Phase D: evaluates a bug-predictor checkpoint on held-out data, with 95% bootstrap CIs.

    python main.py evaluate                         # shipped model on MBPP test + HumanEval
    python main.py evaluate --splits val            # validation split only
    python main.py evaluate --model runs/<run-name> # a model you trained yourself

The decision threshold is the model's own threshold.json value (chosen on validation).
Results are printed and saved to results/evaluation_<model-name>.json.
"""
import argparse
import json
import os

from bug_predictor import BugPredictor
from config import BUG_MODEL_DIR, RESULTS_DIR
from training.common import bootstrap_ci, classification_metrics, load_split


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the bug predictor on held-out splits.")
    p.add_argument("--model", default=BUG_MODEL_DIR)
    p.add_argument("--splits", default="test,humaneval_test", help="comma list of files in data/processed/")
    p.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None, help="only the first N samples per split (quick test)")
    args = p.parse_args(argv)

    bp = BugPredictor(path=args.model, device=args.device)
    print(f"Model {bp.path} on {bp.device}, threshold {bp.threshold}")
    report = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        _, codes, labels, tasks = load_split(split)
        if args.limit:
            codes, labels, tasks = codes[: args.limit], labels[: args.limit], tasks[: args.limit]
        print(f"\nScoring {len(codes)} samples from '{split}' ...")
        scores = [bp.score(c) for c in codes]
        m = classification_metrics(scores, labels, tasks, bp.threshold)
        m["ci95"] = bootstrap_ci(scores, labels, tasks, bp.threshold, n_boot=args.bootstrap)
        report[split] = m
        for k in ("roc_auc", "accuracy", "precision", "recall", "f1", "flag_rate"):
            lo, hi = m["ci95"][k]
            print(f"  {k:10s} {m[k]:.4f}  [95% CI {lo:.4f} - {hi:.4f}]")
        print(f"  pairwise   {m['pairwise_accuracy']:.4f} over {m['num_pairs']} pairs")
        print(f"  best-of-N  {m['best_of_n_pass_rate']:.4f} vs random {m['random_pick_pass_rate']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"evaluation_{os.path.basename(os.path.normpath(args.model))}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
