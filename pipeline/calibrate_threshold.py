"""Chooses the bug-agent risk threshold on the VALIDATION split only (data/val.jsonl, MBPP 511-600).

The F1-optimal tau=0.34 flags 83-92% of code, which leaves nothing to prioritise.
This script scores every validation function and picks the threshold that hits a
target flag rate (default 40%), reporting precision/recall at each candidate.

  python main.py calibrate --target-flag-rate 0.4
Writes models/bug_predictor/pipeline_threshold.json (used by the experiment and the website).
"""
import os
import json
import argparse

import numpy as np

from config import PIPELINE_THRESHOLD_FILE, PROCESSED_DIR, PROJECT_ROOT


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--val-file", default=os.path.join(PROCESSED_DIR, "val.jsonl"))
    p.add_argument("--target-flag-rate", type=float, default=0.4)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=PIPELINE_THRESHOLD_FILE)
    args = p.parse_args(argv)

    from bug_predictor import BugPredictor
    from sklearn.metrics import roc_auc_score

    rows = [json.loads(l) for l in open(args.val_file, encoding="utf-8") if l.strip()]
    bp = BugPredictor(device=args.device)
    print(f"Scoring {len(rows)} validation functions on {bp.device} ...")
    scores, labels = [], []
    for i, r in enumerate(rows, 1):
        scores.append(bp.score(r["code"]))
        labels.append(int(r["label"]))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    s, y = np.array(scores), np.array(labels)

    def stats(t):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        return {"threshold": round(float(t), 4), "flag_rate": round(float(pred.mean()), 4),
                "precision": round(tp / max(int(pred.sum()), 1), 4), "recall": round(tp / max(int((y == 1).sum()), 1), 4)}

    print(f"\nVal ROC-AUC: {roc_auc_score(y, s):.4f} | base bug rate: {y.mean():.3f}")
    print(f"{'flag_rate':>10} {'threshold':>10} {'precision':>10} {'recall':>8}")
    for fr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        st = stats(np.quantile(s, 1 - fr))
        print(f"{st['flag_rate']:>10.3f} {st['threshold']:>10.4f} {st['precision']:>10.3f} {st['recall']:>8.3f}")
    print(f"Reference tau=0.34 (F1-optimal): {stats(0.34)}")

    chosen = stats(np.quantile(s, 1 - args.target_flag_rate))
    out = {**chosen, "val_flag_rate": chosen["flag_rate"], "target_flag_rate": args.target_flag_rate,
           "val_roc_auc": round(float(roc_auc_score(y, s)), 4), "source": os.path.relpath(args.val_file, PROJECT_ROOT)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nChosen: {chosen}\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
