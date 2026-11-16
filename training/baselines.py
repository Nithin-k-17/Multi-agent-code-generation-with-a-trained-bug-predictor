"""Phase B: trivial baselines, to check that the task cannot be solved with surface shortcuts.

Each baseline is trained on the train split and evaluated on the validation split:
code length, line count, both, TF-IDF (1-2 grams) + logistic regression, and the majority class.

    python main.py baselines
Results are printed and saved to results/baselines.json.
"""
import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from config import RESULTS_DIR
from training.common import classification_metrics, load_split


def _size_features(kind):
    def fn(codes):
        chars = [len(c) for c in codes]
        lines = [len(c.splitlines()) for c in codes]
        cols = {"chars": [chars], "lines": [lines], "chars+lines": [chars, lines]}[kind]
        return np.array(cols, dtype=float).T
    return FunctionTransformer(fn)


BASELINES = {
    "code_length": lambda: make_pipeline(_size_features("chars"), StandardScaler(), LogisticRegression()),
    "line_count": lambda: make_pipeline(_size_features("lines"), StandardScaler(), LogisticRegression()),
    "chars_and_lines": lambda: make_pipeline(_size_features("chars+lines"), StandardScaler(), LogisticRegression()),
    "tfidf_logreg": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=5000, token_pattern=r"\S+"),
                                          LogisticRegression(max_iter=1000)),
}


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate trivial baselines on the validation split.")
    p.add_argument("--train", default="train")
    p.add_argument("--val", default="val")
    args = p.parse_args(argv)

    results = {}
    for train_name in (args.train, f"{args.train}_balanced"):
        _, x_tr, y_tr, _ = load_split(train_name)
        _, x_va, y_va, t_va = load_split(args.val)
        for name, build in BASELINES.items():
            model = build().fit(x_tr, y_tr)
            scores = model.predict_proba(x_va)[:, 1]
            results[f"{name} [{train_name}]"] = classification_metrics(scores, y_va, t_va, threshold=0.5)
        majority = int(np.mean(y_tr) >= 0.5)
        results[f"majority_class [{train_name}]"] = classification_metrics(
            np.full(len(y_va), float(majority)), y_va, t_va, threshold=0.5)

    print(f"{'baseline':42s} {'ROC-AUC':>8} {'Acc':>7} {'F1':>7} {'Pairwise':>9}")
    for name, m in results.items():
        print(f"{name:42s} {m['roc_auc']:8.4f} {m['accuracy']:7.4f} {m['f1']:7.4f} {m['pairwise_accuracy']:9.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "baselines.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
