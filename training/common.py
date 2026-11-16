"""Shared helpers: JSONL I/O, dataset loading and evaluation metrics."""
import json
import os
import random
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from config import PROCESSED_DIR


# --------------------------------------------------------------------------- I/O
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(name):
    """Loads data/processed/<name>.jsonl and returns (rows, codes, labels, task_ids)."""
    path = name if name.endswith(".jsonl") else os.path.join(PROCESSED_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. See README, section 'Model and dataset'.")
    rows = read_jsonl(path)
    codes = [r.get("code") or r.get("text", "") for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    tasks = [str(r.get("task_id")) for r in rows]
    return rows, codes, labels, tasks


def balance_by_task(rows, seed=42):
    """Per problem, keeps the same number of clean and buggy samples (problems with one label are dropped)."""
    rng = random.Random(seed)
    by_task = defaultdict(lambda: ([], []))
    for r in rows:
        by_task[r["task_id"]][int(r["label"])].append(r)
    out = []
    for clean, buggy in by_task.values():
        k = min(len(clean), len(buggy))
        out += rng.sample(clean, k) + rng.sample(buggy, k)
    return out


# --------------------------------------------------------------------------- metrics
def pairwise_accuracy(scores, labels, tasks):
    """Share of (buggy, clean) pairs from the same problem where the buggy one gets the higher score."""
    by_task = defaultdict(lambda: ([], []))
    for s, y, t in zip(scores, labels, tasks):
        by_task[t][int(y)].append(s)
    wins = total = 0
    for clean, buggy in by_task.values():
        for b in buggy:
            for c in clean:
                wins += (b > c) + 0.5 * (b == c)
                total += 1
    return wins / total if total else float("nan"), total


def best_of_n(scores, labels, tasks):
    """Picks the lowest-risk candidate per problem. Returns (pass rate of picks, expected random pass rate)."""
    by_task = defaultdict(list)
    for s, y, t in zip(scores, labels, tasks):
        by_task[t].append((s, int(y)))
    picked, rand = [], []
    for cands in by_task.values():
        if len(cands) < 2:
            continue
        picked.append(1 - min(cands)[1])
        rand.append(1 - np.mean([y for _, y in cands]))
    return (float(np.mean(picked)) if picked else float("nan"),
            float(np.mean(rand)) if rand else float("nan"))


def classification_metrics(scores, labels, tasks, threshold):
    scores, labels = np.asarray(scores), np.asarray(labels)
    pred = (scores >= threshold).astype(int)
    pair_acc, n_pairs = pairwise_accuracy(scores, labels, tasks)
    bon, bon_random = best_of_n(scores, labels, tasks)
    return {
        "n": int(len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        "pairwise_accuracy": pair_acc,
        "num_pairs": n_pairs,
        "best_of_n_pass_rate": bon,
        "random_pick_pass_rate": bon_random,
        "threshold": float(threshold),
    }


def best_f1_threshold(scores, labels):
    """Threshold in [0.01, 0.99] that maximises F1 (chosen on validation only)."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    grid = np.arange(0.01, 1.0, 0.01)
    f1s = [f1_score(labels, (scores >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(f1s))], 2))


def bootstrap_ci(scores, labels, tasks, threshold, n_boot=2000, seed=0):
    """95% bootstrap confidence intervals for the main metrics (resampling samples)."""
    rng = np.random.default_rng(seed)
    scores, labels, tasks = np.asarray(scores), np.asarray(labels), np.asarray(tasks)
    keys = ("roc_auc", "accuracy", "precision", "recall", "f1", "flag_rate")
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) < 2:
            continue
        pred = (scores[idx] >= threshold).astype(int)
        y = labels[idx]
        draws["roc_auc"].append(roc_auc_score(y, scores[idx]))
        draws["accuracy"].append(accuracy_score(y, pred))
        draws["precision"].append(precision_score(y, pred, zero_division=0))
        draws["recall"].append(recall_score(y, pred, zero_division=0))
        draws["f1"].append(f1_score(y, pred, zero_division=0))
        draws["flag_rate"].append(pred.mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}
