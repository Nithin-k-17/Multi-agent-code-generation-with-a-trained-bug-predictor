"""Inference wrapper around the fine-tuned CodeBERT bug classifier.

    from bug_predictor import BugPredictor
    bp = BugPredictor(device="cpu")
    bp.score(code)            -> probability in [0, 1] that the code is buggy
    bp.is_risky(code)         -> score >= decision threshold
    bp.score_functions(code)  -> one score per function, with line numbers
"""
import os

# Skip TensorFlow/Flax auto-import in transformers: cuts model load time ~2.5x (PyTorch only).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import ast
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import BUG_MODEL_DIR, DEFAULT_THRESHOLD, MAX_LEN

_CACHE_LIMIT = 2048


def _pick_device(device):
    if device is not None:
        return torch.device(device)
    if os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1"):
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BugPredictor:
    def __init__(self, path=BUG_MODEL_DIR, device=None):
        if not os.path.isfile(os.path.join(path, "config.json")):
            raise FileNotFoundError(
                f"Bug-predictor checkpoint not found in {path}. "
                "Place the model files there (see README, section 'Model and dataset').")
        self.path = path
        self.device = _pick_device(device)
        self.threshold, self.max_length = self._read_metadata(path)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self._cache = {}

    @staticmethod
    def _read_metadata(path):
        meta_path = os.path.join(path, "threshold.json")
        if not os.path.exists(meta_path):
            return DEFAULT_THRESHOLD, MAX_LEN
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        threshold = meta.get("optimal_threshold", meta.get("best_threshold", DEFAULT_THRESHOLD))
        return float(threshold), int(meta.get("max_len", MAX_LEN))

    @torch.no_grad()
    def score(self, code: str) -> float:
        """Probability (0..1) that `code` is buggy."""
        key = (code or "").strip()
        if not key:
            return 0.0
        if key in self._cache:
            return self._cache[key]
        enc = self.tok(code, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
        prob = float(torch.softmax(self.model(**enc).logits, dim=-1)[0, 1])
        if len(self._cache) > _CACHE_LIMIT:
            self._cache.clear()
        self._cache[key] = prob
        return prob

    def is_risky(self, code: str) -> bool:
        return self.score(code) >= self.threshold

    def score_functions(self, code: str) -> list:
        """Scores every function separately. Falls back to the whole snippet if it has no functions
        or does not parse."""
        if not code or not code.strip():
            return []
        units = []
        try:
            lines = code.splitlines(keepends=True)
            for node in ast.walk(ast.parse(code)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    end = getattr(node, "end_lineno", len(lines))
                    units.append((node.name, node.lineno, end, "".join(lines[node.lineno - 1:end])))
        except SyntaxError:
            units = []
        if not units:
            units = [("<module_or_script>", 1, len(code.splitlines()), code)]

        results = []
        for name, start, end, src in units:
            s = self.score(src)
            results.append({"name": name, "line_start": start, "line_end": end,
                            "score": round(s, 4), "is_risky": s >= self.threshold, "code": src})
        return results
