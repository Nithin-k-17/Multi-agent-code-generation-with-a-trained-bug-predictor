"""The five pipeline agents (proposal Section 3, Steps 1-5).

  RequirementAgent   (Step 1) task text          -> structured spec
  DeveloperAgent     (Step 2) spec               -> code
  BugPredictionAgent (Step 3) code               -> risk score, risky functions, hot lines   [NEW]
  TesterAgent        (Step 4) spec (+ risk info) -> assert tests, executed in the sandbox
  FixerAgent         (Step 5) code + failures (+ risk info) -> repaired code
"""
import ast
import json
import random
import re
import zlib

from .llm import extract_code


# --------------------------------------------------------------------------- helpers

def sanitize_code(code):
    """Keeps only top-level imports, defs, classes and constant assignments.
    Drops example calls, prints, `if __name__ == "__main__":` blocks and stray asserts."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    keep = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)
    lines = code.splitlines()
    out = []
    for node in tree.body:
        if isinstance(node, keep):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            out.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n\n".join(out) if out else code


def _defines(code, name):
    try:
        tree = ast.parse(code)
        return (any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree))
                or any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
                       for n in tree.body))
    except SyntaxError:
        return False


def ensure_entry_point(code, name):
    """Makes sure `code` defines a top-level function `name`.

    Small code models often answer in LeetCode style (`class Solution: def someMethod(self, ...)`)
    or pick their own function name. If exactly one candidate is found, a thin wrapper is appended
    so the tests can call `name(...)`; otherwise the code is returned unchanged."""
    if not code.strip() or not name:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    top_funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if name in top_funcs:
        return code
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        methods = [m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and not m.name.startswith("_")]
        if len(methods) == 1:
            return code + f"\n\n\ndef {name}(*args, **kwargs):\n    return {cls.name}().{methods[0]}(*args, **kwargs)\n"
    public = [f for f in top_funcs if not f.startswith("_")]
    if len(public) == 1:
        return code + f"\n\n\n{name} = {public[0]}\n"
    return code


def _parse_asserts(text, entry_point, limit):
    """Pulls valid single-statement `assert ...` lines that call the entry point."""
    tests, seen = [], set()
    for raw in extract_code(text).splitlines() + text.splitlines():
        line = raw.strip()
        if not line.startswith("assert ") or entry_point not in line or line in seen:
            continue
        try:
            node = ast.parse(line)
        except SyntaxError:
            continue
        if len(node.body) == 1 and isinstance(node.body[0], ast.Assert):
            seen.add(line)
            tests.append(line)
        if len(tests) >= limit:
            break
    return tests


def _problem_block(problem):
    parts = [f"Task:\n{problem.description.strip()}"]
    if problem.starter_code and problem.starter_code != problem.description:
        parts.append(f"Starter code:\n```python\n{problem.starter_code.rstrip()}\n```")
    parts.append(f"Required function name: `{problem.entry_point}`")
    if problem.public_tests:
        parts.append("Example test(s) the solution must pass:\n" + "\n".join(problem.public_tests))
    return "\n\n".join(parts)


def _spec_block(spec):
    return json.dumps(spec, indent=2) if isinstance(spec, dict) else str(spec)


def _risk_block(risk):
    if not risk or not risk.get("flagged"):
        return ""
    lines = [f"The bug-prediction agent rates this code as HIGH RISK (bug-proneness score {risk['score']:.2f}, "
             f"threshold {risk['threshold']:.2f})."]
    if risk.get("risky_functions"):
        lines.append("Highest-risk functions: " + ", ".join(
            f"`{f['name']}` (lines {f['line_start']}-{f['line_end']}, score {f['score']:.2f})" for f in risk["risky_functions"]))
    if risk.get("hot_lines"):
        lines.append("Lines that contribute most to the risk score:")
        lines += [f"  line {h['line_start']}-{h['line_end']}: {h['text'].strip()}" for h in risk["hot_lines"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- Step 1

class RequirementAgent:
    SYSTEM = ("You are a requirements engineer. Turn a programming task into a precise technical "
              "specification. Respond with a single JSON object and nothing else.")

    def __init__(self, llm):
        self.llm = llm

    def run(self, problem):
        user = (_problem_block(problem) + "\n\nWrite the specification as JSON with keys:\n"
                '"function_name", "signature", "inputs" (list of {name, type, description}), '
                '"output" ({type, description}), "behaviour" (step-by-step description), '
                '"edge_cases" (list of strings).\n'
                "Be concise: one short sentence per field, at most 5 edge cases.")
        # Pre-filling the JSON opening stops the model from answering with code instead of a spec
        # (it tends to write a LeetCode `class Solution` for competitive-programming style tasks).
        prefix = '```json\n{\n  "function_name": "' + problem.entry_point + '",\n  "signature": "'
        text = self.llm.chat(self.SYSTEM, user, max_new_tokens=320, prefix=prefix)
        m = re.search(r"\{.*\}", text, re.S)
        spec = None
        if m:
            try:
                spec = json.loads(m.group(0))
            except json.JSONDecodeError:
                spec = None
        if not isinstance(spec, dict):
            spec = {"description": text.strip()}
        spec["function_name"] = problem.entry_point
        return spec


# --------------------------------------------------------------------------- Step 2

class DeveloperAgent:
    SYSTEM = ("You are an expert Python developer. Write correct, self-contained Python code. "
              "Return only one ```python code block with the implementation (imports included), "
              "no example usage, no prints, no tests.")

    def __init__(self, llm):
        self.llm = llm

    def run(self, problem, spec, temperature=None, seed=None):
        user = (_problem_block(problem) + "\n\nSpecification:\n" + _spec_block(spec) +
                f"\n\nImplement it as a top-level function named `{problem.entry_point}` (not a class method).")
        raw = self.llm.chat(self.SYSTEM, user, max_new_tokens=512, stop_at_code_end=True, temperature=temperature, seed=seed)
        return ensure_entry_point(sanitize_code(extract_code(raw)), problem.entry_point)

    def candidates(self, problem, spec, k, temperature=0.8):
        """k-1 alternative solutions sampled with fixed seeds (reproducible). Used to check which
        generated tests are trustworthy and to pick the best starting code."""
        return [self.run(problem, spec, temperature=temperature, seed=i) for i in range(1, k)]


# --------------------------------------------------------------------------- Step 3

class BugPredictionAgent:
    """Wraps the trained CodeBERT classifier (bug_predictor.BugPredictor).

    mode="model"  : real predictions (the proposed contribution)
    mode="random" : ablation control; flags code at random with probability `random_flag_rate`
                    and picks random functions/lines, so we can tell whether gains come from the
                    learned signal or simply from spending more tests on some problems.
    """

    def __init__(self, predictor=None, mode="model", threshold=None, top_k_lines=3, random_flag_rate=0.5, seed=0):
        self.predictor = predictor
        self.mode = mode
        self.threshold = threshold if threshold is not None else (predictor.threshold if predictor else 0.5)
        self.top_k_lines = top_k_lines
        self.random_flag_rate = random_flag_rate
        self.seed = seed

    def _statements(self, code):
        """Statements inside function bodies, as (line_start, line_end) spans."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        spans = []
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if isinstance(node, ast.stmt) and node is not fn and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if isinstance(node, ast.Expr) and isinstance(getattr(node, "value", None), ast.Constant):
                            continue  # docstring
                        spans.append((node.lineno, node.end_lineno))
        return sorted(set(spans))[:40]

    def line_saliency(self, code):
        """Gradient x input attribution of the 'buggy' logit, summed per source line.
        Positive values push the classifier towards 'buggy'. (Occluding statements with `pass`
        was tried first, but CodeBERT rates the edited code as *more* buggy, so it is unusable.)"""
        cache = self.__dict__.setdefault("_sal_cache", {})
        if code in cache:
            return cache[code]
        if len(cache) > 256:
            cache.clear()
        cache[code] = self._line_saliency(code)
        return cache[code]

    def _line_saliency(self, code):
        import torch
        tok, model = self.predictor.tok, self.predictor.model
        enc = tok(code, truncation=True, max_length=self.predictor.max_length,
                  return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.predictor.device) for k, v in enc.items()}
        emb_layer = model.get_input_embeddings()
        with torch.enable_grad():
            embeds = emb_layer(enc["input_ids"]).detach().requires_grad_(True)
            logits = model(inputs_embeds=embeds, attention_mask=enc["attention_mask"]).logits
            (logits[0, 1] - logits[0, 0]).backward()
        attr = (embeds.grad * embeds).sum(-1)[0].tolist()
        line_starts = [0]
        for i, ch in enumerate(code):
            if ch == "\n":
                line_starts.append(i + 1)
        per_line = {}
        for (s, e), a in zip(offsets, attr):
            if e <= s:
                continue  # special tokens
            ln = next(i for i in range(len(line_starts) - 1, -1, -1) if line_starts[i] <= s) + 1
            per_line[ln] = per_line.get(ln, 0.0) + a
        return per_line

    def localize(self, code):
        """Top-k statements (inside functions) with the highest positive per-line saliency."""
        lines = code.splitlines()
        spans = {s for s, _ in self._statements(code)}  # first line of every statement
        if not spans:
            return []
        sal = self.line_saliency(code)
        hot = [{"line_start": ln, "line_end": ln, "saliency": round(sal.get(ln, 0.0), 4), "text": lines[ln - 1].strip()[:160]}
               for ln in sorted(spans) if sal.get(ln, 0.0) > 0 and not lines[ln - 1].strip().startswith("#")]
        hot.sort(key=lambda h: -h["saliency"])
        return hot[:self.top_k_lines]

    def analyze(self, code, task_id=""):
        if self.mode == "random":
            return self._random_analysis(code, task_id)
        score = self.predictor.score(code)
        flagged = score >= self.threshold
        funcs = [f for f in self.predictor.score_functions(code) if f["name"] != "<module_or_script>"]
        funcs.sort(key=lambda f: -f["score"])
        risk = {
            "mode": "model", "score": round(score, 4), "threshold": self.threshold, "flagged": flagged,
            "functions": [{k: f[k] for k in ("name", "line_start", "line_end", "score")} for f in funcs],
            "risky_functions": [{k: f[k] for k in ("name", "line_start", "line_end", "score")} for f in funcs[:2]] if flagged else [],
            "hot_lines": self.localize(code) if flagged and self.top_k_lines > 0 else [],
        }
        return risk

    def _random_analysis(self, code, task_id):
        rng = random.Random(zlib.crc32(f"{self.seed}:{task_id}:{code}".encode()))
        flagged = rng.random() < self.random_flag_rate
        lines = code.splitlines()
        spans = self._statements(code)
        rng.shuffle(spans)
        funcs = []
        try:
            funcs = [{"name": n.name, "line_start": n.lineno, "line_end": n.end_lineno, "score": 1.0}
                     for n in ast.walk(ast.parse(code)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        except SyntaxError:
            pass
        return {
            "mode": "random", "score": 1.0 if flagged else 0.0, "threshold": 0.5, "flagged": flagged,
            "functions": funcs,
            "risky_functions": rng.sample(funcs, min(2, len(funcs))) if flagged else [],
            "hot_lines": [{"line_start": s, "line_end": e, "saliency": 0.0,
                           "text": " | ".join(l.strip() for l in lines[s - 1:e])[:160]}
                          for s, e in spans[:self.top_k_lines]] if flagged else [],
        }


# --------------------------------------------------------------------------- Step 4

class TesterAgent:
    SYSTEM = ("You are a meticulous software tester. Write Python assert statements that check a "
              "function against its specification. Do NOT implement the function. Each assert must be a "
              "single line of the form `assert <call> == <expected>` with the exact expected value. "
              "Return only the assert lines in one ```python block.")

    @staticmethod
    def _prefix(problem):
        """Pre-fills the answer with the public example (to show the format) and the start of a call
        to the entry point. Without it the 1.5B model often implements the function or writes
        trivial asserts such as `assert 2 > 1`."""
        seed = problem.public_tests[0] + "\n" if problem.public_tests else ""
        return f"```python\n{seed}assert {problem.entry_point}("

    def __init__(self, llm, n_base=5, n_targeted=5):
        self.llm = llm
        self.n_base = n_base
        self.n_targeted = n_targeted

    def base_tests(self, problem, spec):
        """Black-box tests written from the spec only (shared by every pipeline variant)."""
        if self.n_base <= 0:
            return []
        user = (_problem_block(problem) + "\n\nSpecification:\n" + _spec_block(spec) +
                f"\n\nWrite {self.n_base} diverse assert tests for `{problem.entry_point}`: "
                "typical inputs first, then edge cases. Be certain every expected value is correct.")
        raw = self.llm.chat(self.SYSTEM, user, max_new_tokens=40 * self.n_base + 20, stop_at_code_end=True,
                            prefix=self._prefix(problem))
        tests = [t for t in _parse_asserts(raw, problem.entry_point, self.n_base + 1) if t not in problem.public_tests]
        return tests[:self.n_base]

    def targeted_tests(self, problem, spec, code, risk, existing):
        """Extra tests that deliberately probe the sections the bug-prediction agent flagged."""
        if self.n_targeted <= 0 or not risk or not risk.get("flagged"):
            return []
        numbered = "\n".join(f"{i + 1:3d}| {l}" for i, l in enumerate(code.splitlines()))
        user = (_problem_block(problem) + "\n\nSpecification:\n" + _spec_block(spec) +
                f"\n\nCandidate implementation:\n```python\n{numbered}\n```\n\n" + _risk_block(risk) +
                "\n\nTests already planned:\n" + "\n".join(existing) +
                f"\n\nWrite {self.n_targeted} NEW assert tests that specifically exercise the flagged "
                "functions/lines: boundary values, empty inputs, off-by-one cases, unusual types or orderings "
                "that the specification allows. Expected values must follow the SPECIFICATION, not the code.")
        raw = self.llm.chat(self.SYSTEM, user, max_new_tokens=40 * self.n_targeted + 20, stop_at_code_end=True,
                            prefix=self._prefix(problem))
        tests = [t for t in _parse_asserts(raw, problem.entry_point, self.n_targeted + len(existing)) if t not in existing]
        return tests[:self.n_targeted]


# --------------------------------------------------------------------------- Step 5

class FixerAgent:
    SYSTEM = ("You are an expert Python debugger. Fix the implementation so it satisfies the task. "
              "Return only the complete corrected code in one ```python block, no tests, no prints.")

    def __init__(self, llm, max_failures_shown=5):
        self.llm = llm
        self.max_failures_shown = max_failures_shown

    def run(self, problem, spec, code, report, risk=None, trusted=()):
        """`trusted` tests (the task's own examples) are listed first and marked as certainly correct."""
        trusted = set(trusted)
        failures = sorted(report.failures, key=lambda r: r.test not in trusted)[:self.max_failures_shown]
        fail_txt = "\n".join(f"- {r.test}{'   [given example, certainly correct]' if r.test in trusted else ''}"
                             f"\n    -> {r.error[:300]}" for r in failures)
        if report.load_error:
            fail_txt = f"The code fails to load: {report.load_error[:300]}\n" + fail_txt
        risk_txt = _risk_block(risk)
        user = (_problem_block(problem) + "\n\nSpecification:\n" + _spec_block(spec) +
                f"\n\nCurrent implementation:\n```python\n{code}\n```\n\nFailing tests:\n{fail_txt}\n\n" +
                (risk_txt + "\nInspect the flagged code first.\n\n" if risk_txt else "") +
                "Some generated tests may themselves be wrong; when a test contradicts the task description "
                "or the example test, follow the task description. "
                f"Return the full corrected implementation as a top-level function `{problem.entry_point}`.")
        raw = self.llm.chat(self.SYSTEM, user, max_new_tokens=512, stop_at_code_end=True)
        new_code = ensure_entry_point(sanitize_code(extract_code(raw)), problem.entry_point)
        return new_code if _defines(new_code, problem.entry_point) else code
