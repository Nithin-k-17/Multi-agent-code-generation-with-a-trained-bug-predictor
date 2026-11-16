"""Benchmark loaders for the multi-agent pipeline (MBPP and HumanEval).

Each problem is normalised into a `Problem` with:
  - `public_tests`: tests the agents are allowed to see (MBPP's first assert,
    HumanEval's docstring examples). These mirror what a developer is given.
  - `hidden_program`: the official test suite, used ONLY for final pass@1
    scoring. No agent ever sees it.

Splits follow the bug-predictor's data card so the evaluation never touches
problems the classifier was trained on:
  mbpp_test : MBPP task IDs 601-974 (held-out, use for the main experiment)
  mbpp_val  : MBPP task IDs 511-600 (use for tuning threshold / top-k)
  humaneval : all 164 HumanEval problems (zero-shot for the bug predictor)
"""
import os
import re
import ast
import gzip
import json
import doctest
import urllib.request
from dataclasses import dataclass, field

from config import BENCHMARK_DIR as BENCH_DIR

SOURCES = {
    "mbpp": ("https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl", "mbpp.jsonl"),
    "humaneval": ("https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz", "HumanEval.jsonl.gz"),
}

SPLITS = {
    "mbpp_test": ("mbpp", 601, 974),
    "mbpp_val": ("mbpp", 511, 600),
    "humaneval": ("humaneval", None, None),
}


@dataclass
class Problem:
    task_id: str
    dataset: str
    description: str          # natural-language task given to the Requirement Agent
    entry_point: str          # function name the solution must define
    starter_code: str         # signature (+ docstring for HumanEval), may be ""
    public_tests: list = field(default_factory=list)   # list[str] of assert statements
    hidden_program: str = ""  # official tests; append to solution and run
    setup_code: str = ""      # MBPP test_setup_code (imports etc.)


def _path(name):
    return os.path.join(BENCH_DIR, SOURCES[name][1])


def download(name):
    """Downloads a benchmark file into data/benchmarks/ if it is not already there."""
    url, fname = SOURCES[name]
    dest = os.path.join(BENCH_DIR, fname)
    if os.path.exists(dest):
        return dest
    os.makedirs(BENCH_DIR, exist_ok=True)
    print(f"Downloading {name} from {url} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def _read_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _mbpp_entry_point(test):
    """First non-builtin function called in an MBPP assert, e.g. `assert set(foo(x)) == ...` -> foo."""
    import builtins
    for node in ast.walk(ast.parse(test)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not hasattr(builtins, node.func.id):
            return node.func.id
    m = re.search(r"([A-Za-z_]\w*)\s*\(", test)
    return m.group(1) if m else ""


def _mbpp_problems(lo, hi):
    problems = []
    for r in _read_jsonl(_path("mbpp")):
        tid = int(r["task_id"])
        if not (lo <= tid <= hi):
            continue
        tests = r["test_list"]
        setup = r.get("test_setup_code", "") or ""
        problems.append(Problem(
            task_id=f"mbpp/{tid}",
            dataset="mbpp",
            description=r["text"],
            entry_point=_mbpp_entry_point(tests[0]),
            starter_code="",
            public_tests=[tests[0]],
            hidden_program=setup + "\n" + "\n".join(tests),
            setup_code=setup,
        ))
    return problems


def _literal_call(src, entry_point):
    """True if `src` is `entry_point(<literal args>)`, so the assert cannot hit an undefined name."""
    try:
        node = ast.parse(src, mode="eval").body
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry_point):
            return False
        for arg in list(node.args) + [k.value for k in node.keywords]:
            ast.literal_eval(arg)
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def _humaneval_public_tests(prompt, entry_point):
    """Turns docstring examples (`>>> f(x)` + output, or `f(x) => y` / `f(x) ==> y` / `f(x) == y`)
    into assert statements. Only examples with literal arguments and values are kept."""
    pairs = []
    try:
        for ex in doctest.DocTestParser().get_examples(prompt):
            want = ex.want.strip().splitlines()[0].strip() if ex.want.strip() else ""
            pairs.append((ex.source.strip(), want.rstrip("\"' ").strip()))
    except ValueError:
        pass
    for m in re.finditer(rf"^\s*({re.escape(entry_point)}\(.*\))\s*(?:==>|=>|==|➞|->)\s*(.+?)\s*$", prompt, re.M):
        pairs.append((m.group(1), m.group(2)))
    tests = []
    for src, want in pairs:
        if not want or "\n" in src or not _literal_call(src, entry_point):
            continue
        try:
            ast.literal_eval(want)
        except (SyntaxError, ValueError):
            continue
        t = f"assert {src} == {want}"
        if t not in tests:
            tests.append(t)
    return tests


_IO_EXAMPLE = re.compile(r"Input\s*:\s*(.+?)\s*\n\s*Output\s*:\s*(.+?)\s*(?:\n|$)", re.I)
_ARG_SPLIT = re.compile(r",\s*(?=[A-Za-z_]\w*\s*=)")


def _pyliteral(text):
    """LeetCode examples use true/false/null; returns the Python source of the literal or None."""
    text = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", text.strip())))
    try:
        ast.literal_eval(text)
        return text
    except (SyntaxError, ValueError):
        return None


def tests_from_examples(description, entry_point, limit=5):
    """Builds assert statements from examples written in a free-text task:
    LeetCode style ("Input: nums = [1,2], k = 3" / "Output: 2"), `>>> f(x)` doctests and `f(x) => y` lines.
    Used for custom playground tasks when the user gives no example tests."""
    tests = []
    for inp, out in _IO_EXAMPLE.findall(description):
        args = [a.split("=", 1)[1] if re.match(r"\s*[A-Za-z_]\w*\s*=", a) else a for a in _ARG_SPLIT.split(inp)]
        args = [_pyliteral(a) for a in args]
        expected = _pyliteral(out)
        if expected is None or any(a is None for a in args):
            continue
        t = f"assert {entry_point}({', '.join(args)}) == {expected}"
        if t not in tests:
            tests.append(t)
    for t in _humaneval_public_tests(description, entry_point):
        if t not in tests:
            tests.append(t)
    return tests[:limit]


def _humaneval_problems():
    problems = []
    for r in _read_jsonl(_path("humaneval")):
        problems.append(Problem(
            task_id=r["task_id"],
            dataset="humaneval",
            description=r["prompt"],
            entry_point=r["entry_point"],
            starter_code=r["prompt"],
            public_tests=_humaneval_public_tests(r["prompt"], r["entry_point"]),
            hidden_program=r["test"] + f"\ncheck({r['entry_point']})\n",
        ))
    return problems


def load_split(split, limit=None, allow_download=True):
    """Loads a split name from SPLITS and returns a list of Problem."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; choose from {list(SPLITS)}")
    name, lo, hi = SPLITS[split]
    if not os.path.exists(_path(name)):
        if not allow_download:
            raise FileNotFoundError(f"{_path(name)} missing; run with downloads allowed or place the file there.")
        download(name)
    problems = _mbpp_problems(lo, hi) if name == "mbpp" else _humaneval_problems()
    return problems[:limit] if limit else problems
