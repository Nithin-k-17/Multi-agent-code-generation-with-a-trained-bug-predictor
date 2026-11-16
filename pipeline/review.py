"""AI code review for the Bug Scanner, using the same local Qwen2.5-Coder model the agents use.

Small code models are much better at *fixing* code than at listing bugs in a strict format, so:
  1. The model writes a corrected version of the code (exact static findings are given as hints).
     If exact issues remain in its fix, it gets one more attempt with those issues listed.
  2. Bug locations are the lines the fix actually had to change (a diff against the user's code),
     so every reported location is grounded in a concrete change, never a guessed line number.
  3. A second short call explains each change: bug type, severity, what was wrong, how it was fixed.
Purely cosmetic changes (whitespace, comments, docstrings) are ignored.
"""
import ast
import difflib
import re

from .llm import extract_code

TYPES = ["Syntax error", "Runtime error", "Logic error", "Edge case", "Performance", "Code quality"]
SEVERITIES = ["Critical", "High", "Medium", "Low"]

FIX_SYSTEM = ("You are an expert Python debugger. Fix every real bug (crashes, wrong results, unhandled edge cases) "
              "while keeping the same function names and behaviour otherwise. Do not add prints, tests or examples. "
              "If the code is already correct, return it unchanged.")
EXPLAIN_SYSTEM = "You are a senior Python code reviewer who explains bug fixes briefly and precisely."



def _norm(value, allowed, default):
    v = value.strip().lower()
    for a in allowed:
        if a.lower() in v or (v and v in a.lower()):
            return a
    return default


def _strip_comments(line):
    s = line.split("#", 1)[0].strip()
    return re.sub(r"\s+", " ", s)


def _parses(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _fix(llm, code, issues):
    hints = ""
    if issues:
        hints = "\n\nKnown problems you must fix:\n" + "\n".join(f"- line {f['line']}: {f['title']}. {f['explanation']}" for f in issues)
    user = f"Fix the bugs in this Python code.\n```python\n{code}\n```{hints}\n\nReturn the complete corrected code in one ```python block."
    text = llm.chat(FIX_SYSTEM, user, max_new_tokens=700, stop_at_code_end=True, prefix="```python\n")
    fixed = extract_code(text).strip("\n")
    return fixed if fixed.strip() and _parses(fixed) else ""


def _hunks(code, fixed):
    """Changed regions as (old_start, old_end, old_text, new_text), 1-based lines in the user's code."""
    a, b = code.splitlines(), fixed.splitlines()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        old, new = a[i1:i2], b[j1:j2]
        if [_strip_comments(l) for l in old if _strip_comments(l)] == [_strip_comments(l) for l in new if _strip_comments(l)]:
            continue  # whitespace / comment only
        if all(re.match(r'^\s*("""|\'\'\'|#)', l) or not l.strip() for l in old + new):
            continue  # docstring / comment lines
        start = i1 + 1 if i2 > i1 else max(1, i1)  # pure insertion: anchor on the line before
        end = i2 if i2 > i1 else start
        out.append((start, end, "\n".join(old), "\n".join(new)))
    return out


def _explain_one(llm, code, old, new):
    """One focused call per change: small models answer this far more reliably than a multi-item list."""
    user = (f"Original code:\n```python\n{code}\n```\n\nThis part:\n```python\n{old or '(nothing here)'}\n```\n"
            f"was changed to:\n```python\n{new or '(removed)'}\n```\n\n"
            "Answer in exactly two short lines:\nBUG: <what was wrong in the original code, and an input where it fails>\n"
            "FIX: <what the change does>")
    text = llm.chat(EXPLAIN_SYSTEM, user, max_new_tokens=140, prefix="BUG:")
    bug = re.search(r"BUG:\s*(.+)", text)
    fix = re.search(r"FIX:\s*(.+)", text)
    return (bug.group(1).strip() if bug else ""), (fix.group(1).strip() if fix else "")


_RUNTIME = re.compile(r"\b(error|exception|crash|raise|IndexError|ZeroDivision|TypeError|KeyError|ValueError|NameError|"
                      r"AttributeError|RecursionError|out of range|divid\w* by zero|infinite)\b", re.I)
_EDGE = re.compile(r"\b(empty|zero|negative|edge|boundary|None|less than 2|<= ?1|< ?2|single|duplicate|0 or 1|n = 1|n == 1)\b", re.I)
_PERF = re.compile(r"\b(performance|efficien\w*|faster|slow|square root|sqrt|complexity|optimi[sz]\w*)\b", re.I)


def _classify(text):
    """Bug type and severity from the explanation, so labels are consistent from one review to the next."""
    if _RUNTIME.search(text):
        return "Runtime error", "High"
    if _EDGE.search(text):
        return "Edge case", "Medium"
    if _PERF.search(text) and not re.search(r"\b(wrong|incorrect)\b", text, re.I):
        return "Performance", "Low"
    return "Logic error", "High"


def review(llm, code, static_findings=()):
    # Same code -> same prompt -> same answer: trailing whitespace must not change the review.
    code = "\n".join(l.rstrip() for l in code.rstrip().splitlines())
    fixed = _fix(llm, code, list(static_findings))
    if fixed:
        from .static_check import analyze
        remaining = [f for f in analyze(fixed, execute=False) if f["type"] != "Code quality"]
        for _ in range(2):  # up to two more attempts on what is still wrong
            if not remaining:
                break
            again = _fix(llm, fixed, remaining)
            left = [f for f in analyze(again, execute=False) if f["type"] != "Code quality"] if again else remaining
            if again and len(left) < len(remaining):
                fixed, remaining = again, left
            else:
                break
    if not fixed or fixed.strip() == code.strip():
        return {"findings": [], "fixed_code": "", "says_clean": True}

    hunks = _hunks(code, fixed)
    if not hunks:
        return {"findings": [], "fixed_code": "", "says_clean": True}
    findings = []
    for s, e, old, new in hunks[:6]:
        exact = [f for f in static_findings if s <= f["line"] <= e]
        if exact:
            f0 = exact[0]
            typ, sev, expl, fix = f0["type"], f0["severity"], f0["explanation"], f0["suggestion"]
        else:
            expl, fix = _explain_one(llm, code, old, new)
            if not expl:
                expl, fix = "The reviewer rewrote this part; compare it with the corrected code.", ""
            typ, sev = _classify(expl)
        findings.append({"line": s, "end_line": e, "type": typ, "severity": sev, "title": expl.split(". ")[0][:90],
                         "explanation": expl, "suggestion": fix or "See the corrected code.", "before": old, "after": new,
                         "source": "ai"})
    return {"findings": findings, "fixed_code": fixed, "says_clean": False}
