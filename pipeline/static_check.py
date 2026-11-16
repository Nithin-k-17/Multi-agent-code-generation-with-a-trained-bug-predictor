"""Exact, rule-based bug checks for the Bug Scanner.

The CodeBERT classifier scores a whole snippet; it cannot say *which* line is wrong. These checks
complement it with findings whose line numbers are exact because they come from the Python parser,
the syntax tree, or an actual sandboxed run of the module:

  - syntax errors (parser)                       - calls to undefined names (NameError)
  - errors raised while defining the module      - off-by-one loops: range(len(x) + 1) indexing x[i]
  - x[len(x)] indexing                            - division by len(x) with no empty-input guard
  - infinite `while True` loops without break     - recursion without any base case
  - functions that return a value on some paths but fall off the end (implicit None)
  - loops that return on the very first iteration - removing from / appending to a list while iterating it
  - mutable default arguments                     - bare `except:` / swallowed exceptions
  - `== None` comparisons, `is` with literals     - shadowing built-ins

Every finding: {line, end_line, type, severity, title, explanation, suggestion, source: "static"}.
"""
import ast
import builtins
import re
import subprocess
import sys
import tempfile

from pipeline.executor import sandbox_limits

BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__builtins__", "self", "cls"}
SHADOWABLE = {"list", "dict", "str", "int", "float", "set", "tuple", "sum", "min", "max", "len", "input", "id",
              "type", "map", "filter", "range", "sorted", "any", "all", "abs", "round", "open", "object", "iter", "next"}


def _f(node, typ, sev, title, expl, fix, end=None):
    line = getattr(node, "lineno", node) if not isinstance(node, int) else node
    return {"line": line, "end_line": end or getattr(node, "end_lineno", line) or line, "type": typ, "severity": sev,
            "title": title, "explanation": expl, "suggestion": fix, "source": "static"}


def _is_len_call(node, name=None):
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "len" and node.args
            and (name is None or (isinstance(node.args[0], ast.Name) and node.args[0].id == name)))


def _len_arg(node):
    a = node.args[0]
    return a.id if isinstance(a, ast.Name) else ast.unparse(a)


def _always_returns(stmts):
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return _always_returns(last.body) and _always_returns(last.orelse)
    if isinstance(last, ast.Try):
        return (_always_returns(last.body) or _always_returns(last.finalbody)) and all(_always_returns(h.body) for h in last.handlers)
    if isinstance(last, ast.While) and isinstance(last.test, ast.Constant) and last.test.value is True:
        return not any(isinstance(n, ast.Break) for n in ast.walk(last))
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _always_returns(last.body)
    if isinstance(last, ast.Match):
        return all(_always_returns(c.body) for c in last.cases) and any(
            isinstance(c.pattern, ast.MatchAs) and c.pattern.pattern is None for c in last.cases)
    return False


def _own_nodes(fn):
    """Nodes inside a function, not descending into nested functions/classes."""
    stack = list(fn.body)
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if not isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(c)


def _defined_names(tree):
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs + ([a.vararg] if a.vararg else []) + ([a.kwarg] if a.kwarg else []):
                names.add(arg.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                names.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif isinstance(n, ast.MatchAs) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.MatchStar) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
    return names


def _has_empty_guard(fn, var):
    """True if the function checks `not var`, `len(var) == 0`, `if var`, try/except ZeroDivisionError, etc."""
    v = re.escape(var)
    guard = re.compile(rf"(\bnot\s+{v}\b|\blen\(\s*{v}\s*\)|^\s*{v}\s*$|\b{v}\s*(==|!=|is)\b|\b{v}\s+and\b|\band\s+{v}\b)")
    for n in _own_nodes(fn):
        if isinstance(n, (ast.If, ast.IfExp, ast.While, ast.Assert)):
            if guard.search(ast.unparse(n.test)):
                return True
        if isinstance(n, ast.Try) and any(h.type is None or "ZeroDivision" in ast.unparse(h.type) or "Exception" in ast.unparse(h.type)
                                         for h in n.handlers):
            return True
        if isinstance(n, ast.BoolOp) and guard.search(ast.unparse(n)):
            return True
    return False


def _check_function(fn, out):
    name = fn.name
    nodes = list(_own_nodes(fn))

    # mutable default arguments
    for d in fn.args.defaults + [d for d in fn.args.kw_defaults if d is not None]:
        if isinstance(d, (ast.List, ast.Dict, ast.Set)) or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in ("list", "dict", "set")):
            out.append(_f(d, "Logic error", "Medium", "Mutable default argument",
                          f"The default value `{ast.unparse(d)}` is created once and shared by every call of `{name}`, so changes leak between calls.",
                          f"Use `None` as the default and create a new object inside the function, e.g. `if x is None: x = {ast.unparse(d)}`."))

    # implicit None on some paths
    returns_value = [n for n in nodes if isinstance(n, ast.Return) and n.value is not None]
    if returns_value and not _always_returns(fn.body) and not any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in nodes):
        last = fn.body[-1]
        out.append(_f(getattr(last, "end_lineno", last.lineno), "Logic error", "Medium", "Missing return on some paths",
                      f"`{name}` returns a value on some paths but can reach the end without a `return`, so it silently returns `None`.",
                      "Add a final `return` (or `raise`) that covers the remaining cases."))

    # recursion without a base case
    calls_self = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name for n in nodes)
    if calls_self and not any(isinstance(n, (ast.If, ast.IfExp, ast.Match, ast.Try, ast.While, ast.For)) for n in nodes):
        out.append(_f(fn, "Runtime error", "High", "Recursion without a base case",
                      f"`{name}` calls itself unconditionally, which ends in RecursionError.",
                      "Add a stopping condition (e.g. `if n <= 1: return ...`) before the recursive call.", end=fn.lineno))

    for n in nodes:
        # off-by-one: for i in range(len(x) + 1) ... x[i]
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Call) and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range" and isinstance(n.target, ast.Name):
            stop = n.iter.args[-1] if len(n.iter.args) in (1, 2) else (n.iter.args[1] if len(n.iter.args) == 3 else None)
            if isinstance(stop, ast.BinOp) and isinstance(stop.op, ast.Add) and _is_len_call(stop.left) and isinstance(stop.right, ast.Constant) and stop.right.value == 1:
                seq = _len_arg(stop.left)
                idx = n.target.id
                uses = [s for s in ast.walk(n) if isinstance(s, ast.Subscript) and ast.unparse(s.value) == seq and ast.unparse(s.slice) == idx]
                if uses:
                    out.append(_f(n.iter, "Runtime error", "High", "Off-by-one: index out of range",
                                  f"`range(len({seq}) + 1)` makes `{idx}` reach `len({seq})`, but the last valid index is `len({seq}) - 1`, so `{seq}[{idx}]` raises IndexError on the final iteration.",
                                  f"Loop over `range(len({seq}))`, or iterate directly with `for item in {seq}:`."))
        # x[len(x)]
        if isinstance(n, ast.Subscript) and _is_len_call(n.slice) and ast.unparse(n.slice.args[0]) == ast.unparse(n.value):
            seq = ast.unparse(n.value)
            out.append(_f(n, "Runtime error", "High", "Index equals length",
                          f"`{seq}[len({seq})]` is always one past the end and raises IndexError.",
                          f"Use `{seq}[-1]` (last element) or `{seq}[len({seq}) - 1]`."))
        # division by len(x) without an empty guard
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv, ast.Mod)) and _is_len_call(n.right):
            var = _len_arg(n.right)
            if not _has_empty_guard(fn, var):
                out.append(_f(n, "Runtime error", "Medium", "Division by zero on empty input",
                              f"`{ast.unparse(n)}` divides by `len({var})`, which is 0 when `{var}` is empty → ZeroDivisionError.",
                              f"Handle the empty case first, e.g. `if not {var}: return 0` (or whatever the task expects)."))
        # infinite loop
        if isinstance(n, ast.While) and isinstance(n.test, ast.Constant) and n.test.value is True:
            if not any(isinstance(s, (ast.Break, ast.Return, ast.Raise)) for s in ast.walk(n)):
                out.append(_f(n, "Runtime error", "High", "Infinite loop",
                              "`while True` has no `break`, `return` or `raise`, so it never ends.",
                              "Add an exit condition inside the loop.", end=n.lineno))
        # loop that returns on its first iteration
        if isinstance(n, (ast.For, ast.While)) and n.body and isinstance(n.body[-1], ast.If):
            last = n.body[-1]
            if _always_returns(last.body) and _always_returns(last.orelse):
                out.append(_f(last, "Logic error", "High", "Loop exits on the first iteration",
                              "Both branches of this `if` return, so the loop never gets past its first element.",
                              "Return the 'not found' result after the loop finishes instead of in the `else` branch.", end=last.lineno))
        # modifying a list while iterating over it
        if isinstance(n, ast.For) and isinstance(n.iter, ast.Name):
            seq = n.iter.id
            for s in ast.walk(n):
                if (isinstance(s, ast.Call) and isinstance(s.func, ast.Attribute) and isinstance(s.func.value, ast.Name)
                        and s.func.value.id == seq and s.func.attr in ("remove", "append", "pop", "insert", "extend", "clear")):
                    out.append(_f(s, "Logic error", "High", "List modified while iterating",
                                  f"`{seq}.{s.func.attr}(...)` changes `{seq}` while the `for` loop is iterating over it, so elements are skipped or the loop never ends.",
                                  f"Iterate over a copy (`for x in {seq}[:]`) or build a new list instead."))
                    break


def _check_module(tree, code, out):
    defined = _defined_names(tree)
    reported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined and n.id not in BUILTINS and n.id not in reported:
            reported.add(n.id)
            out.append(_f(n, "Runtime error", "High", f"Undefined name `{n.id}`",
                          f"`{n.id}` is used but never defined or imported, so this line raises NameError when it runs.",
                          f"Define `{n.id}`, fix the spelling, or add the missing import."))
        if isinstance(n, ast.ExceptHandler):
            if n.type is None:
                out.append(_f(n, "Code quality", "Low", "Bare except",
                              "`except:` also catches KeyboardInterrupt/SystemExit and hides real errors.",
                              "Catch the specific exception, e.g. `except ValueError:`.", end=n.lineno))
            elif len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
                out.append(_f(n, "Code quality", "Low", "Exception silently ignored",
                              "The error is caught and discarded with `pass`, which hides bugs.",
                              "Log or handle the error, or let it propagate.", end=n.lineno))
        if isinstance(n, ast.Compare):
            for op, right in zip(n.ops, n.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(right, ast.Constant) and right.value is None:
                    out.append(_f(n, "Code quality", "Low", "Comparison to None with ==",
                                  "`== None` can be fooled by objects that override `__eq__`.",
                                  "Use `is None` / `is not None`."))
                if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and isinstance(right.value, (int, float, str)) and not isinstance(right.value, bool):
                    out.append(_f(n, "Logic error", "Medium", "Identity comparison with a literal",
                                  f"`is` checks object identity, not value, so `{ast.unparse(n)}` can be False even when the values are equal.",
                                  "Use `==` / `!=` to compare values."))
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in SHADOWABLE:
                    out.append(_f(n, "Code quality", "Low", f"Built-in `{t.id}` shadowed",
                                  f"Assigning to `{t.id}` hides Python's built-in `{t.id}()` for the rest of this scope.",
                                  f"Rename the variable, e.g. `{t.id}_`."))
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(n, out)


def _run_module(code, timeout=3.0):
    """Defines the module in an isolated interpreter; reports an exception raised at definition time."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], input="", capture_output=True, text=True,
                                  timeout=timeout, cwd=tmp, encoding="utf-8", errors="replace",
                                  preexec_fn=sandbox_limits(int(timeout) + 2))
        except subprocess.TimeoutExpired:
            return {"line": 1, "end_line": 1, "type": "Runtime error", "severity": "High", "title": "Code does not finish",
                    "explanation": f"Running the file did not finish within {timeout:.0f} s (an infinite loop or blocking call at top level).",
                    "suggestion": "Move top-level work into a function, or fix the loop's exit condition.", "source": "execution"}
    if proc.returncode == 0:
        return None
    err = proc.stderr.strip().splitlines()
    if not err or err[-1].startswith(("EOFError", "KeyboardInterrupt")):
        return None
    lines = [int(m.group(1)) for m in re.finditer(r'File "<string>", line (\d+)', proc.stderr)]
    exc = err[-1]
    return {"line": lines[-1] if lines else 1, "end_line": lines[-1] if lines else 1, "type": "Runtime error", "severity": "Critical",
            "title": exc.split(":")[0], "explanation": f"Running the file raised `{exc}`.",
            "suggestion": "Fix this line so the module can load; the functions cannot be used until it does.", "source": "execution"}


def analyze(code, execute=True):
    """Returns a list of findings sorted by line."""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        line = e.lineno or 1
        return [{"line": line, "end_line": line, "type": "Syntax error", "severity": "Critical", "title": e.msg.capitalize(),
                 "explanation": f"Python cannot parse line {line}: {e.msg}. Nothing in the file can run until this is fixed.",
                 "suggestion": "Check brackets, colons and indentation on this line (and the line before it).", "source": "static"}]
    _check_module(tree, code, out)
    if execute and not any(f["title"].startswith("Undefined name") for f in out):
        r = _run_module(code)
        if r:
            out.append(r)
    seen, uniq = set(), []
    for f in sorted(out, key=lambda f: (f["line"], f["title"])):
        k = (f["line"], f["title"])
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq
