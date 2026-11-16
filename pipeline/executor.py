"""Sandboxed execution of generated code in an isolated Python subprocess.

Same policy as the dataset generator used to train the bug predictor:
isolated interpreter (`python -I`), temporary working directory, hard timeout.
"""
import os
import sys
import json
import tempfile
import subprocess
from dataclasses import dataclass, field

DEFAULT_TIMEOUT = 5.0

# The harness defines the solution once, then runs every assert separately so
# we know exactly which tests failed and why. One JSON line per test is
# flushed immediately, so tests that never report back were killed by the timeout.
_HARNESS = r'''
import json, sys, traceback
_payload = json.loads(sys.stdin.read())
_ns = {"__name__": "__main__"}
def _emit(obj):
    sys.__stdout__.write(json.dumps(obj) + "\n"); sys.__stdout__.flush()
try:
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_payload["code"] + "\n" + _payload["setup"], "<solution>", "exec"), _ns)
except BaseException as e:
    _emit({"idx": -1, "ok": False, "error": f"{type(e).__name__}: {e}"})
    sys.exit(0)
for _i, _t in enumerate(_payload["tests"]):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(_t, f"<test_{_i}>", "exec"), _ns)
        _emit({"idx": _i, "ok": True, "error": ""})
    except AssertionError as e:
        _emit({"idx": _i, "ok": False, "error": ("AssertionError: " + str(e)) if str(e) else "AssertionError"})
    except BaseException as e:
        _emit({"idx": _i, "ok": False, "error": f"{type(e).__name__}: {e}"})
'''


@dataclass
class TestResult:
    test: str
    ok: bool
    error: str = ""


@dataclass
class ExecReport:
    results: list = field(default_factory=list)   # list[TestResult]
    load_error: str = ""                           # solution itself failed to import/define
    timed_out: bool = False

    @property
    def all_passed(self):
        return not self.load_error and bool(self.results) and all(r.ok for r in self.results)

    @property
    def failures(self):
        return [r for r in self.results if not r.ok]


def sandbox_limits(cpu_seconds=10):
    """`preexec_fn` for code runs on Linux/macOS (e.g. the public Hugging Face Space): caps CPU time,
    memory, written file size and open files of the child process, so code typed by a visitor cannot
    exhaust the server. Returns None on Windows, where `resource` does not exist."""
    if os.name != "posix" or os.environ.get("PIPELINE_SANDBOX_LIMITS", "1") == "0":
        return None

    def apply():
        import resource
        mem = int(os.environ.get("PIPELINE_SANDBOX_MEM_MB", "1024")) * 1024 * 1024
        for res, value in ((resource.RLIMIT_CPU, cpu_seconds), (resource.RLIMIT_AS, mem),
                           (resource.RLIMIT_FSIZE, 10 * 1024 * 1024), (resource.RLIMIT_NOFILE, 64),
                           (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass
        os.setsid()  # own process group, so nothing it starts outlives it as part of the server's group
    return apply


def _run(script, stdin, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                input=stdin, capture_output=True, text=True, timeout=timeout, cwd=tmp,
                encoding="utf-8", errors="replace", preexec_fn=sandbox_limits(int(timeout) + 2),
            )
            return proc.stdout, proc.returncode, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return out, None, True


def run_tests(code, tests, setup="", timeout=DEFAULT_TIMEOUT):
    """Runs each assert in `tests` against `code`; returns an ExecReport with per-test outcomes."""
    report = ExecReport()
    if not tests:
        return report
    stdout, _, timed_out = _run(_HARNESS, json.dumps({"code": code, "tests": tests, "setup": setup}), timeout)
    report.timed_out = timed_out
    seen = {}
    for line in stdout.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj["idx"] == -1:
            report.load_error = obj["error"]
        else:
            seen[obj["idx"]] = obj
    for i, t in enumerate(tests):
        if report.load_error:
            report.results.append(TestResult(t, False, report.load_error))
        elif i in seen:
            report.results.append(TestResult(t, seen[i]["ok"], seen[i]["error"]))
        else:
            report.results.append(TestResult(t, False, "Timeout (possible infinite loop)" if timed_out else "No result"))
    return report


def run_program(program, timeout=DEFAULT_TIMEOUT):
    """Runs a whole program (solution + official tests). Returns (passed, error_text)."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", program],
                capture_output=True, text=True, timeout=timeout, cwd=tmp,
                encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, "Timeout"
    if proc.returncode == 0:
        return True, ""
    err = proc.stderr.strip().splitlines()
    return False, err[-1] if err else f"exit code {proc.returncode}"
