"""Pipeline orchestration (proposal Section 3).

Paired design: Steps 1-2 (spec + first draft + alternative candidates) and the black-box base tests
are produced ONCE per problem and shared by every variant, so any difference between variants is
caused only by the bug-prediction agent (Step 3) and what the other agents do with its signal.

Variants
  baseline : 1 -> 2 -> 4 -> 5          (FlowGen / AgentCoder style, purely reactive)
  proposed : 1 -> 2 -> 3 -> 4 -> 5     (trained CodeBERT risk signal guides selection, testing and fixing)
  random   : like proposed, but Step 3 flags / ranks at random (ablation control)

v2 (default) adds three things that every variant gets, because the v1 diagnosis showed that only 48%
of agent-written tests were correct, so the Fixer spent its effort "fixing" correct code:
  * candidates   - the Developer also samples k-1 alternative solutions (fixed seeds, reproducible)
  * test voting  - a generated test is kept only if most candidates that pass the task's own example
                   test also pass it (wrong expected values disagree with most candidates)
  * fix guard    - a fix is kept only if it passes more trusted tests than the code it replaces;
                   the task's own example tests are always shown to the Fixer first
The bug-prediction agent's extra role in v2: when several candidates tie on the tests, `proposed`
starts from the one CodeBERT rates least bug-prone (`random` picks at random, `baseline` the first).
`legacy=True` reproduces the v1 pipeline exactly.
"""
import random
import time
import zlib

from .agents import RequirementAgent, DeveloperAgent, TesterAgent, FixerAgent
from .executor import run_tests, run_program
from .llm import Cancelled


def hidden_pass(problem, code, timeout):
    """Official benchmark tests -> pass@1 label. Never shown to any agent.
    Returns (None, "") for custom tasks that have no hidden tests (e.g. from the web playground)."""
    if not problem.hidden_program.strip():
        return None, ""
    ok, err = run_program(code + "\n\n" + problem.hidden_program, timeout=timeout)
    return ok, err


def pass_matrix(problem, candidates, tests, timeout):
    """pass_matrix[i][j] = candidate i passes test j (a load error fails every test)."""
    out = []
    for code in candidates:
        rep = run_tests(code, tests, setup=problem.setup_code, timeout=timeout) if tests else None
        out.append([r.ok for r in rep.results] if rep else [])
    return out


def vote_tests(public, tests, matrix):
    """Keeps generated tests that a majority of the trusted candidates pass.
    Trusted = candidates passing every public (given) test; if none do, all candidates.
    Returns (kept_tests, info). `matrix` covers public + tests, in that order."""
    n_pub = len(public)
    loaded = [row for row in matrix if row]
    anchored = [row for row in loaded if n_pub and all(row[:n_pub])]
    trusted = anchored or loaded
    # A lone candidate that no given example vouches for is not evidence: voting with it would just
    # delete every test it fails, so no test could ever catch its bugs (e.g. a 1-candidate run on a
    # task without examples). Keep all tests then.
    if not trusted or not tests or (not anchored and len(trusted) < 2):
        return list(tests), {"pool": len(trusted), "dropped": 0}
    kept = [t for j, t in enumerate(tests) if 2 * sum(row[n_pub + j] for row in trusted) > len(trusted)]
    return kept, {"pool": len(trusted), "dropped": len(tests) - len(kept)}


class Pipeline:
    def __init__(self, llm, bug_agents, n_base_tests=5, n_targeted_tests=5, max_fix_iters=3, timeout=5.0, on_event=None,
                 n_candidates=4, legacy=False, sample_temperature=0.8):
        """`bug_agents` maps variant name -> BugPredictionAgent or None (None = no Step 3).
        `on_event(dict)` is called at every agent step (used by the web UI to stream progress)."""
        self.llm = llm
        self.req = RequirementAgent(llm)
        self.dev = DeveloperAgent(llm)
        self.tester = TesterAgent(llm, n_base=n_base_tests, n_targeted=n_targeted_tests)
        self.fixer = FixerAgent(llm)
        self.bug_agents = bug_agents
        self.max_fix_iters = max_fix_iters
        self.timeout = timeout
        self.on_event = on_event
        self.legacy = legacy
        self.n_candidates = 1 if legacy else max(1, n_candidates)
        self.sample_temperature = sample_temperature

    def emit(self, **event):
        """Sends a progress event. Agent events also drive token streaming and per-step timing:
        while an agent is "running", every generated text delta is forwarded as a `token` event,
        and the matching "done" event carries the step's wall-clock `seconds`."""
        if self.llm.cancel_event.is_set():
            raise Cancelled()
        if not self.on_event:
            return
        if event.get("type") == "agent":
            key = (event.get("variant"), event["agent"], event.get("iter"))
            if event["status"] == "running":
                self._step_started = getattr(self, "_step_started", {})
                self._step_started[key] = time.time()
                tag = {k: event[k] for k in ("agent", "variant", "iter") if event.get(k) is not None}
                self.llm.on_token = lambda text: self.on_event({"type": "token", "text": text,
                                                                "n": getattr(self.llm, "delta_tokens", 1), **tag})
            else:
                started = getattr(self, "_step_started", {}).pop(key, None)
                if started is not None:
                    event["seconds"] = round(time.time() - started, 2)
                self.llm.on_token = None
        self.on_event(event)

    def _llm_snapshot(self):
        return dict(self.llm.stats())

    def _llm_delta(self, before):
        after = self.llm.stats()
        return {k: round(after[k] - before[k], 2) for k in after}

    # ------------------------------------------------------------------ shared Steps 1-2 (+ base tests)
    def shared_stage(self, problem):
        t0, before = time.time(), self._llm_snapshot()
        self.emit(type="agent", agent="requirement", status="running")
        spec = self.req.run(problem)
        self.emit(type="agent", agent="requirement", status="done", spec=spec)
        self.emit(type="agent", agent="developer", status="running")
        code0 = self.dev.run(problem, spec)
        candidates = [code0]
        if self.n_candidates > 1:
            # alternatives are not streamed to the UI (they would interleave with the main draft)
            on_token, self.llm.on_token = self.llm.on_token, None
            try:
                candidates += self.dev.candidates(problem, spec, self.n_candidates, temperature=self.sample_temperature)
            finally:
                self.llm.on_token = on_token
        self.emit(type="agent", agent="developer", status="done", code=code0)
        self.emit(type="agent", agent="tester_base", status="running")
        raw_tests = self.tester.base_tests(problem, spec)
        public = list(problem.public_tests)
        matrix = pass_matrix(problem, candidates, public + raw_tests, self.timeout) if not self.legacy else []
        base_tests, vote = (vote_tests(public, raw_tests, matrix) if not self.legacy
                            else (raw_tests, {"pool": 0, "dropped": 0}))
        self.emit(type="agent", agent="tester_base", status="done", tests=base_tests, dropped=vote["dropped"])
        init_ok, init_err = hidden_pass(problem, code0, self.timeout)
        return {
            "task_id": problem.task_id, "spec": spec, "initial_code": code0, "base_tests": base_tests,
            "base_tests_raw": raw_tests, "candidates": candidates, "candidate_matrix": matrix, "vote": vote,
            "candidates_hidden_pass": [hidden_pass(problem, c, self.timeout)[0] for c in candidates] if len(candidates) > 1 else [init_ok],
            "initial_hidden_pass": init_ok, "initial_hidden_error": init_err,
            "shared_llm": self._llm_delta(before), "shared_seconds": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------ v2 helpers
    def _score(self, problem, code, tests):
        """(given-example tests passed, all trusted tests passed) - compared lexicographically."""
        rep = run_tests(code, tests, setup=problem.setup_code, timeout=self.timeout)
        n_pub = len(problem.public_tests)
        oks = [r.ok for r in rep.results]
        return (sum(oks[:n_pub]), sum(oks)), rep

    def _select(self, problem, shared, bug):
        """Starting code for this variant: the candidate(s) passing most trusted tests; ties broken by
        the bug agent (proposed: lowest CodeBERT risk, random: random) or by order (baseline)."""
        cands = shared.get("candidates") or [shared["initial_code"]]
        if self.legacy or len(cands) == 1 or not shared.get("candidate_matrix"):
            return 0, None
        n_pub = len(problem.public_tests)
        kept = set(shared["base_tests"])
        keep = [j for j, t in enumerate(shared["base_tests_raw"]) if t in kept]
        scores = []
        for row in shared["candidate_matrix"]:
            if not row:
                scores.append((-1, -1))
                continue
            scores.append((sum(row[:n_pub]), sum(row[:n_pub]) + sum(row[n_pub + j] for j in keep)))
        best = max(scores)
        uniq = []  # tied candidates, identical code counted once
        for i, s in enumerate(scores):
            if s == best and cands[i] not in [cands[u] for u in uniq]:
                uniq.append(i)
        if len(uniq) == 1 or bug is None:
            return uniq[0], None
        if bug.mode == "random":
            rng = random.Random(zlib.crc32(f"{bug.seed}:{problem.task_id}:select".encode()))
            return rng.choice(uniq), {"tied": len(uniq), "by": "random"}
        risks = {i: bug.predictor.score(cands[i]) for i in uniq}
        pick = min(uniq, key=lambda i: (risks[i], i))
        return pick, {"tied": len(uniq), "by": "risk", "risks": {str(i): round(r, 4) for i, r in risks.items()}}

    # ------------------------------------------------------------------ Steps 3-5 per variant
    def run_variant(self, problem, shared, variant):
        bug = self.bug_agents.get(variant)
        t0, before = time.time(), self._llm_snapshot()
        spec = shared["spec"]
        cands = shared.get("candidates") or [shared["initial_code"]]
        sel_idx, sel_info = self._select(problem, shared, bug)
        code = cands[sel_idx]
        public = list(problem.public_tests)
        tests = public + list(shared["base_tests"])

        bug_seconds = 0.0
        risk = None
        targeted, targeted_raw = [], []
        if bug is not None:
            self.emit(type="agent", variant=variant, agent="bug_predictor", status="running")
            tb = time.time()
            risk = bug.analyze(code, problem.task_id)
            bug_seconds += time.time() - tb
            self.emit(type="agent", variant=variant, agent="bug_predictor", status="done", risk=risk, code=code)
            if risk["flagged"]:
                self.emit(type="agent", variant=variant, agent="tester_targeted", status="running")
            targeted_raw = self.tester.targeted_tests(problem, spec, code, risk, tests)
            targeted = targeted_raw
            if targeted_raw and not self.legacy and len(cands) > 1:
                m = pass_matrix(problem, cands, public + targeted_raw, self.timeout)
                targeted, _ = vote_tests(public, targeted_raw, m)
            if risk["flagged"]:
                self.emit(type="agent", variant=variant, agent="tester_targeted", status="done", tests=targeted)
            # Tester prioritises flagged sections: targeted tests run first
            tests = targeted + tests
        initial_risk = risk

        history = []
        fix_iters = fixes_rejected = test_executions = 0
        hidden_ok_after = []
        report = score = None
        for it in range(self.max_fix_iters + 1):
            self.emit(type="agent", variant=variant, agent="tester", status="running", iter=it)
            if report is None and tests:
                if self.legacy:
                    report = run_tests(code, tests, setup=problem.setup_code, timeout=self.timeout)
                else:
                    score, report = self._score(problem, code, tests)
            self.emit(type="agent", variant=variant, agent="tester", status="done", iter=it,
                      results=[{"test": r.test, "ok": r.ok, "error": r.error[:300],
                                "targeted": r.test in targeted} for r in (report.results if report else [])],
                      load_error=report.load_error[:300] if report else "")
            test_executions += len(tests)
            h_ok, _ = hidden_pass(problem, code, self.timeout)  # analysis only, never fed back to agents
            hidden_ok_after.append(h_ok)
            history.append({
                "iter": it, "n_tests": len(tests), "n_failed": len(report.failures) if report else 0,
                "load_error": report.load_error[:200] if report else "", "timed_out": report.timed_out if report else False,
                "risk_score": risk["score"] if risk else None, "flagged": risk["flagged"] if risk else None,
                "hidden_pass": h_ok,
            })
            if report is None or report.all_passed or it == self.max_fix_iters:
                break
            self.emit(type="agent", variant=variant, agent="fixer", status="running", iter=it + 1)
            new_code = self.fixer.run(problem, spec, code, report, risk=risk, trusted=() if self.legacy else public)
            fix_iters += 1
            if self.legacy:
                code, report = new_code, None
            else:
                new_score, new_report = self._score(problem, new_code, tests)
                if new_code == code or new_score <= score:
                    # Guard: the fix does not pass more trusted tests, so keep the old code. With greedy
                    # decoding the same prompt would give the same answer again, so stop here.
                    fixes_rejected += 1
                    history[-1]["fix_rejected"] = True
                    self.emit(type="agent", variant=variant, agent="fixer", status="done", iter=it + 1, code=code, rejected=True)
                    break
                code, report, score = new_code, new_report, new_score
            self.emit(type="agent", variant=variant, agent="fixer", status="done", iter=it + 1, code=code)
            if bug is not None:
                tb = time.time()
                risk = bug.analyze(code, problem.task_id)
                bug_seconds += time.time() - tb
                self.emit(type="agent", variant=variant, agent="bug_predictor", status="done", risk=risk, code=code, iter=it + 1)

        final_ok, final_err = hidden_pass(problem, code, self.timeout)
        first_hidden_ok = next((i for i, ok in enumerate(hidden_ok_after) if ok), None)
        cand_hidden = shared.get("candidates_hidden_pass") or [shared["initial_hidden_pass"]]
        return {
            "task_id": problem.task_id, "variant": variant,
            "initial_hidden_pass": shared["initial_hidden_pass"],
            "selected_candidate": sel_idx, "selection": sel_info,
            "selected_hidden_pass": cand_hidden[sel_idx] if sel_idx < len(cand_hidden) else None,
            "final_hidden_pass": final_ok, "final_hidden_error": final_err,
            "internal_tests_passed": bool(history) and history[-1]["n_failed"] == 0 and not history[-1]["load_error"],
            "fix_iterations": fix_iters, "fixes_rejected": fixes_rejected,
            "fix_iters_to_hidden_pass": first_hidden_ok,
            "n_public_tests": len(problem.public_tests), "n_base_tests": len(shared["base_tests"]),
            "n_targeted_tests": len(targeted), "n_targeted_raw": len(targeted_raw), "test_executions": test_executions,
            "flagged_initial": initial_risk["flagged"] if initial_risk else None,
            "risk_score_initial": initial_risk["score"] if initial_risk else None,
            "risk_initial": initial_risk,
            "targeted_tests": targeted,
            "history": history,
            "final_code": code,
            "variant_llm": self._llm_delta(before),
            "bug_agent_seconds": round(bug_seconds, 2),
            "variant_seconds": round(time.time() - t0, 2),
        }
