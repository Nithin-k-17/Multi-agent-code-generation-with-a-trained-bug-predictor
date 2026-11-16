"""Web API for the multi-agent bug-aware code generation platform.

  python main.py serve                         (from the project folder; then open http://localhost:8000)

Environment:
  PIPELINE_LLM       auto | ollama | hf | gguf | openai (default auto: GGUF if installed, else Ollama if running, else PyTorch)
  PIPELINE_LOOKUP    prompt-lookup draft length for the llama.cpp backend (default 10 here; 0 = off)
  PIPELINE_MODEL     model id / served model name
  PIPELINE_BASE_URL  OpenAI-compatible base URL (for openai backend, e.g. http://localhost:11434/v1)
  OPENAI_API_KEY     key for that endpoint, if needed

Endpoints
  GET  /api/health                 status, which components are loaded, LLM config
  POST /api/scan                   {code} -> CodeBERT bug score, per-function scores, hot lines
  POST /api/run                    custom task or benchmark task -> Server-Sent Events of every agent step
  GET  /api/benchmarks/{split}     sample problems for the playground
  GET  /api/runs                   experiment result folders (results/*) with summaries
  GET  /api/runs/{name}            one run: summary + per-problem rows
  GET  /api/model                  bug-predictor metadata (threshold, validation metrics, calibration)
  GET  /api/models                 code-generator models to choose from ("default" = Qwen2.5-Coder 1.5B, plus Ollama models)
If web/dist exists (npm run build), the React app is served at /.
"""
import os
# Skip TensorFlow/Flax auto-import in transformers: cuts model load time ~2.5x (PyTorch only).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Interactive use: prompt-lookup decoding makes the Fixer ~1.7x faster (llama.cpp backend).
os.environ.setdefault("PIPELINE_LOOKUP", "10")
import json
import time
import queue
import threading
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.benchmarks import Problem, load_split, tests_from_examples, SPLITS  # noqa: E402
from pipeline.agents import BugPredictionAgent  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402
from pipeline.llm import Cancelled  # noqa: E402
from config import BUG_MODEL_DIR, PIPELINE_THRESHOLD_FILE, RESULTS_DIR, WEB_DIST_DIR  # noqa: E402

CALIBRATED = PIPELINE_THRESHOLD_FILE
DIST_DIR = WEB_DIST_DIR

app = FastAPI(title="BugAware Agents API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state = {"predictor": None, "llm": None, "llm_error": None, "warmup": "idle",
          "ollama_llms": {}, "active_model": "default"}
_pred_lock = threading.Lock()
_llm_lock = threading.Lock()
_run_lock = threading.Lock()


def _calibrated_threshold():
    if os.path.exists(CALIBRATED):
        with open(CALIBRATED, encoding="utf-8") as f:
            return json.load(f)
    return None


def get_predictor():
    with _pred_lock:
        if _state["predictor"] is None:
            from bug_predictor import BugPredictor
            _state["predictor"] = BugPredictor(device=os.environ.get("BUG_DEVICE", "cpu"))
        return _state["predictor"]


def get_llm(model=None):
    """The default generator (Qwen2.5-Coder 1.5B), or, if `model` names an Ollama model picked on the
    website (e.g. "qwen2.5-coder:7b"), that model served by Ollama."""
    with _llm_lock:
        if model and model != "default":
            if model not in _state["ollama_llms"]:
                from pipeline.llm import OllamaLLM
                _state["ollama_llms"][model] = OllamaLLM(model)
            llm = _state["ollama_llms"][model]
        else:
            if _state["llm"] is None:
                from pipeline.llm import make_llm
                _state["llm"] = make_llm(os.environ.get("PIPELINE_LLM", "auto"), os.environ.get("PIPELINE_MODEL"),
                                         os.environ.get("PIPELINE_BASE_URL"))
            llm = _state["llm"]
        _unload_other_models(keep=llm)
        _state["active_model"] = model or "default"
        return llm


def _unload_other_models(keep):
    """Big Ollama models hold several GB of RAM for up to an hour: free the ones not in use now.
    The default 1.5B model, if it lives in this process (llama.cpp), is dropped too while another
    model runs (about 1 GB); it reloads in a few seconds when it is picked again."""
    if _state["llm"] is not None and keep is not _state["llm"] and type(_state["llm"]).__name__ == "GGUFLLM":
        import gc
        _state["llm"] = None
        gc.collect()
    for other in list(_state["ollama_llms"].values()) + [_state["llm"]]:
        if other is not None and other is not keep and hasattr(other, "unload") and getattr(other, "_used", False):
            try:
                other.unload()
            except Exception:
                pass
            other._used = False
    keep._used = True


def _model_label(name):
    """"qwen2.5-coder:7b" -> "Qwen2.5-Coder 7B"."""
    family, _, tag = name.partition(":")
    family = "-".join(p[:1].upper() + p[1:] for p in family.split("-"))
    return f"{family} {tag.upper()}" if tag and tag != "latest" else family


def _param_billions(text):
    try:
        return float(text.lower().rstrip("b"))
    except ValueError:
        return 0.0


def llm_config():
    from pipeline.llm import resolve_backend
    active = _state["ollama_llms"].get(_state["active_model"])
    if active is not None:  # a model picked on the website (served by Ollama)
        return {"backend": "ollama", "model": active.name, "base_url": None}
    llm = _state["llm"]
    if llm is not None:  # don't probe the network on every health poll once the LLM is loaded
        backend = {"OllamaLLM": "ollama", "GGUFLLM": "gguf", "HFLLM": "hf"}.get(type(llm).__name__, "openai")
    else:
        backend = resolve_backend(os.environ.get("PIPELINE_LLM", "auto"))
    model = getattr(llm, "name", None) or os.environ.get("PIPELINE_MODEL")
    return {"backend": backend, "model": model, "base_url": os.environ.get("PIPELINE_BASE_URL")}


def bug_agent(mode="model"):
    cal = _calibrated_threshold()
    pred = get_predictor() if mode == "model" else None
    if mode == "random":
        return BugPredictionAgent(None, mode="random", random_flag_rate=(cal or {}).get("val_flag_rate", 0.5))
    return BugPredictionAgent(pred, mode="model", threshold=cal["threshold"] if cal else None)


# --------------------------------------------------------------------------- basic endpoints

def _warmup():
    """Loads CodeBERT and the LLM in the background at startup so the first request is fast."""
    try:
        _state["warmup"] = "loading_predictor"
        get_predictor().score("def f(x):\n    return x")
        _state["warmup"] = "loading_llm"
        llm = get_llm()
        if hasattr(llm, "warmup"):
            llm.warmup()  # Ollama: load the model into memory now
        _state["warmup"] = "ready"
    except Exception as e:  # keep the API up; the error is shown in /api/health
        _state["warmup"] = "error"
        _state["llm_error"] = f"{type(e).__name__}: {e}"


@app.on_event("startup")
def _start_warmup():
    if os.environ.get("PIPELINE_WARMUP", "1") != "0":
        threading.Thread(target=_warmup, daemon=True).start()


@app.get("/api/health")
def health():
    import sys as _sys
    torch = _sys.modules.get("torch")
    llm = _state["llm"]
    return {"ok": True, "predictor_loaded": _state["predictor"] is not None, "llm_loaded": llm is not None,
            "warmup": _state["warmup"], "llm_error": _state["llm_error"],
            "llm": {**llm_config(), "quantize": getattr(llm, "quantize", None)},
            "cuda": bool(torch and torch.cuda.is_available()), "busy": _run_lock.locked(),
            "threshold": (_calibrated_threshold() or {}).get("threshold")}


@app.get("/api/model")
def model_info():
    mdir = BUG_MODEL_DIR
    out = {"calibrated": _calibrated_threshold()}
    for name in ("threshold.json", "val_metrics.json"):
        path = os.path.join(mdir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                out[name.replace(".json", "")] = json.load(f)
    return out


@app.get("/api/models")
def models():
    """Models for the website's model picker. The 1.5B default is always first; every other model
    pulled in Ollama (e.g. `ollama pull qwen2.5-coder:7b`) is listed after it."""
    from config import OLLAMA_MODEL
    from pipeline.llm import ollama_models
    out = [{"id": "default", "label": "Qwen2.5-Coder 1.5B", "detail": "Fastest, default", "params": "1.5B",
            "size_gb": 1.0, "default": True}]
    for m in ollama_models():
        if m["name"] in (OLLAMA_MODEL, f"{OLLAMA_MODEL}:latest") or "embed" in m["name"]:
            continue  # the default is already listed; embedding models cannot write code
        big = _param_billions(m["params"]) >= 3
        out.append({"id": m["name"], "label": _model_label(m["name"]), "params": m["params"], "size_gb": m["size_gb"],
                    "detail": "Better code, about 4x slower on CPU" if big else "Ollama model", "default": False})
    return {"models": out, "active": _state["active_model"]}


class ScanRequest(BaseModel):
    code: str = Field(..., max_length=20000)
    top_k_lines: int = 5


@app.post("/api/scan")
def scan(req: ScanRequest):
    if not req.code.strip():
        raise HTTPException(400, "code is empty")
    # Standalone scanning uses the model's own decision threshold (F1-optimal on validation, tau = 0.34),
    # the one all reported accuracy / recall numbers are measured at. The pipeline keeps the stricter
    # calibrated tau (flags ~40%) because there the score is used to prioritise testing effort.
    pred = get_predictor()
    agent = BugPredictionAgent(pred, mode="model", threshold=pred.threshold)
    agent.top_k_lines = req.top_k_lines
    risk = agent.analyze(req.code)
    try:
        sal = agent.line_saliency(req.code)
    except Exception:
        sal = {}
    # Hot lines are reported even below threshold in the scanner so users can still explore
    if not risk["hot_lines"]:
        risk["hot_lines"] = agent.localize(req.code)
    risk["line_saliency"] = {str(k): round(v, 4) for k, v in sal.items()}
    # Exact findings (parser, syntax tree, sandboxed module load): real line numbers, unlike the snippet-level score.
    from pipeline.static_check import analyze as static_analyze
    risk["findings"] = static_analyze(req.code)
    return risk


class ReviewRequest(BaseModel):
    code: str = Field(..., max_length=12000)


@app.post("/api/review")
def review_code(req: ReviewRequest):
    """AI code review: bug explanations, types, severities and a corrected version, from the local Qwen model.
    Exact static findings take precedence; AI findings on the same line are folded into them."""
    if not req.code.strip():
        raise HTTPException(400, "code is empty")
    from pipeline.static_check import analyze as static_analyze
    from pipeline.review import review
    static = static_analyze(req.code)
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "The agent pipeline is running; the AI review will be available when it finishes.")
    t0 = time.time()
    try:
        llm = get_llm()
        llm.on_token = None
        out = review(llm, req.code, static)
    finally:
        _run_lock.release()
    static_lines = {f["line"] for f in static}
    merged = static + [f for f in out["findings"] if f["line"] not in static_lines]
    merged.sort(key=lambda f: (f["line"], f["source"] != "static"))
    fix_check = None
    if out["fixed_code"]:
        pred = get_predictor()
        fix_check = {"remaining": static_analyze(out["fixed_code"]), "score": round(pred.score(out["fixed_code"]), 4),
                     "threshold": pred.threshold}
    return {"findings": merged, "fixed_code": out["fixed_code"], "fix_check": fix_check,
            "says_clean": out["says_clean"] and not static, "seconds": round(time.time() - t0, 2),
            "reviewer": getattr(_state["llm"], "name", "")}


@app.get("/api/benchmarks/{split}")
def benchmarks(split: str, limit: int = 60):
    if split not in SPLITS:
        raise HTTPException(404, f"unknown split; choose from {list(SPLITS)}")
    try:
        probs = load_split(split, limit=limit, allow_download=False)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return [{"task_id": p.task_id, "description": p.description, "entry_point": p.entry_point,
             "starter_code": p.starter_code, "public_tests": p.public_tests, "has_hidden": bool(p.hidden_program)}
            for p in probs]


# --------------------------------------------------------------------------- experiments

def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _summary(run_dir):
    sp = os.path.join(run_dir, "summary.json")
    rp = os.path.join(run_dir, "results.jsonl")
    if os.path.exists(rp) and (not os.path.exists(sp) or os.path.getmtime(sp) < os.path.getmtime(rp)):
        from pipeline import analyze
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            analyze.main(run_dir)
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            out = json.load(f)
        pp = os.path.join(run_dir, "policies.json")  # written by `python main.py policies RUN_DIR`
        if os.path.exists(pp):
            with open(pp, encoding="utf-8") as f:
                out["policies"] = json.load(f)
        return out
    return None


@app.get("/api/runs")
def runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        d = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(d):
            continue
        try:
            s = _summary(d)
        except Exception as e:  # a half-written run should not break the list
            s = {"error": str(e)}
        out.append({"name": name, "summary": s, "modified": os.path.getmtime(d)})
    return sorted(out, key=lambda r: -r["modified"])


@app.get("/api/runs/{name}")
def run_detail(name: str):
    d = os.path.join(RESULTS_DIR, os.path.basename(name))
    if not os.path.isdir(d):
        raise HTTPException(404, "run not found")
    rows = _read_jsonl(os.path.join(d, "results.jsonl"))
    light = [{k: r.get(k) for k in ("task_id", "variant", "initial_hidden_pass", "final_hidden_pass", "fix_iterations",
                                    "n_targeted_tests", "test_executions", "flagged_initial", "risk_score_initial",
                                    "variant_seconds")} for r in rows]
    return {"name": name, "summary": _summary(d), "rows": light}


# --------------------------------------------------------------------------- live pipeline run (SSE)

class RunRequest(BaseModel):
    description: str = ""
    entry_point: str = ""
    starter_code: str = ""
    public_tests: list[str] = []
    benchmark_split: str | None = None
    task_id: str | None = None
    variants: list[str] = ["baseline", "proposed"]
    max_fix_iters: int = Field(3, ge=0, le=6)
    n_base_tests: int = Field(5, ge=0, le=12)
    n_targeted_tests: int = Field(5, ge=0, le=12)
    n_candidates: int = Field(3, ge=1, le=6)  # v2: candidate solutions used to vote out wrong tests
    model: str | None = Field(None, max_length=100)  # code generator: None/"default" = 1.5B, else an Ollama model


def _resolve_problem(req: RunRequest):
    if req.task_id and req.benchmark_split:
        for p in load_split(req.benchmark_split, allow_download=False):
            if p.task_id == req.task_id:
                return p
        raise HTTPException(404, "task not found")
    if not req.description.strip() or not req.entry_point.strip():
        raise HTTPException(400, "description and entry_point are required")
    entry = req.entry_point.strip()
    tests = [t.strip() for t in req.public_tests if t.strip()]
    if not tests:
        # No example tests given: derive them from "Input: ... Output: ..." / doctest examples in the text
        tests = tests_from_examples(req.description, entry)
    return Problem(task_id="custom", dataset="custom", description=req.description, entry_point=entry,
                   starter_code=req.starter_code, public_tests=tests)


def _sse(obj):
    return f"data: {json.dumps(obj, default=str)}\n\n"


@app.post("/api/run")
def run(req: RunRequest):
    problem = _resolve_problem(req)
    variants = [v for v in req.variants if v in ("baseline", "proposed", "random")] or ["baseline", "proposed"]
    if req.model and req.model != "default":
        from pipeline.llm import ollama_models
        if req.model not in {m["name"] for m in ollama_models()}:
            raise HTTPException(400, f"Model '{req.model}' is not available. Is Ollama running and the model pulled?")
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(409, "another run is in progress; try again when it finishes")

    events = queue.Queue()

    run_start = time.time()
    cancel = threading.Event()

    def worker():
        llm = None
        try:
            if req.model and req.model != "default":
                events.put({"type": "status", "message": f"Loading {_model_label(req.model)} (the first run takes longer)..."})
            elif _state["llm"] is None:
                events.put({"type": "status", "message": "Loading the language model (first run only)..."})
            llm = get_llm(req.model)
            llm.cancel_event = cancel
            agents = {"baseline": None}
            if "proposed" in variants:
                agents["proposed"] = bug_agent("model")
            if "random" in variants:
                agents["random"] = bug_agent("random")
            pipe = Pipeline(llm, agents, n_base_tests=req.n_base_tests, n_targeted_tests=req.n_targeted_tests,
                            max_fix_iters=req.max_fix_iters, on_event=events.put, n_candidates=req.n_candidates)
            events.put({"type": "start", "task_id": problem.task_id, "variants": variants,
                        "llm": getattr(llm, "name", ""), "has_hidden": bool(problem.hidden_program),
                        "public_tests": problem.public_tests,
                        "auto_tests": not any(t.strip() for t in req.public_tests) and bool(problem.public_tests)})
            shared = pipe.shared_stage(problem)
            if not shared["initial_code"].strip():
                events.put({"type": "warning", "message": "The Developer agent returned no code. The CPU may be busy "
                            "(close heavy browser tabs) or the task may be too long for this model."})
            events.put({"type": "shared_done", "initial_hidden_pass": shared["initial_hidden_pass"]})
            for v in variants:
                events.put({"type": "variant_start", "variant": v})
                row = pipe.run_variant(problem, shared, v)
                row.pop("risk_initial", None)
                events.put({"type": "variant_done", "variant": v, "result": row})
            events.put({"type": "done", "seconds": round(time.time() - run_start, 2)})
        except Cancelled:
            events.put({"type": "error", "message": "Run cancelled"})
        except Exception as e:
            events.put({"type": "error", "message": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})
        finally:
            if llm is not None:
                llm.on_token = None
                llm.cancel_event = threading.Event()
            events.put(None)
            _run_lock.release()

    threading.Thread(target=worker, daemon=True).start()

    def stream():
        try:
            while True:
                ev = events.get()
                if ev is None:
                    break
                yield _sse(ev)
        finally:
            cancel.set()  # client disconnected or finished: stop generating immediately

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- built frontend

if os.path.isdir(DIST_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(DIST_DIR, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        target = os.path.join(DIST_DIR, path)
        if path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
