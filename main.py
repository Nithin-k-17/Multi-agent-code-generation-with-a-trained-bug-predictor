"""Main entry point for the project. Run `python main.py --help` to see every command.

    python main.py check                 check Python, libraries, model files and Ollama
    python main.py demo                  score sample buggy / clean functions with the trained model
    python main.py scan FILE.py          score every function in a Python file
    python main.py serve                 start the website + API at http://localhost:8000

    python main.py generate-data ...     Phase A: build the labelled dataset (GPU)
    python main.py baselines             Phase B: trivial baselines
    python main.py train ...             Phase C: fine-tune CodeBERT (GPU)
    python main.py evaluate ...          Phase D: held-out evaluation with confidence intervals
    python main.py calibrate ...         choose the pipeline's risk threshold on validation

    python main.py experiment ...        Phase E: multi-agent experiment (baseline / proposed / random)
    python main.py analyze RUN_DIR       summary table + significance tests for an experiment
    python main.py policies RUN_DIR      offline: when is the bug predictor worth using? (v2 runs)
    python main.py selftest [--model]    offline end-to-end test of the pipeline (no LLM needed)

Add --help after any command for its options, e.g. `python main.py train --help`.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def cmd_check(_args):
    import importlib
    import platform
    from config import BUG_MODEL_DIR, OLLAMA_MODEL, PROCESSED_DIR, WEB_DIST_DIR

    ok = True
    print(f"Python {platform.python_version()} on {platform.system()}")
    if sys.version_info < (3, 10):
        print("  !! Python 3.10 or newer is required")
        ok = False
    for module in ("torch", "transformers", "sklearn", "numpy", "fastapi", "uvicorn", "openai"):
        try:
            mod = importlib.import_module(module)
            print(f"  [ok] {module} {getattr(mod, '__version__', '')}")
        except ImportError:
            print(f"  [missing] {module}  -> pip install -r requirements.txt")
            ok = False
    try:
        import torch
        print(f"  GPU (CUDA): {'yes - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no (CPU only)'}")
    except ImportError:
        pass

    model_file = os.path.join(BUG_MODEL_DIR, "model.safetensors")
    print(f"  [{'ok' if os.path.exists(model_file) else 'missing'}] bug-predictor model: {model_file}")
    ok &= os.path.exists(model_file)
    n_data = len([f for f in os.listdir(PROCESSED_DIR) if f.endswith(".jsonl")]) if os.path.isdir(PROCESSED_DIR) else 0
    print(f"  [{'ok' if n_data else 'missing'}] dataset files in data/processed: {n_data}")
    print(f"  [{'ok' if os.path.isdir(WEB_DIST_DIR) else 'missing'}] built website (web/dist)")

    from pipeline.llm import gguf_available, ollama_available, resolve_backend
    if gguf_available():
        print("  [ok] llama.cpp backend (llama-cpp-python): fastest, supports --fast prompt-lookup decoding")
    else:
        print("  [optional] llama-cpp-python not installed (pip install llama-cpp-python) - the fastest backend")
    if ollama_available():
        print(f"  [ok] Ollama is running with {OLLAMA_MODEL}")
    else:
        print(f"  [optional] Ollama not running or '{OLLAMA_MODEL}' not pulled")
    names = {"gguf": "llama.cpp", "ollama": "Ollama", "hf": "PyTorch / Hugging Face (slow on CPU)"}
    print(f"  -> code generator in use: {names.get(resolve_backend('auto'), resolve_backend('auto'))}")
    print("\nAll required parts are in place." if ok else "\nSome required parts are missing (see above).")
    return 0 if ok else 1


def cmd_demo(args):
    from bug_predictor.demo import main
    main(device=args[0] if args else "cpu")


def cmd_scan(args):
    if not args:
        raise SystemExit("usage: python main.py scan FILE.py")
    from bug_predictor import BugPredictor
    with open(args[0], encoding="utf-8") as f:
        code = f.read()
    bp = BugPredictor(device="cpu")
    print(f"{args[0]}  (threshold {bp.threshold:.2f})")
    for fn in bp.score_functions(code):
        flag = "RISKY" if fn["is_risky"] else "ok"
        print(f"  {fn['name']:30s} lines {fn['line_start']:>4}-{fn['line_end']:<4} score {fn['score']:.4f}  {flag}")


def cmd_serve(args):
    import uvicorn
    port = int(args[args.index("--port") + 1]) if "--port" in args else 8000
    print(f"Open http://localhost:{port} in your browser (Ctrl+C to stop).")
    uvicorn.run("server.app:app", host="127.0.0.1", port=port)


def _module_main(module, takes_argv=True):
    def run(args):
        mod = __import__(module, fromlist=["main"])
        if takes_argv:
            return mod.main(args)
        sys.argv = [module] + args
        return mod.main()
    return run


def cmd_analyze(args):
    if len(args) != 1:
        raise SystemExit("usage: python main.py analyze results/<run-name>")
    from pipeline import analyze
    analyze.main(args[0])


def cmd_policies(args):
    if len(args) != 1:
        raise SystemExit("usage: python main.py policies results/<run-name>")
    from pipeline import policies
    policies.main(args[0])


COMMANDS = {
    "check": cmd_check,
    "demo": cmd_demo,
    "scan": cmd_scan,
    "serve": cmd_serve,
    "generate-data": _module_main("training.generate_data"),
    "baselines": _module_main("training.baselines"),
    "train": _module_main("training.train"),
    "evaluate": _module_main("training.evaluate"),
    "calibrate": _module_main("pipeline.calibrate_threshold"),
    "experiment": _module_main("pipeline.run_experiment"),
    "analyze": cmd_analyze,
    "policies": cmd_policies,
    "selftest": _module_main("pipeline.selftest", takes_argv=False),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    command, args = sys.argv[1], sys.argv[2:]
    if command not in COMMANDS:
        print(f"Unknown command '{command}'.\n{__doc__}")
        return 2
    return COMMANDS[command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
