"""Entry point of the Hugging Face Space (Gradio SDK).

The Space only needs a server on port 7860, so this starts the project's own FastAPI server, which
serves the full website (web/dist) and the API. Same app as `python main.py serve` on a laptop.
"""
import os

try:  # CPUs actually available to this container
    cpus = len(os.sched_getaffinity(0))
except AttributeError:
    cpus = os.cpu_count() or 2
os.environ.setdefault("PIPELINE_THREADS", str(min(max(cpus, 2), 8)))
os.environ.setdefault("OMP_NUM_THREADS", os.environ["PIPELINE_THREADS"])
os.environ.setdefault("PIPELINE_LLM", "gguf")   # Qwen2.5-Coder-1.5B, 4-bit, llama.cpp (downloaded on first start)
os.environ.setdefault("BUG_DEVICE", "cpu")

# ZeroGPU hardware (the free option for Gradio Spaces) stops any app that declares no GPU function.
# This app runs entirely on CPU, so it declares one placeholder that is never called, and reports
# start-up itself (normally done by Gradio's launch(), which this app does not use).
# `spaces` must be imported before torch. On other hardware or a laptop this block does nothing.
try:
    import spaces
    from spaces.config import Config as _ZeroConfig
except ImportError:
    spaces = None

if spaces is not None and _ZeroConfig.zero_gpu:
    @spaces.GPU(duration=10)
    def _zerogpu_placeholder():
        return None

    spaces.zero.startup()

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
