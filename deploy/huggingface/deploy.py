"""Builds the Hugging Face Space folder and (optionally) uploads it.

  python deploy/huggingface/deploy.py                       # only build deploy/huggingface/space/
  python deploy/huggingface/deploy.py USERNAME/SPACE_NAME   # build, then upload to that Space

Before uploading: create the Space on huggingface.co (SDK: Gradio, template: Blank, hardware: CPU basic
or ZeroGPU)
and log in once on this computer with `python -c "from huggingface_hub import login; login()"`
(paste a token with "write" access from huggingface.co/settings/tokens).
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "space")

# Only what the running website needs (no training data, results or docs).
INCLUDE = [
    "bug_predictor", "pipeline", "server", "config.py", "main.py",
    "models/bug_predictor", "web/dist", "data/benchmarks",
]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".cache")


def build():
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(OUT):  # empty it (not delete it: a terminal may have it open as its folder)
        path = os.path.join(OUT, name)
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    for item in INCLUDE:
        src, dst = os.path.join(ROOT, item), os.path.join(OUT, item)
        if not os.path.exists(src):
            raise SystemExit(f"missing: {item}")
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=SKIP)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    for name in ("app.py", "requirements.txt", "README.md"):  # Space entry point, CPU libraries, Space page
        shutil.copy2(os.path.join(HERE, name), os.path.join(OUT, name))
    weights = os.path.join(OUT, "models", "bug_predictor", "model.safetensors")
    if not os.path.exists(weights):
        raise SystemExit("models/bug_predictor/model.safetensors is missing: the Space needs the trained model.")
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs)
    print(f"Built {OUT} ({size / 1e6:.0f} MB)")


def upload(space_id):
    from huggingface_hub import HfApi
    api = HfApi()
    print(f"Uploading to https://huggingface.co/spaces/{space_id} (the 500 MB model takes a few minutes)...")
    api.upload_folder(folder_path=OUT, repo_id=space_id, repo_type="space",
                      commit_message="Deploy BugAware Agents")
    print(f"Done. The Space now installs and starts (about 10 minutes): https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    build()
    if len(sys.argv) > 1:
        upload(sys.argv[1])
