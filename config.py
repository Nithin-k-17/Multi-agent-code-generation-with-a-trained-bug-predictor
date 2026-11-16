"""Central configuration: every path and default setting used by the project lives here.

All paths are built from the project root, so the project runs from any folder,
on Windows, macOS and Linux, without editing code.
Settings marked "env" can be overridden with environment variables.
"""
import os

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmarks")      # raw MBPP / HumanEval files
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")       # labelled train/val/test .jsonl

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BUG_MODEL_DIR = os.path.join(MODELS_DIR, "bug_predictor")  # trained CodeBERT checkpoint
PIPELINE_THRESHOLD_FILE = os.path.join(BUG_MODEL_DIR, "pipeline_threshold.json")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")       # multi-agent experiment outputs
TRAINING_RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")    # checkpoints written by training/train.py
CACHE_DIR = os.path.join(PROJECT_ROOT, ".cache")
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_cache.jsonl")
WEB_DIST_DIR = os.path.join(PROJECT_ROOT, "web", "dist")  # built website served by the API

# --------------------------------------------------------------------------- bug predictor
BASE_ENCODER = "microsoft/codebert-base"
MAX_LEN = 512
DEFAULT_THRESHOLD = 0.34          # F1-optimal on validation (used when threshold.json is missing)

# --------------------------------------------------------------------------- code-generator LLM
GENERATOR_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"     # Hugging Face id (backend "hf")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")                 # env
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1")             # env

# --------------------------------------------------------------------------- dataset splits (MBPP task ids)
SPLIT_RANGES = {
    "train": (11, 510),
    "val": (511, 600),
    "test": (601, 974),
}
