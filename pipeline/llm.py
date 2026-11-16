"""LLM backends shared by every agent.

  hf      : local Hugging Face model (default Qwen/Qwen2.5-Coder-1.5B-Instruct,
            the same generator used to build the bug predictor's training data).
  ollama  : a local Ollama server, native raw-prompt API (fastest option on a laptop CPU;
            `ollama pull qwen2.5-coder:1.5b`).
  openai  : any OpenAI-compatible endpoint (vLLM, LM Studio, Groq, OpenAI...).
            Configure with --base-url and the OPENAI_API_KEY env var.
  gguf    : llama.cpp (needs the optional llama-cpp-python package).
  auto    : gguf if installed, else Ollama if it is running, else hf.
            (llama.cpp and Ollama run at the same speed on CPU; only llama.cpp supports
             prompt-lookup decoding, see GGUFLLM._chat_lookup.)

Decoding is greedy by default so both pipeline variants see the same behaviour
for the same prompt; all randomness in the comparison comes from the bug agent.

Speed (measured on an i5-1235U laptop CPU, Qwen2.5-Coder-1.5B):
  fp32, 10 threads  1.96 tok/s     <- previous default
  fp32,  4 threads  2.22 tok/s
  int8,  4 threads  4.54 tok/s     <- default on CPU now (dynamic int8 quantisation of Linear layers)
On top of that, generation stops as soon as the answer's code block is closed
(`stop_at_code_end`), instead of running on into explanations up to max_new_tokens.
"""
import os
# Skip TensorFlow/Flax auto-import in transformers: cuts model load time ~2.5x (PyTorch only).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
import re
import json
import time
import hashlib
import threading

from config import GENERATOR_MODEL, LLM_CACHE_FILE as CACHE_PATH, MODELS_DIR, OLLAMA_MODEL, OLLAMA_URL


class Cancelled(Exception):
    """Raised when a run is cancelled (e.g. the web client pressed Stop)."""


class LLM:
    def __init__(self, cache=True):
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.seconds = 0.0
        self.cache_hits = 0
        self.on_token = None                  # callable(str) receiving streamed text deltas
        self.delta_tokens = 1                 # how many tokens the delta passed to on_token holds
        self.cancel_event = threading.Event()
        self._cache = {}
        self._cache_enabled = cache
        self._cache_lock = threading.Lock()
        if cache and os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        self._cache[row["key"]] = row["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    def stats(self):
        return {"llm_calls": self.calls, "prompt_tokens": self.prompt_tokens, "cache_hits": self.cache_hits,
                "completion_tokens": self.completion_tokens, "llm_seconds": round(self.seconds, 2)}

    def _key(self, system, user, max_new_tokens, stop_at_code_end, prefix=""):
        raw = json.dumps([getattr(self, "name", ""), getattr(self, "temperature", 0.0), system, user,
                          max_new_tokens, stop_at_code_end] + ([prefix] if prefix else [])
                         + ([self.seed] if getattr(self, "seed", None) is not None else []))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def chat(self, system, user, max_new_tokens=512, stop_at_code_end=False, prefix="", temperature=None, seed=None):
        """Returns the completion text. Identical greedy calls are answered from a persistent cache.
        `prefix` pre-fills the start of the assistant's answer (e.g. a code fence followed by "assert ")
        so the model can only continue in that format; the returned text includes it.
        `temperature` / `seed` override the defaults for this call (used to sample alternative candidate
        solutions); seeded samples are reproducible, so they are cached too."""
        if temperature is None and seed is None:
            return self._chat_cached(system, user, max_new_tokens, stop_at_code_end, prefix)
        saved = (getattr(self, "temperature", 0.0), getattr(self, "seed", None))
        self.temperature = saved[0] if temperature is None else temperature
        self.seed = seed
        try:
            return self._chat_cached(system, user, max_new_tokens, stop_at_code_end, prefix)
        finally:
            self.temperature, self.seed = saved

    def _chat_cached(self, system, user, max_new_tokens, stop_at_code_end, prefix):
        if self.cancel_event.is_set():
            raise Cancelled()
        cacheable = self._cache_enabled and (getattr(self, "temperature", 0.0) == 0 or getattr(self, "seed", None) is not None)
        key = self._key(system, user, max_new_tokens, stop_at_code_end, prefix) if cacheable else None
        if key and key in self._cache:
            self.calls += 1
            self.cache_hits += 1
            text = self._cache[key]
            if self.on_token:
                self.on_token(text)
            return text
        t0 = time.time()
        if prefix and self.on_token:
            self.on_token(prefix)
        text, p_tok, c_tok = self._chat(system, user, max_new_tokens, stop_at_code_end, prefix)
        text = prefix + text
        if self.cancel_event.is_set():
            raise Cancelled()
        self.calls += 1
        self.prompt_tokens += p_tok
        self.completion_tokens += c_tok
        self.seconds += time.time() - t0
        self.last_timed_out = getattr(self, "_timed_out", False)
        # Never cache an empty or time-capped answer: a busy CPU must not poison later runs.
        if key and text[len(prefix):].strip() and not self.last_timed_out:
            with self._cache_lock:
                self._cache[key] = text
                os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
                with open(CACHE_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": text}) + "\n")
        return text

    def _chat(self, system, user, max_new_tokens, stop_at_code_end, prefix=""):
        raise NotImplementedError


_CLOSED_BLOCK = re.compile(r"```[^\n`]*\n.*?\n\s*```", re.S)


def code_block_closed(text):
    """True once the text contains a complete fenced code block."""
    return bool(_CLOSED_BLOCK.search(text))


class HFLLM(LLM):
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", device=None, temperature=0.0,
                 quantize=None, threads=None, max_time=None, cache=True):
        super().__init__(cache=cache)
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            # Hybrid laptop CPUs (P + E cores) decode faster with few threads than with all of them.
            torch.set_num_threads(int(threads or os.environ.get("PIPELINE_THREADS", 4)))
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        # Use the local HF cache first (no network round-trips / retries at startup); download only if missing.
        try:
            self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, local_files_only=True)
        except OSError:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = model.to(self.device).eval()
        quantize = quantize if quantize is not None else os.environ.get("PIPELINE_QUANTIZE", "int8" if self.device == "cpu" else "none")
        if quantize == "int8" and self.device == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.quantize = quantize
        self.temperature = temperature
        self.max_time = float(max_time or os.environ.get("PIPELINE_MAX_SECONDS_PER_CALL", 420))
        self.name = model_name

    def _chat(self, system, user, max_new_tokens, stop_at_code_end, prefix=""):
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + prefix
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        prompt_len = ids.shape[1]
        llm = self

        class Stream(BaseStreamer):
            """Decodes new tokens as they arrive; forwards text deltas to `on_token`."""
            def __init__(self):
                self.tokens, self.text, self.skip_prompt = [], "", True

            def put(self, value):
                if self.skip_prompt:          # first call carries the prompt
                    self.skip_prompt = False
                    return
                self.tokens.extend(value.reshape(-1).tolist())
                text = llm.tok.decode(self.tokens, skip_special_tokens=True)
                if llm.on_token and len(text) > len(self.text) and not text.endswith("�"):
                    llm.on_token(text[len(self.text):])
                    self.text = text

            def end(self):
                pass

        stream = Stream()

        class Stop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if llm.cancel_event.is_set():
                    return True
                return stop_at_code_end and code_block_closed(prefix + (stream.text or llm.tok.decode(input_ids[0, prompt_len:], skip_special_tokens=True)))

        gen_kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.eos_token_id, "max_time": self.max_time,
                      "streamer": stream, "stopping_criteria": StoppingCriteriaList([Stop()])}
        if self.temperature > 0:
            if getattr(self, "seed", None) is not None:
                self.torch.manual_seed(self.seed)
            gen_kwargs.update(do_sample=True, temperature=self.temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        t_start = time.time()
        with self.torch.no_grad():
            out = self.model.generate(ids, attention_mask=self.torch.ones_like(ids), **gen_kwargs)
        self._timed_out = time.time() - t_start >= self.max_time - 0.5
        new = out[0, prompt_len:]
        if self.device == "cuda":
            self.torch.cuda.empty_cache()
        return self.tok.decode(new, skip_special_tokens=True), prompt_len, int(new.shape[0])


class OpenAILLM(LLM):
    def __init__(self, model_name, base_url=None, temperature=0.0, cache=True):
        super().__init__(cache=cache)
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
        self.name = model_name
        self.temperature = temperature

    def _chat(self, system, user, max_new_tokens, stop_at_code_end, prefix=""):
        # Chat APIs cannot reliably pre-fill the answer; ask for the format in the prompt instead.
        if prefix:
            user += f"\n\nStart your answer exactly with:\n{prefix}"
        stream = self.client.chat.completions.create(
            model=self.name, temperature=self.temperature, max_tokens=max_new_tokens,
            seed=getattr(self, "seed", None) or 0, stream=True,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text, n = "", 0
        for chunk in stream:
            if self.cancel_event.is_set():
                stream.close()
                break
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            text += delta
            n += 1
            if self.on_token:
                self.on_token(delta)
            if stop_at_code_end and code_block_closed(prefix + text):
                stream.close()
                break
        if prefix and text.startswith(prefix):
            text = text[len(prefix):]  # the model echoed the requested prefix
        return text, 0, n


GGUF_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"


def gguf_available():
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False


class GGUFLLM(LLM):
    """Same Qwen2.5-Coder-1.5B-Instruct model, run with llama.cpp (4-bit GGUF, official Qwen build).
    On a laptop CPU this is ~5-8x faster than PyTorch: ~20-30 tok/s instead of ~4 tok/s,
    and prompt processing is much faster too, so a full pipeline run drops from minutes to seconds."""

    def __init__(self, model_path=None, temperature=0.0, threads=None, n_ctx=4096, cache=True):
        super().__init__(cache=cache)
        from llama_cpp import Llama
        if model_path is None:
            local = os.path.join(MODELS_DIR, GGUF_FILE)
            if os.path.exists(local):
                model_path = local
            else:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(GGUF_REPO, GGUF_FILE)
        n_threads = int(threads or os.environ.get("PIPELINE_THREADS", max(4, (os.cpu_count() or 8) // 2)))
        self.model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads, n_batch=512, verbose=False)
        self.temperature = temperature
        self.quantize = "gguf-q4_k_m"
        # Prompt-lookup decoding (PIPELINE_LOOKUP = max draft length, 0 = off). Off by default so experiment
        # runs reproduce plain greedy decoding exactly; the web server turns it on for speed. Batched
        # verification can flip rare near-ties, so answers can differ slightly: it gets its own cache key.
        self.lookup = int(os.environ.get("PIPELINE_LOOKUP", 0))
        self.name = "Qwen/Qwen2.5-Coder-1.5B-Instruct (GGUF q4_k_m, llama.cpp)" + (" +lookup" if self.lookup > 0 else "")
        self._stop_ids = {t for t in (self.model.token_eos(),
                                      *self.model.tokenize(b"<|im_end|>", add_bos=False, special=True),
                                      *self.model.tokenize(b"<|endoftext|>", add_bos=False, special=True))}

    # ------------------------------------------------------------------ prompt-lookup decoding
    @staticmethod
    def _draft(history, k, max_ngram=3, min_ngram=2):
        """Proposes the k tokens that followed the most recent earlier occurrence of the last n tokens
        (n = 3, then 2). Agents mostly copy from their prompt (the Fixer rewrites the code it was given,
        the Tester repeats the function name and arguments), so these guesses are often right."""
        n_hist = len(history)
        for n in range(min(max_ngram, n_hist - 1), min_ngram - 1, -1):
            tail = history[-n:]
            last = tail[-1]
            for start in range(n_hist - n - 1, -1, -1):
                if history[start + n - 1] == last and history[start:start + n] == tail:
                    return history[start + n:start + n + k]
        return []

    def _decode_greedy(self, tokens):
        """Evaluates `tokens` after the current KV cache and returns the greedy next-token prediction
        after each of them (logits are requested only for these few tokens, not the whole prompt)."""
        import numpy as np
        m = self.model
        m._batch.set_batch(batch=tokens, n_past=m.n_tokens, logits_all=True)
        m._ctx.decode(m._batch)
        logits = np.ctypeslib.as_array(m._ctx.get_logits(), shape=(len(tokens), m._n_vocab))
        m.input_ids[m.n_tokens:m.n_tokens + len(tokens)] = tokens
        m.n_tokens += len(tokens)
        return logits.argmax(axis=1).tolist()

    def _rewind(self, n_tokens):
        self.model.n_tokens = n_tokens
        self.model._ctx.kv_cache_seq_rm(-1, n_tokens, -1)

    def _chat_lookup(self, prompt, max_new_tokens, stop_at_code_end, prefix):
        """Greedy decoding with prompt-lookup speculation: each step checks the current token plus a few
        drafted tokens in ONE forward pass and keeps every draft token the model agrees with.
        Measured on a laptop CPU: Fixer calls ~1.7x faster (it re-writes code that is in its prompt),
        other agents about the same. Output matches plain greedy decoding except for rare near-ties."""
        m = self.model
        toks = m.tokenize(prompt.encode("utf-8"), special=True)
        # reuse the KV cache of the longest common prefix with the previous prompt
        common = 0
        limit = min(m.n_tokens, len(toks) - 1)
        prev = m.input_ids[:limit].tolist()
        while common < limit and prev[common] == toks[common]:
            common += 1
        self._rewind(common)
        if len(toks) - 1 > common:
            m.eval(toks[common:-1])
        history, out, pending = list(toks), [], toks[-1]
        text, done, streamed = "", False, 0
        k = min(4, self.lookup)  # adaptive draft length: grows while guesses are accepted, shrinks when not
        while not done and len(out) < max_new_tokens and not self.cancel_event.is_set():
            room = min(max_new_tokens - len(out) - 1, m.n_ctx() - m.n_tokens - 1)
            draft = self._draft(history + out, k)[:max(0, room)]
            base = m.n_tokens
            preds = self._decode_greedy([pending] + draft)
            accepted = [preds[0]]
            for i, d in enumerate(draft):
                if d != preds[i]:
                    break
                accepted.append(preds[i + 1])
            if draft:
                k = min(self.lookup * 2, k * 2) if len(accepted) - 1 == len(draft) else max(2, k // 2)
            # KV now holds pending + all drafts; keep pending + the drafts that were accepted
            self._rewind(base + len(accepted))
            for t in accepted:
                if t in self._stop_ids or len(out) >= max_new_tokens:
                    done = True
                    break
                out.append(t)
            # The KV cache now holds everything except the newest accepted token, which is fed next step.
            pending = out[-1] if out else pending
            new_text = m.detokenize(out).decode("utf-8", errors="ignore")
            if len(new_text) > len(text):
                if self.on_token:
                    self.delta_tokens = max(1, len(out) - streamed)
                    self.on_token(new_text[len(text):])
                    self.delta_tokens = 1
                text, streamed = new_text, len(out)
            if stop_at_code_end and code_block_closed(prefix + text):
                break
        self._timed_out = False
        return m.detokenize(out).decode("utf-8", errors="replace"), len(toks), len(out)

    def _chat(self, system, user, max_new_tokens, stop_at_code_end, prefix=""):
        prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n{prefix}")
        if self.lookup > 0 and self.temperature == 0:
            return self._chat_lookup(prompt, max_new_tokens, stop_at_code_end, prefix)
        text, n = "", 0
        stream = self.model.create_completion(prompt, max_tokens=max_new_tokens, temperature=self.temperature,
                                              top_p=0.95 if self.temperature > 0 else 1.0, top_k=0 if self.temperature > 0 else 1,
                                              seed=getattr(self, "seed", None) or 0,
                                              stop=["<|im_end|>", "<|endoftext|>"], stream=True)
        for chunk in stream:
            if self.cancel_event.is_set():
                break
            delta = chunk["choices"][0]["text"]
            if not delta:
                continue
            text += delta
            n += 1
            if self.on_token:
                self.on_token(delta)
            if stop_at_code_end and code_block_closed(prefix + text):
                break
        self._timed_out = False
        p_tok = len(self.model.tokenize(prompt.encode("utf-8"), special=True))
        return text, p_tok, n


class OllamaLLM(LLM):
    """Same Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) served by a local Ollama, through its native
    /api/generate endpoint in raw mode. Raw mode sends exactly the same chat-template prompt as the
    llama.cpp backend, so answer pre-filling (`prefix`) works properly, unlike the OpenAI-compatible API.

    Speed on a laptop CPU is the same as llama-cpp-python (both are llama.cpp: ~95 tok/s prompt,
    ~25-28 tok/s generation). Ollama keeps the model loaded in its own process (`keep_alive`), so the
    web server starts faster and uses less memory; it cannot do prompt-lookup decoding."""

    def __init__(self, model_name=OLLAMA_MODEL, base_url=OLLAMA_URL, temperature=0.0, n_ctx=4096, cache=True):
        super().__init__(cache=cache)
        self.model_name = model_name
        self.host = (base_url or OLLAMA_URL).rstrip("/").removesuffix("/v1")
        self.temperature = temperature
        self.n_ctx = n_ctx
        self.keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "60m")
        self.quantize = "ollama-q4_k_m"
        self.name = f"ollama:{model_name}"

    def _post(self, path, body, timeout=600):
        import urllib.request
        req = urllib.request.Request(self.host + path, json.dumps(body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def warmup(self):
        """Loads the model into memory now so the first real request does not wait for it."""
        with self._post("/api/generate", {"model": self.model_name, "prompt": "", "keep_alive": self.keep_alive}) as r:
            r.read()

    def unload(self):
        """Frees the model's memory in Ollama now (e.g. when the website switches to another model)."""
        with self._post("/api/generate", {"model": self.model_name, "prompt": "", "keep_alive": 0}, timeout=30) as r:
            r.read()

    def _chat(self, system, user, max_new_tokens, stop_at_code_end, prefix=""):
        prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n{prefix}")
        options = {"temperature": self.temperature, "num_predict": max_new_tokens, "num_ctx": self.n_ctx,
                   "seed": getattr(self, "seed", None) or 0, "stop": ["<|im_end|>", "<|endoftext|>"]}
        if self.temperature > 0:
            options["top_p"] = 0.95
        else:
            options["top_k"] = 1
        body = {"model": self.model_name, "prompt": prompt, "raw": True, "stream": True,
                "keep_alive": self.keep_alive, "options": options}
        text, n, p_tok = "", 0, 0
        with self._post("/api/generate", body) as resp:
            for line in resp:  # one JSON object per line
                if self.cancel_event.is_set():
                    break      # closing the connection stops generation on the server
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise RuntimeError(f"Ollama: {chunk['error']}")
                delta = chunk.get("response", "")
                if delta:
                    text += delta
                    n += 1
                    if self.on_token:
                        self.on_token(delta)
                    if stop_at_code_end and code_block_closed(prefix + text):
                        break
                if chunk.get("done"):
                    p_tok = chunk.get("prompt_eval_count", 0)
                    n = chunk.get("eval_count", n)
                    break
        self._timed_out = False
        return text, p_tok, n


def ollama_available(base_url=OLLAMA_URL, model=OLLAMA_MODEL):
    """True if an Ollama server is reachable and has `model` pulled."""
    import urllib.request
    tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=2) as resp:
            names = {m.get("name", "") for m in json.load(resp).get("models", [])}
    except Exception:
        return False
    return model in names or f"{model}:latest" in names


def ollama_models(base_url=OLLAMA_URL):
    """Models pulled in the local Ollama: [{name, size_gb, params}], or [] if Ollama is not running."""
    import urllib.request
    tags_url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=2) as resp:
            models = json.load(resp).get("models", [])
    except Exception:
        return []
    return [{"name": m.get("name", ""), "size_gb": round(m.get("size", 0) / 1e9, 1),
             "params": (m.get("details") or {}).get("parameter_size", "")} for m in models if m.get("name")]


def resolve_backend(backend):
    """Turns "auto" into a concrete backend name."""
    if backend != "auto":
        return backend
    if gguf_available():
        return "gguf"
    return "ollama" if ollama_available() else "hf"


def make_llm(backend, model_name=None, base_url=None, temperature=0.0):
    """`PIPELINE_CACHE=0` turns off the persistent answer cache (e.g. for timing runs)."""
    backend = resolve_backend(backend)
    cache = os.environ.get("PIPELINE_CACHE", "1") != "0"
    if backend == "gguf":
        return GGUFLLM(model_name if model_name and model_name.endswith(".gguf") else None, temperature=temperature, cache=cache)
    if backend == "hf":
        return HFLLM(model_name or GENERATOR_MODEL, temperature=temperature, cache=cache)
    if backend == "ollama":
        return OllamaLLM(model_name or OLLAMA_MODEL, base_url=base_url or OLLAMA_URL, temperature=temperature, cache=cache)
    if backend == "openai":
        if not model_name:
            raise ValueError("--model is required for the openai backend (e.g. qwen2.5-coder:1.5b for Ollama)")
        return OpenAILLM(model_name, base_url=base_url, temperature=temperature, cache=cache)
    raise ValueError(f"Unknown LLM backend {backend!r}")


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    """Returns the longest ```python fenced block, or the raw text if there is none."""
    blocks = _FENCE.findall(text)
    if blocks:
        return max(blocks, key=len).strip("\n")
    # Unterminated fence (generation cut off)
    if "```" in text:
        return re.split(r"```(?:python|py)?\s*\n", text, maxsplit=1)[-1].replace("```", "").strip("\n")
    return text.strip("\n")
