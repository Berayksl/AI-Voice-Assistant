import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from llama_cpp import Llama

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "./llama model/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
N_GPU_LAYERS = -1       # -1 = offload ALL layers to GPU (max speed on 8GB VRAM)
CONTEXT_LEN  = 4096     # context window size
N_THREADS    = 8        # CPU threads
VERBOSE      = False    # set True to see llama.cpp internals
# ─────────────────────────────────────────────────────────────────────────────

# Llama 3 chat template tokens
B_INST  = "<|start_header_id|>user<|end_header_id|>\n\n"
E_INST  = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
B_SYS   = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
E_SYS   = "<|eot_id|>"


def load_model(model_path: str = MODEL_PATH):
    print(f"[LLM] Loading model: {model_path}")
    print(f"[LLM] GPU layers: {'ALL' if N_GPU_LAYERS == -1 else N_GPU_LAYERS}")
    t0 = time.time()
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=CONTEXT_LEN,
        n_threads=N_THREADS,
        verbose=VERBOSE,
    )
    print(f"[LLM] Model loaded in {time.time() - t0:.2f}s\n")
    return llm


def chat(llm: Llama, user_message: str, system_prompt: str = None, max_tokens: int = 512):
    """
    Send a message and get a response. Returns (response_text, stats_dict).
    Uses Llama 3 instruct chat format.
    """
    # Build prompt using Llama 3 chat template
    if system_prompt:
        prompt = f"{B_SYS}{system_prompt}{E_SYS}{B_INST}{user_message}{E_INST}"
    else:
        prompt = f"{B_INST}{user_message}{E_INST}"

    print(f"[LLM] Prompt: {user_message}")
    t0 = time.time()

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        echo=False,
    )

    elapsed       = time.time() - t0
    response_text = output["choices"][0]["text"].strip()
    n_tokens      = output["usage"]["completion_tokens"]
    tokens_per_s  = n_tokens / elapsed if elapsed > 0 else 0

    print(f"[LLM] Response: {response_text}")
    print(f"[LLM] ── Stats ─────────────────────────────")
    print(f"  Tokens generated : {n_tokens}")
    print(f"  Inference time   : {elapsed:.2f}s")
    print(f"  Speed            : {tokens_per_s:.1f} tokens/sec")
    print(f"────────────────────────────────────────────\n")

    return response_text, {
        "tokens": n_tokens,
        "inference_time": elapsed,
        "tokens_per_sec": tokens_per_s,
    }


def benchmark(llm: Llama):
    """Run a few representative test prompts and report speeds."""
    test_cases = [
        {
            "label": "General Q&A",
            "system": "You are a helpful assistant. Be concise.",
            "user": "What is the difference between RAM and VRAM?",
        },
        {
            "label": "Translation (EN→FR)",
            "system": "You are a translator. Translate the user's text to French. Output only the translation.",
            "user": "The weather today is sunny and warm. I enjoy going for walks in the park.",
        },
        {
            "label": "Summarization",
            "system": "Summarize the following text in one sentence.",
            "user": (
                "Deep learning is a subset of machine learning that uses neural networks "
                "with many layers to learn representations of data with multiple levels of abstraction. "
                "It has achieved state-of-the-art results in image recognition, natural language processing, "
                "and speech recognition."
            ),
        },
    ]

    print("=" * 50)
    print("[LLM] Running benchmark...")
    print("=" * 50)
    all_stats = []
    for tc in test_cases:
        print(f"\n── {tc['label']} ──")
        _, stats = chat(llm, tc["user"], system_prompt=tc["system"])
        all_stats.append({"label": tc["label"], **stats})

    print("\n[LLM] ── Benchmark Summary ─────────────────")
    print(f"  {'Task':<25} {'Tokens':>7} {'Time':>8} {'Tok/s':>8}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8}")
    for s in all_stats:
        print(f"  {s['label']:<25} {s['tokens']:>7} {s['inference_time']:>7.2f}s {s['tokens_per_sec']:>7.1f}")
    print("─────────────────────────────────────────────\n")
    return all_stats


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[LLM] Model not found at {MODEL_PATH}")
        print("  → Download with:")
        print("    python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download(repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF', "
              "filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf', local_dir='./models')\"")
    else:
        llm = load_model()

        #quick single test
        chat(llm, "Explain what a transformer neural network is in 3 sentences.",
             system_prompt="You are a helpful AI assistant. Be concise.")

        #full benchmark
        benchmark(llm)