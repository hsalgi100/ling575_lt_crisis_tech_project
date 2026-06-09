#!/usr/bin/env python3
"""
SETUP (run once):
  python3 -m venv qwen-env
  source qwen-env/bin/activate
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
  pip install huggingface_hub

  NOTE: Qwen3.5 is a recent architecture. If the model fails to load with an
  "unknown architecture" / "unsupported model" error, your llama-cpp-python is
  built against a too-old llama.cpp. Re-run the install line above to rebuild
  against the latest, or `pip install llama-cpp-python --pre` for a newer build.

DOWNLOAD MODEL (run once):
  hf auth login            # only if you aren't already logged in
  hf download bartowski/Qwen_Qwen3.5-9B-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./models

RUN:
  python scripts/qwen3_5_9b.py
"""

import os
import re
import sys
import glob
from llama_cpp import Llama

# Configuration

# Folder to look for .gguf files
MODELS_DIR = "models"

MODEL_PATH = None 

# Number of layers to offload to Metal GPU
# -1 = offload everything
N_GPU_LAYERS = -1

# Context window size.
N_CTX = 8192

# Max tokens to generate per response.
MAX_NEW_TOKENS = 2048

# Reasoning toggle.
#   True  -> model thinks first 
#   False -> thinking is suppressed; you get a direct answer.
ENABLE_THINKING = True

# When thinking is enabled, also print the reasoning trace?
SHOW_THINKING = False

# Sampling parameters.
# Qwen's recommended settings for thinking mode (general tasks):
#   temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5
# For non-thinking / instruct mode they suggest temperature=0.7, top_p=0.8.
TEMPERATURE = 1.0 if ENABLE_THINKING else 0.7
TOP_P = 0.95 if ENABLE_THINKING else 0.8
TOP_K = 20
MIN_P = 0.0
PRESENCE_PENALTY = 1.5
REPEAT_PENALTY = 1.0  # Qwen recommends repetition_penalty = 1.0

SYSTEM_PROMPT = "You are a helpful, friendly assistant."

# ANSI dim for displaying the thinking trace
DIM = "\033[2m"
RESET = "\033[0m"


# Model Detection

def find_gguf_model() -> str:
    """Auto-detect a GGUF file in the models directory."""
    if MODEL_PATH:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at '{MODEL_PATH}'")
            sys.exit(1)
        return MODEL_PATH

    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory '{MODELS_DIR}' not found.")
        print("\nCreate it and download a model with:")
        print("  huggingface-cli download bartowski/Qwen_Qwen3.5-9B-GGUF \\")
        print('    --include "*Q4_K_M*" --local-dir ./models')
        sys.exit(1)

    gguf_files = glob.glob(os.path.join(MODELS_DIR, "**/*.gguf"), recursive=True)
    gguf_files += glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
    gguf_files = list(set(gguf_files))  # deduplicate

    if not gguf_files:
        print(f"Error: No .gguf files found in '{MODELS_DIR}'.")
        print("\nDownload a model with:")
        print("  huggingface-cli download bartowski/Qwen_Qwen3.5-9B-GGUF \\")
        print('    --include "*Q4_K_M*" --local-dir ./models')
        sys.exit(1)

    if len(gguf_files) > 1:
        print("Multiple GGUF files found, using the first one:")
        for f in gguf_files:
            print(f"  {f}")
        print()

    return gguf_files[0]


# Model Loading

def load_model(model_path: str) -> Llama:
    print(f"Loading model: {os.path.basename(model_path)}")
    print(f"GPU layers: {'all (Metal)' if N_GPU_LAYERS == -1 else N_GPU_LAYERS}")
    print(f"Context window: {N_CTX} tokens")
    print(f"Thinking mode: {'ON' if ENABLE_THINKING else 'OFF'}\n")

    llm = Llama(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=False,
    )

    print("Model loaded successfully!\n")
    return llm


# Chat Logic

def build_prompt(conversation_history: list) -> str:
    """
    Build a Qwen3.5 chat prompt from conversation history.
    Qwen uses the ChatML format with <|im_start|> / <|im_end|> tokens.

    When ENABLE_THINKING is False we append an empty <think></think> block to
    the assistant turn, which is how Qwen's own chat template suppresses the
    reasoning step.
    """
    prompt = ""

    for message in conversation_history:
        role = message["role"]          # "system" | "user" | "assistant"
        content = message["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # Generation prompt
    prompt += "<|im_start|>assistant\n"
    if not ENABLE_THINKING:
        prompt += "<think>\n\n</think>\n\n"

    return prompt


def split_thinking(text: str):
    """
    Separate a reasoning trace from the final answer.
    Returns (thinking_or_None, answer).
    """
    # The opening <think> may or may not be present in the raw output depending
    # on how generation started; match on the closing tag.
    if "</think>" in text:
        thinking, _, answer = text.partition("</think>")
        thinking = re.sub(r"^\s*<think>", "", thinking).strip()
        return (thinking or None), answer.strip()
    return None, text.strip()


def generate_response(llm: Llama, conversation_history: list):
    """Returns (thinking_or_None, answer)."""
    prompt = build_prompt(conversation_history)

    output = llm(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
        presence_penalty=PRESENCE_PENALTY,
        repeat_penalty=REPEAT_PENALTY,
        stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        echo=False,
    )

    raw = output["choices"][0]["text"]
    return split_thinking(raw)


# Main Chat Loop

def main():
    print("=" * 55)
    print("  Qwen3.5 9B -- llama.cpp Chat (Apple Silicon)")
    print("=" * 55)

    model_path = find_gguf_model()

    try:
        llm = load_model(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nMake sure llama-cpp-python was installed with Metal support and")
        print("is recent enough to know the Qwen3.5 architecture:")
        print('  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir')
        sys.exit(1)

    print("Commands:  'clear' = reset history  |  'quit' or 'exit' = stop")
    print("=" * 55 + "\n")

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})
            print("-- Conversation cleared --\n")
            continue

        conversation_history.append({"role": "user", "content": user_input})

        print("\nQwen: ", end="", flush=True)
        try:
            thinking, answer = generate_response(llm, conversation_history)

            if SHOW_THINKING and thinking:
                print(f"{DIM}[thinking] {thinking}{RESET}\n")

            print(answer)
            print()

            # Keep only the final answer in history (Qwen recommends not
            # feeding prior reasoning traces back into multi-turn context).
            conversation_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"[Error generating response: {e}]")
            conversation_history.pop()


if __name__ == "__main__":
    main()