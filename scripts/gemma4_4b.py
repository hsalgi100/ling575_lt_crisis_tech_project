#!/usr/bin/env python3
"""
SETUP (run once):
  python3 -m venv gemma-env
  source gemma-env/bin/activate
  CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
  pip install huggingface_hub

DOWNLOAD MODEL (run once):
  huggingface-cli login
  huggingface-cli download bartowski/google_gemma-4-E4B-it-GGUF \
    --include "*Q4_K_M*" \
    --local-dir ./models

RUN:
  python gemma4_4b.py
"""

import os
import sys
import glob
from llama_cpp import Llama

# ── Configuration ─────────────────────────────────────────────────────────────

# Folder to look for .gguf files (relative to where you run this script)
MODELS_DIR = "models"

MODEL_PATH = None  # Leave None to auto-detect from MODELS_DIR

# Number of layers to offload to Metal GPU
# -1 = offload everything (recommended for Apple Silicon)
N_GPU_LAYERS = -1

# Context window size (max 128k for Gemma 4 E4B, lower = less RAM)
N_CTX = 8192

# Max tokens to generate per response
MAX_NEW_TOKENS = 512

# Sampling parameters (recommended by Google for Gemma 4)
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64

SYSTEM_PROMPT = "You are a helpful, friendly assistant."

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
        print("  huggingface-cli download bartowski/google_gemma-4-E4B-it-GGUF \\")
        print('    --include "*Q4_K_M*" --local-dir ./models')
        sys.exit(1)

    gguf_files = glob.glob(os.path.join(MODELS_DIR, "**/*.gguf"), recursive=True)
    gguf_files += glob.glob(os.path.join(MODELS_DIR, "*.gguf"))
    gguf_files = list(set(gguf_files))  # deduplicate

    if not gguf_files:
        print(f"Error: No .gguf files found in '{MODELS_DIR}'.")
        print("\nDownload a model with:")
        print("  huggingface-cli download bartowski/google_gemma-4-E4B-it-GGUF \\")
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
    print(f"Context window: {N_CTX} tokens\n")

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
    Build a Gemma 4 chat prompt from conversation history.
    Gemma 4 uses the standard ChatML-style format with <start_of_turn> tokens.
    """
    prompt = ""

    for message in conversation_history:
        role = message["role"]
        content = message["content"]

        if role == "system":
            prompt += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            prompt += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            prompt += f"<start_of_turn>model\n{content}<end_of_turn>\n"

    # Add generation prompt
    prompt += "<start_of_turn>model\n"
    return prompt


def generate_response(llm: Llama, conversation_history: list) -> str:
    prompt = build_prompt(conversation_history)

    output = llm(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        stop=["<end_of_turn>", "<start_of_turn>"],
        echo=False,
    )

    response = output["choices"][0]["text"].strip()
    return response


# Main Chat Loop

def main():
    print("=" * 55)
    print("  Gemma 4 E4B -- llama.cpp Chat (Apple Silicon)")
    print("=" * 55)

    model_path = find_gguf_model()

    try:
        llm = load_model(model_path)
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("\nMake sure llama-cpp-python was installed with Metal support:")
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

        print("\nGemma: ", end="", flush=True)
        try:
            response = generate_response(llm, conversation_history)
            print(response)
            print()
            conversation_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"[Error generating response: {e}]")
            conversation_history.pop()


if __name__ == "__main__":
    main()