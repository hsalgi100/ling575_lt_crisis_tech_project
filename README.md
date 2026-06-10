# ling575_lt_crisis_tech_project
AI benchmarking for Alerts Dataset

# Set up a venv
```python3 -m venv ai_venv```

# Run this setup for llama-cpp and huggingface
```CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir```
```pip install huggingface_hub```

- The "-DGGML_METAL=on" is a mac setting
- Drop this flag entirely if you are doing CPU-only
- NVIDIA needs CMAKE_ARGS="-DGGML_CUDA=on"
- AMD needs CMAKE_ARGS="-DGGML_HIP=on"
- If you are not sure what you have, try CMAKE_ARGS="-DGGML_VULKAN=1"

# Download Qwen3.5_9B
```hf auth login  (only if you aren't already logged in)```
```hf download bartowski/Qwen_Qwen3.5-9B-GGUF --include "*Q4_K_M*" --local-dir ./models```

# Download Gemma4_4B
```hf auth login```
```hf download bartowski/google_gemma-4-E4B-it-GGUF --include "*Q4_K_M*" --local-dir ./models```

# COMET Eval for Open Sourced Models
Run the following bash script:
```bash open_source_comet_evaluation.sh```
- In the bash script, the /models directory is inside of the /scripts directory because that is how we have it locally, however you will need to change the filepath in the script to point to where the models folder is if you try to run this yourself
