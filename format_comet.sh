Usage:
    python scripts/formatting_for_comet.py \
        --comet-summary machine_translations/kc/kc_Gemma_Qwen_MT_metrics/kc_Gemma4_Qwen3.5_MT_comet.json \
        --gemma-translations machine_translations/kc/kc_Gemma4_MT.jsonl \
        --qwen-translations  machine_translations/kc/kc_Qwen3.5_MT.jsonl \
        --gemma-out kc_Gemma4_MT_comet.jsonl \
        --qwen-out  kc_Qwen3.5_MT_comet.jsonl \
        --source-lang en