python scripts/formatting_for_bleu.py \
        --bleu-summary machine_translations/kc/kc_Gemma4_Qwen3.5_MT_bleu_chrf.json \
        --gemma-translations machine_translations/kc/kc_Gemma4_MT.jsonl \
        --qwen-translations  machine_translations/kc/kc_Qwen3.5_MT.jsonl \
        --gemma-out gemma_results.jsonl --qwen-out qwen_results.jsonl \
        --source-lang en