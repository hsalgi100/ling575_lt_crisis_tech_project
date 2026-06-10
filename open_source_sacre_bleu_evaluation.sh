python3 scripts/open_source_sacrebleu_eval.py --input translation_results.json --output sacrebleu_results.json --tokenizer 13a
#   python3 scripts/open_source_sacrebleu_eval --tokenizer 13a            # force a tokenizer for all langs
#   python3 scripts/open_source_sacrebleu_eval--data /data/kc            # also read gold from source jsonl