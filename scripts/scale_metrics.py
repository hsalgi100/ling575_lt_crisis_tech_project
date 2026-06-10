# A simple file to scale the metrics to where they should be for consistency with comparisons

import json,os,re,sys

the_read_file = open("../machine_translations/kc/kc_Gemma_Qwen_MT_metrics/kc_Qwen3.5_MT_bleu_chrf.jsonl",'r')
with open("../machine_translations/kc/kc_Qwen3.5_MT_bleu_chrf_SCALED.jsonl",'w') as output:
    for i,line in enumerate(the_read_file.readlines()):
        print(f"Reading Line {i}")
        curr = json.loads(line)
        curr["bleu"] = curr["bleu"] / 100
        output.write(json.dumps(curr,ensure_ascii=False) + "\n")

