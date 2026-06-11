import os,re,sys,json
from eval_fns import *

main_args = sys.argv # 1st arg is gold data path, 2nd is corresponding MT file path, 3rd is name of output file
# main_args = [
#     "",
#     "../../TranslationToolEvaluation/data/processed/kc",
#     "../machine_translations/kc/kc_GoogleTranslate_MT.jsonl",
#     "../machine_translations/kc/kc_GoogleTranslate_MT_metrics/kc_GoogleTranslate_MT_bleu_chrf.jsonl"
# ]

# Step 1: load the gold data into a hashable form
gold = {}
for filename in os.listdir(main_args[1]):
    path = main_args[1] + "/" + filename
    if os.path.isdir(path):
        continue

    with open(path,'r') as file:
        for line in file.readlines():
            curr = json.loads(line) # current jsonl
            if curr["id"] not in gold:
                gold[curr["id"]] = {}
            for key in curr["translations"].keys():
                if key not in gold[curr["id"]]:
                    gold[curr["id"]][key] = curr["translations"][key]


# Step 2: do evals on MT
output = open(main_args[3],'w')

with open(main_args[2], 'r') as file:
    for i,line in enumerate(file.readlines()):
        curr = json.loads(line) # current machine translation jsonl
        # print(curr.keys())
        MT_iso_codes = [re.search("(.+)_MT",s).group(1) for s in curr.keys() if "_MT" in s]
        # MT_iso_codes = [iso for iso in curr["translations"].keys()]
        # print(MT_iso_codes)
        print(f"\n\n\nprocessing {curr['id']}")

        # iterate over languages
        for i,iso in enumerate(MT_iso_codes):
            # print(iso)
            source_text = gold[curr["id"]]["en"] # source is English
            human_translation = gold[curr["id"]][iso] # gold translation
            machine_translation = curr[iso+"_MT"] # machine translation
            # machine_translation = curr["translations"][iso] # machine translation

            d = {} # dictionary to write to
            d["id"] = curr["id"]
            d["scenario"] = curr["scenario"]
            d["technology"] = "Google Translate"
            d["source_language"] = "en"
            d["target_language"] = iso

            # Metrics to evaluate
            # d["bleu"] = bleu(machine_translation, human_translation)["bleu"]
            # d["chrf"] = chrf(machine_translation, human_translation)["chrf"]
            # d["chrf++"] = chrf(machine_translation, human_translation)["chrf++"]
            d["comet"] = comet(machine_translation,human_translation,source_text)["comet"]
            d["sacrebleu"] = sacrebleu(machine_translation,human_translation)["sacrebleu"]
            
            output.write(json.dumps(d,ensure_ascii=False) + "\n")

output.close()