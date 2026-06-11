# A simple file to scale the metrics to where they should be for consistency with comparisons
import json,os,re,sys

path = "../../TranslationToolEvaluation/data/processed/kc/"

for name in os.listdir(path):
    the_read_file = open(path+"/"+name,'r')
    with open("../data/kc_metadata/kc_"+name[0:-6]+"_Gold_metadata.jsonl",'w') as output:
        for i,line in enumerate(the_read_file.readlines()):
            print(f"Reading Line {i}")
            curr = json.loads(line)
            d = {}
            d["id"] = curr["id"]
            d["technology"] = "Gold"
            # MT_iso_codes = [re.search("(.+)_MT",s).group(1) for s in curr.keys() if "_MT" in s]
            for iso in curr["translations"].keys():
            # for iso in MT_iso_codes:
                d[iso] = len(curr["translations"][iso])
                # d[iso] = len(curr[iso+"_MT"])
            # curr["bleu"] = curr["bleu"] / 100

            output.write(json.dumps(d,ensure_ascii=False) + "\n")

