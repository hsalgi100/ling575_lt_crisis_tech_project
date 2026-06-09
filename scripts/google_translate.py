import time
import sys, os, json
from google.cloud import translate # translate_v2 as translate
from eval_fns import *

# Retrieve Service Account Credentials to call the API
service_account_credentials_path = "/Users/anantha/.apikeys/wea-translation-497802-3e63f30e8ae5.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_credentials_path
PROJECT_ID = "wea-translation-497802" #environ.get("PROJECT_ID", "")
PARENT = f"projects/{PROJECT_ID}/locations/global"

# Define client to be used in subsequent functions
client = translate.TranslationServiceClient()


def print_supported_languages(display_language_code: str):

    # client = translate.TranslationServiceClient()
    response = client.get_supported_languages(
        parent=PARENT,
        display_language_code=display_language_code,
    )

    languages = response.languages
    print(f" Languages: {len(languages)} ".center(60, "-"))
    for language in languages:
        language_code = language.language_code
        display_name = language.display_name
        print(f"{language_code:10}{display_name}")

def translate_text(text: str, target_language_code: str) -> translate.Translation:
    # client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        source_language_code="en",
        target_language_code=target_language_code,
    )

    return response.translations[0]
    

def translate_with_backoff(text:str, target_language_code:str, max_retries:int = 3, base_delay:int = 1, max_delay:int = 1):
    for attempt in range(max_retries):
        try:
            return translate_text(text,target_language_code).translated_text # Return the successful response
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts. Error: {e}")
                raise
                
            # Calculate exponential backoff: base_delay * 2^attempt
            sleep_time = min(base_delay * (2 ** attempt), max_delay)
            print(f"Attempt {attempt + 1} failed. Retrying in {sleep_time}s... (Error: {e})")
            time.sleep(sleep_time)

# Main Inputs:
main_args = ["","../../TranslationToolEvaluation/data/processed/kc","../machine_translations/kc/kc_MT.jsonl"]
# main_args = sys.argv
"""Add check to ensure the second argument is a .jsonl so that the extension gets correctly stripped off for the LOGS.txt file"""

data = []
for filename in os.listdir(main_args[1]):
    path = main_args[1] + "/" + filename
    if os.path.isdir(path):
        continue

    with open(path,'r') as file:
        for line in file.readlines():
            data.append(
                json.loads(line)
            )


start_time = time.time()
output = open(main_args[2],'w',encoding="utf-8")
logs = open(main_args[2][0:-6] +"_LOGS.txt", 'w',encoding='utf-8')
for i,d in enumerate(data):
    print(f"processing: {d["id"]}")
    source_text = d["source"]["eng"]
    line = {"id": d["id"], "scenario": d["scenario"], "english_source_text": source_text }
    
    for key in d["translations"].keys():
        if key == "en":
            continue
        iso = key
        if key == "zh-Hans": # Do conversion mappings
            iso = "zh-CN"
        elif key == "zh-Hant":
            iso = "zh-TW"
        else:
            iso = key

        human_translation = d["translations"][key]
        machine_translation = translate_with_backoff(source_text, target_language_code = iso) #translate_text(source_text,target_language_code=key).translated_text
        
        line[key+"_MT"] = machine_translation
        logs.write(f"{d["id"]} translated into: {key}\n")
        # line[key +"_MT_bleu"] = bleu(MT_sent=machine_translation, human_reference= human_translation)["bleu"]
        # line[key+"_MT_chrf"] = chrf(MT_sent=machine_translation, human_reference= human_translation)["chrf"]
        # line[key+"_MT_chrf++"] = chrf(MT_sent=machine_translation, human_reference= human_translation)["chrf++"]

    logs.write(f"----- {d["id"]} Elapsed Time: {time.time() - start_time} -----\n")
    output.write(json.dumps(line, ensure_ascii=False) + "\n")

logs.write(f"Total Elapsed Time: {time.time() - start_time}")
logs.close()
output.close()
