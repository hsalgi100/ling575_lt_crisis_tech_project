import sys, os
from google.cloud import translate # translate_v2 as translate
from eval_fns import *
# from os import environ
# from google.cloud import translate

service_account_credentials_path = "/Users/anantha/.apikeys/wea-translation-497802-3e63f30e8ae5.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_credentials_path
PROJECT_ID = "wea-translation-497802" #environ.get("PROJECT_ID", "")
PARENT = f"projects/{PROJECT_ID}/locations/global"


# from STEP 4 in tutorial
def print_supported_languages(display_language_code: str):
    client = translate.TranslationServiceClient()

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
    client = translate.TranslationServiceClient()

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        target_language_code=target_language_code,
    )

    return response.translations[0]
    
