# Ling 575: Language Technology in Crisis Communication Term Project: AI benchmarking for Alerts Dataset
This project represent the work of Helen Salgi and Anantha Rao in evaluating the readiness of various commercial and open-source models for the purposes of translating [Wireless Emergency Alerts (WEA) Dataset](https://zepam.github.io/TranslationToolEvaluation/) from English into a variety of other languages. The WEA dataset we used to evaluate is one put together by [Jen Wilson](https://github.com/zepam).This work draws inspiration heavily from
[Bansal et al (2026): "Is MT Ready for the Next Crisis or Pandemic?"](https://arxiv.org/pdf/2601.10082).

Specifically, we evaluated the ability of Google Translate, Gemma4, and Qwen3.5 to translate emergency alerts from the WEA dataset from English into a variety of languages. We used the reference translations from the same dataset and looked at the following metrics:
- [BLEU](https://github.com/huggingface/evaluate/tree/main/metrics/bleu)
- [chrf(++)](https://github.com/huggingface/evaluate/tree/main/metrics/chrf)
- [COMET](https://github.com/huggingface/evaluate/tree/main/metrics/comet)
- [SacreBLEU](https://github.com/huggingface/evaluate/tree/main/metrics/sacrebleu)

>**Table of Contents**
>1. [Repository Structure]()
>2. [Google Translate API - Setup Instructions]()
>3. [Open Source LLM - Setup Instructions]()

## Repository Structure
1. ```./analysis/```: this directory contains R scripts for generating stats and graphs as well as statistical and visual content itself.
2. ```./data/```: this directory contains a duplicate of the data found in the WEA dataset.
3. ```./machine_translations/```: this directory contains each of the Machine Translations for a subset of the WEA Dataset (sub-divided by location). 'kc' = King County.
4. ```./scripts/```: this directory contains python scripts we used for processing data, running evaluations, calling API endpoints and running open source LLMs.
5. ```./*.sh```: the bash files at the root of this repository are used to call python scripts in ```./scripts/``` in the desired order to accomplish a specific task.
6. ```./prompt.md```: this file represents the prompt we used with the LLMs to do translation.
7. ```./requirements.txt```: the packages we used to do this project.
8. ```./comet_requirements.txt```: the packages used to evaluate COMET scores after generating translations (as opposed to do evaluation at time of translation)

# Google Translate API - Setup Instructions

### Step 1: Set up Google Cloud Account
[Google Cloud Welcome Page](https://console.cloud.google.com/welcome?authuser=2&organizationId=657476903663)

- Set up an account
- Create a New Project and note its project id!
- Set up Billing Account and link your Project to it

### Step 2: Create a Service Account

This service account allows other apps (including a python script) to make API calls

- On the pop-out Navigation Menu, click "APIs & Services > Credentials".
- Click "Manage service Account". Then click "Create service account".
- Download the JSON containing authentication info. Put this in a known location on your computer so your python script can access it.
- Give your Service Account permission to access whatever resource you want by "assigning a role". You'll probably want the "Cloud Translation API Admin"

### Step 3: Write your Python App

Now we'll access this service account with a python script!

```python
import os
from google.cloud import translate

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/downloaded/json/credentials.json"
PROJECT_ID = "project-id-of-your-project-123"
PARENT = f"projects/{PROJECT_ID}/locations/global"

client = translate.TranslateionServiceClient()

def translate_text(text: str, target_language_code: str) -> translate.Translation:

    response = client.translate_text(
        parent=PARENT,
        contents=[text],
        source_language_code="en",
        target_language_code=target_language_code,
    )

    return response.translations

translate_text("Mera naam Anantha hai", target_langauge_code = 'en')

```
Some notes:
- You'll need to run ```pip install google-cloud-translate```. You made need to install some other packages to run the above script. Look at the error messages for which packages need to yet be installed. If none show up you're good.
- Substitute the "/path/to/your/downloaded...." path with the path to the JSON credentials.
- language iso codes are the 2 letter ones ("en", "es", "hi", etc.)

Happy Translating!

# Open Source LLM - Setup Instructions
## Set up a venv
```python3 -m venv ai_venv```

## Run this setup for llama-cpp and huggingface
```CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir```

```pip install huggingface_hub```

- The "-DGGML_METAL=on" is a mac setting
- Drop this flag entirely if you are doing CPU-only
- NVIDIA needs CMAKE_ARGS="-DGGML_CUDA=on"
- AMD needs CMAKE_ARGS="-DGGML_HIP=on"
- If you are not sure what you have, try CMAKE_ARGS="-DGGML_VULKAN=1"

## Download Qwen3.5_9B
```hf auth login  (only if you aren't already logged in)```

```hf download bartowski/Qwen_Qwen3.5-9B-GGUF --include "*Q4_K_M*" --local-dir ./models```

## Download Gemma4_4B
```hf auth login```

```hf download bartowski/google_gemma-4-E4B-it-GGUF --include "*Q4_K_M*" --local-dir ./models```

## COMET Eval for Open Sourced Models
``` pip install llama-cpp-python unbabel-comet```

Run the following bash script:

```bash open_source_comet_evaluation.sh```

- In the bash script, the /models directory is inside of the /scripts directory because that is how we have it locally, however you will need to change the filepath in the script to point to where the models folder is if you try to run this yourself

## SacreBLEU Eval for Open Sourced Models
```pip install sacrebleu sentencepiece```

```bash open_source_scare_bleu_evaluation.sh```

On first use it downloads a ~5 MB SPM model from dl.fbaipublicfiles.com (cached in ~/.sacrebleu afterward), so the first run needs internet.

## For reformatting Gemma and Qwen results for the R scripts:
```bash format_bleu.sh```

```bash format_comet.sh```

As a note, the comet and bleu results from the open sourced models were designed to show the average results of each language.