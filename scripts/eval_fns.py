import evaluate
from comet import download_model, load_from_checkpoint

#from nltk.translate.bleu_score import sentence_bleu
#from nltk.translate.chrf_score import sentence_chrf

# Things to load beforehand:
_chrf = evaluate.load("chrf")
_bleu = evaluate.load("bleu")
_sacrebleu = evaluate.load("sacrebleu")
_comet_model = download_model("Unbabel/wmt22-comet-da")
_comet = load_from_checkpoint(_comet_model)
# _meteor = evaluate.load("meteor")

# from torch.utils.data import DataLoader
# import torch.multiprocessing as mp
# # You may need to apply this globally at the top of your file or inside main
# mp.set_start_method('spawn', force=True) 

def bleu(MT_sent:str, human_reference:str):
    """Computes the bleu score for BLEU-1, BLEU-2, BLEU-3, and BLEU-4

    Args:
        - MT_sent: the machine translated sentence
        - human_reference: the gold reference / human translation
    
    Returns:
    A dictionary with BLEU scores and precision
    """
    bleu_scores = _bleu.compute(
        predictions = [MT_sent],
        references = [human_reference],
        smooth = True
    )

    return {
        "bleu": bleu_scores["bleu"],
        "bleu1_precision": bleu_scores["precisions"][0],
        "bleu2_precision": bleu_scores["precisions"][1], 
        "bleu3_precision": bleu_scores["precisions"][2],
        "bleu4_precision": bleu_scores["precisions"][3]
    }

def chrf(MT_sent:str, human_reference:str):
    """Computes chrf++. The ++ represents the word order the character level
    precision, recall, and f are computed over.
    
    Args:
        - MT_sent: the machine translated sentence
        - human_reference: the gold reference / human translation
    
    Returns:
    A dictionary with chrf, and chrf++ scores
    """

    chrf_scores = _chrf.compute(
        predictions = [MT_sent],
        references = [human_reference],
        word_order = 0
    )

    chrfplusplus_scores = _chrf.compute(
        predictions = [MT_sent],
        references = [human_reference],
        word_order = 2
    )

    return {
        "chrf": chrf_scores["score"],
        "chrf++": chrfplusplus_scores["score"]
    }

def sacrebleu(MT_sent:str, human_reference:str):
    """Computes sacrebleu
    
    Args:
        - MT_sent: the machine translated sentence
        - human_reference: the gold reference / human translation
    
    Returns:
    A dictionary with sacrebleu scores
    """
    sb_scores = _sacrebleu.compute(
        predictions=[MT_sent],
        references=[human_reference]
    )

    return {
        "sacrebleu": sb_scores["score"] / 100
    }

# _comet = evaluate.load("comet")
def comet(MT_sent:str, human_reference:str, source_sentence:str):
    """Computes Comet for a triple of sentences.
    
    Args:
        - MT_sent: the machine translated sentence
        - human_reference: the gold reference / human translation
        - source_sentence: the sentence in the source language (before translation)
    
    Returns:
    A dictionary with COMET scores
    """
    # HuggingFace Way of doing it that wasn't working
    # comet_score = _comet.compute(
    #     predictions = [MT_sent],
    #     references = [human_reference],
    #     sources = [source_sentence],
    #     gpus = None
    # )

    # Direct way by loading the model
    data = [
        {
            "src": source_sentence,
            "mt": MT_sent,
            "ref": human_reference
        }
    ]

    comet_score = _comet.predict(data, batch_size=1)

    return {
        "comet":comet_score.scores[0]
    }
