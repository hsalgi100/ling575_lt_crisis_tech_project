import evaluate
#from nltk.translate.bleu_score import sentence_bleu
#from nltk.translate.chrf_score import sentence_chrf

# Things to load beforehand:
_chrf = evaluate.load("chrf")
_bleu = evaluate.load("bleu")
_comet = evaluate.load("comet")

# def nltk_bleu(MT_sent:str, human_reference:str):
#     """Computes the NLTK bleu score for BLEU-2, BLEU-3, and BLEU-4

#     Args:
#         - MT_sent: the machine translated sentence
#         - human_reference: the gold reference / human translation
    
#     Returns:
#     A dictionary with BLEU-2, BLEU-3, and BLEU-4 score
#     """

#     hypo = MT_sent.split(" ")
#     ref = human_reference.split(" ")

#     b2, b3, b4 = sentence_bleu(
#     [ref],
#     hypo,
#     weights=[
#         (1/2,1/2),
#         (1/3,1/3,1/3),
#         (1/4,1/4,1/4,1/4)
#         ]
#     )

#     return {
#         "bleu2": b2, 
#         "bleu3": b3,
#         "bleu4": b4
#     }

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

def comet(MT_sent:str, human_reference:str, source_sentence:str):
    """Computes Comet for a triple of sentences.
    
    Args:
        - MT_sent: the machine translated sentence
        - human_reference: the gold reference / human translation
        - source_sentence: the sentence in the source language (before translation)
    
    Returns:
    A dictionary with COMET scores
    """

    comet_score = _comet.compute(
        predictions = [MT_sent],
        reference = [human_reference],
        sources = [source_sentence]
    )
