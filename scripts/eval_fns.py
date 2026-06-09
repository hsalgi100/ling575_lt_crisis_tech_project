from nltk.translate.bleu_score import sentence_bleu

def bleu(hypothesis_sent:str, human_reference:str):
    """Computes the NLTK bleu score for BLEU-2, BLEU-3, and BLEU-4"""

    hypo = hypothesis_sent.split(" ")
    ref = human_reference.split(" ")

    b2, b3, b4 = sentence_bleu(
    [ref],
    hypo,
    weights=[
        (1/2,1/2),
        (1/3,1/3,1/3),
        (1/4,1/4,1/4,1/4)
        ]
    )

    return {
        "bleu2": b2, 
        "bleu3": b3,
        "bleu4": b4
    }


