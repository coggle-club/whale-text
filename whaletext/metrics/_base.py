from scipy import stats
from nltk.translate.bleu_score import sentence_bleu

def pearson_corr(arr1, arr2):
    return stats.pearsonr(arr1, arr2)

def BLEU(references, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
    return sentence_bleu(references, candidate, weights=weights)
