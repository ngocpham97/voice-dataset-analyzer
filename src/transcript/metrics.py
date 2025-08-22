from typing import List, Tuple

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Character Error Rate (CER) between the reference and hypothesis strings.
    """
    reference = reference.replace(" ", "")
    hypothesis = hypothesis.replace(" ", "")
    
    if len(reference) == 0:
        return float(hypothesis != "")
    
    edit_distance = levenshtein_distance(reference, hypothesis)
    cer = edit_distance / len(reference)
    return cer

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Word Error Rate (WER) between the reference and hypothesis strings.
    """
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    
    if len(reference_words) == 0:
        return float(hypothesis_words != [])
    
    edit_distance = levenshtein_distance(reference_words, hypothesis_words)
    wer = edit_distance / len(reference_words)
    return wer

def levenshtein_distance(ref: List[str], hyp: List[str]) -> int:
    """
    Calculate the Levenshtein distance between two sequences.
    """
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
            elif ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    
    return d[len(ref)][len(hyp)]