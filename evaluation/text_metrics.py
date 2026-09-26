"""
Text normalization evaluation metrics: Character Error Rate (CER) and Word Error Rate (WER).
"""

from typing import List, Dict, Any, Tuple


def _levenshtein_distance(s1: List[Any], s2: List[Any]) -> int:
    """Computes minimum edit distance between two sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )
            
    return dp[m][n]


def compute_cer(ref: str, hyp: str) -> float:
    """Computes Character Error Rate (CER)."""
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    dist = _levenshtein_distance(ref_chars, hyp_chars)
    return dist / len(ref_chars)


def compute_wer(ref: str, hyp: str) -> float:
    """Computes Word Error Rate (WER)."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = _levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


def evaluate_text_normalization(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates normalized text quality against gold normalized text.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}

    total_evaluated = 0
    total_cer = 0.0
    total_wer = 0.0

    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        g_norm = gold_rec.get("normalized_text")
        p_norm = pred_by_id[rec_id].get("normalized_text")

        # Skip if gold normalized text was not set
        if not g_norm:
            continue

        p_norm = p_norm or ""
        total_cer += compute_cer(g_norm, p_norm)
        total_wer += compute_wer(g_norm, p_norm)
        total_evaluated += 1

    if total_evaluated == 0:
        return {
            "status": "NO_GOLD_NORMALIZED_TEXT",
            "message": "No gold records had 'normalized_text' set."
        }

    return {
        "total_evaluated": total_evaluated,
        "mean_cer": round(total_cer / total_evaluated, 4),
        "mean_wer": round(total_wer / total_evaluated, 4)
    }

