"""
Uncertainty calibration and evaluation metrics: AUROC, ECE, Brier score.
Answers Research Questions RQ1 and RQ2.
"""

from typing import List, Dict, Any, Tuple, Optional
import math


def compute_roc_curve_and_auroc(
    y_true: List[int],
    y_scores: List[float]
) -> Dict[str, Any]:
    """
    Computes AUROC (Area Under the ROC Curve) and ROC curve coordinates.
    Pure Python/NumPy implementation with no hard external dependency.
    
    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_scores: Predicted continuous uncertainty / confidence scores (e.g. [0.0, 1.0]).
        
    Returns:
        Dict with auroc, fpr, tpr, thresholds.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length.")
    
    if len(set(y_true)) < 2:
        # Both positive and negative classes must be present
        return {
            "auroc": float("nan"),
            "warning": "Only one class present in y_true; AUROC is undefined.",
            "fpr": [],
            "tpr": []
        }

    # Sort descending by predicted scores
    desc_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    y_true_sorted = [y_true[i] for i in desc_indices]
    y_scores_sorted = [y_scores[i] for i in desc_indices]
    
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    
    tpr_list = [0.0]
    fpr_list = [0.0]
    
    tp = 0
    fp = 0
    auroc = 0.0
    prev_fpr = 0.0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
            
        cur_tpr = tp / pos_count
        cur_fpr = fp / neg_count
        
        # Trapezoidal rule for AUC
        auroc += cur_tpr * (cur_fpr - prev_fpr)
        prev_fpr = cur_fpr
        
        tpr_list.append(round(cur_tpr, 4))
        fpr_list.append(round(cur_fpr, 4))

    return {
        "auroc": round(auroc, 4),
        "pos_count": pos_count,
        "neg_count": neg_count
    }


def compute_ece(
    y_true: List[int],
    y_probs: List[float],
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and binning statistics.
    
    ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if len(y_true) != len(y_probs):
        raise ValueError("y_true and y_probs must have the same length.")
    
    if len(y_true) == 0:
        return {"ece": 0.0, "bins": []}
    
    bin_boundaries = [i / n_bins for i in range(n_bins + 1)]
    bins_data = []
    
    total_samples = len(y_true)
    ece = 0.0
    
    for b in range(n_bins):
        low = bin_boundaries[b]
        high = bin_boundaries[b + 1]
        
        # For the last bin, include the upper edge 1.0
        if b == n_bins - 1:
            indices = [i for i, p in enumerate(y_probs) if low <= p <= high]
        else:
            indices = [i for i, p in enumerate(y_probs) if low <= p < high]
            
        count = len(indices)
        if count > 0:
            bin_acc = sum(y_true[i] for i in indices) / count
            bin_conf = sum(y_probs[i] for i in indices) / count
            bin_err = abs(bin_acc - bin_conf)
            ece += (count / total_samples) * bin_err
            
            bins_data.append({
                "bin": b,
                "range": [round(low, 2), round(high, 2)],
                "count": count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(bin_err, 4)
            })
        else:
            bins_data.append({
                "bin": b,
                "range": [round(low, 2), round(high, 2)],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "error": 0.0
            })

    return {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "bins": bins_data
    }


def compute_brier_score(
    y_true: List[int],
    y_probs: List[float]
) -> float:
    """Computes the mean squared error (Brier Score) between probabilities and binary labels."""
    if len(y_true) == 0:
        return 0.0
    squared_errors = [(p - y) ** 2 for p, y in zip(y_probs, y_true)]
    return round(sum(squared_errors) / len(squared_errors), 4)


def evaluate_span_uncertainty(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates predicted span-level aleatoric and epistemic uncertainties against gold annotations.
    - Evaluates RQ1: Does predicted aleatoric uncertainty correlate with gold 'human_ambiguous: true'?
    - Computes AUROC, ECE, and Brier Score.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}
    
    y_human_ambiguous = []
    y_pred_aleatoric = []
    y_pred_epistemic = []
    
    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        pred_rec = pred_by_id[rec_id]
        
        gold_uncs = gold_rec.get("uncertainties", [])
        pred_uncs = pred_rec.get("uncertainties", [])
        
        # Match spans between gold and pred
        for g_unc in gold_uncs:
            human_amb = g_unc.get("human_ambiguous")
            if human_amb is None:
                continue
                
            label = 1 if human_amb is True else 0
            g_span = g_unc.get("span", "")
            
            # Find best matching pred uncertainty span
            matched_pred = None
            for p_unc in pred_uncs:
                if p_unc.get("span", "") == g_span:
                    matched_pred = p_unc
                    break
                    
            if matched_pred and matched_pred.get("aleatoric") is not None:
                aleatoric_score = float(matched_pred["aleatoric"])
                epistemic_score = float(matched_pred.get("epistemic", 0.0))
            else:
                # If model missed predicting uncertainty on an ambiguous span, default to 0.0
                aleatoric_score = 0.0
                epistemic_score = 0.0
                
            y_human_ambiguous.append(label)
            y_pred_aleatoric.append(aleatoric_score)
            y_pred_epistemic.append(epistemic_score)

    if not y_human_ambiguous:
        return {
            "status": "NO_GOLD_AMBIGUITY_ANNOTATIONS",
            "message": "No gold records had 'human_ambiguous' set to true/false."
        }

    auroc_result = compute_roc_curve_and_auroc(y_human_ambiguous, y_pred_aleatoric)
    ece_result = compute_ece(y_human_ambiguous, y_pred_aleatoric)
    brier = compute_brier_score(y_human_ambiguous, y_pred_aleatoric)

    return {
        "sample_count": len(y_human_ambiguous),
        "positive_ambiguous_count": sum(y_human_ambiguous),
        "ambiguity_auroc": auroc_result.get("auroc"),
        "ambiguity_ece": ece_result.get("ece"),
        "ambiguity_brier_score": brier,
        "calibration_bins": ece_result.get("bins", [])
    }

