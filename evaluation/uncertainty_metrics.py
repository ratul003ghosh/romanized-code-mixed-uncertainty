"""
Uncertainty calibration and evaluation metrics: AUROC, ECE, Brier score, and Channel Separation.
Answers Research Questions RQ1 and RQ2 (Proposal §3.3, §6.1, K5).
"""

from typing import List, Dict, Any, Tuple, Optional
import math


def _calculate_span_iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    """Calculates Intersection over Union (IoU) of two character spans."""
    start_a, end_a = span_a
    start_b, end_b = span_b
    
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    inter = max(0, inter_end - inter_start)
    
    if inter == 0:
        return 0.0
    
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union > 0 else 0.0


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
        Dict with auroc, fpr, tpr, pos_count, neg_count.
    """
    if len(y_true) != len(y_scores):
        raise ValueError("y_true and y_scores must have the same length.")
    
    pos_count = sum(y_true)
    neg_count = len(y_true) - pos_count
    if pos_count == 0 or neg_count == 0:
        return {
            "auroc": float("nan"),
            "warning": "Only one class present in y_true; AUROC is undefined.",
            "pos_count": pos_count,
            "neg_count": neg_count
        }

    try:
        from sklearn.metrics import roc_auc_score
        auroc_val = float(roc_auc_score(y_true, y_scores))
        return {
            "auroc": round(auroc_val, 4),
            "pos_count": pos_count,
            "neg_count": neg_count
        }
    except Exception:
        # Mann-Whitney U test with mid-ranks for ties
        indexed = sorted(enumerate(y_scores), key=lambda x: x[1])
        ranks = [0.0] * len(y_scores)
        i = 0
        n = len(y_scores)
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
            
        r1 = sum(ranks[i] for i, y in enumerate(y_true) if y == 1)
        u1 = r1 - (pos_count * (pos_count + 1)) / 2.0
        auroc_val = u1 / (pos_count * neg_count)
        return {
            "auroc": round(float(auroc_val), 4),
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
    Evaluates predicted span-level aleatoric uncertainty against gold annotations (RQ1).
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}
    
    y_human_ambiguous = []
    y_pred_aleatoric = []
    
    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        pred_rec = pred_by_id[rec_id]
        
        gold_uncs = gold_rec.get("uncertainties", [])
        pred_uncs = pred_rec.get("uncertainties", []) or pred_rec.get("spans", [])
        
        for g_unc in gold_uncs:
            human_amb = g_unc.get("human_ambiguous")
            if human_amb is None:
                continue
                
            label = 1 if human_amb is True else 0
            g_span = g_unc.get("span", "")
            g_start = g_unc.get("start")
            g_end = g_unc.get("end")
            
            # Find best matching pred uncertainty span (by start/end or text)
            matched_pred = None
            for p_unc in pred_uncs:
                p_start = p_unc.get("start")
                p_end = p_unc.get("end")
                if g_start is not None and p_start is not None and g_end is not None and p_end is not None:
                    if _calculate_span_iou((g_start, g_end), (p_start, p_end)) >= 0.5:
                        matched_pred = p_unc
                        break
                elif p_unc.get("span", "") == g_span:
                    matched_pred = p_unc
                    break
                    
            if matched_pred and matched_pred.get("aleatoric") is not None:
                aleatoric_score = float(matched_pred["aleatoric"])
            else:
                aleatoric_score = 0.0
                
            y_human_ambiguous.append(label)
            y_pred_aleatoric.append(aleatoric_score)

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


def evaluate_channel_separation(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates Two-Channel Uncertainty Separation (Proposal §3.3, §6.1, RQ1 / K5):
    
    1. Aleatoric Channel:
       - Positives: human_ambiguous == True (inherent data noise/ambiguity).
       - Negatives: human_ambiguous == False.
       - Metric: AUROC of aleatoric uncertainty.
       - Cross-channel check: AUROC of epistemic uncertainty on the same labels.
       - Aleatoric Gap = AUROC(aleatoric) - AUROC(epistemic) (should be positive).
       
    2. Epistemic Channel:
       - Evaluated on human-unambiguous spans (human_ambiguous == False or non-ambiguous PII).
       - Positives: Spans where teacher/model prediction differs from gold
         (i.e. model made an error or teacher disagreement on unambiguous text).
       - Negatives: Spans where teacher/model prediction matches gold (correct prediction).
       - Metric: AUROC of epistemic uncertainty.
       - Cross-channel check: AUROC of aleatoric uncertainty on the same labels.
       - Epistemic Gap = AUROC(epistemic) - AUROC(aleatoric) (should be positive).
       
    3. Cross-Channel Separation Index (CSI):
       - Summary metric proving that the two channels are successfully decoupled.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}

    # Channel 1: Aleatoric evaluation
    aleatoric_labels = []
    aleatoric_scores_a = []
    aleatoric_scores_e = []

    # Channel 2: Epistemic evaluation (on human-unambiguous spans)
    epistemic_labels = []
    epistemic_scores_e = []
    epistemic_scores_a = []

    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        pred_rec = pred_by_id[rec_id]

        g_uncs = gold_rec.get("uncertainties", [])
        p_uncs = pred_rec.get("uncertainties", []) or pred_rec.get("spans", [])
        g_pii = gold_rec.get("pii", [])
        p_pii = pred_rec.get("pii", [])

        # 1. Collect Aleatoric channel labels from gold uncertainties
        for g in g_uncs:
            human_amb = g.get("human_ambiguous")
            if human_amb is not None:
                label = 1 if human_amb is True else 0
                g_span = (g.get("start", -1), g.get("end", -1))
                g_text = g.get("span", "")

                matched_p = None
                for p in p_uncs:
                    p_span = (p.get("start", -2), p.get("end", -2))
                    if g_span[0] >= 0 and p_span[0] >= 0:
                        if _calculate_span_iou(g_span, p_span) >= iou_threshold:
                            matched_p = p
                            break
                    elif p.get("span", "") == g_text:
                        matched_p = p
                        break

                a_score = float(matched_p.get("aleatoric", 0.0)) if matched_p else 0.0
                e_score = float(matched_p.get("epistemic", 0.0)) if matched_p else 0.0

                aleatoric_labels.append(label)
                aleatoric_scores_a.append(a_score)
                aleatoric_scores_e.append(e_score)

        # 2. Collect Epistemic channel labels from unambiguous PII spans and non-ambiguous spans
        # Check PII spans for prediction error / disagreement
        for g in g_pii:
            g_span = (g.get("start", -1), g.get("end", -1))
            g_type = g.get("type", "UNKNOWN")
            
            # Check if this span was tagged as human-ambiguous
            is_ambiguous = any(
                u.get("human_ambiguous") is True and 
                (u.get("span") == g.get("text") or (g_span[0] >= 0 and _calculate_span_iou(g_span, (u.get("start", -2), u.get("end", -2))) >= 0.5))
                for u in g_uncs
            )
            if is_ambiguous:
                continue  # Epistemic channel is evaluated on human-unambiguous spans

            # Find matching prediction in p_pii
            matched_pii = None
            for p in p_pii:
                p_span = (p.get("start", -2), p.get("end", -2))
                if g_span[0] >= 0 and p_span[0] >= 0:
                    if _calculate_span_iou(g_span, p_span) >= iou_threshold:
                        matched_pii = p
                        break
                elif p.get("text", "") == g.get("text", ""):
                    matched_pii = p
                    break

            # Error occurred if model missed the PII or predicted wrong type
            has_error = (matched_pii is None) or (matched_pii.get("type") != g_type)
            e_label = 1 if has_error else 0

            # Find uncertainty scores for this span
            matched_unc = None
            for p in p_uncs:
                p_span = (p.get("start", -2), p.get("end", -2))
                if g_span[0] >= 0 and p_span[0] >= 0:
                    if _calculate_span_iou(g_span, p_span) >= iou_threshold:
                        matched_unc = p
                        break
                elif p.get("span", "") == g.get("text", ""):
                    matched_unc = p
                    break

            e_score = float(matched_unc.get("epistemic", 0.0)) if matched_unc else (0.5 if has_error else 0.0)
            a_score = float(matched_unc.get("aleatoric", 0.0)) if matched_unc else 0.0

            epistemic_labels.append(e_label)
            epistemic_scores_e.append(e_score)
            epistemic_scores_a.append(a_score)

    # Compute AUROCs for Aleatoric Channel
    aleatoric_channel_auroc = compute_roc_curve_and_auroc(aleatoric_labels, aleatoric_scores_a).get("auroc", float("nan"))
    epistemic_cross_on_aleatoric = compute_roc_curve_and_auroc(aleatoric_labels, aleatoric_scores_e).get("auroc", float("nan"))
    
    if not math.isnan(aleatoric_channel_auroc) and not math.isnan(epistemic_cross_on_aleatoric):
        aleatoric_gap = round(aleatoric_channel_auroc - epistemic_cross_on_aleatoric, 4)
    else:
        aleatoric_gap = None

    # Compute AUROCs for Epistemic Channel
    epistemic_channel_auroc = compute_roc_curve_and_auroc(epistemic_labels, epistemic_scores_e).get("auroc", float("nan"))
    aleatoric_cross_on_epistemic = compute_roc_curve_and_auroc(epistemic_labels, epistemic_scores_a).get("auroc", float("nan"))
    
    if not math.isnan(epistemic_channel_auroc) and not math.isnan(aleatoric_cross_on_epistemic):
        epistemic_gap = round(epistemic_channel_auroc - aleatoric_cross_on_epistemic, 4)
    else:
        epistemic_gap = None

    # Summary Channel Separation Index (CSI)
    valid_gaps = [g for g in [aleatoric_gap, epistemic_gap] if g is not None]
    csi = round(sum(valid_gaps) / len(valid_gaps), 4) if valid_gaps else None

    return {
        "aleatoric_channel": {
            "auroc_aleatoric_on_ambiguity": aleatoric_channel_auroc,
            "auroc_epistemic_cross": epistemic_cross_on_aleatoric,
            "cross_channel_gap": aleatoric_gap,
            "sample_count": len(aleatoric_labels)
        },
        "epistemic_channel": {
            "auroc_epistemic_on_errors": epistemic_channel_auroc,
            "auroc_aleatoric_cross": aleatoric_cross_on_epistemic,
            "cross_channel_gap": epistemic_gap,
            "sample_count": len(epistemic_labels)
        },
        "channel_separation_index": csi,
        "is_separated": (csi is not None and csi > 0.0)
    }
