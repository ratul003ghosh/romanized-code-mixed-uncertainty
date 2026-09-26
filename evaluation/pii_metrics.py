"""
PII evaluation metrics: Span F1, Precision, Recall, Leakage Rate, and Over-masking Rate.
Conforms to project Schema v0.2.
"""

from typing import List, Dict, Any, Tuple, Optional
import re


def _calculate_span_overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
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


def evaluate_pii_spans(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    match_type: bool = True
) -> Dict[str, Any]:
    """
    Evaluates PII span detection accuracy (Exact match and IoU partial match).
    
    Args:
        gold_records: List of gold records containing 'pii' lists.
        pred_records: List of prediction records containing 'pii' lists.
        iou_threshold: Minimum IoU for partial overlap match (default: 0.5).
        match_type: Whether PII entity type must match for a true positive.
        
    Returns:
        Dict with precision, recall, f1 (exact and partial), and per-type breakdown.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}
    
    common_ids = set(gold_by_id.keys()) & set(pred_by_id.keys())
    if not common_ids:
        raise ValueError("No common record IDs found between gold and prediction sets.")
        
    # Global counts
    exact_tp = 0
    exact_fp = 0
    exact_fn = 0
    
    partial_tp = 0
    partial_fp = 0
    partial_fn = 0
    
    per_type_counts: Dict[str, Dict[str, int]] = {}

    for rec_id in common_ids:
        g_pii = gold_by_id[rec_id].get("pii", [])
        p_pii = pred_by_id[rec_id].get("pii", [])
        
        # Track matched predictions and gold items
        matched_gold_exact = set()
        matched_pred_exact = set()
        
        matched_gold_partial = set()
        matched_pred_partial = set()
        
        # 1. Exact match pass
        for p_idx, p in enumerate(p_pii):
            p_span = (p.get("start", -1), p.get("end", -1))
            p_type = p.get("type", "UNKNOWN")
            
            for g_idx, g in enumerate(g_pii):
                if g_idx in matched_gold_exact:
                    continue
                g_span = (g.get("start", -2), g.get("end", -2))
                g_type = g.get("type", "UNKNOWN")
                
                type_matches = (not match_type) or (p_type == g_type)
                if p_span == g_span and type_matches:
                    exact_tp += 1
                    matched_gold_exact.add(g_idx)
                    matched_pred_exact.add(p_idx)
                    
                    if g_type not in per_type_counts:
                        per_type_counts[g_type] = {"tp": 0, "fp": 0, "fn": 0}
                    per_type_counts[g_type]["tp"] += 1
                    break

        exact_fp += len(p_pii) - len(matched_pred_exact)
        exact_fn += len(g_pii) - len(matched_gold_exact)
        
        # Account for per-type FP and FN in exact match
        for p_idx, p in enumerate(p_pii):
            if p_idx not in matched_pred_exact:
                p_type = p.get("type", "UNKNOWN")
                if p_type not in per_type_counts:
                    per_type_counts[p_type] = {"tp": 0, "fp": 0, "fn": 0}
                per_type_counts[p_type]["fp"] += 1
                
        for g_idx, g in enumerate(g_pii):
            if g_idx not in matched_gold_exact:
                g_type = g.get("type", "UNKNOWN")
                if g_type not in per_type_counts:
                    per_type_counts[g_type] = {"tp": 0, "fp": 0, "fn": 0}
                per_type_counts[g_type]["fn"] += 1

        # 2. Partial match pass (IoU >= threshold)
        for p_idx, p in enumerate(p_pii):
            p_span = (p.get("start", -1), p.get("end", -1))
            p_type = p.get("type", "UNKNOWN")
            best_iou = 0.0
            best_g_idx = -1
            
            for g_idx, g in enumerate(g_pii):
                if g_idx in matched_gold_partial:
                    continue
                g_type = g.get("type", "UNKNOWN")
                if match_type and (p_type != g_type):
                    continue
                g_span = (g.get("start", -2), g.get("end", -2))
                iou = _calculate_span_overlap(p_span, g_span)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx
            
            if best_g_idx >= 0:
                partial_tp += 1
                matched_gold_partial.add(best_g_idx)
                matched_pred_partial.add(p_idx)
                
        partial_fp += len(p_pii) - len(matched_pred_partial)
        partial_fn += len(g_pii) - len(matched_gold_partial)

    def _calc_prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    per_type_metrics = {}
    for pii_type, counts in per_type_counts.items():
        per_type_metrics[pii_type] = _calc_prf(counts["tp"], counts["fp"], counts["fn"])

    return {
        "exact_match": _calc_prf(exact_tp, exact_fp, exact_fn),
        "partial_match": _calc_prf(partial_tp, partial_fp, partial_fn),
        "per_type": per_type_metrics
    }


def compute_pii_leakage(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes PII Leakage Rate: percentage of ground truth PII strings
    that leaked into the model's sanitized prompt.
    
    A PII item leaks if its raw text is found unmasked inside sanitized_prompt.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}
    
    total_gold_pii = 0
    leaked_pii_count = 0
    leaked_details = []

    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        pred_rec = pred_by_id[rec_id]
        sanitized = pred_rec.get("sanitized_prompt") or ""
        
        for p in gold_rec.get("pii", []):
            raw_val = p.get("text")
            if not raw_val or len(raw_val.strip()) == 0:
                continue
            total_gold_pii += 1
            
            # Check if raw sensitive text still appears verbatim in sanitized output
            # (allowing boundary word boundaries when appropriate)
            escaped_val = re.escape(raw_val)
            if re.search(escaped_val, sanitized, re.IGNORECASE):
                leaked_pii_count += 1
                leaked_details.append({
                    "id": rec_id,
                    "type": p.get("type"),
                    "text": raw_val,
                    "sanitized_prompt": sanitized
                })

    leakage_rate = leaked_pii_count / total_gold_pii if total_gold_pii > 0 else 0.0

    return {
        "total_gold_pii": total_gold_pii,
        "leaked_pii_count": leaked_pii_count,
        "leakage_rate": round(leakage_rate, 4),
        "leakage_percentage": round(leakage_rate * 100, 2),
        "sample_leaks": leaked_details[:10]
    }


def compute_overmasking_rate(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes Over-Masking Rate (Utility loss):
    Evaluates whether non-sensitive context or preserved entities (e.g. monetary amounts,
    dates, service names like bKash) were mistakenly redacted or classified as PII.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}

    total_preserved = 0
    erroneously_masked = 0
    overmasked_details = []

    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        pred_rec = pred_by_id[rec_id]
        pred_pii = pred_rec.get("pii", [])
        sanitized = pred_rec.get("sanitized_prompt") or ""

        preserved_entities = gold_rec.get("preserved_entities", [])
        for ent in preserved_entities:
            val = ent.get("value")
            if not val:
                continue
            total_preserved += 1
            
            # Check if this preserved value was marked as PII in prediction
            # or if it was stripped/replaced with a placeholder in sanitized prompt
            was_tagged_as_pii = any(val.lower() in (p.get("text", "").lower()) for p in pred_pii)
            was_lost_in_sanitization = (val.lower() not in sanitized.lower())
            
            if was_tagged_as_pii or was_lost_in_sanitization:
                erroneously_masked += 1
                overmasked_details.append({
                    "id": rec_id,
                    "entity": ent,
                    "was_tagged_as_pii": was_tagged_as_pii,
                    "was_lost_in_sanitization": was_lost_in_sanitization
                })

    overmasking_rate = erroneously_masked / total_preserved if total_preserved > 0 else 0.0
    utility_preservation_rate = 1.0 - overmasking_rate

    return {
        "total_preserved_entities": total_preserved,
        "erroneously_masked_count": erroneously_masked,
        "overmasking_rate": round(overmasking_rate, 4),
        "utility_preservation_rate": round(utility_preservation_rate, 4),
        "sample_overmasked": overmasked_details[:10]
    }

