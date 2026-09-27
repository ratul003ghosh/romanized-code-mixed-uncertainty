"""
Routing decision evaluation metrics: Routing Accuracy, Confusion Matrix, and Risk-Coverage Curves.
Answers Research Question RQ3.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict


VALID_ROUTES = ["PROCEED", "PROCEED_WITH_FLAGS", "ASK_USER", "ESCALATE"]


def evaluate_routing_decisions(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates routing decisions against gold standard routing.
    Computes overall accuracy, macro F1, per-class precision/recall, and confusion matrix.
    """
    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}
    
    total = 0
    correct = 0
    
    confusion_matrix = {r1: {r2: 0 for r2 in VALID_ROUTES} for r1 in VALID_ROUTES}
    gold_route_counts = defaultdict(int)
    pred_route_counts = defaultdict(int)

    for rec_id, gold_rec in gold_by_id.items():
        if rec_id not in pred_by_id:
            continue
        pred_rec = pred_by_id[rec_id]
        
        g_route = (gold_rec.get("routing") or "").strip().upper()
        p_route = (pred_rec.get("routing") or "").strip().upper()
        
        if not g_route or g_route not in VALID_ROUTES:
            continue
            
        total += 1
        gold_route_counts[g_route] += 1
        pred_route_counts[p_route] += 1
        
        if p_route in VALID_ROUTES:
            confusion_matrix[g_route][p_route] += 1
        
        if g_route == p_route:
            correct += 1

    if total == 0:
        return {
            "status": "NO_GOLD_ROUTING_LABELS",
            "message": "No valid gold routing labels found among evaluated records."
        }

    accuracy = correct / total

    # Per-class precision, recall, f1
    per_class = {}
    f1_sum = 0.0
    valid_class_count = 0
    
    for route in VALID_ROUTES:
        tp = confusion_matrix[route][route]
        fp = sum(confusion_matrix[other][route] for other in VALID_ROUTES if other != route)
        fn = sum(confusion_matrix[route][other] for other in VALID_ROUTES if other != route)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        per_class[route] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "gold_support": gold_route_counts[route]
        }
        
        if gold_route_counts[route] > 0:
            f1_sum += f1
            valid_class_count += 1

    macro_f1 = (f1_sum / valid_class_count) if valid_class_count > 0 else 0.0

    return {
        "total_evaluated": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix
    }


def compute_risk_coverage(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes Risk-Coverage tradeoff:
    - Coverage: % of prompts processed automatically (routed to PROCEED or PROCEED_WITH_FLAGS).
    - Escalation Rate: % of prompts routed to ASK_USER or ESCALATE.
    - Risk (Residual Leakage): PII leakage rate on covered prompts vs. escalated prompts.
    """
    from evaluation.pii_metrics import compute_pii_leakage

    gold_by_id = {rec["id"]: rec for rec in gold_records}
    pred_by_id = {rec["id"]: rec for rec in pred_records}

    covered_ids = []
    escalated_ids = []

    for rec_id, pred_rec in pred_by_id.items():
        if rec_id not in gold_by_id:
            continue
        route = (pred_rec.get("routing") or "").strip().upper()
        if route in ["PROCEED", "PROCEED_WITH_FLAGS"]:
            covered_ids.append(rec_id)
        else:
            escalated_ids.append(rec_id)

    total_prompts = len(covered_ids) + len(escalated_ids)
    if total_prompts == 0:
        return {"coverage": 0.0, "risk": 0.0}

    coverage_rate = len(covered_ids) / total_prompts
    escalation_rate = len(escalated_ids) / total_prompts

    # Leakage in covered vs escalated
    covered_gold = [gold_by_id[i] for i in covered_ids]
    covered_pred = [pred_by_id[i] for i in covered_ids]
    covered_leakage = compute_pii_leakage(covered_gold, covered_pred)

    escalated_gold = [gold_by_id[i] for i in escalated_ids]
    escalated_pred = [pred_by_id[i] for i in escalated_ids]
    escalated_leakage = compute_pii_leakage(escalated_gold, escalated_pred)

    return {
        "total_prompts": total_prompts,
        "coverage_rate": round(coverage_rate, 4),
        "escalation_rate": round(escalation_rate, 4),
        "covered_prompts_count": len(covered_ids),
        "escalated_prompts_count": len(escalated_ids),
        "covered_leakage_rate": covered_leakage["leakage_rate"],
        "escalated_leakage_rate": escalated_leakage["leakage_rate"],
        "leakage_reduction_by_routing": round(
            max(0.0, escalated_leakage["leakage_rate"] - covered_leakage["leakage_rate"]), 4
        )
    }

