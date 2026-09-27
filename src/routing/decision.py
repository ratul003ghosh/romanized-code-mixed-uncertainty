"""
Risk-Controlled Dynamic Routing Decision Layer.
Maps span-level two-channel uncertainties (aleatoric & epistemic) into 4 discrete actions:
  - PROCEED: Low uncertainty, safe to pass to downstream LLM.
  - PROCEED_WITH_FLAGS: PII masked or moderate uncertainty, proceed with audit logging.
  - ASK_USER: High aleatoric uncertainty (inherent ambiguity like "5oo tk"), prompt user for clarification.
  - ESCALATE: High epistemic uncertainty (model doubt / disagreement), escalate to human moderator.

Conforms to Proposal §4.6, §5.1, and Schema v0.2.
"""

from typing import List, Dict, Any, Tuple, Optional
import copy


VALID_ROUTES = ["PROCEED", "PROCEED_WITH_FLAGS", "ASK_USER", "ESCALATE"]


class RoutingDecisionEngine:
    """
    Threshold-based risk-controlled router.
    """

    def __init__(
        self,
        tau_epistemic: float = 0.60,
        tau_aleatoric: float = 0.60,
        flag_threshold: float = 0.30
    ):
        """
        Args:
            tau_epistemic: Threshold for escalating to human due to model confusion/disagreement.
            tau_aleatoric: Threshold for asking user due to inherent data ambiguity.
            flag_threshold: Threshold above which outputs are flagged for audit logging.
        """
        self.tau_epistemic = tau_epistemic
        self.tau_aleatoric = tau_aleatoric
        self.flag_threshold = flag_threshold

    def decide_route(self, record: Dict[str, Any]) -> str:
        """
        Determines the routing decision for a single record.
        """
        uncertainties = record.get("uncertainties", []) or record.get("spans", [])
        pii = record.get("pii", [])

        max_e = 0.0
        max_a = 0.0

        for unc in uncertainties:
            e = float(unc.get("epistemic", 0.0))
            a = float(unc.get("aleatoric", 0.0))
            if e > max_e:
                max_e = e
            if a > max_a:
                max_a = a

        # Rule 1: High epistemic uncertainty -> ESCALATE to human / heavier model
        if max_e >= self.tau_epistemic:
            return "ESCALATE"

        # Rule 2: High aleatoric uncertainty -> ASK_USER for clarification
        if max_a >= self.tau_aleatoric:
            return "ASK_USER"

        # Rule 3: PII masked or mild uncertainty -> PROCEED_WITH_FLAGS
        if len(pii) > 0 or max_e >= self.flag_threshold or max_a >= self.flag_threshold:
            return "PROCEED_WITH_FLAGS"

        # Rule 4: Clean, low uncertainty -> PROCEED
        return "PROCEED"

    def apply(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies routing decision to a copy of all records.
        """
        routed_records = []
        for rec in records:
            rec_copy = copy.deepcopy(rec)
            rec_copy["routing"] = self.decide_route(rec_copy)
            routed_records.append(rec_copy)
        return routed_records

    @staticmethod
    def tune_thresholds(
        dev_gold_records: List[Dict[str, Any]],
        dev_pred_records: List[Dict[str, Any]],
        target_max_leakage: float = 0.05,
        grid_step: float = 0.05
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Tunes (tau_epistemic, tau_aleatoric) on dev split via grid search to maximize
        automated coverage (PROCEED + PROCEED_WITH_FLAGS) while maintaining residual
        PII leakage below target_max_leakage (Risk-Controlled Routing).
        
        Returns:
            (best_tau_e, best_tau_a, best_stats)
        """
        from evaluation.routing_metrics import compute_risk_coverage

        best_tau_e = 0.60
        best_tau_a = 0.60
        best_coverage = -1.0
        best_stats = {}

        # Grid search over thresholds in [0.20, 0.90]
        steps = [round(x * grid_step, 2) for x in range(4, 19)]  # 0.20 to 0.90
        for tau_e in steps:
            for tau_a in steps:
                engine = RoutingDecisionEngine(tau_epistemic=tau_e, tau_aleatoric=tau_a)
                routed_preds = engine.apply(dev_pred_records)
                rc = compute_risk_coverage(dev_gold_records, routed_preds)

                coverage = rc.get("coverage_rate", 0.0)
                covered_leakage = rc.get("covered_leakage_rate", 0.0)

                # Feasible operating point: risk constraint satisfied
                if covered_leakage <= target_max_leakage:
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tau_e = tau_e
                        best_tau_a = tau_a
                        best_stats = {
                            "tau_epistemic": tau_e,
                            "tau_aleatoric": tau_a,
                            "coverage_rate": coverage,
                            "covered_leakage_rate": covered_leakage,
                            "escalation_rate": rc.get("escalation_rate", 0.0)
                        }

        if best_coverage == -1.0:
            # Fallback to default if no threshold strictly satisfied target
            best_stats = {
                "tau_epistemic": 0.60,
                "tau_aleatoric": 0.60,
                "note": "Target risk not fully met in grid; defaulted to 0.60"
            }

        return best_tau_e, best_tau_a, best_stats
