
from typing import Dict
from src.signals.types import Trade

class RuleBasedOverrides:
    """
    Hard-coded rules that override ML predictions for edge cases.

    These capture domain knowledge that may not be learnable.
    """

    def apply_overrides(
        self,
        trade: Trade,
        features: Dict[str, float],
        ml_score: float
    ) -> float:
        """
        Apply rule-based overrides to ML score.

        Returns adjusted score.
        """
        adjusted = ml_score

        # Override 1: Privacy funding + zero history = always high
        if (features.get("wallet_has_privacy_funding", 0) == 1 and
            features.get("sig_has_zero_history", 0) == 1):
            adjusted = max(adjusted, 0.9)

        # Override 2: Known entity = always low
        # Assuming we have a feature for this, or check against 0 if not present
        if features.get("wallet_is_known_entity", 0) == 1:
            adjusted = min(adjusted, 0.1)

        # Override 3: Retail behavior + long history = cap at 0.3
        if (features.get("wallet_retail_likelihood", 0) > 0.8 and
            features.get("wallet_days_active", 0) > 180):
            adjusted = min(adjusted, 0.3)

        # Override 4: Cluster coordination + whale = boost
        if (features.get("sig_has_cluster_coord", 0) == 1 and
            features.get("wallet_is_whale", 0) == 1):
            adjusted = min(adjusted * 1.2, 1.0)
            
        # Refine Score to be within [0, 1] just in case
        return max(0.0, min(1.0, adjusted))
