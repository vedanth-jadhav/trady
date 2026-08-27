
import pandas as pd
from typing import List, Tuple, Dict
from src.signals.types import Trade
from .types import ScoredTrade
from .model import InsiderScoringModel
from .features import FeatureEngineer
from .overrides import RuleBasedOverrides

class InsiderScoringService:
    """
    Main service for scoring trades.

    Combines:
    - Feature engineering
    - ML model prediction
    - Rule-based overrides
    - Confidence thresholding
    """

    def __init__(
        self,
        model: InsiderScoringModel,
        feature_engineer: FeatureEngineer,
        overrides: RuleBasedOverrides = None
    ):
        self.model = model
        self.features = feature_engineer
        self.overrides = overrides or RuleBasedOverrides()

    def score_trade(self, trade: Trade) -> ScoredTrade:
        """
        Score a single trade.
        """
        # Extract features
        features = self.features.extract_features(trade)
        features_df = pd.DataFrame([features])

        # ML prediction
        # Handle case where model is not fitted (e.g. dev/testing)
        if self.model.is_fitted:
             ml_score = self.model.predict_proba(features_df)[0]
        else:
             ml_score = 0.5 # Default neutral score

        # Apply overrides
        final_score = self.overrides.apply_overrides(trade, features, ml_score)

        # Determine confidence tier
        confidence_tier = self._get_confidence_tier(final_score)

        return ScoredTrade(
            trade=trade,
            ml_score=ml_score,
            final_score=final_score,
            confidence_tier=confidence_tier,
            features=features,
            top_features=self._get_top_features(features)
        )

    def score_batch(self, trades: List[Trade]) -> List[ScoredTrade]:
        """
        Score multiple trades efficiently.
        """
        if not trades:
            return []
            
        # Batch feature extraction
        features_list = [self.features.extract_features(t) for t in trades]
        features_df = pd.DataFrame(features_list)

        # Batch ML prediction
        if self.model.is_fitted:
            ml_scores = self.model.predict_proba(features_df)
        else:
            ml_scores = [0.5] * len(trades)

        # Apply overrides and build results
        results = []
        for i, trade in enumerate(trades):
            final_score = self.overrides.apply_overrides(
                trade, features_list[i], ml_scores[i]
            )
            results.append(ScoredTrade(
                trade=trade,
                ml_score=ml_scores[i],
                final_score=final_score,
                confidence_tier=self._get_confidence_tier(final_score),
                features=features_list[i],
                top_features=self._get_top_features(features_list[i])
            ))

        return results

    def _get_confidence_tier(self, score: float) -> str:
        """Map score to confidence tier."""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "very_low"

    def _get_top_features(
        self,
        features: Dict[str, float],
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top contributing features."""
        # Sort by absolute value (simplified, ideal would be contribution to score)
        # For now just returning largest feature values as proxy for importance/interest
        return sorted(
            features.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n]
