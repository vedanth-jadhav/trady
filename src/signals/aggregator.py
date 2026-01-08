"""
Signal aggregator.

Combines all signals into a final composite insider score.
"""

from typing import Dict, List

from .types import (
    AggregatedSignal,
    Signal,
    SignalCategory,
)


class SignalAggregator:
    """
    Aggregates all signals into composite insider score.

    Weighting Strategy:
    - Freshness signals: 0.25
    - Timing signals: 0.20
    - Sizing signals: 0.20
    - Funding signals: 0.25
    - Cluster signals: 0.10

    Applies negative signal discount and market context boost.
    """

    CATEGORY_WEIGHTS = {
        SignalCategory.FRESHNESS.value: 0.25,
        SignalCategory.TIMING.value: 0.20,
        SignalCategory.SIZING.value: 0.20,
        SignalCategory.FUNDING.value: 0.25,
        SignalCategory.CLUSTER.value: 0.10,
    }

    def __init__(self, custom_weights: Dict[str, float] = None):
        """
        Initialize aggregator with optional custom weights.

        Args:
            custom_weights: Optional dict to override default category weights
        """
        self.weights = self.CATEGORY_WEIGHTS.copy()
        if custom_weights:
            self.weights.update(custom_weights)

    def aggregate(
        self,
        signals: List[Signal],
        negative_score: float,
        market_context_score: float,
    ) -> AggregatedSignal:
        """
        Aggregate all signals into final score.

        Steps:
        1. Group signals by category
        2. Take max confidence per category
        3. Apply category weights
        4. Discount by negative score
        5. Boost by market context

        Args:
            signals: List of detected signals
            negative_score: Score from NegativeSignalFilter (0-1)
            market_context_score: Score from MarketContextAnalyzer (0-1.5)

        Returns:
            AggregatedSignal with final score and breakdown
        """
        if not signals:
            return AggregatedSignal(
                final_score=0.0,
                raw_score=0.0,
                negative_discount=negative_score,
                market_boost=market_context_score,
                category_scores={},
                signals=[],
            )

        # Group by category
        category_signals: Dict[str, List[float]] = {}
        for signal in signals:
            category = signal.category.value
            if category not in category_signals:
                category_signals[category] = []
            category_signals[category].append(signal.confidence)

        # Max per category
        category_scores = {
            cat: max(scores) for cat, scores in category_signals.items()
        }

        # Weighted sum
        raw_score = sum(
            category_scores.get(cat, 0) * weight
            for cat, weight in self.weights.items()
        )

        # Apply negative discount (1 - negative_score * 0.5)
        # This means a high negative score reduces the final score
        discounted_score = raw_score * (1 - negative_score * 0.5)

        # Apply market context boost
        # Market boost is 0.5-1.5, so we scale accordingly
        # At boost=1.0, no change; at boost=1.5, 25% increase
        final_score = discounted_score * (0.5 + market_context_score * 0.5)

        # Clamp to 0-1
        final_score = max(0.0, min(1.0, final_score))

        return AggregatedSignal(
            final_score=final_score,
            raw_score=raw_score,
            negative_discount=negative_score,
            market_boost=market_context_score,
            category_scores=category_scores,
            signals=signals,
        )

    def get_category(self, signal: Signal) -> str:
        """Map signal to its category string."""
        return signal.category.value

    def explain_score(self, aggregated: AggregatedSignal) -> Dict:
        """
        Generate human-readable explanation of score.

        Args:
            aggregated: Aggregated signal to explain

        Returns:
            Dict with explanation components
        """
        explanation = {
            "final_score": round(aggregated.final_score, 3),
            "raw_score": round(aggregated.raw_score, 3),
            "category_breakdown": {},
            "modifiers": {
                "negative_discount": f"-{aggregated.negative_discount * 50:.0f}%",
                "market_boost": f"+{(aggregated.market_boost - 1.0) * 100:.0f}%"
                if aggregated.market_boost > 1.0
                else f"{(aggregated.market_boost - 1.0) * 100:.0f}%",
            },
            "signal_count": len(aggregated.signals),
        }

        # Category breakdown
        for cat, score in aggregated.category_scores.items():
            weight = self.weights.get(cat, 0)
            contribution = score * weight
            explanation["category_breakdown"][cat] = {
                "score": round(score, 3),
                "weight": weight,
                "contribution": round(contribution, 3),
            }

        # Top signals
        top_signals = sorted(
            aggregated.signals,
            key=lambda s: s.confidence,
            reverse=True,
        )[:5]

        explanation["top_signals"] = [
            {
                "type": s.type.value,
                "category": s.category.value,
                "confidence": round(s.confidence, 3),
            }
            for s in top_signals
        ]

        return explanation
