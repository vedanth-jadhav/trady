from dataclasses import dataclass
from typing import Dict, List, Tuple
from src.signals.types import Trade

@dataclass
class ScoredTrade:
    """
    Final output of the scoring pipeline.
    """
    trade: Trade
    ml_score: float
    final_score: float
    confidence_tier: str
    features: Dict[str, float]
    top_features: List[Tuple[str, float]]
    
    def to_dict(self):
        return {
            "trade_id": self.trade.trade_id,
            "ml_score": self.ml_score,
            "final_score": self.final_score,
            "confidence_tier": self.confidence_tier,
            "top_features":  [{"name": f[0], "value": f[1]} for f in self.top_features],
        }
