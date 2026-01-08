"""
Signal types and dataclasses for insider detection.

Defines the core signal types, aggregation structures, and output formats
for the signal detection pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class SignalCategory(Enum):
    """Signal category classification."""
    FRESHNESS = "freshness"
    TIMING = "timing"
    SIZING = "sizing"
    FUNDING = "funding"
    CLUSTER = "cluster"


class FreshnessSignalType(Enum):
    """Freshness-based signal types."""
    ZERO_HISTORY = "zero_history"
    NEW_WALLET = "new_wallet"
    RECENT_FUNDING = "recent_funding"
    LOW_ACTIVITY = "low_activity"


class TimingSignalType(Enum):
    """Timing-based signal types."""
    PRE_NEWS = "pre_news"
    OFF_HOURS = "off_hours"
    PRE_SPIKE = "pre_spike"
    RAPID_SUCCESSION = "rapid_succession"


class SizingSignalType(Enum):
    """Sizing-based signal types."""
    WHALE_SIZE = "whale_size"
    UNUSUAL_SIZE = "unusual_size"
    CONCENTRATED_POSITION = "concentrated_position"


class FundingSignalType(Enum):
    """Funding-based signal types."""
    PRIVACY_FUNDING = "privacy_funding"
    TRACKED_WALLET_FUNDING = "tracked_wallet_funding"
    BRIDGE_FUNDING = "bridge_funding"
    CEX_FUNDING = "cex_funding"


class ClusterSignalType(Enum):
    """Cluster-based signal types."""
    CLUSTER_COORDINATION = "cluster_coordination"
    CLUSTER_CONCENTRATION = "cluster_concentration"
    NEW_CLUSTER_MEMBER = "new_cluster_member"


# Union type for all signal types
SignalType = (
    FreshnessSignalType
    | TimingSignalType
    | SizingSignalType
    | FundingSignalType
    | ClusterSignalType
)


@dataclass
class Signal:
    """A single detected signal."""
    type: SignalType
    category: SignalCategory
    confidence: float  # 0-1, higher = more confident
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "details": self.details,
        }


@dataclass
class AggregatedSignal:
    """Aggregated signals with final score."""
    final_score: float  # 0-1, final insider likelihood
    raw_score: float  # Before discounts/boosts
    negative_discount: float  # Negative signal discount applied
    market_boost: float  # Market context boost applied
    category_scores: Dict[str, float] = field(default_factory=dict)
    signals: List[Signal] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "final_score": self.final_score,
            "raw_score": self.raw_score,
            "negative_discount": self.negative_discount,
            "market_boost": self.market_boost,
            "category_scores": self.category_scores,
            "signals": [s.to_dict() for s in self.signals],
        }


@dataclass
class Trade:
    """Trade data for signal detection."""
    trade_id: str
    market_id: str
    timestamp: datetime
    wallet: str  # The wallet we're analyzing (maker or taker)
    side: str  # "BUY" or "SELL"
    outcome: str
    size: float
    price: float
    notional: float
    tx_hash: Optional[str] = None

    @classmethod
    def from_row(cls, row, wallet_column: str = "maker_address") -> "Trade":
        """Create Trade from DataFrame row."""
        return cls(
            trade_id=str(row.get("trade_id", "")),
            market_id=str(row.get("market_id", "")),
            timestamp=row["timestamp"],
            wallet=str(row.get(wallet_column, "")),
            side=str(row.get("side", "")),
            outcome=str(row.get("outcome", "")),
            size=float(row.get("size", 0)),
            price=float(row.get("price", 0)),
            notional=float(row.get("notional", 0)),
            tx_hash=row.get("tx_hash"),
        )


@dataclass
class Market:
    """Market data for signal detection."""
    market_id: str
    question: str
    category: str
    volume: float
    liquidity: float
    is_resolved: bool
    resolution: Optional[str] = None
    resolution_timestamp: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "Market":
        """Create Market from DataFrame row."""
        return cls(
            market_id=str(row.get("market_id", "")),
            question=str(row.get("question", "")),
            category=str(row.get("category", "")),
            volume=float(row.get("volume", 0)),
            liquidity=float(row.get("liquidity", 0)),
            is_resolved=bool(row.get("is_resolved", False)),
            resolution=row.get("resolution"),
            resolution_timestamp=row.get("resolution_timestamp"),
        )


@dataclass
class TradeSignal:
    """Complete signal output for a trade."""
    trade: Trade
    wallet_profile: Any  # WalletProfile from analysis module
    signals: List[Signal]
    aggregated: AggregatedSignal
    final_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "trade_id": self.trade.trade_id,
            "market_id": self.trade.market_id,
            "wallet": self.trade.wallet,
            "timestamp": self.trade.timestamp,
            "side": self.trade.side,
            "outcome": self.trade.outcome,
            "size": self.trade.size,
            "price": self.trade.price,
            "notional": self.trade.notional,
            "final_score": self.final_score,
            "raw_score": self.aggregated.raw_score,
            "negative_discount": self.aggregated.negative_discount,
            "market_boost": self.aggregated.market_boost,
            "freshness_score": self.aggregated.category_scores.get("freshness", 0),
            "timing_score": self.aggregated.category_scores.get("timing", 0),
            "sizing_score": self.aggregated.category_scores.get("sizing", 0),
            "funding_score": self.aggregated.category_scores.get("funding", 0),
            "cluster_score": self.aggregated.category_scores.get("cluster", 0),
            "signals_json": self.aggregated.to_dict(),
        }
