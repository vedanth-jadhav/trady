from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class FundingSource(Enum):
    """Funding source classification with risk scores."""
    PRIVACY_TOOL = ("privacy_tool", 1.0)      # Critical - tornado cash etc
    TRACKED_WALLET = ("tracked_wallet", 0.8)  # High - from wallet we're watching
    CROSS_CHAIN = ("cross_chain", 0.6)        # Medium-High - bridge
    CEX_WITHDRAWAL = ("cex", 0.4)             # Medium - exchange withdrawal
    DIRECT_WALLET = ("direct", 0.2)           # Low - normal transfer
    UNKNOWN = ("unknown", 0.3)                # Medium-Low

    @property
    def name_str(self) -> str:
        return self.value[0]

    @property
    def risk_score(self) -> float:
        return self.value[1]


@dataclass
class FreshnessProfile:
    """Wallet freshness analysis results."""
    wallet: str
    freshness_score: float  # 0-1, higher = more fresh/suspicious
    is_zero_history: bool
    is_new_to_polymarket: bool
    is_recently_funded: bool
    first_polymarket_trade: Optional[datetime]  # None if no trades found
    total_polymarket_trades: int
    days_active: int


@dataclass
class FundingProfile:
    """Wallet funding source analysis results."""
    wallet: str
    primary_funding_source: FundingSource
    funding_sources: List[FundingSource]
    funding_risk_score: float  # 0-1, weighted by source priority
    total_funded_amount: float
    from_tracked_wallets: List[str]  # If funded from wallets we're tracking


@dataclass
class BehaviorProfile:
    """Wallet trading behavior profile."""
    wallet: str

    # Trading frequency
    avg_trades_per_day: float
    burst_episodes: int
    off_hours_ratio: float

    # Sizing
    avg_trade_size: float
    max_trade_size: float
    size_variance: float

    # Market diversity
    unique_markets: int
    market_concentration: float  # Herfindahl index
    niche_market_ratio: float

    # Profitability (if resolved markets available)
    total_pnl: Optional[float] = None
    win_rate: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # Exit patterns
    holds_to_resolution_ratio: float = 0.0
    early_exit_ratio: float = 0.0

    # Computed scores
    retail_likelihood: float = 0.5
    sophistication_score: float = 0.5


@dataclass
class WalletCluster:
    """A cluster of related wallets."""
    cluster_id: str
    wallets: List[str]
    primary_wallet: str  # Largest wallet in cluster
    clustering_method: str  # "funding", "behavior", "correlation", "merged"
    confidence: float
    total_volume: float
    combined_trades: int


@dataclass
class ClusteringResult:
    """Results from wallet clustering."""
    clusters: List[WalletCluster]
    wallet_to_cluster: Dict[str, str]  # Wallet -> cluster_id mapping
    unclustered_wallets: List[str]


@dataclass
class WalletProfile:
    """Complete wallet profile combining all analyses."""
    # Identity
    address: str
    cluster_id: Optional[str] = None
    cluster_members: List[str] = field(default_factory=list)

    # Basic stats
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_trades: int = 0
    total_volume: float = 0.0
    unique_markets: int = 0
    is_whale: bool = False

    # Freshness
    freshness_score: float = 0.0
    is_zero_history: bool = False
    is_new_to_polymarket: bool = False
    is_recently_funded: bool = False
    days_on_polymarket: int = 0

    # Funding
    primary_funding_source: str = "unknown"
    funding_risk_score: float = 0.0
    has_privacy_funding: bool = False
    from_tracked_wallets: List[str] = field(default_factory=list)

    # Behavior
    retail_likelihood: float = 0.5
    sophistication_score: float = 0.5
    holds_to_resolution_ratio: float = 0.0
    win_rate: Optional[float] = None
    off_hours_ratio: float = 0.0
    burst_episodes: int = 0
    avg_trade_size: float = 0.0
    max_trade_size: float = 0.0

    # Computed scores
    preliminary_insider_score: float = 0.0
    negative_signal_score: float = 0.0
