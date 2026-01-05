# Phase 2: Wallet Analysis

## Objective

Build comprehensive wallet profiles that capture trading behavior, funding patterns, and relationships between wallets. This phase transforms raw wallet data into rich analytical profiles that feed the signal detection system.

---

## Scope

| Item | Details |
|------|---------|
| Input | `wallets.parquet`, `trades.parquet` from Phase 1 |
| Output | Enriched wallet profiles, wallet clusters |
| Key Analyses | Freshness, funding source, behavior patterns, clustering |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     WALLET ANALYSIS PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Wallet     │  From Phase 1                                 │
│  │   Index      │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Freshness  │    │   Funding    │    │   Behavior   │       │
│  │   Analyzer   │    │   Tracker    │    │   Profiler   │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └─────────┬─────────┴─────────┬─────────┘                │
│                   ▼                   ▼                          │
│            ┌──────────────┐    ┌──────────────┐                 │
│            │   Wallet     │    │   Wallet     │                 │
│            │   Profiles   │    │   Clusterer  │                 │
│            └──────────────┘    └──────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### 1. FreshnessAnalyzer

**Purpose**: Determine how "fresh" or new a wallet is based on multiple factors.

```python
class FreshnessAnalyzer:
    """
    Analyzes wallet freshness using multiple indicators.

    Freshness Factors:
    1. Zero history: Brand new wallet, first-ever transaction
    2. New to Polymarket: Existing wallet, first Polymarket trade
    3. Recently funded: Wallet received funds shortly before trading
    4. Account age: Time since wallet creation (if determinable)

    Each factor scored 0-1, combined into composite freshness score.
    """

    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df

    def compute_polymarket_age(self, wallet: str) -> Dict:
        """
        Compute wallet's Polymarket trading history.

        Returns:
            {
                "first_trade_date": datetime,
                "days_on_polymarket": int,
                "total_trades_before_current": int,
                "is_first_trade": bool,
            }
        """
        pass

    def compute_funding_recency(
        self,
        wallet: str,
        trade_timestamp: datetime
    ) -> Dict:
        """
        Check if wallet was recently funded before this trade.

        Requires on-chain data (may need external source).

        Returns:
            {
                "funding_timestamp": Optional[datetime],
                "hours_since_funding": Optional[float],
                "is_recently_funded": bool,  # < 24 hours
            }
        """
        pass

    def compute_freshness_score(
        self,
        wallet: str,
        trade_timestamp: datetime = None
    ) -> FreshnessProfile:
        """
        Compute composite freshness score.

        Score Weights:
        - Zero history: 0.4
        - New to Polymarket: 0.3
        - Recently funded: 0.2
        - Low trade count: 0.1

        Returns FreshnessProfile with score 0-1.
        """
        pass


@dataclass
class FreshnessProfile:
    wallet: str
    freshness_score: float  # 0-1, higher = more fresh/suspicious
    is_zero_history: bool
    is_new_to_polymarket: bool
    is_recently_funded: bool
    first_polymarket_trade: datetime
    total_polymarket_trades: int
    days_active: int
```

---

### 2. FundingTracker

**Purpose**: Track and categorize wallet funding sources.

```python
class FundingTracker:
    """
    Tracks funding sources for wallets.

    Funding Source Priority (for insider detection):
    1. Privacy tools (Tornado Cash, etc.) - CRITICAL
    2. From known tracked wallets - HIGH
    3. Cross-chain bridge - MEDIUM-HIGH
    4. CEX withdrawals - MEDIUM

    Note: Full funding analysis requires on-chain data beyond
    Polymarket API. This module provides hooks for integration
    with blockchain data providers (Etherscan, Alchemy, etc.)
    """

    # Known privacy tool contract addresses
    PRIVACY_CONTRACTS = {
        "0x...": "tornado_cash_eth",
        "0x...": "tornado_cash_100",
        # Add more as identified
    }

    # Known CEX hot wallet addresses
    CEX_WALLETS = {
        "0x...": "binance",
        "0x...": "coinbase",
        "0x...": "kraken",
        # Add more
    }

    # Known bridge contract addresses
    BRIDGE_CONTRACTS = {
        "0x...": "polygon_bridge",
        "0x...": "arbitrum_bridge",
        # Add more
    }

    def __init__(self, blockchain_client=None):
        """
        Args:
            blockchain_client: Optional client for on-chain queries
                              (Etherscan, Alchemy, etc.)
        """
        self.blockchain = blockchain_client

    def get_funding_transactions(
        self,
        wallet: str,
        before_timestamp: datetime = None
    ) -> List[FundingTx]:
        """
        Get funding transactions for a wallet.

        Returns list of incoming value transfers.
        """
        pass

    def classify_funding_source(
        self,
        from_address: str
    ) -> FundingSource:
        """
        Classify the source of funds.

        Returns:
            FundingSource enum with risk level
        """
        pass

    def compute_funding_profile(self, wallet: str) -> FundingProfile:
        """
        Build complete funding profile for wallet.
        """
        pass


class FundingSource(Enum):
    PRIVACY_TOOL = ("privacy_tool", 1.0)      # Critical
    TRACKED_WALLET = ("tracked_wallet", 0.8)  # High
    CROSS_CHAIN = ("cross_chain", 0.6)        # Medium-High
    CEX_WITHDRAWAL = ("cex", 0.4)             # Medium
    DIRECT_WALLET = ("direct", 0.2)           # Low
    UNKNOWN = ("unknown", 0.3)                # Medium-Low


@dataclass
class FundingProfile:
    wallet: str
    primary_funding_source: FundingSource
    funding_sources: List[FundingSource]
    funding_risk_score: float  # 0-1, weighted by source priority
    total_funded_amount: float
    funding_transactions: List[FundingTx]
    from_tracked_wallets: List[str]  # If funded from wallets we're tracking
```

---

### 3. BehaviorProfiler

**Purpose**: Analyze trading behavior patterns to distinguish retail from sophisticated traders.

```python
class BehaviorProfiler:
    """
    Profiles wallet trading behavior.

    Behavioral Indicators:
    - Trade frequency patterns
    - Position sizing consistency
    - Market diversity
    - Timing patterns (time of day, day of week)
    - Win rate and profitability
    - Exit behavior (early exit vs hold to resolution)
    """

    def __init__(self, trades_df: pd.DataFrame, markets_df: pd.DataFrame):
        self.trades = trades_df
        self.markets = markets_df

    def compute_trading_frequency(self, wallet: str) -> Dict:
        """
        Analyze trading frequency patterns.

        Returns:
            {
                "avg_trades_per_day": float,
                "trade_frequency_variance": float,
                "burst_trading_episodes": int,  # Rapid succession
                "typical_hours": List[int],     # Hours of day (UTC)
                "off_hours_ratio": float,       # % trades outside 9-5
            }
        """
        pass

    def compute_sizing_patterns(self, wallet: str) -> Dict:
        """
        Analyze position sizing behavior.

        Returns:
            {
                "avg_trade_size": float,
                "max_trade_size": float,
                "size_variance": float,
                "uses_round_numbers": bool,  # Retail indicator
                "consistent_sizing": bool,   # Same size each time
            }
        """
        pass

    def compute_market_diversity(self, wallet: str) -> Dict:
        """
        Analyze market selection patterns.

        Returns:
            {
                "unique_markets": int,
                "market_concentration": float,  # Herfindahl index
                "category_preference": Dict[str, float],
                "niche_market_ratio": float,  # % in low-volume markets
            }
        """
        pass

    def compute_profitability(self, wallet: str) -> Dict:
        """
        Analyze trading profitability.

        Requires market resolution data.

        Returns:
            {
                "total_pnl": float,
                "win_rate": float,
                "avg_profit_per_win": float,
                "avg_loss_per_loss": float,
                "sharpe_ratio": float,
                "best_trade_pnl": float,
                "worst_trade_pnl": float,
            }
        """
        pass

    def compute_exit_patterns(self, wallet: str) -> Dict:
        """
        Analyze how wallet exits positions.

        Key insight: Insiders hold to resolution, retail exits early.

        Returns:
            {
                "holds_to_resolution_ratio": float,  # % positions held
                "avg_exit_probability": float,       # At what prob they exit
                "early_exit_ratio": float,           # % exits before 90%
            }
        """
        pass

    def build_behavior_profile(self, wallet: str) -> BehaviorProfile:
        """
        Build complete behavior profile.
        """
        pass

    def compute_retail_likelihood(self, profile: BehaviorProfile) -> float:
        """
        Score how likely this is retail behavior (negative signal).

        High retail score = NOT an insider.
        """
        pass


@dataclass
class BehaviorProfile:
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
    market_concentration: float
    niche_market_ratio: float

    # Profitability
    total_pnl: float
    win_rate: float
    sharpe_ratio: float

    # Exit patterns
    holds_to_resolution_ratio: float
    early_exit_ratio: float

    # Computed scores
    retail_likelihood: float
    sophistication_score: float
```

---

### 4. WalletClusterer

**Purpose**: Identify related wallets that may belong to the same entity.

```python
class WalletClusterer:
    """
    Clusters related wallets using multiple methods.

    Clustering Methods:
    1. Funding Source Clustering: Same funding source
    2. Behavioral Clustering: Similar trading patterns
    3. Position Correlation: Correlated position changes

    Uses combination of heuristic rules and ML clustering.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        funding_profiles: Dict[str, FundingProfile],
        behavior_profiles: Dict[str, BehaviorProfile]
    ):
        self.trades = trades_df
        self.funding = funding_profiles
        self.behavior = behavior_profiles

    def cluster_by_funding(self) -> List[WalletCluster]:
        """
        Cluster wallets funded from the same source.

        Method:
        - Group wallets by funding source address
        - Wallets funded from same address likely same entity
        """
        pass

    def cluster_by_behavior(
        self,
        similarity_threshold: float = 0.8
    ) -> List[WalletCluster]:
        """
        Cluster wallets with similar trading behavior.

        Uses cosine similarity on behavior feature vectors:
        - Trading time patterns
        - Size patterns
        - Market selection overlap
        """
        pass

    def cluster_by_correlation(
        self,
        correlation_threshold: float = 0.7
    ) -> List[WalletCluster]:
        """
        Cluster wallets with correlated position changes.

        Method:
        - Build time series of position changes per wallet
        - Compute pairwise correlation matrix
        - Cluster highly correlated wallets
        """
        pass

    def merge_clusters(
        self,
        funding_clusters: List[WalletCluster],
        behavior_clusters: List[WalletCluster],
        correlation_clusters: List[WalletCluster]
    ) -> List[WalletCluster]:
        """
        Merge clusters from different methods.

        Uses union-find to combine overlapping clusters.
        """
        pass

    def run_full_clustering(self) -> ClusteringResult:
        """
        Run all clustering methods and merge results.
        """
        pass


@dataclass
class WalletCluster:
    cluster_id: str
    wallets: List[str]
    primary_wallet: str  # Largest wallet in cluster
    clustering_method: str  # "funding", "behavior", "correlation"
    confidence: float
    total_volume: float
    combined_trades: int


@dataclass
class ClusteringResult:
    clusters: List[WalletCluster]
    wallet_to_cluster: Dict[str, str]  # Wallet -> cluster_id mapping
    unclustered_wallets: List[str]
```

---

### 5. WalletProfileBuilder

**Purpose**: Combine all analyses into unified wallet profiles.

```python
class WalletProfileBuilder:
    """
    Combines all wallet analyses into unified profiles.

    Final profile includes:
    - Basic stats (from Phase 1)
    - Freshness analysis
    - Funding profile
    - Behavior profile
    - Cluster membership
    - Preliminary insider score
    """

    def __init__(
        self,
        freshness_analyzer: FreshnessAnalyzer,
        funding_tracker: FundingTracker,
        behavior_profiler: BehaviorProfiler,
        clusterer: WalletClusterer
    ):
        self.freshness = freshness_analyzer
        self.funding = funding_tracker
        self.behavior = behavior_profiler
        self.clusterer = clusterer

    def build_profile(self, wallet: str) -> WalletProfile:
        """
        Build complete profile for a single wallet.
        """
        pass

    def build_all_profiles(
        self,
        wallets: List[str],
        progress_callback: Callable = None
    ) -> Dict[str, WalletProfile]:
        """
        Build profiles for all wallets.
        """
        pass

    def compute_preliminary_insider_score(
        self,
        profile: WalletProfile
    ) -> float:
        """
        Compute preliminary insider likelihood score.

        This is a heuristic score before ML.
        Combines:
        - Freshness score (weight: 0.3)
        - Funding risk score (weight: 0.3)
        - Inverse retail likelihood (weight: 0.2)
        - Win rate (weight: 0.2)
        """
        pass


@dataclass
class WalletProfile:
    # Identity
    address: str
    cluster_id: Optional[str]
    cluster_members: List[str]

    # Basic stats
    first_seen: datetime
    last_seen: datetime
    total_trades: int
    total_volume: float
    unique_markets: int

    # Freshness
    freshness_score: float
    is_zero_history: bool
    is_new_to_polymarket: bool
    is_recently_funded: bool

    # Funding
    primary_funding_source: FundingSource
    funding_risk_score: float
    from_tracked_wallets: List[str]

    # Behavior
    retail_likelihood: float
    sophistication_score: float
    holds_to_resolution_ratio: float
    win_rate: float
    sharpe_ratio: float

    # Timing
    off_hours_ratio: float
    burst_episodes: int

    # Sizing
    avg_trade_size: float
    max_trade_size: float

    # Computed
    preliminary_insider_score: float
    is_whale: bool
```

---

## Negative Signal Detection

Wallets with these characteristics are LESS likely to be insiders:

```python
class NegativeSignalDetector:
    """
    Detects signals that DECREASE insider likelihood.
    Used to filter out false positives.
    """

    # Known public wallets (funds, market makers, etc.)
    KNOWN_ENTITIES = {
        "0x...": "market_maker_1",
        "0x...": "known_fund",
        # Populated from research
    }

    def is_known_entity(self, wallet: str) -> bool:
        """Check if wallet is a known public entity."""
        pass

    def has_long_history(
        self,
        profile: WalletProfile,
        threshold_days: int = 180
    ) -> bool:
        """Check if wallet has extensive trading history."""
        pass

    def exhibits_retail_behavior(
        self,
        profile: WalletProfile,
        threshold: float = 0.7
    ) -> bool:
        """Check if wallet shows clear retail patterns."""
        pass

    def exits_early_frequently(
        self,
        profile: WalletProfile,
        threshold: float = 0.5
    ) -> bool:
        """Check if wallet frequently exits before resolution."""
        pass

    def compute_negative_signal_score(
        self,
        profile: WalletProfile
    ) -> float:
        """
        Compute aggregate negative signal score.

        Higher score = LESS likely to be insider.
        """
        pass
```

---

## Output Schemas

### `wallet_profiles.parquet`

```python
{
    "address": str,
    "cluster_id": Optional[str],

    # Basic
    "first_seen": datetime,
    "last_seen": datetime,
    "total_trades": int,
    "total_volume": float,
    "unique_markets": int,
    "is_whale": bool,

    # Freshness
    "freshness_score": float,
    "is_zero_history": bool,
    "is_new_to_polymarket": bool,
    "is_recently_funded": bool,
    "days_on_polymarket": int,

    # Funding
    "primary_funding_source": str,
    "funding_risk_score": float,
    "has_privacy_funding": bool,

    # Behavior
    "retail_likelihood": float,
    "sophistication_score": float,
    "holds_to_resolution_ratio": float,
    "win_rate": float,
    "off_hours_ratio": float,
    "burst_episodes": int,
    "avg_trade_size": float,
    "max_trade_size": float,

    # Scores
    "preliminary_insider_score": float,
    "negative_signal_score": float,
}
```

### `wallet_clusters.parquet`

```python
{
    "cluster_id": str,
    "wallets": List[str],
    "primary_wallet": str,
    "wallet_count": int,
    "clustering_method": str,
    "confidence": float,
    "total_volume": float,
    "combined_trades": int,
}
```

---

## CLI Interface

```python
# src/analysis/cli.py

@app.command()
def analyze_freshness(
    trades_file: Path,
    output_file: Path
):
    """Analyze wallet freshness."""
    pass

@app.command()
def analyze_funding(
    wallets_file: Path,
    output_file: Path,
    blockchain_api_key: str = None
):
    """Analyze wallet funding sources."""
    pass

@app.command()
def analyze_behavior(
    trades_file: Path,
    markets_file: Path,
    output_file: Path
):
    """Build behavior profiles."""
    pass

@app.command()
def cluster_wallets(
    profiles_file: Path,
    output_file: Path
):
    """Run wallet clustering."""
    pass

@app.command()
def build_all_profiles(
    trades_file: Path,
    markets_file: Path,
    wallets_file: Path,
    output_dir: Path
):
    """Run complete wallet analysis pipeline."""
    pass
```

---

## Dependencies

```python
# Additional requirements for Phase 2

scikit-learn>=1.3.0     # For clustering algorithms
scipy>=1.11.0           # Statistical functions
networkx>=3.2           # Graph-based clustering
web3>=6.0.0             # Ethereum interaction (optional)
```

---

## Success Criteria

- [ ] Freshness scores computed for all wallets
- [ ] Funding profiles for whale wallets (may be partial without blockchain data)
- [ ] Behavior profiles for all wallets with 3+ trades
- [ ] Wallet clusters identified with >80% precision
- [ ] Preliminary insider scores computed
- [ ] Negative signals properly filter obvious non-insiders

---

## Next Phase

After completing Phase 2, proceed to:
→ **Phase 3: Signal Detection** (`03_signal_detection/spec.md`)

Wallet profiles will be used to:
- Detect suspicious trade patterns in real-time
- Score individual trades for insider likelihood
- Track wallet behavior changes over time
