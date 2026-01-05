# Phase 3: Signal Detection

## Objective

Build a comprehensive signal detection system that identifies suspicious trading patterns indicating potential insider activity. This phase transforms wallet profiles and trade data into actionable trading signals.

---

## Scope

| Item | Details |
|------|---------|
| Input | `wallet_profiles.parquet`, `trades.parquet`, `markets.parquet` |
| Output | Trade-level signals with confidence scores |
| Signal Categories | Freshness, Timing, Sizing, Funding, Clustering |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIGNAL DETECTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Trade      │  Individual trade from data                   │
│  │   Event      │                                               │
│  └──────┬───────┘                                               │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              SIGNAL DETECTORS (Parallel)              │       │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │       │
│  │  │Freshness│ │ Timing  │ │ Sizing  │ │ Funding │    │       │
│  │  │Detector │ │Detector │ │Detector │ │Detector │    │       │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │       │
│  │       │           │           │           │          │       │
│  │  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐    │       │
│  │  │Cluster  │ │ Market  │ │Behavior │ │Negative │    │       │
│  │  │Detector │ │Context  │ │Anomaly  │ │ Filter  │    │       │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │       │
│  └───────┼───────────┼───────────┼───────────┼──────────┘       │
│          │           │           │           │                   │
│          └─────────┬─┴───────────┴───────────┘                   │
│                    ▼                                             │
│             ┌──────────────┐                                    │
│             │   Signal     │                                    │
│             │  Aggregator  │                                    │
│             └──────┬───────┘                                    │
│                    │                                             │
│                    ▼                                             │
│             ┌──────────────┐                                    │
│             │   Signal     │  Ranked list of signals            │
│             │   Output     │                                    │
│             └──────────────┘                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Signal Categories

### 1. Freshness Signals

Detect trades from suspiciously new or fresh wallets.

```python
class FreshnessSignalDetector:
    """
    Detects freshness-based insider signals.

    Signals:
    - ZERO_HISTORY: First-ever trade on Polymarket
    - NEW_WALLET: Wallet created recently
    - RECENT_FUNDING: Funded shortly before trade
    - LOW_ACTIVITY: Very few prior trades
    """

    class SignalType(Enum):
        ZERO_HISTORY = "zero_history"
        NEW_WALLET = "new_wallet"
        RECENT_FUNDING = "recent_funding"
        LOW_ACTIVITY = "low_activity"

    def __init__(self, wallet_profiles: Dict[str, WalletProfile]):
        self.profiles = wallet_profiles

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile
    ) -> List[Signal]:
        """
        Detect freshness signals for a trade.

        Returns list of signals with confidence scores.
        """
        signals = []

        # Zero history: First trade ever
        if wallet_profile.total_trades == 1:
            signals.append(Signal(
                type=self.SignalType.ZERO_HISTORY,
                confidence=1.0,
                details={"first_trade": True}
            ))

        # New wallet: Less than 7 days on platform
        if wallet_profile.days_on_polymarket < 7:
            confidence = 1.0 - (wallet_profile.days_on_polymarket / 7)
            signals.append(Signal(
                type=self.SignalType.NEW_WALLET,
                confidence=confidence,
                details={"days_active": wallet_profile.days_on_polymarket}
            ))

        # Recent funding: Funded within 24 hours of trade
        if wallet_profile.is_recently_funded:
            signals.append(Signal(
                type=self.SignalType.RECENT_FUNDING,
                confidence=0.8,
                details={"recently_funded": True}
            ))

        # Low activity: Less than 5 prior trades
        if wallet_profile.total_trades < 5:
            confidence = 1.0 - (wallet_profile.total_trades / 5)
            signals.append(Signal(
                type=self.SignalType.LOW_ACTIVITY,
                confidence=confidence * 0.7,  # Lower weight
                details={"total_trades": wallet_profile.total_trades}
            ))

        return signals
```

---

### 2. Timing Signals

Detect trades with suspicious timing patterns.

```python
class TimingSignalDetector:
    """
    Detects timing-based insider signals.

    Signals:
    - PRE_NEWS: Trade shortly before news/resolution
    - OFF_HOURS: Trade during off-peak hours
    - PRE_SPIKE: Trade before unusual price movement
    - RAPID_SUCCESSION: Multiple trades in quick burst
    """

    class SignalType(Enum):
        PRE_NEWS = "pre_news"
        OFF_HOURS = "off_hours"
        PRE_SPIKE = "pre_spike"
        RAPID_SUCCESSION = "rapid_succession"

    # Off-hours defined as outside 9am-9pm ET
    OFF_HOURS_START = 21  # 9pm ET = 01:00 UTC next day
    OFF_HOURS_END = 13    # 9am ET = 13:00 UTC

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame
    ):
        self.trades = trades_df
        self.markets = markets_df
        self._precompute_price_spikes()

    def _precompute_price_spikes(self):
        """
        Identify all significant price movements.

        A spike is defined as >10% price change in 1 hour.
        """
        self.price_spikes = {}  # market_id -> List[spike_timestamps]
        pass

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
        market: Market
    ) -> List[Signal]:
        """
        Detect timing signals for a trade.
        """
        signals = []

        # Pre-news: Trade within 24h before resolution
        if market.resolution_timestamp:
            hours_before = (market.resolution_timestamp - trade.timestamp).total_seconds() / 3600
            if 0 < hours_before < 24:
                confidence = 1.0 - (hours_before / 24)
                signals.append(Signal(
                    type=self.SignalType.PRE_NEWS,
                    confidence=confidence,
                    details={"hours_before_resolution": hours_before}
                ))

        # Off-hours trading
        trade_hour = trade.timestamp.hour
        if self._is_off_hours(trade_hour):
            signals.append(Signal(
                type=self.SignalType.OFF_HOURS,
                confidence=0.5,  # Moderate signal
                details={"trade_hour_utc": trade_hour}
            ))

        # Pre-spike: Trade before significant price movement
        if self._is_before_spike(trade, market):
            signals.append(Signal(
                type=self.SignalType.PRE_SPIKE,
                confidence=0.9,
                details={"spike_detected": True}
            ))

        # Rapid succession: Part of burst trading
        if self._is_rapid_succession(trade, wallet_profile):
            signals.append(Signal(
                type=self.SignalType.RAPID_SUCCESSION,
                confidence=0.6,
                details={"burst_trade": True}
            ))

        return signals

    def _is_off_hours(self, hour_utc: int) -> bool:
        """Check if hour is outside peak trading times."""
        return hour_utc >= self.OFF_HOURS_START or hour_utc < self.OFF_HOURS_END

    def _is_before_spike(
        self,
        trade: Trade,
        market: Market,
        lookforward_hours: int = 4
    ) -> bool:
        """Check if trade occurred before a price spike."""
        pass

    def _is_rapid_succession(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
        window_minutes: int = 10
    ) -> bool:
        """Check if trade is part of rapid succession."""
        pass
```

---

### 3. Sizing Signals

Detect trades with unusual position sizes relative to market.

```python
class SizingSignalDetector:
    """
    Detects sizing-based insider signals.

    Uses DYNAMIC thresholds based on market context.

    Signals:
    - WHALE_SIZE: Trade is large % of market volume
    - UNUSUAL_SIZE: Trade is unusual for this wallet
    - CONCENTRATED_POSITION: Building large position
    """

    class SignalType(Enum):
        WHALE_SIZE = "whale_size"
        UNUSUAL_SIZE = "unusual_size"
        CONCENTRATED_POSITION = "concentrated_position"

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame
    ):
        self.trades = trades_df
        self.markets = markets_df
        self._compute_market_baselines()

    def _compute_market_baselines(self):
        """
        Compute baseline trade sizes for each market.

        Metrics:
        - Median trade size
        - 95th percentile trade size
        - Total volume
        """
        self.market_baselines = {}
        for market_id, group in self.trades.groupby("market_id"):
            self.market_baselines[market_id] = {
                "median_size": group["notional"].median(),
                "p95_size": group["notional"].quantile(0.95),
                "total_volume": group["notional"].sum(),
            }

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
        market: Market
    ) -> List[Signal]:
        """
        Detect sizing signals for a trade.
        """
        signals = []
        baseline = self.market_baselines.get(trade.market_id, {})

        # Whale size: Trade is >5% of market volume
        if baseline.get("total_volume", 0) > 0:
            pct_of_volume = trade.notional / baseline["total_volume"]
            if pct_of_volume > 0.05:
                signals.append(Signal(
                    type=self.SignalType.WHALE_SIZE,
                    confidence=min(pct_of_volume * 10, 1.0),
                    details={
                        "pct_of_volume": pct_of_volume,
                        "trade_size": trade.notional,
                        "market_volume": baseline["total_volume"]
                    }
                ))

        # Unusual size: Trade is >3x wallet's average
        if wallet_profile.avg_trade_size > 0:
            size_ratio = trade.notional / wallet_profile.avg_trade_size
            if size_ratio > 3:
                signals.append(Signal(
                    type=self.SignalType.UNUSUAL_SIZE,
                    confidence=min((size_ratio - 3) / 7, 1.0),  # Scale 3-10x to 0-1
                    details={
                        "size_ratio": size_ratio,
                        "trade_size": trade.notional,
                        "avg_size": wallet_profile.avg_trade_size
                    }
                ))

        # Concentrated position: Check if building large position in market
        wallet_positions = self._get_wallet_market_positions(
            wallet_profile.address,
            trade.market_id
        )
        if wallet_positions:
            total_position = sum(p.notional for p in wallet_positions)
            if total_position > baseline.get("p95_size", float("inf")):
                signals.append(Signal(
                    type=self.SignalType.CONCENTRATED_POSITION,
                    confidence=0.7,
                    details={"total_position": total_position}
                ))

        return signals

    def _get_wallet_market_positions(
        self,
        wallet: str,
        market_id: str
    ) -> List[Trade]:
        """Get all trades by wallet in this market."""
        pass
```

---

### 4. Funding Signals

Detect trades from wallets with suspicious funding patterns.

```python
class FundingSignalDetector:
    """
    Detects funding-based insider signals.

    Signals based on funding source priority:
    - PRIVACY_FUNDING: Funded via privacy tools (CRITICAL)
    - TRACKED_WALLET_FUNDING: Funded from another tracked wallet
    - BRIDGE_FUNDING: Funded via cross-chain bridge
    - CEX_FUNDING: Funded from centralized exchange
    """

    class SignalType(Enum):
        PRIVACY_FUNDING = "privacy_funding"
        TRACKED_WALLET_FUNDING = "tracked_wallet_funding"
        BRIDGE_FUNDING = "bridge_funding"
        CEX_FUNDING = "cex_funding"

    # Signal confidence by funding type
    FUNDING_CONFIDENCE = {
        FundingSource.PRIVACY_TOOL: 1.0,      # Critical
        FundingSource.TRACKED_WALLET: 0.8,    # High
        FundingSource.CROSS_CHAIN: 0.6,       # Medium-High
        FundingSource.CEX_WITHDRAWAL: 0.4,    # Medium
    }

    def __init__(self, wallet_profiles: Dict[str, WalletProfile]):
        self.profiles = wallet_profiles

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile
    ) -> List[Signal]:
        """
        Detect funding signals for a trade.
        """
        signals = []
        source = wallet_profile.primary_funding_source

        if source == FundingSource.PRIVACY_TOOL:
            signals.append(Signal(
                type=self.SignalType.PRIVACY_FUNDING,
                confidence=1.0,
                details={"funding_source": "privacy_tool"}
            ))

        elif source == FundingSource.TRACKED_WALLET:
            signals.append(Signal(
                type=self.SignalType.TRACKED_WALLET_FUNDING,
                confidence=0.8,
                details={
                    "funding_source": "tracked_wallet",
                    "source_wallets": wallet_profile.from_tracked_wallets
                }
            ))

        elif source == FundingSource.CROSS_CHAIN:
            signals.append(Signal(
                type=self.SignalType.BRIDGE_FUNDING,
                confidence=0.6,
                details={"funding_source": "bridge"}
            ))

        elif source == FundingSource.CEX_WITHDRAWAL:
            signals.append(Signal(
                type=self.SignalType.CEX_FUNDING,
                confidence=0.4,
                details={"funding_source": "cex"}
            ))

        return signals
```

---

### 5. Cluster Signals

Detect coordinated activity across related wallets.

```python
class ClusterSignalDetector:
    """
    Detects signals based on wallet cluster behavior.

    Signals:
    - CLUSTER_COORDINATION: Multiple cluster members trading same market
    - CLUSTER_CONCENTRATION: Cluster building concentrated position
    - NEW_CLUSTER_MEMBER: Trade from newly identified cluster member
    """

    class SignalType(Enum):
        CLUSTER_COORDINATION = "cluster_coordination"
        CLUSTER_CONCENTRATION = "cluster_concentration"
        NEW_CLUSTER_MEMBER = "new_cluster_member"

    def __init__(
        self,
        wallet_profiles: Dict[str, WalletProfile],
        clusters: List[WalletCluster],
        trades_df: pd.DataFrame
    ):
        self.profiles = wallet_profiles
        self.clusters = clusters
        self.trades = trades_df
        self.wallet_to_cluster = self._build_lookup()

    def _build_lookup(self) -> Dict[str, WalletCluster]:
        """Build wallet -> cluster lookup."""
        pass

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile
    ) -> List[Signal]:
        """
        Detect cluster signals for a trade.
        """
        signals = []

        cluster = self.wallet_to_cluster.get(wallet_profile.address)
        if not cluster:
            return signals

        # Cluster coordination: Other members also traded this market
        other_members = [w for w in cluster.wallets if w != wallet_profile.address]
        same_market_trades = self._get_cluster_market_activity(
            other_members,
            trade.market_id,
            window_hours=48
        )

        if len(same_market_trades) >= 2:
            signals.append(Signal(
                type=self.SignalType.CLUSTER_COORDINATION,
                confidence=min(len(same_market_trades) * 0.3, 1.0),
                details={
                    "cluster_id": cluster.cluster_id,
                    "coordinated_trades": len(same_market_trades),
                    "other_members": other_members
                }
            ))

        # Cluster concentration: Total cluster position is large
        cluster_position = self._get_cluster_market_position(
            cluster.wallets,
            trade.market_id
        )
        if cluster_position > 10000:  # $10k+
            signals.append(Signal(
                type=self.SignalType.CLUSTER_CONCENTRATION,
                confidence=min(cluster_position / 50000, 1.0),
                details={
                    "cluster_id": cluster.cluster_id,
                    "total_position": cluster_position
                }
            ))

        return signals

    def _get_cluster_market_activity(
        self,
        wallets: List[str],
        market_id: str,
        window_hours: int
    ) -> List[Trade]:
        """Get trades by cluster members in market."""
        pass

    def _get_cluster_market_position(
        self,
        wallets: List[str],
        market_id: str
    ) -> float:
        """Get total cluster position in market."""
        pass
```

---

### 6. Market Context Analyzer

Determine if market is "high-insider" category.

```python
class MarketContextAnalyzer:
    """
    Analyzes market context for insider likelihood.

    High-insider categories:
    - Political events (elections, appointments)
    - Corporate events (earnings, M&A)
    - Regulatory decisions

    Low-insider categories:
    - Sports outcomes
    - Weather events
    - Random events
    """

    HIGH_INSIDER_CATEGORIES = [
        "Politics",
        "Business",
        "Crypto",  # Insider trading common
        "Science",  # Research announcements
    ]

    LOW_INSIDER_CATEGORIES = [
        "Sports",
        "Entertainment",
        "Weather",
    ]

    def __init__(self, markets_df: pd.DataFrame):
        self.markets = markets_df

    def get_insider_category_score(self, market: Market) -> float:
        """
        Score market's insider likelihood based on category.

        Returns 0-1, higher = more likely to have insider activity.
        """
        if market.category in self.HIGH_INSIDER_CATEGORIES:
            return 0.8
        elif market.category in self.LOW_INSIDER_CATEGORIES:
            return 0.2
        else:
            return 0.5

    def get_market_characteristics(self, market: Market) -> Dict:
        """
        Analyze market characteristics.

        Returns:
        - Is niche market (low participation)
        - Near resolution
        - High/low liquidity
        """
        pass
```

---

### 7. Negative Signal Filter

Filter out obvious non-insiders.

```python
class NegativeSignalFilter:
    """
    Filters signals based on negative indicators.

    Negative signals (DECREASE insider likelihood):
    - Long trading history
    - Known entity
    - Retail-like behavior
    - Frequent early exits
    """

    def __init__(self, wallet_profiles: Dict[str, WalletProfile]):
        self.profiles = wallet_profiles

    def compute_negative_score(self, wallet_profile: WalletProfile) -> float:
        """
        Compute negative signal score (0-1).

        Higher = MORE likely to be non-insider.
        """
        score = 0.0

        # Long history: >6 months on platform
        if wallet_profile.days_on_polymarket > 180:
            score += 0.3

        # High retail likelihood
        if wallet_profile.retail_likelihood > 0.7:
            score += 0.3

        # Frequent early exits (non-insider behavior)
        if wallet_profile.early_exit_ratio > 0.5:
            score += 0.2

        # Known entity (from whitelist)
        if self._is_known_entity(wallet_profile.address):
            score += 0.2

        return min(score, 1.0)

    def should_filter(
        self,
        wallet_profile: WalletProfile,
        threshold: float = 0.7
    ) -> bool:
        """Check if wallet should be filtered out."""
        return self.compute_negative_score(wallet_profile) > threshold

    def _is_known_entity(self, wallet: str) -> bool:
        """Check against known entities list."""
        pass
```

---

### 8. Signal Aggregator

Combine all signals into final score.

```python
class SignalAggregator:
    """
    Aggregates all signals into composite insider score.

    Weighting Strategy:
    - Freshness signals: 0.25
    - Timing signals: 0.20
    - Sizing signals: 0.20
    - Funding signals: 0.25
    - Cluster signals: 0.10

    Applies negative signal discount.
    """

    CATEGORY_WEIGHTS = {
        "freshness": 0.25,
        "timing": 0.20,
        "sizing": 0.20,
        "funding": 0.25,
        "cluster": 0.10,
    }

    def __init__(self):
        pass

    def aggregate(
        self,
        signals: List[Signal],
        negative_score: float,
        market_context_score: float
    ) -> AggregatedSignal:
        """
        Aggregate all signals into final score.

        Steps:
        1. Group signals by category
        2. Take max confidence per category
        3. Apply category weights
        4. Discount by negative score
        5. Boost by market context
        """
        # Group by category
        category_scores = {}
        for signal in signals:
            category = self._get_category(signal)
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(signal.confidence)

        # Max per category
        max_scores = {
            cat: max(scores) for cat, scores in category_scores.items()
        }

        # Weighted sum
        raw_score = sum(
            max_scores.get(cat, 0) * weight
            for cat, weight in self.CATEGORY_WEIGHTS.items()
        )

        # Apply negative discount (1 - negative_score)
        discounted_score = raw_score * (1 - negative_score * 0.5)

        # Apply market context boost
        final_score = discounted_score * (0.5 + market_context_score * 0.5)

        return AggregatedSignal(
            final_score=final_score,
            raw_score=raw_score,
            negative_discount=negative_score,
            market_boost=market_context_score,
            category_scores=max_scores,
            signals=signals
        )

    def _get_category(self, signal: Signal) -> str:
        """Map signal type to category."""
        pass


@dataclass
class AggregatedSignal:
    final_score: float         # 0-1, final insider likelihood
    raw_score: float           # Before discounts/boosts
    negative_discount: float   # Negative signal discount applied
    market_boost: float        # Market context boost applied
    category_scores: Dict[str, float]
    signals: List[Signal]
```

---

## Main Signal Detector

```python
class InsiderSignalDetector:
    """
    Main orchestrator for signal detection.

    Combines all detectors and produces trade-level signals.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        wallet_profiles: Dict[str, WalletProfile],
        clusters: List[WalletCluster]
    ):
        # Initialize all detectors
        self.freshness_detector = FreshnessSignalDetector(wallet_profiles)
        self.timing_detector = TimingSignalDetector(trades_df, markets_df)
        self.sizing_detector = SizingSignalDetector(trades_df, markets_df)
        self.funding_detector = FundingSignalDetector(wallet_profiles)
        self.cluster_detector = ClusterSignalDetector(
            wallet_profiles, clusters, trades_df
        )
        self.market_analyzer = MarketContextAnalyzer(markets_df)
        self.negative_filter = NegativeSignalFilter(wallet_profiles)
        self.aggregator = SignalAggregator()

        self.wallet_profiles = wallet_profiles
        self.markets = markets_df.set_index("market_id")

    def detect_signals(self, trade: Trade) -> TradeSignal:
        """
        Detect all signals for a single trade.
        """
        wallet_profile = self.wallet_profiles.get(trade.wallet)
        market = self.markets.loc[trade.market_id]

        if not wallet_profile:
            return TradeSignal(trade=trade, signals=[], final_score=0)

        # Collect signals from all detectors
        all_signals = []

        all_signals.extend(
            self.freshness_detector.detect(trade, wallet_profile)
        )
        all_signals.extend(
            self.timing_detector.detect(trade, wallet_profile, market)
        )
        all_signals.extend(
            self.sizing_detector.detect(trade, wallet_profile, market)
        )
        all_signals.extend(
            self.funding_detector.detect(trade, wallet_profile)
        )
        all_signals.extend(
            self.cluster_detector.detect(trade, wallet_profile)
        )

        # Get negative score
        negative_score = self.negative_filter.compute_negative_score(
            wallet_profile
        )

        # Get market context
        market_context = self.market_analyzer.get_insider_category_score(market)

        # Aggregate
        aggregated = self.aggregator.aggregate(
            all_signals, negative_score, market_context
        )

        return TradeSignal(
            trade=trade,
            wallet_profile=wallet_profile,
            signals=all_signals,
            aggregated=aggregated,
            final_score=aggregated.final_score
        )

    def detect_all_signals(
        self,
        trades: List[Trade],
        min_score: float = 0.3,
        progress_callback: Callable = None
    ) -> List[TradeSignal]:
        """
        Detect signals for all trades.

        Args:
            min_score: Only return signals above this threshold

        Returns:
            List of TradeSignals sorted by final_score descending
        """
        results = []
        for i, trade in enumerate(trades):
            signal = self.detect_signals(trade)
            if signal.final_score >= min_score:
                results.append(signal)

            if progress_callback and i % 1000 == 0:
                progress_callback(i, len(trades))

        return sorted(results, key=lambda x: x.final_score, reverse=True)


@dataclass
class TradeSignal:
    trade: Trade
    wallet_profile: WalletProfile
    signals: List[Signal]
    aggregated: AggregatedSignal
    final_score: float
```

---

## Output Schema

### `signals.parquet`

```python
{
    "trade_id": str,
    "market_id": str,
    "wallet": str,
    "timestamp": datetime,
    "side": str,
    "outcome": str,
    "size": float,
    "price": float,

    # Signal scores
    "final_score": float,
    "raw_score": float,
    "negative_discount": float,
    "market_boost": float,

    # Category scores
    "freshness_score": float,
    "timing_score": float,
    "sizing_score": float,
    "funding_score": float,
    "cluster_score": float,

    # Individual signals (JSON)
    "signals_json": str,

    # Wallet context
    "wallet_cluster_id": Optional[str],
    "wallet_is_whale": bool,
    "wallet_freshness": float,
    "wallet_retail_likelihood": float,
}
```

---

## CLI Interface

```python
@app.command()
def detect_signals(
    trades_file: Path,
    markets_file: Path,
    profiles_file: Path,
    clusters_file: Path,
    output_file: Path,
    min_score: float = 0.3
):
    """Run signal detection on all trades."""
    pass

@app.command()
def analyze_signal(
    trade_id: str,
    trades_file: Path,
    profiles_file: Path
):
    """Analyze signals for a specific trade (debugging)."""
    pass

@app.command()
def calibrate_weights(
    signals_file: Path,
    labels_file: Path,
    output_file: Path
):
    """Calibrate signal weights based on labeled data."""
    pass
```

---

## Success Criteria

- [ ] All 5 signal categories implemented and tested
- [ ] Signals properly weighted by category
- [ ] Negative signals filter out obvious non-insiders
- [ ] Market context applied correctly
- [ ] Signal output includes all required fields
- [ ] Performance: Process 100K trades in <5 minutes

---

## Next Phase

After completing Phase 3, proceed to:
→ **Phase 4: ML Scoring** (`04_ml_scoring/spec.md`)

Rule-based signals will be used as features for XGBoost model training.
