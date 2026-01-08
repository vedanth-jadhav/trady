"""
Tests for signal detection module.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.analysis import WalletCluster, WalletProfile
from src.signals import (
    AggregatedSignal,
    ClusterSignalDetector,
    FreshnessSignalDetector,
    FundingSignalDetector,
    InsiderSignalDetector,
    Market,
    MarketContextAnalyzer,
    NegativeSignalFilter,
    Signal,
    SignalAggregator,
    SignalCategory,
    SizingSignalDetector,
    TimingSignalDetector,
    Trade,
    FreshnessSignalType,
    TimingSignalType,
    SizingSignalType,
)


# Test fixtures
@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
        trade_id="test_trade_1",
        market_id="test_market_1",
        timestamp=datetime.now(timezone.utc),
        wallet="0x1234567890abcdef",
        side="BUY",
        outcome="Yes",
        size=100.0,
        price=0.65,
        notional=65.0,
    )


@pytest.fixture
def sample_wallet_profile():
    """Create a sample wallet profile for testing."""
    return WalletProfile(
        address="0x1234567890abcdef",
        total_trades=5,
        total_volume=1000.0,
        unique_markets=3,
        is_whale=False,
        freshness_score=0.8,
        is_zero_history=False,
        is_new_to_polymarket=True,
        is_recently_funded=True,
        days_on_polymarket=3,
        primary_funding_source="cex",
        funding_risk_score=0.4,
        has_privacy_funding=False,
        retail_likelihood=0.3,
        sophistication_score=0.7,
        avg_trade_size=50.0,
        max_trade_size=100.0,
    )


@pytest.fixture
def sample_market():
    """Create a sample market for testing."""
    return Market(
        market_id="test_market_1",
        question="Will Bitcoin reach $100k by end of year?",
        category="Crypto",
        volume=50000.0,
        liquidity=10000.0,
        is_resolved=False,
    )


@pytest.fixture
def sample_trades_df():
    """Create a sample trades DataFrame."""
    now = datetime.now(timezone.utc)
    return pd.DataFrame({
        "trade_id": ["t1", "t2", "t3", "t4", "t5"],
        "market_id": ["m1", "m1", "m1", "m2", "m2"],
        "timestamp": [
            now - timedelta(hours=5),
            now - timedelta(hours=4),
            now - timedelta(hours=3),
            now - timedelta(hours=2),
            now - timedelta(hours=1),
        ],
        "maker_address": ["0xaaa", "0xbbb", "0xaaa", "0xccc", "0xaaa"],
        "taker_address": ["0xddd", "0xaaa", "0xeee", "0xaaa", "0xfff"],
        "side": ["BUY", "SELL", "BUY", "BUY", "SELL"],
        "outcome": ["Yes", "No", "Yes", "Yes", "No"],
        "size": [100, 200, 150, 300, 50],
        "price": [0.5, 0.6, 0.55, 0.7, 0.3],
        "notional": [50, 120, 82.5, 210, 15],
    })


@pytest.fixture
def sample_markets_df():
    """Create a sample markets DataFrame."""
    return pd.DataFrame({
        "market_id": ["m1", "m2"],
        "question": ["Will event A happen?", "Will event B happen?"],
        "category": ["Politics", "Sports"],
        "volume": [100000, 50000],
        "liquidity": [20000, 10000],
        "is_resolved": [False, False],
    })


class TestFreshnessSignalDetector:
    """Tests for FreshnessSignalDetector."""

    def test_detect_zero_history(self, sample_trade):
        """Test detection of zero history signal."""
        profile = WalletProfile(
            address=sample_trade.wallet,
            total_trades=1,
            days_on_polymarket=0,
        )
        profiles = {sample_trade.wallet: profile}

        detector = FreshnessSignalDetector(profiles)
        signals = detector.detect(sample_trade, profile)

        # Should detect ZERO_HISTORY
        signal_types = [s.type for s in signals]
        assert FreshnessSignalType.ZERO_HISTORY in signal_types

    def test_detect_new_wallet(self, sample_trade):
        """Test detection of new wallet signal."""
        profile = WalletProfile(
            address=sample_trade.wallet,
            total_trades=10,
            days_on_polymarket=3,
        )
        profiles = {sample_trade.wallet: profile}

        detector = FreshnessSignalDetector(profiles)
        signals = detector.detect(sample_trade, profile)

        # Should detect NEW_WALLET
        signal_types = [s.type for s in signals]
        assert FreshnessSignalType.NEW_WALLET in signal_types

    def test_detect_recent_funding(self, sample_trade):
        """Test detection of recent funding signal."""
        profile = WalletProfile(
            address=sample_trade.wallet,
            total_trades=10,
            days_on_polymarket=30,
            is_recently_funded=True,
        )
        profiles = {sample_trade.wallet: profile}

        detector = FreshnessSignalDetector(profiles)
        signals = detector.detect(sample_trade, profile)

        # Should detect RECENT_FUNDING
        signal_types = [s.type for s in signals]
        assert FreshnessSignalType.RECENT_FUNDING in signal_types

    def test_no_signals_for_old_wallet(self, sample_trade):
        """Test that old active wallets don't trigger freshness signals."""
        profile = WalletProfile(
            address=sample_trade.wallet,
            total_trades=100,
            days_on_polymarket=180,
            is_recently_funded=False,
        )
        profiles = {sample_trade.wallet: profile}

        detector = FreshnessSignalDetector(profiles)
        signals = detector.detect(sample_trade, profile)

        # Should not detect any freshness signals
        assert len(signals) == 0


class TestTimingSignalDetector:
    """Tests for TimingSignalDetector."""

    def test_detect_off_hours(self, sample_trades_df, sample_markets_df):
        """Test detection of off-hours trading."""
        detector = TimingSignalDetector(sample_trades_df, sample_markets_df)

        # Create trade at 3 AM UTC (off hours)
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            timestamp=datetime(2024, 1, 15, 3, 0, 0, tzinfo=timezone.utc),
            wallet="0xaaa",
            side="BUY",
            outcome="Yes",
            size=100,
            price=0.5,
            notional=50,
        )
        profile = WalletProfile(address="0xaaa", total_trades=10)
        market = Market(
            market_id="m1",
            question="Test",
            category="Politics",
            volume=100000,
            liquidity=20000,
            is_resolved=False,
        )

        signals = detector.detect(trade, profile, market)

        # Should detect OFF_HOURS
        signal_types = [s.type for s in signals]
        assert TimingSignalType.OFF_HOURS in signal_types

    def test_no_off_hours_during_peak(self, sample_trades_df, sample_markets_df):
        """Test that peak hours don't trigger off-hours signal."""
        detector = TimingSignalDetector(sample_trades_df, sample_markets_df)

        # Create trade at 3 PM UTC (peak hours)
        trade = Trade(
            trade_id="t1",
            market_id="m1",
            timestamp=datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc),
            wallet="0xaaa",
            side="BUY",
            outcome="Yes",
            size=100,
            price=0.5,
            notional=50,
        )
        profile = WalletProfile(address="0xaaa", total_trades=10)
        market = Market(
            market_id="m1",
            question="Test",
            category="Politics",
            volume=100000,
            liquidity=20000,
            is_resolved=False,
        )

        signals = detector.detect(trade, profile, market)

        # Should NOT detect OFF_HOURS
        signal_types = [s.type for s in signals]
        assert TimingSignalType.OFF_HOURS not in signal_types


class TestSizingSignalDetector:
    """Tests for SizingSignalDetector."""

    def test_detect_whale_size(self, sample_trades_df, sample_markets_df):
        """Test detection of whale-sized trades."""
        detector = SizingSignalDetector(sample_trades_df, sample_markets_df)

        # Create trade that's >5% of market volume
        trade = Trade(
            trade_id="whale_trade",
            market_id="m1",
            timestamp=datetime.now(timezone.utc),
            wallet="0xwhale",
            side="BUY",
            outcome="Yes",
            size=1000,
            price=0.5,
            notional=500,  # Much larger than existing trades
        )
        profile = WalletProfile(
            address="0xwhale",
            total_trades=10,
            avg_trade_size=50,
        )
        market = Market(
            market_id="m1",
            question="Test",
            category="Politics",
            volume=1000,  # Small market
            liquidity=200,
            is_resolved=False,
        )

        signals = detector.detect(trade, profile, market)

        # Should detect WHALE_SIZE
        signal_types = [s.type for s in signals]
        assert SizingSignalType.WHALE_SIZE in signal_types

    def test_detect_unusual_size(self, sample_trades_df, sample_markets_df):
        """Test detection of unusual trade size for wallet."""
        detector = SizingSignalDetector(sample_trades_df, sample_markets_df)

        trade = Trade(
            trade_id="unusual_trade",
            market_id="m1",
            timestamp=datetime.now(timezone.utc),
            wallet="0xunusual",
            side="BUY",
            outcome="Yes",
            size=1000,
            price=0.5,
            notional=500,  # 10x their average
        )
        profile = WalletProfile(
            address="0xunusual",
            total_trades=50,
            avg_trade_size=50,  # Normally trades $50
        )
        market = Market(
            market_id="m1",
            question="Test",
            category="Politics",
            volume=1000000,  # Large market
            liquidity=200000,
            is_resolved=False,
        )

        signals = detector.detect(trade, profile, market)

        # Should detect UNUSUAL_SIZE
        signal_types = [s.type for s in signals]
        assert SizingSignalType.UNUSUAL_SIZE in signal_types


class TestMarketContextAnalyzer:
    """Tests for MarketContextAnalyzer."""

    def test_high_insider_category(self, sample_markets_df):
        """Test scoring of high-insider category markets."""
        analyzer = MarketContextAnalyzer(sample_markets_df)

        market = Market(
            market_id="m1",
            question="Will the president resign?",
            category="Politics",
            volume=100000,
            liquidity=20000,
            is_resolved=False,
        )

        score = analyzer.get_insider_category_score(market)

        # Politics is high-insider, plus "resign" keyword
        assert score >= 0.8

    def test_low_insider_category(self, sample_markets_df):
        """Test scoring of low-insider category markets."""
        analyzer = MarketContextAnalyzer(sample_markets_df)

        market = Market(
            market_id="m2",
            question="Will team X win the championship?",
            category="Sports",
            volume=50000,
            liquidity=10000,
            is_resolved=False,
        )

        score = analyzer.get_insider_category_score(market)

        # Sports is low-insider
        assert score <= 0.3


class TestNegativeSignalFilter:
    """Tests for NegativeSignalFilter."""

    def test_filter_long_history_wallet(self):
        """Test that long-history wallets get high negative score."""
        profile = WalletProfile(
            address="0xold",
            days_on_polymarket=200,
            retail_likelihood=0.3,
            total_trades=50,
        )
        profiles = {"0xold": profile}

        filter = NegativeSignalFilter(profiles)
        score = filter.compute_negative_score(profile)

        # Long history should add 0.3
        assert score >= 0.3

    def test_filter_retail_wallet(self):
        """Test that retail-like wallets get high negative score."""
        profile = WalletProfile(
            address="0xretail",
            days_on_polymarket=30,
            retail_likelihood=0.9,
            total_trades=100,
            unique_markets=20,
        )
        profiles = {"0xretail": profile}

        filter = NegativeSignalFilter(profiles)
        score = filter.compute_negative_score(profile)

        # High retail + many trades should give high negative score
        assert score >= 0.4

    def test_should_filter(self):
        """Test should_filter method."""
        profile = WalletProfile(
            address="0xfiltered",
            days_on_polymarket=200,
            retail_likelihood=0.9,
            total_trades=100,
            unique_markets=20,
        )
        profiles = {"0xfiltered": profile}

        filter = NegativeSignalFilter(profiles)

        # Should be filtered with default threshold
        assert filter.should_filter(profile, threshold=0.6)


class TestSignalAggregator:
    """Tests for SignalAggregator."""

    def test_aggregate_empty_signals(self):
        """Test aggregation with no signals."""
        aggregator = SignalAggregator()
        result = aggregator.aggregate([], 0.0, 1.0)

        assert result.final_score == 0.0
        assert result.raw_score == 0.0
        assert len(result.signals) == 0

    def test_aggregate_single_category(self):
        """Test aggregation with signals from single category."""
        aggregator = SignalAggregator()

        signals = [
            Signal(
                type=FreshnessSignalType.NEW_WALLET,
                category=SignalCategory.FRESHNESS,
                confidence=0.8,
            ),
            Signal(
                type=FreshnessSignalType.LOW_ACTIVITY,
                category=SignalCategory.FRESHNESS,
                confidence=0.5,
            ),
        ]

        result = aggregator.aggregate(signals, negative_score=0.0, market_context_score=1.0)

        # Should use max confidence (0.8) * weight (0.25) = 0.2 raw score
        assert result.category_scores["freshness"] == 0.8
        assert result.raw_score == pytest.approx(0.2, rel=0.01)

    def test_aggregate_with_negative_discount(self):
        """Test that negative score reduces final score."""
        aggregator = SignalAggregator()

        signals = [
            Signal(
                type=FreshnessSignalType.NEW_WALLET,
                category=SignalCategory.FRESHNESS,
                confidence=1.0,
            ),
        ]

        # Without negative
        result_no_neg = aggregator.aggregate(signals, negative_score=0.0, market_context_score=1.0)

        # With negative
        result_neg = aggregator.aggregate(signals, negative_score=0.8, market_context_score=1.0)

        # Negative score should reduce final score
        assert result_neg.final_score < result_no_neg.final_score

    def test_explain_score(self):
        """Test score explanation."""
        aggregator = SignalAggregator()

        signals = [
            Signal(
                type=FreshnessSignalType.NEW_WALLET,
                category=SignalCategory.FRESHNESS,
                confidence=0.8,
            ),
        ]

        result = aggregator.aggregate(signals, 0.1, 1.0)
        explanation = aggregator.explain_score(result)

        assert "final_score" in explanation
        assert "raw_score" in explanation
        assert "category_breakdown" in explanation
        assert "top_signals" in explanation


class TestInsiderSignalDetector:
    """Tests for InsiderSignalDetector."""

    def test_detect_signals_complete(
        self,
        sample_trades_df,
        sample_markets_df,
        sample_trade,
        sample_wallet_profile,
    ):
        """Test complete signal detection."""
        profiles = {sample_wallet_profile.address: sample_wallet_profile}
        clusters = []

        detector = InsiderSignalDetector(
            trades_df=sample_trades_df,
            markets_df=sample_markets_df,
            wallet_profiles=profiles,
            clusters=clusters,
        )

        # Detect signals for sample trade
        result = detector.detect_signals(sample_trade)

        assert result is not None
        assert result.trade == sample_trade
        assert hasattr(result, "final_score")
        assert hasattr(result, "aggregated")

    def test_detect_signals_missing_profile(self, sample_trades_df, sample_markets_df, sample_trade):
        """Test handling of trade with missing wallet profile."""
        detector = InsiderSignalDetector(
            trades_df=sample_trades_df,
            markets_df=sample_markets_df,
            wallet_profiles={},  # Empty profiles
            clusters=[],
        )

        result = detector.detect_signals(sample_trade)

        # Should return signal with score 0
        assert result.final_score == 0.0
        assert len(result.signals) == 0

    def test_to_dataframe(
        self,
        sample_trades_df,
        sample_markets_df,
        sample_trade,
        sample_wallet_profile,
    ):
        """Test conversion to DataFrame."""
        profiles = {sample_wallet_profile.address: sample_wallet_profile}
        clusters = []

        detector = InsiderSignalDetector(
            trades_df=sample_trades_df,
            markets_df=sample_markets_df,
            wallet_profiles=profiles,
            clusters=clusters,
        )

        result = detector.detect_signals(sample_trade)
        df = detector.to_dataframe([result])

        assert len(df) == 1
        assert "trade_id" in df.columns
        assert "final_score" in df.columns
        assert "freshness_score" in df.columns
        assert "signals_json" in df.columns


class TestTradeDataclass:
    """Tests for Trade dataclass."""

    def test_from_row(self):
        """Test creating Trade from DataFrame row."""
        row = {
            "trade_id": "t1",
            "market_id": "m1",
            "timestamp": datetime.now(timezone.utc),
            "maker_address": "0xmaker",
            "taker_address": "0xtaker",
            "side": "BUY",
            "outcome": "Yes",
            "size": 100,
            "price": 0.5,
            "notional": 50,
        }

        trade = Trade.from_row(row, wallet_column="maker_address")

        assert trade.trade_id == "t1"
        assert trade.wallet == "0xmaker"
        assert trade.notional == 50

    def test_from_row_taker(self):
        """Test creating Trade from DataFrame row for taker."""
        row = {
            "trade_id": "t1",
            "market_id": "m1",
            "timestamp": datetime.now(timezone.utc),
            "maker_address": "0xmaker",
            "taker_address": "0xtaker",
            "side": "BUY",
            "outcome": "Yes",
            "size": 100,
            "price": 0.5,
            "notional": 50,
        }

        trade = Trade.from_row(row, wallet_column="taker_address")

        assert trade.wallet == "0xtaker"


class TestSignalDataclass:
    """Tests for Signal dataclass."""

    def test_to_dict(self):
        """Test Signal serialization."""
        signal = Signal(
            type=FreshnessSignalType.NEW_WALLET,
            category=SignalCategory.FRESHNESS,
            confidence=0.8,
            details={"days_active": 3},
        )

        d = signal.to_dict()

        assert d["type"] == "new_wallet"
        assert d["category"] == "freshness"
        assert d["confidence"] == 0.8
        assert d["details"]["days_active"] == 3
