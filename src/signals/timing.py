"""
Timing signal detector.

Detects trades with suspicious timing patterns.
"""

from datetime import timedelta
from typing import Dict, List, Optional, Set

import pandas as pd

from src.analysis import WalletProfile

from .types import (
    Market,
    Signal,
    SignalCategory,
    TimingSignalType,
    Trade,
)


class TimingSignalDetector:
    """
    Detects timing-based insider signals.

    Signals:
    - PRE_NEWS: Trade shortly before news/resolution
    - OFF_HOURS: Trade during off-peak hours
    - PRE_SPIKE: Trade before unusual price movement
    - RAPID_SUCCESSION: Multiple trades in quick burst
    """

    # Off-hours defined as outside 9am-9pm ET (13:00-01:00 UTC)
    OFF_HOURS_START = 1   # 1:00 UTC = 9pm ET (previous day)
    OFF_HOURS_END = 13    # 13:00 UTC = 9am ET

    # Pre-news window (hours before resolution)
    PRE_NEWS_HOURS = 24

    # Price spike threshold (10% change)
    SPIKE_THRESHOLD = 0.10

    # Rapid succession window (minutes)
    RAPID_WINDOW_MINUTES = 10
    RAPID_MIN_TRADES = 3

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame,
    ):
        """
        Initialize detector with trade and market data.

        Args:
            trades_df: DataFrame with all trades
            markets_df: DataFrame with market data
        """
        self.trades = trades_df
        self.markets = markets_df
        self._price_spikes: Dict[str, List[pd.Timestamp]] = {}
        self._wallet_trades_cache: Dict[str, pd.DataFrame] = {}
        self._precompute_price_spikes()

    def _precompute_price_spikes(self) -> None:
        """
        Identify all significant price movements.

        A spike is defined as >10% price change in 1 hour.
        """
        if self.trades.empty:
            return

        # Group trades by market and compute hourly price changes
        for market_id, group in self.trades.groupby("market_id"):
            if len(group) < 2:
                continue

            # Sort by timestamp
            group = group.sort_values("timestamp")

            # Compute rolling price change over 1-hour windows
            spikes = []
            group = group.set_index("timestamp")

            # Resample to hourly and compute price change
            if len(group) > 1:
                hourly = group["price"].resample("1h").mean().dropna()
                if len(hourly) > 1:
                    pct_change = hourly.pct_change().abs()
                    spike_times = pct_change[pct_change > self.SPIKE_THRESHOLD].index.tolist()
                    spikes.extend(spike_times)

            self._price_spikes[str(market_id)] = spikes

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
        market: Market,
    ) -> List[Signal]:
        """
        Detect timing signals for a trade.

        Args:
            trade: The trade to analyze
            wallet_profile: Profile of the wallet
            market: Market the trade was made in

        Returns:
            List of detected signals
        """
        signals = []

        # Pre-news: Trade within 24h before resolution
        pre_news_signal = self._detect_pre_news(trade, market)
        if pre_news_signal:
            signals.append(pre_news_signal)

        # Off-hours trading
        if self._is_off_hours(trade.timestamp.hour):
            signals.append(Signal(
                type=TimingSignalType.OFF_HOURS,
                category=SignalCategory.TIMING,
                confidence=0.5,  # Moderate signal
                details={"trade_hour_utc": trade.timestamp.hour}
            ))

        # Pre-spike: Trade before significant price movement
        pre_spike_signal = self._detect_pre_spike(trade, market)
        if pre_spike_signal:
            signals.append(pre_spike_signal)

        # Rapid succession: Part of burst trading
        rapid_signal = self._detect_rapid_succession(trade, wallet_profile)
        if rapid_signal:
            signals.append(rapid_signal)

        return signals

    def _detect_pre_news(self, trade: Trade, market: Market) -> Optional[Signal]:
        """Detect if trade occurred shortly before market resolution."""
        if not market.resolution_timestamp:
            return None

        time_diff = market.resolution_timestamp - trade.timestamp
        hours_before = time_diff.total_seconds() / 3600

        if 0 < hours_before < self.PRE_NEWS_HOURS:
            # Confidence increases as we get closer to resolution
            confidence = 1.0 - (hours_before / self.PRE_NEWS_HOURS)
            return Signal(
                type=TimingSignalType.PRE_NEWS,
                category=SignalCategory.TIMING,
                confidence=confidence,
                details={"hours_before_resolution": round(hours_before, 2)}
            )

        return None

    def _is_off_hours(self, hour_utc: int) -> bool:
        """Check if hour is outside peak trading times (9am-9pm ET)."""
        # Off hours: 01:00-13:00 UTC (which is 8pm-8am ET approximately)
        return self.OFF_HOURS_START <= hour_utc < self.OFF_HOURS_END

    def _detect_pre_spike(
        self,
        trade: Trade,
        market: Market,
        lookforward_hours: int = 4,
    ) -> Optional[Signal]:
        """Check if trade occurred before a price spike."""
        spikes = self._price_spikes.get(trade.market_id, [])

        if not spikes:
            return None

        trade_ts = trade.timestamp
        lookforward = timedelta(hours=lookforward_hours)

        for spike_ts in spikes:
            time_to_spike = spike_ts - trade_ts
            if timedelta(0) < time_to_spike <= lookforward:
                hours_before = time_to_spike.total_seconds() / 3600
                return Signal(
                    type=TimingSignalType.PRE_SPIKE,
                    category=SignalCategory.TIMING,
                    confidence=0.9,
                    details={
                        "spike_detected": True,
                        "hours_before_spike": round(hours_before, 2),
                    }
                )

        return None

    def _detect_rapid_succession(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
    ) -> Optional[Signal]:
        """Check if trade is part of rapid succession."""
        wallet = trade.wallet

        # Get wallet's trades from cache or compute
        if wallet not in self._wallet_trades_cache:
            # Find all trades for this wallet (as maker or taker)
            mask = (
                (self.trades["maker_address"] == wallet) |
                (self.trades["taker_address"] == wallet)
            )
            wallet_trades = self.trades[mask].copy()
            wallet_trades = wallet_trades.sort_values("timestamp")
            self._wallet_trades_cache[wallet] = wallet_trades
        else:
            wallet_trades = self._wallet_trades_cache[wallet]

        if len(wallet_trades) < self.RAPID_MIN_TRADES:
            return None

        # Check for burst around this trade's timestamp
        window = timedelta(minutes=self.RAPID_WINDOW_MINUTES)
        trade_ts = trade.timestamp

        # Count trades within window
        mask = (
            (wallet_trades["timestamp"] >= trade_ts - window) &
            (wallet_trades["timestamp"] <= trade_ts + window)
        )
        trades_in_window = len(wallet_trades[mask])

        if trades_in_window >= self.RAPID_MIN_TRADES:
            return Signal(
                type=TimingSignalType.RAPID_SUCCESSION,
                category=SignalCategory.TIMING,
                confidence=min(0.3 + (trades_in_window - self.RAPID_MIN_TRADES) * 0.1, 0.8),
                details={
                    "burst_trade": True,
                    "trades_in_window": trades_in_window,
                    "window_minutes": self.RAPID_WINDOW_MINUTES,
                }
            )

        return None
