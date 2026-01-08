"""
Sizing signal detector.

Detects trades with unusual position sizes relative to market and wallet history.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.analysis import WalletProfile

from .types import (
    Market,
    Signal,
    SignalCategory,
    SizingSignalType,
    Trade,
)


class SizingSignalDetector:
    """
    Detects sizing-based insider signals.

    Uses DYNAMIC thresholds based on market context.

    Signals:
    - WHALE_SIZE: Trade is large % of market volume
    - UNUSUAL_SIZE: Trade is unusual for this wallet
    - CONCENTRATED_POSITION: Building large position
    """

    # Thresholds
    WHALE_VOLUME_PCT = 0.05  # >5% of market volume
    UNUSUAL_SIZE_MULTIPLIER = 3.0  # >3x wallet's average

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
        self.market_baselines: Dict[str, Dict[str, float]] = {}
        self._wallet_positions_cache: Dict[str, Dict[str, List[Trade]]] = {}
        self._compute_market_baselines()

    def _compute_market_baselines(self) -> None:
        """
        Compute baseline trade sizes for each market.

        Metrics:
        - Median trade size
        - 95th percentile trade size
        - Total volume
        """
        if self.trades.empty:
            return

        for market_id, group in self.trades.groupby("market_id"):
            notional = group["notional"]
            self.market_baselines[str(market_id)] = {
                "median_size": float(notional.median()),
                "p95_size": float(notional.quantile(0.95)),
                "total_volume": float(notional.sum()),
                "trade_count": len(group),
            }

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
        market: Market,
    ) -> List[Signal]:
        """
        Detect sizing signals for a trade.

        Args:
            trade: The trade to analyze
            wallet_profile: Profile of the wallet
            market: Market the trade was made in

        Returns:
            List of detected signals
        """
        signals = []
        baseline = self.market_baselines.get(trade.market_id, {})

        # Whale size: Trade is >5% of market volume
        whale_signal = self._detect_whale_size(trade, baseline)
        if whale_signal:
            signals.append(whale_signal)

        # Unusual size: Trade is >3x wallet's average
        unusual_signal = self._detect_unusual_size(trade, wallet_profile)
        if unusual_signal:
            signals.append(unusual_signal)

        # Concentrated position: Check if building large position in market
        concentrated_signal = self._detect_concentrated_position(
            trade, wallet_profile, baseline
        )
        if concentrated_signal:
            signals.append(concentrated_signal)

        return signals

    def _detect_whale_size(
        self,
        trade: Trade,
        baseline: Dict[str, float],
    ) -> Optional[Signal]:
        """Detect if trade is a whale-sized trade."""
        total_volume = baseline.get("total_volume", 0)

        if total_volume <= 0:
            return None

        pct_of_volume = trade.notional / total_volume

        if pct_of_volume > self.WHALE_VOLUME_PCT:
            # Confidence scales with percentage (caps at 1.0)
            confidence = min(pct_of_volume * 10, 1.0)
            return Signal(
                type=SizingSignalType.WHALE_SIZE,
                category=SignalCategory.SIZING,
                confidence=confidence,
                details={
                    "pct_of_volume": round(pct_of_volume, 4),
                    "trade_size": trade.notional,
                    "market_volume": total_volume,
                }
            )

        return None

    def _detect_unusual_size(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
    ) -> Optional[Signal]:
        """Detect if trade size is unusual for this wallet."""
        avg_size = wallet_profile.avg_trade_size

        if avg_size <= 0:
            return None

        size_ratio = trade.notional / avg_size

        if size_ratio > self.UNUSUAL_SIZE_MULTIPLIER:
            # Scale confidence from 0 at 3x to 1.0 at 10x
            confidence = min((size_ratio - 3) / 7, 1.0)
            return Signal(
                type=SizingSignalType.UNUSUAL_SIZE,
                category=SignalCategory.SIZING,
                confidence=confidence,
                details={
                    "size_ratio": round(size_ratio, 2),
                    "trade_size": trade.notional,
                    "avg_size": round(avg_size, 2),
                }
            )

        return None

    def _detect_concentrated_position(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
        baseline: Dict[str, float],
    ) -> Optional[Signal]:
        """Check if wallet is building a concentrated position."""
        wallet = trade.wallet
        market_id = trade.market_id

        # Get all wallet trades in this market
        wallet_positions = self._get_wallet_market_positions(wallet, market_id)

        if not wallet_positions:
            return None

        # Calculate total position
        total_position = sum(t.notional for t in wallet_positions)

        # Compare to market's 95th percentile trade size
        p95_size = baseline.get("p95_size", float("inf"))

        if total_position > p95_size:
            return Signal(
                type=SizingSignalType.CONCENTRATED_POSITION,
                category=SignalCategory.SIZING,
                confidence=min(total_position / (p95_size * 2), 0.9),
                details={
                    "total_position": round(total_position, 2),
                    "trade_count": len(wallet_positions),
                    "p95_size": round(p95_size, 2),
                }
            )

        return None

    def _get_wallet_market_positions(
        self,
        wallet: str,
        market_id: str,
    ) -> List[Trade]:
        """Get all trades by wallet in this market."""
        # Check cache
        cache_key = f"{wallet}_{market_id}"
        if cache_key in self._wallet_positions_cache:
            return self._wallet_positions_cache.get(cache_key, [])

        # Find trades
        mask = (
            ((self.trades["maker_address"] == wallet) |
             (self.trades["taker_address"] == wallet)) &
            (self.trades["market_id"] == market_id)
        )
        wallet_trades = self.trades[mask]

        # Convert to Trade objects
        trades = []
        for _, row in wallet_trades.iterrows():
            wallet_col = "maker_address" if row["maker_address"] == wallet else "taker_address"
            trades.append(Trade.from_row(row, wallet_column=wallet_col))

        # Cache and return
        self._wallet_positions_cache[cache_key] = trades
        return trades
