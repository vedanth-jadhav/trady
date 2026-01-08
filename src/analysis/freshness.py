"""
Freshness Analyzer for wallet age and trading history analysis.

Analyzes how "fresh" or new a wallet is - a key indicator for insider detection
since insiders often use fresh wallets to avoid detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import logging

from .types import FreshnessProfile

logger = logging.getLogger(__name__)


class FreshnessAnalyzer:
    """
    Analyzes wallet freshness using multiple indicators.

    Freshness Factors:
    1. Zero history: Brand new wallet, first-ever transaction
    2. New to Polymarket: First Polymarket trade recently
    3. Recently funded: Wallet received funds shortly before trading (needs on-chain data)
    4. Low trade count: Very few trades overall

    Each factor scored 0-1, combined into composite freshness score.
    """

    def __init__(self, trades_df: pd.DataFrame):
        """
        Initialize FreshnessAnalyzer with trade data.

        Args:
            trades_df: DataFrame with trade data (has maker_address, taker_address, timestamp)
        """
        self.trades = trades_df.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.trades['timestamp']):
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'], utc=True)

        self._wallet_stats = None  # Cache for wallet statistics

    def _build_wallet_stats(self) -> pd.DataFrame:
        """
        Build aggregated wallet statistics from trades.

        Returns:
            DataFrame with columns: wallet, first_trade, last_trade, trade_count,
            unique_markets, total_volume
        """
        if self._wallet_stats is not None:
            return self._wallet_stats

        logger.info("Building wallet statistics from trades...")

        # Create a view of trades with wallet column (combining maker and taker)
        maker_trades = self.trades[['maker_address', 'timestamp', 'market_id', 'notional']].copy()
        maker_trades.columns = ['wallet', 'timestamp', 'market_id', 'notional']

        taker_trades = self.trades[['taker_address', 'timestamp', 'market_id', 'notional']].copy()
        taker_trades.columns = ['wallet', 'timestamp', 'market_id', 'notional']

        # Combine all trades
        all_trades = pd.concat([maker_trades, taker_trades], ignore_index=True)

        # Group by wallet and compute statistics
        stats = all_trades.groupby('wallet').agg(
            first_trade=('timestamp', 'min'),
            last_trade=('timestamp', 'max'),
            trade_count=('timestamp', 'count'),
            unique_markets=('market_id', 'nunique'),
            total_volume=('notional', 'sum')
        ).reset_index()

        # Compute days on platform - from first trade until NOW
        # This is the true "age" of the wallet on Polymarket
        now = datetime.now(timezone.utc)
        stats['days_active'] = (now - stats['first_trade']).dt.total_seconds() / 86400
        stats['days_active'] = stats['days_active'].fillna(0).clip(lower=0)

        self._wallet_stats = stats
        logger.info(f"Computed statistics for {len(stats)} wallets")

        return self._wallet_stats

    def compute_polymarket_age(self, wallet: str) -> Dict:
        """
        Compute wallet's Polymarket trading history.

        Args:
            wallet: Wallet address to analyze

        Returns:
            {
                "first_trade_date": datetime,
                "days_on_polymarket": int,
                "total_trades": int,
                "is_first_trade": bool (if only 1 trade),
            }
        """
        stats = self._build_wallet_stats()

        wallet_data = stats[stats['wallet'] == wallet]

        if wallet_data.empty:
            return {
                "first_trade_date": None,
                "days_on_polymarket": 0,
                "total_trades": 0,
                "is_first_trade": True,
            }

        row = wallet_data.iloc[0]

        # Calculate days on Polymarket from first trade to now
        now = datetime.now(timezone.utc)
        days_on_platform = (now - row['first_trade']).total_seconds() / 86400

        return {
            "first_trade_date": row['first_trade'],
            "days_on_polymarket": int(days_on_platform),
            "total_trades": int(row['trade_count']),
            "is_first_trade": row['trade_count'] == 1,
        }

    def compute_freshness_score(
        self,
        wallet: str,
        as_of_date: Optional[datetime] = None
    ) -> FreshnessProfile:
        """
        Compute composite freshness score for a wallet.

        Score Weights:
        - Very new to Polymarket (< 7 days): 0.4
        - Low trade count (< 5 trades): 0.3
        - Single market focus: 0.2
        - Recent first trade: 0.1

        Args:
            wallet: Wallet address to analyze
            as_of_date: Calculate freshness as of this date (default: now)

        Returns:
            FreshnessProfile with score 0-1 (higher = fresher/more suspicious).
        """
        stats = self._build_wallet_stats()
        wallet_data = stats[stats['wallet'] == wallet]

        if wallet_data.empty:
            # Wallet not found - treat as completely fresh
            return FreshnessProfile(
                wallet=wallet,
                freshness_score=1.0,
                is_zero_history=True,
                is_new_to_polymarket=True,
                is_recently_funded=False,  # Unknown without on-chain data
                first_polymarket_trade=None,
                total_polymarket_trades=0,
                days_active=0
            )

        row = wallet_data.iloc[0]

        # Use as_of_date or current time
        reference_date = as_of_date if as_of_date else datetime.now(timezone.utc)

        # Calculate age metrics
        days_since_first_trade = (reference_date - row['first_trade']).total_seconds() / 86400
        trade_count = int(row['trade_count'])
        unique_markets = int(row['unique_markets'])

        # Score components (all 0-1, higher = more suspicious/fresh)

        # 1. Very new to Polymarket (< 7 days gets full weight)
        age_score = max(0, 1 - (days_since_first_trade / 7.0))
        age_score = min(1.0, age_score)  # Cap at 1.0

        # 2. Low trade count (< 5 trades is suspicious)
        trade_count_score = max(0, 1 - (trade_count / 5.0))
        trade_count_score = min(1.0, trade_count_score)

        # 3. Single market focus (trading in only 1-2 markets is suspicious)
        market_diversity_score = max(0, 1 - (unique_markets / 3.0))
        market_diversity_score = min(1.0, market_diversity_score)

        # 4. Recent first trade (first trade very recent)
        # Same as age_score but checking if < 3 days
        recent_score = max(0, 1 - (days_since_first_trade / 3.0))
        recent_score = min(1.0, recent_score)

        # Composite score with weights
        composite_score = (
            0.4 * age_score +
            0.3 * trade_count_score +
            0.2 * market_diversity_score +
            0.1 * recent_score
        )

        # Determine boolean flags
        is_zero_history = trade_count == 1
        is_new_to_polymarket = days_since_first_trade < 7
        is_recently_funded = False  # Would require on-chain data

        return FreshnessProfile(
            wallet=wallet,
            freshness_score=composite_score,
            is_zero_history=is_zero_history,
            is_new_to_polymarket=is_new_to_polymarket,
            is_recently_funded=is_recently_funded,
            first_polymarket_trade=row['first_trade'],
            total_polymarket_trades=trade_count,
            days_active=int(row['days_active'])
        )

    def compute_all_freshness_profiles(
        self,
        wallets: Optional[List[str]] = None
    ) -> Dict[str, FreshnessProfile]:
        """
        Compute freshness profiles for all wallets (or specified list).

        Args:
            wallets: Optional list of wallet addresses to analyze.
                    If None, analyzes all wallets in trade data.

        Returns:
            Dict mapping wallet address to FreshnessProfile.
        """
        stats = self._build_wallet_stats()

        # Use provided wallet list or all wallets from stats
        if wallets is None:
            wallets = stats['wallet'].unique().tolist()

        logger.info(f"Computing freshness profiles for {len(wallets)} wallets...")

        profiles = {}
        for i, wallet in enumerate(wallets):
            if i > 0 and i % 10000 == 0:
                logger.info(f"Processed {i}/{len(wallets)} wallets...")

            profiles[wallet] = self.compute_freshness_score(wallet)

        logger.info(f"Completed freshness analysis for {len(profiles)} wallets")

        return profiles

    def get_freshest_wallets(
        self,
        n: int = 100,
        min_volume: float = 0
    ) -> List[FreshnessProfile]:
        """
        Get the N freshest (most suspicious) wallets by freshness score.

        Args:
            n: Number of freshest wallets to return
            min_volume: Minimum total volume threshold (filters low-volume wallets)

        Returns:
            List of FreshnessProfile objects, sorted by freshness_score descending.
        """
        stats = self._build_wallet_stats()

        # Filter by minimum volume if specified
        if min_volume > 0:
            filtered_wallets = stats[stats['total_volume'] >= min_volume]['wallet'].tolist()
            logger.info(f"Filtered to {len(filtered_wallets)} wallets with volume >= {min_volume}")
        else:
            filtered_wallets = None

        # Compute all profiles
        profiles = self.compute_all_freshness_profiles(wallets=filtered_wallets)

        # Sort by freshness score descending
        sorted_profiles = sorted(
            profiles.values(),
            key=lambda p: p.freshness_score,
            reverse=True
        )

        # Return top N
        result = sorted_profiles[:n]

        logger.info(f"Returning top {len(result)} freshest wallets")
        if result:
            logger.info(f"Freshness score range: {result[0].freshness_score:.3f} to {result[-1].freshness_score:.3f}")

        return result
