import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
import logging
from collections import Counter

from .types import BehaviorProfile

logger = logging.getLogger(__name__)


class BehaviorProfiler:
    """
    Profiles wallet trading behavior.

    Behavioral Indicators:
    - Trade frequency patterns (burst trading = suspicious)
    - Position sizing consistency
    - Market diversity (niche markets = suspicious)
    - Timing patterns (off-hours trading)
    - Win rate (high win rate on resolved markets = suspicious)
    - Exit behavior (holding to resolution = insider signal)
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: Optional[pd.DataFrame] = None
    ):
        self.trades = trades_df
        self.markets = markets_df
        self._wallet_trades = {}  # Cache

        # Ensure timestamp is datetime
        if 'timestamp' in self.trades.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.trades['timestamp']):
                self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'], utc=True)
            elif self.trades['timestamp'].dt.tz is None:
                self.trades['timestamp'] = self.trades['timestamp'].dt.tz_localize('UTC')

    def _get_wallet_trades(self, wallet: str) -> pd.DataFrame:
        """Get all trades for a wallet (as maker or taker)."""
        if wallet in self._wallet_trades:
            return self._wallet_trades[wallet]

        wallet_trades = self.trades[
            (self.trades['maker_address'] == wallet) |
            (self.trades['taker_address'] == wallet)
        ].copy()

        self._wallet_trades[wallet] = wallet_trades
        return wallet_trades

    def compute_trading_frequency(self, wallet: str) -> Dict:
        """
        Analyze trading frequency patterns.

        Returns:
            {
                "avg_trades_per_day": float,
                "trade_frequency_variance": float,
                "burst_trading_episodes": int,  # >5 trades in 1 hour
                "typical_hours": List[int],     # Most active hours (UTC)
                "off_hours_ratio": float,       # % trades outside 9-17 UTC
            }
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0:
            return {
                "avg_trades_per_day": 0.0,
                "trade_frequency_variance": 0.0,
                "burst_trading_episodes": 0,
                "typical_hours": [],
                "off_hours_ratio": 0.0,
            }

        # Sort by timestamp - use copy to avoid mutating cache
        trades = trades.sort_values('timestamp').copy()

        # Calculate time span - use total_seconds for fractional day accuracy
        first_trade = trades['timestamp'].min()
        last_trade = trades['timestamp'].max()
        # Use fractional days to avoid inflating activity for same-day trades
        # A wallet active for 6 hours should show 0.25 days, not 1 day
        days_active_float = (last_trade - first_trade).total_seconds() / 86400
        # Minimum of 1 hour (0.0417 days) to avoid division issues for near-instant trades
        days_active = max(days_active_float, 1/24)

        # Avg trades per day
        avg_trades_per_day = len(trades) / days_active

        # Trade frequency variance (trades per day)
        trades['date'] = trades['timestamp'].dt.date
        trades_per_day = trades.groupby('date').size()
        trade_frequency_variance = float(trades_per_day.var()) if len(trades_per_day) > 1 else 0.0

        # Detect burst episodes (>5 trades in 1 hour)
        burst_episodes = 0
        trades['hour_bucket'] = trades['timestamp'].dt.floor('1h')
        trades_per_hour = trades.groupby('hour_bucket').size()
        burst_episodes = int((trades_per_hour > 5).sum())

        # Typical hours (UTC)
        trades['hour'] = trades['timestamp'].dt.hour
        hour_counts = trades['hour'].value_counts()
        typical_hours = hour_counts.nlargest(3).index.tolist() if len(hour_counts) > 0 else []

        # Off-hours ratio (before 9 or after 17 UTC)
        off_hours_mask = (trades['hour'] < 9) | (trades['hour'] >= 17)
        off_hours_ratio = float(off_hours_mask.sum() / len(trades))

        return {
            "avg_trades_per_day": float(avg_trades_per_day),
            "trade_frequency_variance": float(trade_frequency_variance),
            "burst_trading_episodes": burst_episodes,
            "typical_hours": [int(h) for h in typical_hours],
            "off_hours_ratio": off_hours_ratio,
        }

    def compute_sizing_patterns(self, wallet: str) -> Dict:
        """
        Analyze position sizing behavior.

        Returns:
            {
                "avg_trade_size": float,
                "max_trade_size": float,
                "size_variance": float,
                "uses_round_numbers": bool,  # Retail indicator (e.g., $100, $500)
                "consistent_sizing": bool,   # Low variance = more sophisticated
            }
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0 or 'notional' not in trades.columns:
            return {
                "avg_trade_size": 0.0,
                "max_trade_size": 0.0,
                "size_variance": 0.0,
                "uses_round_numbers": False,
                "consistent_sizing": False,
            }

        # Use notional value for sizing
        sizes = trades['notional'].values

        avg_trade_size = float(np.mean(sizes))
        max_trade_size = float(np.max(sizes))
        size_variance = float(np.var(sizes)) if len(sizes) > 1 else 0.0

        # Detect round numbers - include small retail amounts ($10, $25, $50)
        # and large amounts ($100, $500, $1000, $5000, $10000)
        round_numbers = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        round_number_count = 0

        for size in sizes:
            for rn in round_numbers:
                # Within 5% of a round number (use <= for boundary inclusion)
                if abs(size - rn) / rn <= 0.05:
                    round_number_count += 1
                    break

        uses_round_numbers = (round_number_count / len(sizes)) > 0.3 if len(sizes) > 0 else False

        # Consistent sizing: coefficient of variation < 0.5
        cv = np.sqrt(size_variance) / avg_trade_size if avg_trade_size > 0 else 0
        consistent_sizing = cv < 0.5

        return {
            "avg_trade_size": avg_trade_size,
            "max_trade_size": max_trade_size,
            "size_variance": size_variance,
            "uses_round_numbers": bool(uses_round_numbers),
            "consistent_sizing": bool(consistent_sizing),
        }

    def compute_market_diversity(self, wallet: str) -> Dict:
        """
        Analyze market selection patterns.

        Returns:
            {
                "unique_markets": int,
                "market_concentration": float,  # Herfindahl index 0-1
                "category_distribution": Dict[str, float],
                "niche_market_ratio": float,  # % in low-volume markets
            }
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0 or 'market_id' not in trades.columns:
            return {
                "unique_markets": 0,
                "market_concentration": 0.0,
                "category_distribution": {},
                "niche_market_ratio": 0.0,
            }

        # Unique markets
        unique_markets = trades['market_id'].nunique()

        # Herfindahl index (market concentration)
        # HHI = sum of squared market shares
        market_trades = trades.groupby('market_id').size()
        market_shares = market_trades / len(trades)
        herfindahl_index = float((market_shares ** 2).sum())

        # Category distribution
        category_distribution = {}
        if self.markets is not None:
            # Merge with market data
            trades_with_markets = trades.merge(
                self.markets[['market_id', 'category', 'volume']],
                on='market_id',
                how='left'
            )

            # Category distribution
            if 'category' in trades_with_markets.columns:
                category_counts = trades_with_markets['category'].value_counts()
                category_distribution = {
                    str(cat): float(count / len(trades))
                    for cat, count in category_counts.items()
                }

            # Niche market ratio (low-volume markets)
            # Define niche as bottom 25% of volume
            if 'volume' in trades_with_markets.columns:
                volume_threshold = self.markets['volume'].quantile(0.25)
                niche_trades = trades_with_markets[
                    trades_with_markets['volume'] <= volume_threshold
                ]
                niche_market_ratio = float(len(niche_trades) / len(trades))
            else:
                niche_market_ratio = 0.0
        else:
            niche_market_ratio = 0.0

        return {
            "unique_markets": int(unique_markets),
            "market_concentration": herfindahl_index,
            "category_distribution": category_distribution,
            "niche_market_ratio": niche_market_ratio,
        }

    def compute_timing_patterns(self, wallet: str) -> Dict:
        """
        Analyze when wallet trades.

        Off-hours trading (weekends, late night) can be suspicious.
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0:
            return {
                "off_hours_ratio": 0.0,
                "weekend_ratio": 0.0,
                "peak_hours": [],
            }

        # Use copy to avoid mutating cache
        trades = trades.copy()

        # Hour of day
        trades['hour'] = trades['timestamp'].dt.hour
        trades['day_of_week'] = trades['timestamp'].dt.dayofweek

        # Off-hours ratio (before 9 or after 17 UTC)
        off_hours_mask = (trades['hour'] < 9) | (trades['hour'] >= 17)
        off_hours_ratio = float(off_hours_mask.sum() / len(trades))

        # Weekend ratio (Saturday=5, Sunday=6)
        weekend_mask = trades['day_of_week'] >= 5
        weekend_ratio = float(weekend_mask.sum() / len(trades))

        # Peak hours
        hour_counts = trades['hour'].value_counts()
        peak_hours = hour_counts.nlargest(3).index.tolist() if len(hour_counts) > 0 else []

        return {
            "off_hours_ratio": off_hours_ratio,
            "weekend_ratio": weekend_ratio,
            "peak_hours": [int(h) for h in peak_hours],
        }

    def compute_behavioral_consistency(self, wallet: str) -> Dict:
        """
        Analyze behavioral consistency to detect coordinated/Sybil patterns.

        False positive prevention: Sybil addresses show abnormally CONSISTENT
        patterns (coordinated). Natural users show high variance.

        Returns:
            {
                "timing_consistency": float,  # 0-1, high = very consistent timing
                "sizing_consistency": float,  # 0-1, high = very consistent sizing
                "interval_variance": float,   # Low = suspicious, high = natural
                "is_likely_natural": bool,    # True = likely natural user
            }
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) < 3:
            return {
                "timing_consistency": 0.5,
                "sizing_consistency": 0.5,
                "interval_variance": 0.5,
                "is_likely_natural": True,  # Not enough data to judge
            }

        trades = trades.sort_values('timestamp').copy()

        # Calculate inter-trade intervals
        intervals = trades['timestamp'].diff().dt.total_seconds().dropna()

        if len(intervals) < 2:
            interval_variance = 0.5
            timing_consistency = 0.5
        else:
            # Coefficient of variation for intervals
            # Low CV = very consistent timing = suspicious
            # High CV = irregular timing = natural
            mean_interval = intervals.mean()
            std_interval = intervals.std()
            if mean_interval > 0:
                interval_cv = std_interval / mean_interval
                # Normalize: CV < 0.5 is very consistent, CV > 2 is very irregular
                interval_variance = min(1.0, interval_cv / 2)
                timing_consistency = 1.0 - interval_variance
            else:
                interval_variance = 0.5
                timing_consistency = 0.5

        # Calculate sizing consistency
        if 'notional' in trades.columns and len(trades) >= 3:
            sizes = trades['notional'].values
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            if mean_size > 0:
                size_cv = std_size / mean_size
                # Low CV = very consistent sizing = suspicious
                sizing_consistency = max(0, 1.0 - min(1.0, size_cv / 2))
            else:
                sizing_consistency = 0.5
        else:
            sizing_consistency = 0.5

        # Determine if likely natural user
        # Natural users have: high interval variance, low sizing consistency
        # Sybil/coordinated: low interval variance, high sizing consistency
        is_likely_natural = (
            interval_variance > 0.4 or  # Irregular timing
            sizing_consistency < 0.6 or  # Varied sizing
            len(trades) >= 20  # Long history suggests genuine user
        )

        return {
            "timing_consistency": float(timing_consistency),
            "sizing_consistency": float(sizing_consistency),
            "interval_variance": float(interval_variance),
            "is_likely_natural": bool(is_likely_natural),
        }

    def compute_activity_spread(self, wallet: str) -> Dict:
        """
        Analyze how activity is spread across time.

        False positive prevention: Insiders often show activity clustering
        around specific events. Natural users have spread out activity.

        Returns:
            {
                "activity_duration_days": float,
                "trades_per_active_day": float,
                "active_day_ratio": float,  # What % of days in range had trades
                "is_burst_only": bool,      # All activity in short bursts
            }
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0:
            return {
                "activity_duration_days": 0.0,
                "trades_per_active_day": 0.0,
                "active_day_ratio": 0.0,
                "is_burst_only": False,
            }

        trades = trades.copy()
        trades['date'] = trades['timestamp'].dt.date

        # Activity duration
        first_trade = trades['timestamp'].min()
        last_trade = trades['timestamp'].max()
        duration_days = (last_trade - first_trade).total_seconds() / 86400

        # Active days
        unique_active_days = trades['date'].nunique()
        trades_per_active_day = len(trades) / max(unique_active_days, 1)

        # What percentage of calendar days had activity
        if duration_days >= 1:
            active_day_ratio = unique_active_days / max(duration_days, 1)
        else:
            active_day_ratio = 1.0  # All activity in less than a day

        # Is burst only: high trades per active day, low active day ratio
        is_burst_only = (
            trades_per_active_day > 5 and
            active_day_ratio < 0.3 and
            unique_active_days <= 3
        )

        return {
            "activity_duration_days": float(duration_days),
            "trades_per_active_day": float(trades_per_active_day),
            "active_day_ratio": float(min(1.0, active_day_ratio)),
            "is_burst_only": bool(is_burst_only),
        }

    def compute_retail_likelihood(self, wallet: str) -> float:
        """
        Score how likely this is retail behavior (0-1).

        High retail score = NOT an insider.
        Uses CONTINUOUS scoring for better differentiation.

        Retail indicators:
        - Round number trades
        - High variance in sizing
        - Trades during business hours
        - Low trade frequency
        - Low market concentration (diversified)
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0:
            return 0.5  # Neutral

        retail_score = 0.0

        # Get sizing patterns
        sizing = self.compute_sizing_patterns(wallet)

        # Round numbers = retail (continuous based on ratio)
        # Include small retail amounts for better detection
        sizes = trades['notional'].values
        round_numbers = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        round_count = sum(
            1 for s in sizes
            for rn in round_numbers
            if abs(s - rn) / rn <= 0.05
        )
        round_ratio = round_count / len(sizes) if len(sizes) > 0 else 0
        retail_score += 0.2 * min(1.0, round_ratio * 3)  # Scale up

        # High variance = retail (continuous)
        # Coefficient of variation > 1 is high variance
        if sizing['avg_trade_size'] > 0:
            cv = np.sqrt(sizing['size_variance']) / sizing['avg_trade_size']
            variance_score = min(1.0, cv / 2)  # CV of 2+ = max score
            retail_score += 0.15 * variance_score

        # Low trade frequency = retail (continuous)
        frequency = self.compute_trading_frequency(wallet)
        freq_score = max(0, 1 - frequency['avg_trades_per_day'] / 100)
        retail_score += 0.2 * freq_score

        # Low market concentration = retail (diversified trading)
        diversity = self.compute_market_diversity(wallet)
        # HHI close to 0 = diversified = retail, close to 1 = concentrated = sophisticated
        concentration_retail = 1 - diversity['market_concentration']
        retail_score += 0.2 * concentration_retail

        # Small trade sizes = retail
        # Normalize by percentile of max trade size across all wallets
        max_size_score = min(1.0, 1 - sizing['max_trade_size'] / 100000)
        retail_score += 0.15 * max(0, max_size_score)

        # Few unique markets = retail? Actually sophisticated traders often focus
        # More markets = more retail-like
        if diversity['unique_markets'] > 5:
            retail_score += 0.1

        # === NEW: Behavioral consistency checks for false positive prevention ===

        # Check behavioral consistency - natural users have irregular patterns
        consistency = self.compute_behavioral_consistency(wallet)
        if consistency['is_likely_natural']:
            retail_score += 0.05

        # High interval variance = natural user behavior
        if consistency['interval_variance'] > 0.5:
            retail_score += 0.05

        # Check activity spread - burst-only is suspicious, spread is natural
        activity = self.compute_activity_spread(wallet)
        if not activity['is_burst_only'] and activity['active_day_ratio'] > 0.3:
            retail_score += 0.05

        return min(1.0, max(0.0, retail_score))

    def build_behavior_profile(self, wallet: str) -> BehaviorProfile:
        """
        Build complete behavior profile for a wallet.
        """
        trades = self._get_wallet_trades(wallet)

        if len(trades) == 0:
            logger.warning(f"No trades found for wallet {wallet}")
            return BehaviorProfile(
                wallet=wallet,
                avg_trades_per_day=0.0,
                burst_episodes=0,
                off_hours_ratio=0.0,
                avg_trade_size=0.0,
                max_trade_size=0.0,
                size_variance=0.0,
                unique_markets=0,
                market_concentration=0.0,
                niche_market_ratio=0.0,
                holds_to_resolution_ratio=0.0,
                early_exit_ratio=0.0,
                retail_likelihood=0.5,
                sophistication_score=0.5,
            )

        # Compute all components
        frequency = self.compute_trading_frequency(wallet)
        sizing = self.compute_sizing_patterns(wallet)
        diversity = self.compute_market_diversity(wallet)

        # Compute retail likelihood
        retail_likelihood = self.compute_retail_likelihood(wallet)

        # Sophistication score is inverse of retail
        sophistication_score = 1.0 - retail_likelihood

        # Exit patterns (simplified - would need position tracking for full implementation)
        # For now, we'll use placeholder values
        holds_to_resolution_ratio = 0.0
        early_exit_ratio = 0.0

        # If we have market data with resolution info, we can compute basic exit patterns
        if self.markets is not None and 'is_resolved' in self.markets.columns:
            trades_with_markets = trades.merge(
                self.markets[['market_id', 'is_resolved']],
                on='market_id',
                how='left'
            )

            resolved_trades = trades_with_markets[
                trades_with_markets['is_resolved'] == True
            ]

            if len(resolved_trades) > 0:
                # Simplified: assume if they traded in a resolved market, they held to resolution
                # This would need more sophisticated logic with position tracking
                holds_to_resolution_ratio = float(len(resolved_trades) / len(trades))
                early_exit_ratio = 1.0 - holds_to_resolution_ratio

        # Profitability metrics (optional - would need full position tracking)
        total_pnl = None
        win_rate = None
        sharpe_ratio = None

        return BehaviorProfile(
            wallet=wallet,
            # Trading frequency
            avg_trades_per_day=frequency['avg_trades_per_day'],
            burst_episodes=frequency['burst_trading_episodes'],
            off_hours_ratio=frequency['off_hours_ratio'],
            # Sizing
            avg_trade_size=sizing['avg_trade_size'],
            max_trade_size=sizing['max_trade_size'],
            size_variance=sizing['size_variance'],
            # Market diversity
            unique_markets=diversity['unique_markets'],
            market_concentration=diversity['market_concentration'],
            niche_market_ratio=diversity['niche_market_ratio'],
            # Profitability
            total_pnl=total_pnl,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            # Exit patterns
            holds_to_resolution_ratio=holds_to_resolution_ratio,
            early_exit_ratio=early_exit_ratio,
            # Scores
            retail_likelihood=retail_likelihood,
            sophistication_score=sophistication_score,
        )

    def build_all_profiles(
        self,
        wallets: Optional[List[str]] = None
    ) -> Dict[str, BehaviorProfile]:
        """
        Build behavior profiles for all wallets.
        """
        if wallets is None:
            # Get all unique wallets from trades
            maker_wallets = self.trades['maker_address'].unique()
            taker_wallets = self.trades['taker_address'].unique()
            wallets = list(set(list(maker_wallets) + list(taker_wallets)))

        logger.info(f"Building behavior profiles for {len(wallets)} wallets")

        profiles = {}
        for i, wallet in enumerate(wallets):
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(wallets)} wallets")

            try:
                profile = self.build_behavior_profile(wallet)
                profiles[wallet] = profile
            except Exception as e:
                logger.error(f"Error building profile for wallet {wallet}: {e}")
                continue

        logger.info(f"Successfully built {len(profiles)} behavior profiles")
        return profiles
