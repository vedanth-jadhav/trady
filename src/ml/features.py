
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any
from src.signals.types import Trade, Market
from src.analysis.types import WalletProfile

HIGH_INSIDER = {"Crypto", "Tech", "Science"}  # Example categories

class FeatureEngineer:
    """
    Transforms raw data and signals into ML features.

    Feature Categories:
    1. Rule-based signal scores (from Phase 3)
    2. Raw wallet features
    3. Raw trade features
    4. Market context features
    5. Temporal features
    6. Interaction features
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        wallet_profiles: Dict[str, WalletProfile],
        signals: Dict[str, Any] # TradeSignal as dict or object
    ):
        self.trades = trades_df
        # Create quick lookup for markets
        self.markets = {
            row.market_id: Market.from_row(row) 
            for _, row in markets_df.iterrows()
        }
        self.profiles = wallet_profiles
        self.signals = signals
        
        # Pre-compute market statistics ONCE to avoid O(N²) per-trade filtering
        self._precompute_market_stats()

    def _precompute_market_stats(self):
        """
        Pre-compute market-level statistics ONCE to avoid O(N²) complexity.
        
        Instead of filtering trades DataFrame for each trade (6.6M * 6.6M = 43T ops),
        we compute all market stats in a single groupby (O(N)).
        """
        if self.trades.empty:
            self.market_stats = {}
            return
        
        # Detect wallet column name
        wallet_col = "maker_address" if "maker_address" in self.trades.columns else "wallet"
        market_col = "market_id" if "market_id" in self.trades.columns else "condition_id"
        
        # Single O(N) aggregation for all markets
        stats_df = self.trades.groupby(market_col).agg({
            wallet_col: 'nunique',
            'notional': 'median',
        }).rename(columns={wallet_col: 'participant_count', 'notional': 'median_size'})
        
        # Convert to dict for O(1) lookup
        self.market_stats = stats_df.to_dict('index')

    def extract_features(self, trade: Trade) -> Dict[str, float]:
        """
        Extract all features for a single trade.

        Returns dict of feature_name -> value.
        """
        features = {}

        # 1. Signal scores
        features.update(self._get_signal_features(trade))

        # 2. Wallet features
        features.update(self._get_wallet_features(trade))

        # 3. Trade features
        features.update(self._get_trade_features(trade))

        # 4. Market features
        features.update(self._get_market_features(trade))

        # 5. Temporal features
        features.update(self._get_temporal_features(trade))

        # 6. Interaction features
        features.update(self._get_interaction_features(trade, features))

        return features

    def extract_features_batch(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized batch feature extraction for all trades.
        
        This is 50-100x faster than calling extract_features() in a loop
        because it uses pandas vectorized operations instead of Python loops.
        
        Args:
            trades_df: DataFrame with trade data
            
        Returns:
            DataFrame with all features (one row per trade)
        """
        if trades_df.empty:
            return pd.DataFrame()
        
        # Detect column names
        wallet_col = "maker_address" if "maker_address" in trades_df.columns else "wallet"
        market_col = "market_id" if "market_id" in trades_df.columns else "condition_id"
        
        # --- Trade Features (fully vectorized) ---
        features = pd.DataFrame({
            # Size
            "trade_size": trades_df["size"] if "size" in trades_df.columns else 0,
            "trade_notional": trades_df["notional"],
            "trade_size_log": np.log1p(trades_df["notional"].astype(float)),
            
            # Price/odds
            "trade_price": trades_df["price"],
            "trade_is_longshot": (trades_df["price"] < 0.2).astype(int),
            "trade_is_favorite": (trades_df["price"] > 0.8).astype(int),
            "trade_odds_ratio": trades_df["price"] / (1 - trades_df["price"]).replace(0, 0.001),
            
            # Direction
            "trade_is_buy": (trades_df["side"] == "BUY").astype(int) if "side" in trades_df.columns else 0,
            "trade_is_yes": (trades_df["outcome"] == "Yes").astype(int) if "outcome" in trades_df.columns else 0,
        }, index=trades_df.index)
        
        # --- Market Features (using pre-computed stats) ---
        # Create lookup series from market_stats dict
        market_ids = trades_df[market_col]
        participant_counts = market_ids.map(lambda m: self.market_stats.get(m, {}).get('participant_count', 0))
        median_sizes = market_ids.map(lambda m: self.market_stats.get(m, {}).get('median_size', 0))
        
        # Market metadata lookup
        def get_market_attr(market_id, attr, default=0):
            market = self.markets.get(market_id)
            return getattr(market, attr, default) if market else default
        
        features["market_volume"] = market_ids.map(lambda m: get_market_attr(m, 'volume', 0))
        features["market_volume_log"] = np.log1p(features["market_volume"].astype(float))
        features["market_liquidity"] = market_ids.map(lambda m: get_market_attr(m, 'liquidity', 0))
        features["market_participant_count"] = participant_counts
        features["market_is_niche"] = (participant_counts < 50).astype(int)
        features["market_trade_vs_median"] = np.where(median_sizes > 0, trades_df["notional"] / median_sizes, 0)
        features["market_pct_of_volume"] = np.where(
            features["market_volume"] > 0, 
            trades_df["notional"] / features["market_volume"], 
            0
        )
        
        # Category features
        features["market_is_high_insider_category"] = market_ids.map(
            lambda m: 1 if get_market_attr(m, 'category', '') in HIGH_INSIDER else 0
        )
        features["market_is_political"] = market_ids.map(
            lambda m: 1 if get_market_attr(m, 'category', '') == "Politics" else 0
        )
        features["market_is_business"] = market_ids.map(
            lambda m: 1 if get_market_attr(m, 'category', '') == "Business" else 0
        )
        
        # --- Temporal Features (vectorized) ---
        timestamps = pd.to_datetime(trades_df["timestamp"]) if "timestamp" in trades_df.columns else pd.NaT
        if not timestamps.isna().all():
            features["time_hour"] = timestamps.dt.hour
            features["time_is_off_hours"] = ((timestamps.dt.hour < 9) | (timestamps.dt.hour > 21)).astype(int)
            features["time_is_weekend"] = (timestamps.dt.weekday >= 5).astype(int)
            features["time_hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            features["time_hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            features["time_dow_sin"] = np.sin(2 * np.pi * timestamps.dt.weekday / 7)
            features["time_dow_cos"] = np.cos(2 * np.pi * timestamps.dt.weekday / 7)
        else:
            for col in ["time_hour", "time_is_off_hours", "time_is_weekend", 
                       "time_hour_sin", "time_hour_cos", "time_dow_sin", "time_dow_cos"]:
                features[col] = 0
        
        # --- Wallet Features (using profiles lookup) ---
        wallets = trades_df[wallet_col]
        
        def get_profile_attr(wallet, attr, default=0):
            profile = self.profiles.get(wallet)
            if not profile:
                return default
            val = getattr(profile, attr, default)
            return val if val is not None else default
        
        features["wallet_freshness"] = wallets.map(lambda w: get_profile_attr(w, 'freshness_score', 0))
        features["wallet_days_active"] = wallets.map(lambda w: get_profile_attr(w, 'days_on_polymarket', 0))
        features["wallet_is_new"] = (features["wallet_days_active"] < 7).astype(int)
        features["wallet_total_trades"] = wallets.map(lambda w: get_profile_attr(w, 'total_trades', 0))
        features["wallet_total_volume"] = wallets.map(lambda w: get_profile_attr(w, 'total_volume', 0))
        features["wallet_unique_markets"] = wallets.map(lambda w: get_profile_attr(w, 'unique_markets', 0))
        features["wallet_trades_log"] = np.log1p(features["wallet_total_trades"].astype(float))
        features["wallet_volume_log"] = np.log1p(features["wallet_total_volume"].astype(float))
        features["wallet_retail_likelihood"] = wallets.map(lambda w: get_profile_attr(w, 'retail_likelihood', 0.5))
        features["wallet_win_rate"] = wallets.map(lambda w: get_profile_attr(w, 'win_rate', 0.5))
        features["wallet_sharpe"] = wallets.map(lambda w: get_profile_attr(w, 'sharpe_ratio', 0))
        features["wallet_holds_to_resolution"] = wallets.map(lambda w: get_profile_attr(w, 'holds_to_resolution_ratio', 0))
        features["wallet_early_exit_ratio"] = wallets.map(lambda w: get_profile_attr(w, 'early_exit_ratio', 0))
        features["wallet_funding_risk"] = wallets.map(lambda w: get_profile_attr(w, 'funding_risk_score', 0))
        features["wallet_has_privacy_funding"] = wallets.map(lambda w: 1 if get_profile_attr(w, 'has_privacy_funding', False) else 0)
        features["wallet_is_whale"] = wallets.map(lambda w: 1 if get_profile_attr(w, 'is_whale', False) else 0)
        features["wallet_avg_trade_size"] = wallets.map(lambda w: get_profile_attr(w, 'avg_trade_size', 0))
        features["wallet_max_trade_size"] = wallets.map(lambda w: get_profile_attr(w, 'max_trade_size', 0))
        features["wallet_size_log"] = np.log1p(features["wallet_avg_trade_size"].astype(float))
        features["wallet_off_hours_ratio"] = wallets.map(lambda w: get_profile_attr(w, 'off_hours_ratio', 0))
        features["wallet_burst_episodes"] = wallets.map(lambda w: get_profile_attr(w, 'burst_episodes', 0))
        features["wallet_in_cluster"] = wallets.map(lambda w: 1 if get_profile_attr(w, 'cluster_id', None) else 0)
        features["wallet_cluster_size"] = wallets.map(lambda w: len(get_profile_attr(w, 'cluster_members', []) or []) or 1)
        
        # --- Signal Features (default to 0 if not available) ---
        # Signals are typically sparse, so we default to 0
        for col in ["sig_freshness", "sig_timing", "sig_sizing", "sig_funding", "sig_cluster",
                   "sig_raw_score", "sig_negative_discount", "sig_market_boost",
                   "sig_has_zero_history", "sig_has_pre_news", "sig_has_whale_size",
                   "sig_has_privacy_funding", "sig_has_cluster_coord"]:
            features[col] = 0
        
        # --- Interaction Features (vectorized) ---
        features["inter_fresh_whale"] = features["wallet_freshness"] * features["trade_size_log"]
        features["inter_new_niche"] = features["wallet_is_new"] * features["market_is_niche"]
        features["inter_insider_cat_near_res"] = features["market_is_high_insider_category"] * 0  # No resolution info in batch
        features["inter_privacy_zero_hist"] = features["wallet_has_privacy_funding"] * features["sig_has_zero_history"]
        features["inter_cluster_unusual"] = features["wallet_in_cluster"] * features["sig_has_whale_size"]
        features["inter_offhours_longshot"] = features["time_is_off_hours"] * features["trade_is_longshot"]
        
        # Days to resolution placeholder
        features["market_days_to_resolution"] = 30.0
        features["market_is_near_resolution"] = 0
        
        return features

    def _get_signal_features(self, trade: Trade) -> Dict[str, float]:
        """
        Features from rule-based signal detection.

        These capture domain knowledge from Phase 3.
        """
        signal_data = self.signals.get(trade.trade_id)
        
        # Handle both TradeSignal objects and dicts
        signal = {}
        if signal_data:
             if hasattr(signal_data, 'to_dict'):
                 signal = signal_data.to_dict()
             else:
                 signal = signal_data
        
        # Helper to get nested value safely
        def get_nested(d, key, default=0):
            if not isinstance(d, dict): return default
            return d.get(key, default)

        # Helper to check string IN string representation of signal
        # This is a bit hacky but matches the spec's intent of checking existence
        signal_str = str(signal)

        return {
            # Category scores
            "sig_freshness": get_nested(signal, "freshness_score"),
            "sig_timing": get_nested(signal, "timing_score"),
            "sig_sizing": get_nested(signal, "sizing_score"),
            "sig_funding": get_nested(signal, "funding_score"),
            "sig_cluster": get_nested(signal, "cluster_score"),

            # Aggregated scores
            "sig_raw_score": get_nested(signal, "raw_score"),
            "sig_negative_discount": get_nested(signal, "negative_discount"),
            "sig_market_boost": get_nested(signal, "market_boost"),

            # Individual signal presence (binary)
            "sig_has_zero_history": 1 if "zero_history" in signal_str else 0,
            "sig_has_pre_news": 1 if "pre_news" in signal_str else 0,
            "sig_has_whale_size": 1 if "whale_size" in signal_str else 0,
            "sig_has_privacy_funding": 1 if "privacy_funding" in signal_str else 0,
            "sig_has_cluster_coord": 1 if "cluster_coordination" in signal_str else 0,
        }

    def _get_wallet_features(self, trade: Trade) -> Dict[str, float]:
        """
        Features from wallet profile.
        """
        profile = self.profiles.get(trade.wallet)
        if not profile:
            return {f"wallet_{k}": 0 for k in [
                "freshness", "days_active", "total_trades", "total_volume",
                "unique_markets", "retail_likelihood", "win_rate",
                "holds_to_resolution", "funding_risk", "is_whale",
                "off_hours_ratio", "avg_trade_size", "cluster_size"
            ]}

        return {
            # Freshness
            "wallet_freshness": profile.freshness_score,
            "wallet_days_active": profile.days_on_polymarket,
            "wallet_is_new": 1 if profile.days_on_polymarket < 7 else 0,

            # Activity
            "wallet_total_trades": profile.total_trades,
            "wallet_total_volume": profile.total_volume,
            "wallet_unique_markets": profile.unique_markets,
            "wallet_trades_log": np.log1p(float(profile.total_trades)),
            "wallet_volume_log": np.log1p(float(profile.total_volume)),

            # Behavior
            "wallet_retail_likelihood": profile.retail_likelihood,
            "wallet_win_rate": profile.win_rate if profile.win_rate is not None else 0.5,
            "wallet_sharpe": profile.sharpe_ratio if profile.sharpe_ratio is not None else 0.0,
            "wallet_holds_to_resolution": profile.holds_to_resolution_ratio,
            "wallet_early_exit_ratio": profile.early_exit_ratio,

            # Funding
            "wallet_funding_risk": profile.funding_risk_score,
            "wallet_has_privacy_funding": 1 if profile.has_privacy_funding else 0,

            # Size
            "wallet_is_whale": 1 if profile.is_whale else 0,
            "wallet_avg_trade_size": profile.avg_trade_size,
            "wallet_max_trade_size": profile.max_trade_size,
            "wallet_size_log": np.log1p(float(profile.avg_trade_size)),

            # Timing
            "wallet_off_hours_ratio": profile.off_hours_ratio,
            "wallet_burst_episodes": profile.burst_episodes,

            # Cluster
            "wallet_in_cluster": 1 if profile.cluster_id else 0,
            "wallet_cluster_size": len(profile.cluster_members) if profile.cluster_members else 1,
        }

    def _get_trade_features(self, trade: Trade) -> Dict[str, float]:
        """
        Features from the trade itself.
        """
        return {
            # Size
            "trade_size": trade.size,
            "trade_notional": trade.notional,
            "trade_size_log": np.log1p(float(trade.notional)),

            # Price/odds
            "trade_price": trade.price,
            "trade_is_longshot": 1 if trade.price < 0.2 else 0,
            "trade_is_favorite": 1 if trade.price > 0.8 else 0,
            "trade_odds_ratio": trade.price / (1 - trade.price) if trade.price < 1 else 100,

            # Direction
            "trade_is_buy": 1 if trade.side == "BUY" else 0,
            "trade_is_yes": 1 if trade.outcome == "Yes" else 0,
        }

    def _get_market_features(self, trade: Trade) -> Dict[str, float]:
        """
        Features from market context.
        """
        market = self.markets.get(trade.market_id)
        if not market:
            return {
                "market_volume": 0,
                "market_volume_log": 0,
                "market_liquidity": 0,
                "market_days_to_resolution": 30.0,
                "market_is_high_insider_category": 0,
                "market_is_political": 0,
                "market_is_business": 0,
                "market_participant_count": 0,
                "market_is_niche": 0,
                "market_trade_vs_median": 0,
                "market_pct_of_volume": 0,
                "market_is_near_resolution": 0,
            }

        # Use pre-computed stats - O(1) lookup instead of O(N) filtering
        # This is the critical optimization: avoids 43 trillion DataFrame comparisons
        stats = self.market_stats.get(trade.market_id, {})
        participant_count = stats.get('participant_count', 0)
        median_size = stats.get('median_size', 0)

        features = {
            # Volume
            "market_volume": market.volume,
            "market_volume_log": np.log1p(float(market.volume)),
            "market_liquidity": market.liquidity,

            # Timeline
            "market_days_to_resolution": self._days_until(trade.timestamp, getattr(market, 'end_date', None)),
            
            # Category
            "market_is_high_insider_category": 1 if market.category in HIGH_INSIDER else 0,
            "market_is_political": 1 if market.category == "Politics" else 0,
            "market_is_business": 1 if market.category == "Business" else 0,

            # Participation
            "market_participant_count": participant_count,
            "market_is_niche": 1 if participant_count < 50 else 0,

            # Trade context
            "market_trade_vs_median": trade.notional / median_size if median_size > 0 else 0,
            "market_pct_of_volume": trade.notional / market.volume if market.volume > 0 else 0,
        }
        
        # Timeline percentage needs start date, which we might not have on Market dataclass
        # We can approximate or skip if not available
        features["market_is_near_resolution"] = 1 if features["market_days_to_resolution"] < 7 else 0
        
        return features

    def _get_temporal_features(self, trade: Trade) -> Dict[str, float]:
        """
        Time-based features.
        """
        ts = trade.timestamp
        # Ensure timezone awareness if needed, but assuming UTC for simple logic
         
        return {
            # Time of day
            "time_hour": ts.hour,
            "time_is_off_hours": 1 if ts.hour < 9 or ts.hour > 21 else 0,
            "time_is_weekend": 1 if ts.weekday() >= 5 else 0,

            # Cyclical encoding
            "time_hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "time_hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "time_dow_sin": np.sin(2 * np.pi * ts.weekday() / 7),
            "time_dow_cos": np.cos(2 * np.pi * ts.weekday() / 7),
        }

    def _get_interaction_features(
        self,
        trade: Trade,
        base_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Derived interaction features.

        Captures non-linear relationships between features.
        """
        return {
            # Fresh wallet + large trade
            "inter_fresh_whale": (
                base_features.get("wallet_freshness", 0) *
                base_features.get("trade_size_log", 0)
            ),

            # New wallet + niche market
            "inter_new_niche": (
                base_features.get("wallet_is_new", 0) *
                base_features.get("market_is_niche", 0)
            ),

            # High insider category + near resolution
            "inter_insider_cat_near_res": (
                base_features.get("market_is_high_insider_category", 0) *
                base_features.get("market_is_near_resolution", 0)
            ),

            # Privacy funding + zero history
            "inter_privacy_zero_hist": (
                base_features.get("wallet_has_privacy_funding", 0) *
                base_features.get("sig_has_zero_history", 0)
            ),

            # Cluster + unusual size
            "inter_cluster_unusual": (
                base_features.get("wallet_in_cluster", 0) *
                base_features.get("sig_has_whale_size", 0)
            ),

            # Off hours + longshot
            "inter_offhours_longshot": (
                base_features.get("time_is_off_hours", 0) *
                base_features.get("trade_is_longshot", 0)
            ),
        }

    def _days_until(self, current_ts: datetime, end_date: Any) -> float:
        """Calculate days until end date."""
        if not end_date:
            return 30.0 # Default assumption
        
        # Handle string dates if necessary
        if isinstance(end_date, str):
            try:
                # Basic ISO parsing
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return 30.0
        elif isinstance(end_date, datetime):
            end_dt = end_date
        else:
            return 30.0

        # Ensure timezone compatibility
        if current_ts.tzinfo is None and end_dt.tzinfo is not None:
            current_ts = current_ts.replace(tzinfo=timezone.utc) # Assume UTC
        if end_dt.tzinfo is None and current_ts.tzinfo is not None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        delta = end_dt - current_ts
        return max(0.0, delta.total_seconds() / 86400)
