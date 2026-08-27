import pandas as pd
import numpy as np
from typing import Dict, List
from src.signals.types import Trade, TradeSignal
from src.analysis.types import WalletProfile

class GroundTruthLabeler:
    """
    Creates ground truth labels using hybrid approach.

    Label Sources:
    1. Profitability-based: Wallet profited on resolved markets
    2. Pattern-based: Wallet exhibits multiple insider signals
    3. Manual review: Analyst-labeled examples (optional)

    Final label is weighted combination.
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        markets_df: pd.DataFrame,
        wallet_profiles: Dict[str, WalletProfile]
    ):
        self.trades = trades_df
        # Create quick lookup for markets
        self.markets = {
            row.market_id: row 
            for _, row in markets_df.iterrows()
        }
        self.profiles = wallet_profiles

    def compute_profitability_label(
        self,
        wallet: str,
        trade: Trade
    ) -> float:
        """
        Label based on trade profitability.

        Criteria for positive label:
        - Trade on market that resolved
        - Wallet was on winning side
        - Entry price was favorable (<30% for winning outcome)
        """
        market = self.markets.get(trade.market_id)
        if market is None:
            return 0.0
            
        # Check if resolved
        if not getattr(market, "is_resolved", False):
            return 0.0

        # Check if trade was on winning side
        winning_outcome = getattr(market, "resolution", None)  # "Yes" or "No"
        if not winning_outcome:
            return 0.0
            
        on_winning_side = (
            (trade.outcome == winning_outcome and trade.side == "BUY") or
            (trade.outcome != winning_outcome and trade.side == "SELL")
        )

        if not on_winning_side:
            return 0.0

        # Check if entry was at favorable odds
        if trade.price < 0.3:  # Bought at <30% when it resolved to 100%
            return 1.0
        elif trade.price < 0.5:
            return 0.7
        else:
            return 0.3

    def compute_pattern_label(
        self,
        wallet: str,
        trade: Trade,
        signal: TradeSignal
    ) -> float:
        """
        Label based on signal pattern strength.

        Criteria:
        - Multiple high-confidence signals present
        - Low negative signals
        - Wallet shows consistent profitable pattern
        """
        score = 0.0

        if signal is None:
            return 0.0

        # High raw signal score
        if signal.aggregated.raw_score > 0.7:
            score += 0.4

        # Low negative signals
        if signal.aggregated.negative_discount < 0.3:
            score += 0.3

        # Wallet has high win rate
        profile = self.profiles.get(wallet)
        if profile and profile.win_rate is not None and profile.win_rate > 0.65:
            score += 0.3

        return min(score, 1.0)

    def generate_label(
        self,
        trade: Trade,
        signal: TradeSignal,
        profitability_weight: float = 0.6,
        pattern_weight: float = 0.4
    ) -> float:
        """
        Generate final hybrid label.

        Returns continuous label 0-1.
        For training, can threshold to binary.
        """
        profit_label = self.compute_profitability_label(trade.wallet, trade)
        pattern_label = self.compute_pattern_label(trade.wallet, trade, signal)

        return (
            profitability_weight * profit_label +
            pattern_weight * pattern_label
        )

    def generate_all_labels(
        self,
        trades: List[Trade],
        signals: Dict[str, TradeSignal],
        binary_threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate labels for all trades.

        Returns DataFrame with:
        - trade_id
        - label_continuous (0-1)
        - label_binary (0 or 1)
        - label_source (profit/pattern/mixed)
        """
        labels = []
        for trade in trades:
            signal = signals.get(trade.trade_id)
            if not signal:
                continue

            label = self.generate_label(trade, signal)
            labels.append({
                "trade_id": trade.trade_id,
                "label_continuous": label,
                "label_binary": 1 if label >= binary_threshold else 0,
                "profit_component": self.compute_profitability_label(trade.wallet, trade),
                "pattern_component": self.compute_pattern_label(trade.wallet, trade, signal),
            })

        return pd.DataFrame(labels)

    def generate_labels_batch(
        self,
        trades_df: pd.DataFrame,
        profitability_weight: float = 0.6,
        pattern_weight: float = 0.4
    ) -> pd.Series:
        """
        Vectorized batch label generation for all trades.
        
        This is 10-20x faster than calling generate_label() in a loop.
        
        Args:
            trades_df: DataFrame with trade data
            profitability_weight: Weight for profitability-based labels
            pattern_weight: Weight for pattern-based labels
            
        Returns:
            Series of labels (float 0-1) with same index as trades_df
        """
        if trades_df.empty:
            return pd.Series(dtype=float)
        
        # Detect column names
        wallet_col = "maker_address" if "maker_address" in trades_df.columns else "wallet"
        market_col = "market_id" if "market_id" in trades_df.columns else "condition_id"
        
        # Initialize labels as zeros
        labels = pd.Series(0.0, index=trades_df.index)
        
        # --- Profitability-based labels (vectorized) ---
        # Check if market is resolved and trade was on winning side
        
        def get_market_info(market_id):
            """Get market resolution info."""
            market = self.markets.get(market_id)
            if market is None:
                return None, None
            # Handle both pandas Series and objects
            if isinstance(market, pd.Series):
                is_resolved = market.get('is_resolved', False)
                resolution = market.get('resolution', None)
            else:
                is_resolved = getattr(market, 'is_resolved', False)
                resolution = getattr(market, 'resolution', None)
            return is_resolved, resolution
        
        # Get market resolution info for each trade
        market_info = trades_df[market_col].map(lambda m: get_market_info(m))
        is_resolved = market_info.map(lambda x: x[0] if x else False)
        resolutions = market_info.map(lambda x: x[1] if x else None)
        
        # Check if on winning side
        # on_winning_side = (trade.outcome == winning_outcome and trade.side == "BUY") or
        #                   (trade.outcome != winning_outcome and trade.side == "SELL")
        outcomes = trades_df["outcome"] if "outcome" in trades_df.columns else pd.Series("", index=trades_df.index)
        sides = trades_df["side"] if "side" in trades_df.columns else pd.Series("", index=trades_df.index)
        
        on_winning_side = (
            ((outcomes == resolutions) & (sides == "BUY")) |
            ((outcomes != resolutions) & (sides == "SELL"))
        )
        
        # Calculate profitability score based on entry price
        prices = trades_df["price"]
        profit_score = pd.Series(0.0, index=trades_df.index)
        
        # Only score trades that are resolved and on winning side
        valid_mask = is_resolved & on_winning_side
        
        # Price < 0.3 -> 1.0, price < 0.5 -> 0.7, else 0.3
        profit_score = np.where(
            valid_mask,
            np.where(prices < 0.3, 1.0, np.where(prices < 0.5, 0.7, 0.3)),
            0.0
        )
        
        # --- Pattern-based labels (simplified for batch) ---
        # We use wallet win rate as a proxy since signals are typically sparse
        wallets = trades_df[wallet_col]
        
        def get_win_rate(wallet):
            profile = self.profiles.get(wallet)
            if profile and profile.win_rate is not None:
                return profile.win_rate
            return 0.0
        
        win_rates = wallets.map(get_win_rate)
        pattern_score = np.where(win_rates > 0.65, 0.3, 0.0)
        
        # Combine scores
        labels = profitability_weight * profit_score + pattern_weight * pattern_score
        
        return pd.Series(labels, index=trades_df.index)

