"""
Main insider signal detector.

Orchestrates all signal detectors and produces trade-level signals.
"""

import json
import logging
from typing import Callable, Dict, List, Optional

import pandas as pd

from src.analysis import WalletCluster, WalletProfile

from .aggregator import SignalAggregator
from .cluster import ClusterSignalDetector
from .freshness import FreshnessSignalDetector
from .funding import FundingSignalDetector
from .market_context import MarketContextAnalyzer
from .negative_filter import NegativeSignalFilter
from .sizing import SizingSignalDetector
from .timing import TimingSignalDetector
from .types import (
    AggregatedSignal,
    Market,
    Trade,
    TradeSignal,
)

logger = logging.getLogger(__name__)


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
        clusters: List[WalletCluster],
        known_entities: List[str] = None,
    ):
        """
        Initialize detector with all required data.

        Args:
            trades_df: DataFrame with all trades
            markets_df: DataFrame with market data
            wallet_profiles: Dict mapping wallet address to WalletProfile
            clusters: List of wallet clusters
            known_entities: Optional list of known non-insider wallets
        """
        self.trades_df = trades_df
        self.markets_df = markets_df
        self.wallet_profiles = wallet_profiles
        self.clusters = clusters

        # Initialize all detectors
        self.freshness_detector = FreshnessSignalDetector(wallet_profiles)
        self.timing_detector = TimingSignalDetector(trades_df, markets_df)
        self.sizing_detector = SizingSignalDetector(trades_df, markets_df)
        self.funding_detector = FundingSignalDetector(wallet_profiles)
        self.cluster_detector = ClusterSignalDetector(
            wallet_profiles, clusters, trades_df
        )
        self.market_analyzer = MarketContextAnalyzer(markets_df)
        self.negative_filter = NegativeSignalFilter(
            wallet_profiles, known_entities
        )
        self.aggregator = SignalAggregator()

        # Build market lookup
        self._markets_lookup: Dict[str, Market] = {}
        if markets_df is not None and not markets_df.empty:
            for _, row in markets_df.iterrows():
                market = Market.from_row(row)
                self._markets_lookup[market.market_id] = market

        logger.info(
            f"Initialized InsiderSignalDetector with {len(wallet_profiles)} profiles, "
            f"{len(clusters)} clusters, {len(self._markets_lookup)} markets"
        )

    def detect_signals(self, trade: Trade) -> TradeSignal:
        """
        Detect all signals for a single trade.

        Args:
            trade: Trade to analyze

        Returns:
            TradeSignal with all detected signals and final score
        """
        wallet_profile = self.wallet_profiles.get(trade.wallet)
        market = self._markets_lookup.get(trade.market_id)

        # If no profile found, return empty signal
        if not wallet_profile:
            return TradeSignal(
                trade=trade,
                wallet_profile=None,
                signals=[],
                aggregated=AggregatedSignal(
                    final_score=0.0,
                    raw_score=0.0,
                    negative_discount=0.0,
                    market_boost=1.0,
                    category_scores={},
                    signals=[],
                ),
                final_score=0.0,
            )

        # If no market found, create minimal market object
        if not market:
            market = Market(
                market_id=trade.market_id,
                question="",
                category="Unknown",
                volume=0,
                liquidity=0,
                is_resolved=False,
            )

        # Collect signals from all detectors
        all_signals = []

        # Freshness signals
        all_signals.extend(
            self.freshness_detector.detect(trade, wallet_profile)
        )

        # Timing signals
        all_signals.extend(
            self.timing_detector.detect(trade, wallet_profile, market)
        )

        # Sizing signals
        all_signals.extend(
            self.sizing_detector.detect(trade, wallet_profile, market)
        )

        # Funding signals
        all_signals.extend(
            self.funding_detector.detect(trade, wallet_profile)
        )

        # Cluster signals
        all_signals.extend(
            self.cluster_detector.detect(trade, wallet_profile)
        )

        # Get negative score
        negative_score = self.negative_filter.compute_negative_score(
            wallet_profile
        )

        # Get market context boost
        market_context = self.market_analyzer.compute_market_boost(market)

        # Aggregate all signals
        aggregated = self.aggregator.aggregate(
            all_signals, negative_score, market_context
        )

        return TradeSignal(
            trade=trade,
            wallet_profile=wallet_profile,
            signals=all_signals,
            aggregated=aggregated,
            final_score=aggregated.final_score,
        )

    def detect_all_signals(
        self,
        trades: List[Trade] = None,
        min_score: float = 0.3,
        progress_callback: Callable[[int, int], None] = None,
        max_trades: int = None,
    ) -> List[TradeSignal]:
        """
        Detect signals for all trades.

        Args:
            trades: Optional list of trades (if None, uses vectorized detection)
            min_score: Only return signals above this threshold
            progress_callback: Optional callback(current, total) for progress updates
            max_trades: Optional limit on number of trades to process

        Returns:
            List of TradeSignals sorted by final_score descending
        """
        # Use vectorized detection for better performance
        if trades is None:
            return self._detect_signals_vectorized(
                min_score=min_score,
                progress_callback=progress_callback,
                max_trades=max_trades,
            )

        # Fallback to iterative if trades provided
        results = []
        total = len(trades)

        logger.info(f"Processing {total} trades for signal detection...")

        for i, trade in enumerate(trades):
            signal = self.detect_signals(trade)
            if signal.final_score >= min_score:
                results.append(signal)

            if progress_callback and (i % 5000 == 0 or i == total - 1):
                progress_callback(i + 1, total)

        results.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(
            f"Detected {len(results)} signals above threshold {min_score} "
            f"from {total} trades"
        )

        return results

    def _detect_signals_vectorized(
        self,
        min_score: float = 0.3,
        progress_callback: Callable[[int, int], None] = None,
        max_trades: int = None,
    ) -> List[TradeSignal]:
        """
        Vectorized signal detection for performance.

        Computes signals at the wallet level first, then applies to trades.
        Only processes wallets that show truly suspicious patterns.
        """
        logger.info("Running vectorized signal detection...")

        # Pre-compute wallet-level scores
        wallet_scores = {}
        total_wallets = len(self.wallet_profiles)

        for i, (wallet, profile) in enumerate(self.wallet_profiles.items()):
            # Freshness score - based on BOTH days on platform AND trade count
            # A truly fresh wallet has few trades AND is new to the platform
            freshness_score = 0.0

            # Single trade = highest suspicion (zero history)
            if profile.total_trades == 1:
                freshness_score = 1.0
            # Very few trades (< 5) could be new or casual
            elif profile.total_trades < 5:
                if profile.days_on_polymarket < 7:
                    freshness_score = 0.9  # New AND few trades
                elif profile.days_on_polymarket < 30:
                    freshness_score = 0.7  # Relatively new
                else:
                    freshness_score = 0.3  # Old but inactive - less suspicious
            # Moderate trades (5-20)
            elif profile.total_trades < 20:
                if profile.days_on_polymarket < 7:
                    freshness_score = 0.6  # New with some activity
                elif profile.days_on_polymarket < 30:
                    freshness_score = 0.4
                else:
                    freshness_score = 0.1  # Established user
            # Many trades (20+) - unlikely to be fresh insider
            else:
                freshness_score = 0.0  # Established trader

            # Funding score - only high-risk sources matter
            funding_score = 0.0
            if profile.has_privacy_funding or profile.primary_funding_source == "privacy_tool":
                funding_score = 1.0  # Critical - privacy tools
            elif profile.primary_funding_source == "tracked_wallet":
                funding_score = 0.8  # From another wallet we're watching
            elif profile.primary_funding_source == "cross_chain":
                funding_score = 0.5  # Bridge funding - moderate
            # CEX funding is normal - no score

            # Negative score
            negative_score = self.negative_filter.compute_negative_score(profile)

            wallet_scores[wallet] = {
                "freshness": freshness_score,
                "funding": funding_score,
                "negative": negative_score,
                "profile": profile,
            }

            if progress_callback and i % 100 == 0:
                progress_callback(i, total_wallets + 100)

        logger.info(f"Pre-computed scores for {len(wallet_scores)} wallets")

        # Filter to only truly suspicious wallets
        # Must have significant freshness OR funding signal, AND low negative score
        high_potential_wallets = {
            w for w, scores in wallet_scores.items()
            if (scores["freshness"] >= 0.5 or scores["funding"] >= 0.5)
            and scores["negative"] < 0.5
        }

        logger.info(f"Found {len(high_potential_wallets)} suspicious wallets (out of {len(wallet_scores)})")

        if not high_potential_wallets:
            logger.info("No suspicious wallets found - all wallets appear normal")
            if progress_callback:
                progress_callback(100, 100)
            return []

        # Filter trades where EITHER maker or taker is suspicious
        df = self.trades_df
        mask = df["maker_address"].isin(high_potential_wallets) | df["taker_address"].isin(high_potential_wallets)
        filtered_trades = df[mask]

        if max_trades and len(filtered_trades) > max_trades:
            filtered_trades = filtered_trades.head(max_trades)

        logger.info(f"Processing {len(filtered_trades)} trades from {len(high_potential_wallets)} suspicious wallets")

        results = []
        total = len(filtered_trades)

        if total == 0:
            if progress_callback:
                progress_callback(100, 100)
            return []

        for i, row in enumerate(filtered_trades.itertuples(index=False)):
            # Check both maker and taker
            # A trade can have two suspicious wallets
            sides_to_process = []
            
            maker = getattr(row, "maker_address", "")
            if maker in high_potential_wallets:
                sides_to_process.append((maker, "maker"))
                
            taker = getattr(row, "taker_address", "")
            if taker in high_potential_wallets:
                sides_to_process.append((taker, "taker"))

            for wallet, role in sides_to_process:
                scores = wallet_scores.get(wallet)
                if not scores:
                    continue

                profile = scores["profile"]

                # Create trade object (adjusting side/size if necessary for perspective)
                # Note: Trade object assumes 'wallet' is the primary actor
                trade = Trade(
                    trade_id=str(getattr(row, 'trade_id', '')),
                    market_id=str(getattr(row, 'market_id', '')),
                    timestamp=row.timestamp,
                    wallet=wallet,
                    side=str(getattr(row, 'side', '')),
                    outcome=str(getattr(row, 'outcome', '')),
                    size=float(getattr(row, 'size', 0)),
                    price=float(getattr(row, 'price', 0)),
                    notional=float(getattr(row, 'notional', 0)),
                    tx_hash=getattr(row, 'tx_hash', None),
                )

                # Get market
                market = self._markets_lookup.get(trade.market_id)
                if not market:
                    market = Market(
                        market_id=trade.market_id,
                        question="",
                        category="Unknown",
                        volume=0,
                        liquidity=0,
                        is_resolved=False,
                    )

                # Build signals
                all_signals = []

                # Freshness signals
                if scores["freshness"] >= 0.5:
                    from .types import FreshnessSignalType, SignalCategory, Signal
                    if profile.total_trades == 1:
                        sig_type = FreshnessSignalType.ZERO_HISTORY
                    elif profile.days_on_polymarket < 7:
                        sig_type = FreshnessSignalType.NEW_WALLET
                    else:
                        sig_type = FreshnessSignalType.RECENT_FUNDING

                    all_signals.append(Signal(
                        type=sig_type,
                        category=SignalCategory.FRESHNESS,
                        confidence=scores["freshness"],
                        details={
                            "days_on_platform": profile.days_on_polymarket,
                            "total_trades": profile.total_trades,
                        }
                    ))

                # Funding signals
                if scores["funding"] >= 0.5:
                    from .types import FundingSignalType
                    if scores["funding"] >= 0.9:
                        sig_type = FundingSignalType.PRIVACY_FUNDING
                    elif scores["funding"] >= 0.7:
                        sig_type = FundingSignalType.TRACKED_WALLET_FUNDING
                    else:
                        sig_type = FundingSignalType.BRIDGE_FUNDING

                    all_signals.append(Signal(
                        type=sig_type,
                        category=SignalCategory.FUNDING,
                        confidence=scores["funding"],
                        details={"source": profile.primary_funding_source}
                    ))

                # Sizing signals (trade-specific)
                sizing_signals = self.sizing_detector.detect(trade, profile, market)
                all_signals.extend(sizing_signals)

                # Must have at least one signal
                if not all_signals:
                    continue

                # Market context
                market_context = self.market_analyzer.compute_market_boost(market)

                # Aggregate
                aggregated = self.aggregator.aggregate(
                    all_signals, scores["negative"], market_context
                )

                if aggregated.final_score >= min_score:
                    results.append(TradeSignal(
                        trade=trade,
                        wallet_profile=profile,
                        signals=all_signals,
                        aggregated=aggregated,
                        final_score=aggregated.final_score,
                    ))

            if progress_callback and (i % 1000 == 0 or i == total - 1):
                pct = int(((total_wallets + i) / (total_wallets + total)) * 100)
                progress_callback(pct, 100)
            
            # Sort results strictly at the end
            results.sort(key=lambda x: x.final_score, reverse=True)

        logger.info(f"Detected {len(results)} signals above threshold {min_score}")

        if progress_callback:
            progress_callback(100, 100)

        return results

    def _trades_from_dataframe_optimized(self) -> List[Trade]:
        """
        Convert trades DataFrame to list of Trade objects.

        Optimized to only process trades from wallets we have profiles for.
        """
        trades = []
        known_wallets = set(self.wallet_profiles.keys())

        if not known_wallets:
            logger.warning("No wallet profiles loaded, skipping trade processing")
            return trades

        # Filter trades to only those involving known wallets
        df = self.trades_df

        # Create mask for trades involving known wallets
        maker_mask = df["maker_address"].isin(known_wallets)
        taker_mask = df["taker_address"].isin(known_wallets)

        # Process maker trades
        maker_trades = df[maker_mask]
        logger.info(f"Found {len(maker_trades)} trades with known maker wallets")

        for row in maker_trades.itertuples(index=False):
            trades.append(Trade(
                trade_id=str(getattr(row, 'trade_id', '')),
                market_id=str(getattr(row, 'market_id', '')),
                timestamp=row.timestamp,
                wallet=str(row.maker_address),
                side=str(getattr(row, 'side', '')),
                outcome=str(getattr(row, 'outcome', '')),
                size=float(getattr(row, 'size', 0)),
                price=float(getattr(row, 'price', 0)),
                notional=float(getattr(row, 'notional', 0)),
                tx_hash=getattr(row, 'tx_hash', None),
            ))

        # Process taker trades (only for wallets not already processed as maker)
        taker_only_mask = taker_mask & ~maker_mask
        taker_trades = df[taker_only_mask]
        logger.info(f"Found {len(taker_trades)} additional trades with known taker wallets")

        for row in taker_trades.itertuples(index=False):
            trades.append(Trade(
                trade_id=str(getattr(row, 'trade_id', '')),
                market_id=str(getattr(row, 'market_id', '')),
                timestamp=row.timestamp,
                wallet=str(row.taker_address),
                side=str(getattr(row, 'side', '')),
                outcome=str(getattr(row, 'outcome', '')),
                size=float(getattr(row, 'size', 0)),
                price=float(getattr(row, 'price', 0)),
                notional=float(getattr(row, 'notional', 0)),
                tx_hash=getattr(row, 'tx_hash', None),
            ))

        logger.info(f"Total trades to analyze: {len(trades)}")
        return trades

    def _trades_from_dataframe(self) -> List[Trade]:
        """Convert trades DataFrame to list of Trade objects (legacy, slow)."""
        trades = []

        for _, row in self.trades_df.iterrows():
            # Create trades for both maker and taker
            # We analyze each side separately
            trades.append(Trade.from_row(row, wallet_column="maker_address"))
            trades.append(Trade.from_row(row, wallet_column="taker_address"))

        return trades

    def to_dataframe(self, signals: List[TradeSignal]) -> pd.DataFrame:
        """
        Convert list of TradeSignals to DataFrame for storage.

        Args:
            signals: List of TradeSignals

        Returns:
            DataFrame with signal data
        """
        records = []

        for signal in signals:
            record = {
                "trade_id": signal.trade.trade_id,
                "market_id": signal.trade.market_id,
                "wallet": signal.trade.wallet,
                "timestamp": signal.trade.timestamp,
                "side": signal.trade.side,
                "outcome": signal.trade.outcome,
                "size": signal.trade.size,
                "price": signal.trade.price,
                "notional": signal.trade.notional,
                "final_score": signal.final_score,
                "raw_score": signal.aggregated.raw_score,
                "negative_discount": signal.aggregated.negative_discount,
                "market_boost": signal.aggregated.market_boost,
                "freshness_score": signal.aggregated.category_scores.get("freshness", 0),
                "timing_score": signal.aggregated.category_scores.get("timing", 0),
                "sizing_score": signal.aggregated.category_scores.get("sizing", 0),
                "funding_score": signal.aggregated.category_scores.get("funding", 0),
                "cluster_score": signal.aggregated.category_scores.get("cluster", 0),
                "signal_count": len(signal.signals),
                "signals_json": json.dumps(signal.aggregated.to_dict()),
            }

            # Add wallet profile info if available
            if signal.wallet_profile:
                record.update({
                    "wallet_cluster_id": signal.wallet_profile.cluster_id,
                    "wallet_is_whale": signal.wallet_profile.is_whale,
                    "wallet_freshness": signal.wallet_profile.freshness_score,
                    "wallet_retail_likelihood": signal.wallet_profile.retail_likelihood,
                })

            records.append(record)

        return pd.DataFrame(records)

    def analyze_trade(self, trade_id: str) -> Optional[Dict]:
        """
        Analyze a specific trade for debugging.

        Args:
            trade_id: ID of trade to analyze

        Returns:
            Dict with detailed signal breakdown, or None if trade not found
        """
        # Find trade
        mask = self.trades_df["trade_id"] == trade_id
        if not mask.any():
            return None

        row = self.trades_df[mask].iloc[0]
        trade = Trade.from_row(row, wallet_column="maker_address")

        signal = self.detect_signals(trade)

        return {
            "trade": {
                "trade_id": trade.trade_id,
                "market_id": trade.market_id,
                "wallet": trade.wallet,
                "timestamp": str(trade.timestamp),
                "side": trade.side,
                "size": trade.size,
                "price": trade.price,
            },
            "wallet_profile": {
                "total_trades": signal.wallet_profile.total_trades if signal.wallet_profile else 0,
                "freshness_score": signal.wallet_profile.freshness_score if signal.wallet_profile else 0,
                "retail_likelihood": signal.wallet_profile.retail_likelihood if signal.wallet_profile else 0,
            },
            "score_breakdown": self.aggregator.explain_score(signal.aggregated),
            "signals": [s.to_dict() for s in signal.signals],
            "filter_reasons": self.negative_filter.get_filter_reasons(
                signal.wallet_profile
            ) if signal.wallet_profile else [],
        }

    def get_top_suspicious_trades(
        self,
        n: int = 100,
        min_score: float = 0.5,
    ) -> List[TradeSignal]:
        """
        Get top N most suspicious trades.

        Args:
            n: Number of trades to return
            min_score: Minimum score threshold

        Returns:
            List of top TradeSignals
        """
        all_signals = self.detect_all_signals(min_score=min_score)
        return all_signals[:n]

    def get_suspicious_wallets(
        self,
        min_score: float = 0.5,
        min_trades: int = 1,
    ) -> Dict[str, Dict]:
        """
        Get wallets with suspicious trading activity.

        Args:
            min_score: Minimum average signal score
            min_trades: Minimum number of suspicious trades

        Returns:
            Dict mapping wallet address to summary stats
        """
        all_signals = self.detect_all_signals(min_score=min_score * 0.5)

        # Group by wallet
        wallet_signals: Dict[str, List[TradeSignal]] = {}
        for signal in all_signals:
            wallet = signal.trade.wallet
            if wallet not in wallet_signals:
                wallet_signals[wallet] = []
            wallet_signals[wallet].append(signal)

        # Compute stats per wallet
        suspicious_wallets = {}
        for wallet, signals in wallet_signals.items():
            avg_score = sum(s.final_score for s in signals) / len(signals)

            if avg_score >= min_score and len(signals) >= min_trades:
                suspicious_wallets[wallet] = {
                    "trade_count": len(signals),
                    "avg_score": round(avg_score, 3),
                    "max_score": round(max(s.final_score for s in signals), 3),
                    "total_volume": sum(s.trade.notional for s in signals),
                    "markets": list(set(s.trade.market_id for s in signals)),
                }

        # Sort by avg score
        return dict(
            sorted(
                suspicious_wallets.items(),
                key=lambda x: x[1]["avg_score"],
                reverse=True,
            )
        )
