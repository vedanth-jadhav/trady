"""
Wallet Profile Builder and Negative Signal Detector.

Combines all wallet analyses into unified profiles and detects signals
that indicate a wallet is NOT an insider (negative signals).
"""

import pandas as pd
import numpy as np
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from dataclasses import asdict
import logging

from .types import (
    WalletProfile,
    FreshnessProfile,
    FundingProfile,
    BehaviorProfile,
    FundingSource,
    ClusteringResult,
)
from .freshness import FreshnessAnalyzer
from .behavior import BehaviorProfiler
from .funding import FundingTracker
from .clusterer import WalletClusterer

logger = logging.getLogger(__name__)


class NegativeSignalDetector:
    """
    Detects signals that DECREASE insider likelihood.
    Used to filter out false positives.

    Negative signals indicate a wallet is LESS likely to be an insider:
    - Known public entity (fund, market maker)
    - Long trading history
    - Retail behavior patterns
    - Frequently exits early
    - High behavioral variance (inconsistent = not coordinated)
    - Low volume (not worth tracking)
    - Popular market participation (mainstream interests)
    - Irregular activity intervals (natural user behavior)
    """

    # Known public wallets (funds, market makers, etc.)
    # These would be populated from research
    KNOWN_ENTITIES: Dict[str, str] = {
        # Example: "0x...": "market_maker_xyz",
    }

    # Volume thresholds for false positive filtering
    MIN_VOLUME_THRESHOLD = 100  # $100 minimum to be considered
    MICRO_VOLUME_THRESHOLD = 1000  # <$1000 is likely testing/casual

    def __init__(
        self,
        threshold_days: int = 180,
        retail_threshold: float = 0.7,
        min_trades_for_pattern: int = 3
    ):
        """
        Args:
            threshold_days: Days to consider wallet as having "long history"
            retail_threshold: Score above which wallet is considered retail
            min_trades_for_pattern: Minimum trades needed for reliable pattern detection
        """
        self.threshold_days = threshold_days
        self.retail_threshold = retail_threshold
        self.min_trades_for_pattern = min_trades_for_pattern

    def is_known_entity(self, wallet: str) -> bool:
        """Check if wallet is a known public entity."""
        return wallet.lower() in self.KNOWN_ENTITIES

    def get_entity_name(self, wallet: str) -> Optional[str]:
        """Get the name of a known entity if applicable."""
        return self.KNOWN_ENTITIES.get(wallet.lower())

    def has_long_history(
        self,
        profile: WalletProfile,
        threshold_days: Optional[int] = None
    ) -> bool:
        """Check if wallet has extensive trading history."""
        threshold = threshold_days or self.threshold_days
        return profile.days_on_polymarket >= threshold

    def exhibits_retail_behavior(
        self,
        profile: WalletProfile,
        threshold: Optional[float] = None
    ) -> bool:
        """Check if wallet shows clear retail patterns."""
        threshold = threshold or self.retail_threshold
        return profile.retail_likelihood >= threshold

    def exits_early_frequently(
        self,
        profile: WalletProfile,
        threshold: float = 0.5
    ) -> bool:
        """Check if wallet frequently exits before resolution."""
        # High early exit ratio suggests NOT an insider
        # Insiders tend to hold to resolution
        return profile.holds_to_resolution_ratio < threshold

    def is_micro_volume(self, profile: WalletProfile) -> bool:
        """
        Check if wallet has very low volume (likely testing/casual).

        False positive prevention: Micro-volume wallets are unlikely to be
        sophisticated insiders - they're more likely casual users or test accounts.
        """
        return profile.total_volume < self.MICRO_VOLUME_THRESHOLD

    def is_below_minimum_volume(self, profile: WalletProfile) -> bool:
        """
        Check if wallet is below minimum volume threshold.

        Very low volume wallets should be filtered entirely.
        """
        return profile.total_volume < self.MIN_VOLUME_THRESHOLD

    def has_insufficient_trades(self, profile: WalletProfile) -> bool:
        """
        Check if wallet has too few trades for reliable pattern detection.

        False positive prevention: With very few trades, any pattern detection
        is unreliable and could flag innocent users.
        """
        return profile.total_trades < self.min_trades_for_pattern

    def has_high_behavioral_variance(self, profile: WalletProfile) -> bool:
        """
        Check if wallet has high variance in trading behavior.

        False positive prevention: Sybil/insider wallets typically show
        very CONSISTENT behavior (coordinated). High variance suggests
        natural, uncoordinated human behavior.

        Based on Sybil detection research: coordinated addresses show
        abnormally tight temporal clustering and consistent patterns.
        """
        # High variance in trade sizes = retail behavior
        # Insiders typically use consistent sizing for coordination
        if profile.avg_trade_size > 0:
            # Coefficient of variation > 1.0 indicates high variance
            # (would need size_variance from behavior profile)
            pass

        # Diverse market participation suggests genuine interest
        if profile.unique_markets >= 5:
            return True

        # Low sophistication + inconsistent behavior = natural user
        if profile.sophistication_score < 0.4 and profile.retail_likelihood > 0.5:
            return True

        return False

    def has_natural_activity_pattern(self, profile: WalletProfile) -> bool:
        """
        Check if wallet shows natural (irregular) activity patterns.

        False positive prevention: Legitimate users have irregular intervals
        between activities. Sybil addresses show suspicious temporal clustering
        with minimal intervals between events.

        Natural patterns include:
        - Trading during business hours (not exclusively off-hours)
        - Spread out activity over time (not burst-only)
        - Mix of weekday and weekend activity
        """
        # If most activity is during normal hours, less suspicious
        if profile.off_hours_ratio < 0.3:
            return True

        # No burst episodes suggests steady, natural trading
        if profile.burst_episodes == 0:
            return True

        # Long duration on platform suggests genuine user
        if profile.days_on_polymarket >= 30:
            return True

        return False

    def trades_popular_markets_only(
        self,
        profile: WalletProfile,
        behavior: Optional[BehaviorProfile] = None
    ) -> bool:
        """
        Check if wallet only trades in popular/mainstream markets.

        False positive prevention: Trading in popular markets (elections,
        major events) is normal behavior. Only niche market focus is suspicious.

        Insiders typically focus on obscure markets where they have edge.
        """
        if behavior and hasattr(behavior, 'niche_market_ratio'):
            # Low niche market ratio = trades popular markets
            return behavior.niche_market_ratio < 0.2
        return False

    def has_gradual_onboarding(self, profile: WalletProfile) -> bool:
        """
        Check if wallet shows gradual onboarding pattern.

        False positive prevention: Legitimate new users often start small
        and gradually increase activity. Insiders jump in with large,
        confident trades immediately.
        """
        # If avg trade size is much smaller than max, suggests gradual increase
        if profile.max_trade_size > 0 and profile.avg_trade_size > 0:
            size_ratio = profile.max_trade_size / profile.avg_trade_size
            # High ratio (>5x) suggests starting small then increasing
            if size_ratio > 5:
                return True

        # Many trades with low total volume = gradual small bets
        if profile.total_trades >= 10 and profile.total_volume < 5000:
            return True

        return False

    def compute_negative_signal_score(
        self,
        profile: WalletProfile,
        behavior: Optional[BehaviorProfile] = None
    ) -> float:
        """
        Compute aggregate negative signal score.

        Higher score = LESS likely to be insider (more false positive indicators).

        Components (normalized to sum to 1.0):
        - Known entity: 0.20 (strongest signal)
        - Long history: 0.15
        - Retail behavior: 0.15
        - Early exits: 0.10
        - Micro volume: 0.10
        - High behavioral variance: 0.10
        - Natural activity pattern: 0.08
        - Popular markets only: 0.05
        - Gradual onboarding: 0.05
        - Low freshness: 0.01
        - Low funding risk: 0.01
        """
        score = 0.0

        # === STRONG NEGATIVE SIGNALS (definitely not insider) ===

        # Known entity (strongest negative signal)
        if self.is_known_entity(profile.address):
            score += 0.20

        # Long trading history - established users are rarely insiders
        if self.has_long_history(profile):
            score += 0.15

        # Clear retail behavior patterns
        if self.exhibits_retail_behavior(profile):
            score += 0.15

        # === MODERATE NEGATIVE SIGNALS ===

        # Early exit pattern (insiders hold to resolution)
        if self.exits_early_frequently(profile):
            score += 0.10

        # Micro volume - not worth being an insider for small amounts
        if self.is_micro_volume(profile):
            score += 0.10

        # High behavioral variance - uncoordinated natural behavior
        if self.has_high_behavioral_variance(profile):
            score += 0.10

        # Natural activity patterns (not suspicious timing)
        if self.has_natural_activity_pattern(profile):
            score += 0.08

        # === WEAK NEGATIVE SIGNALS ===

        # Only trades popular/mainstream markets
        if self.trades_popular_markets_only(profile, behavior):
            score += 0.05

        # Shows gradual onboarding (not immediate large trades)
        if self.has_gradual_onboarding(profile):
            score += 0.05

        # Additional factors
        # Low freshness = NOT suspicious (established account)
        if profile.freshness_score < 0.3:
            score += 0.01

        # Low funding risk = NOT suspicious
        if profile.funding_risk_score < 0.3:
            score += 0.01

        # Total weights sum to exactly 1.0
        return min(1.0, score)

    def get_false_positive_reasons(
        self,
        profile: WalletProfile,
        behavior: Optional[BehaviorProfile] = None
    ) -> List[str]:
        """
        Get list of reasons why this wallet might be a false positive.

        Useful for debugging and explaining decisions.
        """
        reasons = []

        if self.is_known_entity(profile.address):
            entity_name = self.get_entity_name(profile.address)
            reasons.append(f"Known entity: {entity_name}")

        if self.has_long_history(profile):
            reasons.append(f"Long history: {profile.days_on_polymarket} days on platform")

        if self.exhibits_retail_behavior(profile):
            reasons.append(f"Retail behavior: {profile.retail_likelihood:.2f} likelihood")

        if self.exits_early_frequently(profile):
            reasons.append(f"Early exits: {1-profile.holds_to_resolution_ratio:.2f} early exit ratio")

        if self.is_micro_volume(profile):
            reasons.append(f"Micro volume: ${profile.total_volume:.2f}")

        if self.has_high_behavioral_variance(profile):
            reasons.append("High behavioral variance (natural user)")

        if self.has_natural_activity_pattern(profile):
            reasons.append("Natural activity pattern (not suspicious timing)")

        if self.trades_popular_markets_only(profile, behavior):
            reasons.append("Trades popular markets only")

        if self.has_gradual_onboarding(profile):
            reasons.append("Gradual onboarding pattern")

        if self.has_insufficient_trades(profile):
            reasons.append(f"Insufficient trades for pattern detection: {profile.total_trades}")

        return reasons

    def should_filter_as_false_positive(
        self,
        profile: WalletProfile,
        insider_score_threshold: float = 0.6,
        negative_score_threshold: float = 0.4
    ) -> tuple[bool, List[str]]:
        """
        Determine if a wallet should be filtered as likely false positive.

        Returns:
            (should_filter, reasons) tuple
        """
        reasons = []

        # Always filter if below minimum volume
        if self.is_below_minimum_volume(profile):
            reasons.append(f"Below minimum volume threshold: ${profile.total_volume:.2f}")
            return True, reasons

        # Always filter known entities
        if self.is_known_entity(profile.address):
            reasons.append(f"Known entity: {self.get_entity_name(profile.address)}")
            return True, reasons

        # Filter if high negative score
        negative_score = self.compute_negative_signal_score(profile)
        if negative_score >= negative_score_threshold:
            reasons = self.get_false_positive_reasons(profile)
            reasons.insert(0, f"High negative signal score: {negative_score:.2f}")
            return True, reasons

        # Filter if insider score is low AND has any negative signals
        if profile.preliminary_insider_score < insider_score_threshold:
            if negative_score > 0.2:
                reasons = self.get_false_positive_reasons(profile)
                reasons.insert(0, f"Low insider score ({profile.preliminary_insider_score:.2f}) with negative signals")
                return True, reasons

        return False, []


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
        trades_df: pd.DataFrame,
        markets_df: Optional[pd.DataFrame] = None,
        wallets_df: Optional[pd.DataFrame] = None,
    ):
        """
        Args:
            trades_df: DataFrame with trade data
            markets_df: Optional DataFrame with market data
            wallets_df: Optional DataFrame with basic wallet stats from Phase 1
        """
        self.trades = trades_df
        self.markets = markets_df
        self.wallets = wallets_df

        # Initialize analyzers
        self.freshness_analyzer = FreshnessAnalyzer(trades_df)
        self.behavior_profiler = BehaviorProfiler(trades_df, markets_df)
        self.funding_tracker = FundingTracker()
        self.negative_signal_detector = NegativeSignalDetector()

        # Caches
        self._freshness_profiles: Optional[Dict[str, FreshnessProfile]] = None
        self._behavior_profiles: Optional[Dict[str, BehaviorProfile]] = None
        self._funding_profiles: Optional[Dict[str, FundingProfile]] = None
        self._clustering_result: Optional[ClusteringResult] = None

    def compute_all_analyses(
        self,
        wallets: Optional[List[str]] = None,
        run_clustering: bool = True,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> None:
        """
        Run all analyses and cache results.

        Args:
            wallets: List of wallet addresses to analyze (None = all)
            run_clustering: Whether to run clustering analysis
            progress_callback: Optional callback(step_name, current, total)
        """
        # Get wallet list
        if wallets is None:
            if self.wallets is not None and 'address' in self.wallets.columns:
                wallets = self.wallets['address'].tolist()
            else:
                # Extract from trades
                makers = self.trades['maker_address'].unique()
                takers = self.trades['taker_address'].unique()
                wallets = list(set(list(makers) + list(takers)))
                wallets = [w for w in wallets if w and w != '']

        logger.info(f"Running analyses for {len(wallets)} wallets")

        # 1. Freshness analysis
        if progress_callback:
            progress_callback("Freshness Analysis", 0, len(wallets))
        logger.info("Computing freshness profiles...")
        self._freshness_profiles = self.freshness_analyzer.compute_all_freshness_profiles(wallets)

        # 2. Behavior analysis
        if progress_callback:
            progress_callback("Behavior Analysis", 0, len(wallets))
        logger.info("Computing behavior profiles...")
        self._behavior_profiles = self.behavior_profiler.build_all_profiles(wallets)

        # 3. Funding analysis (limited without blockchain data)
        if progress_callback:
            progress_callback("Funding Analysis", 0, len(wallets))
        logger.info("Computing funding profiles...")

        # Add wallets to tracked set for cross-detection
        self.funding_tracker.add_tracked_wallets(wallets)
        self._funding_profiles = self.funding_tracker.compute_all_funding_profiles(wallets)

        # 4. Clustering
        if run_clustering:
            if progress_callback:
                progress_callback("Clustering", 0, 1)
            logger.info("Running wallet clustering...")

            # Get wallet volumes for clustering
            wallet_volumes = {}
            if self.wallets is not None and 'total_volume' in self.wallets.columns:
                wallet_volumes = dict(zip(
                    self.wallets['address'],
                    self.wallets['total_volume']
                ))

            # Prepare trades with wallet column
            trades_for_clustering = self.trades.copy()
            trades_for_clustering['wallet'] = trades_for_clustering['maker_address']

            clusterer = WalletClusterer(
                trades_df=trades_for_clustering,
                behavior_profiles=self._behavior_profiles,
                wallet_volumes=wallet_volumes
            )

            # Run simplified clustering (skip correlation which is expensive)
            self._clustering_result = clusterer.run_full_clustering(
                methods=["temporal", "behavior"]
            )

        logger.info("All analyses complete")

    def compute_preliminary_insider_score(
        self,
        freshness: FreshnessProfile,
        funding: FundingProfile,
        behavior: BehaviorProfile,
    ) -> float:
        """
        Compute preliminary insider likelihood score.

        This is a heuristic score before ML.
        Uses multiple features with continuous weighting for better differentiation.

        Insider signals (higher = more suspicious):
        - New to platform (freshness_score)
        - Concentrated in few markets (market_concentration)
        - Large trade sizes (avg_trade_size, max_trade_size)
        - Sophisticated trading patterns (sophistication_score)
        - High-risk funding sources (funding_risk_score)

        Anti-insider signals (lower = more suspicious):
        - Retail behavior patterns (retail_likelihood)
        - Many unique markets (diversified = retail)
        """
        score = 0.0

        # Freshness component (0.20 weight)
        # New wallets are more suspicious
        score += 0.20 * freshness.freshness_score

        # Sophistication component (0.25 weight)
        # Use the continuous sophistication score directly
        score += 0.25 * behavior.sophistication_score

        # Market concentration component (0.15 weight)
        # High concentration = focused betting = potential insider signal
        # BUT: Need to distinguish between:
        #   - Casual users who only care about 1 market (low trade count, low sophistication)
        #   - Sophisticated focused insiders (high trade count, high volume, calculated bets)
        # HHI ranges from ~0.077 (diversified) to 1.0 (single market)
        market_conc_score = behavior.market_concentration

        # Adjust concentration score based on context:
        # If wallet has very few trades (<5) and high HHI, they're likely casual, not insider
        # Only treat high concentration as suspicious if combined with other insider signals
        if behavior.unique_markets == 1 and freshness.total_polymarket_trades < 5:
            # Single-market casual user - reduce the concentration signal
            market_conc_score *= 0.3
        elif behavior.unique_markets <= 2 and behavior.sophistication_score < 0.4:
            # Low sophistication + few markets = casual user
            market_conc_score *= 0.5

        score += 0.15 * market_conc_score

        # Trade size component (0.15 weight)
        # Large trades = confidence in outcome = insider signal
        # Normalize by log scale to handle wide range
        max_size = behavior.max_trade_size or 0
        if max_size >= 100:
            # Log scale: $100 = 0.0, $100,000 = 1.0
            size_score = min(1.0, math.log10(max_size / 100) / 3)
        else:
            # Small trades (<$100) get score of 0
            size_score = 0.0
        score += 0.15 * size_score

        # Funding risk component (0.15 weight)
        # High-risk funding = suspicious
        score += 0.15 * funding.funding_risk_score

        # Inverse retail component (0.10 weight)
        # Low retail likelihood = sophisticated = potential insider
        inverse_retail = 1.0 - behavior.retail_likelihood
        score += 0.10 * inverse_retail

        return round(score, 4)

    def build_profile(self, wallet: str) -> WalletProfile:
        """
        Build complete profile for a single wallet.

        Args:
            wallet: Wallet address to profile

        Returns:
            Complete WalletProfile
        """
        wallet_lower = wallet.lower()

        # Get freshness profile
        if self._freshness_profiles and wallet_lower in self._freshness_profiles:
            freshness = self._freshness_profiles[wallet_lower]
        else:
            freshness = self.freshness_analyzer.compute_freshness_score(wallet)

        # Get behavior profile
        if self._behavior_profiles and wallet_lower in self._behavior_profiles:
            behavior = self._behavior_profiles[wallet_lower]
        else:
            behavior = self.behavior_profiler.build_behavior_profile(wallet)

        # Get funding profile
        if self._funding_profiles and wallet_lower in self._funding_profiles:
            funding = self._funding_profiles[wallet_lower]
        else:
            funding = self.funding_tracker.compute_funding_profile(wallet)

        # Get cluster info
        cluster_id = None
        cluster_members = []
        if self._clustering_result:
            cluster_id = self._clustering_result.wallet_to_cluster.get(wallet_lower)
            if cluster_id:
                for cluster in self._clustering_result.clusters:
                    if cluster.cluster_id == cluster_id:
                        cluster_members = [w for w in cluster.wallets if w != wallet_lower]
                        break

        # Get basic stats from wallets_df if available
        first_seen = freshness.first_polymarket_trade
        last_seen = None
        total_trades = freshness.total_polymarket_trades
        total_volume = 0.0
        unique_markets = behavior.unique_markets
        is_whale = False

        if self.wallets is not None:
            wallet_row = self.wallets[self.wallets['address'] == wallet_lower]
            if not wallet_row.empty:
                row = wallet_row.iloc[0]
                if 'first_seen' in row:
                    first_seen = row['first_seen']
                if 'last_seen' in row:
                    last_seen = row['last_seen']
                if 'total_trades' in row:
                    total_trades = int(row['total_trades'])
                if 'total_volume' in row:
                    total_volume = float(row['total_volume'])
                if 'unique_markets' in row:
                    unique_markets = int(row['unique_markets'])
                if 'is_whale' in row:
                    is_whale = bool(row['is_whale'])

        # Compute insider score
        insider_score = self.compute_preliminary_insider_score(freshness, funding, behavior)

        # Build profile
        profile = WalletProfile(
            address=wallet_lower,
            cluster_id=cluster_id,
            cluster_members=cluster_members,

            # Basic stats
            first_seen=first_seen,
            last_seen=last_seen,
            total_trades=total_trades,
            total_volume=total_volume,
            unique_markets=unique_markets,
            is_whale=is_whale,

            # Freshness
            freshness_score=freshness.freshness_score,
            is_zero_history=freshness.is_zero_history,
            is_new_to_polymarket=freshness.is_new_to_polymarket,
            is_recently_funded=freshness.is_recently_funded,
            days_on_polymarket=freshness.days_active,

            # Funding
            primary_funding_source=funding.primary_funding_source.name_str,
            funding_risk_score=funding.funding_risk_score,
            has_privacy_funding=(funding.primary_funding_source == FundingSource.PRIVACY_TOOL),
            from_tracked_wallets=funding.from_tracked_wallets,

            # Behavior
            retail_likelihood=behavior.retail_likelihood,
            sophistication_score=behavior.sophistication_score,
            holds_to_resolution_ratio=behavior.holds_to_resolution_ratio,
            win_rate=behavior.win_rate,
            off_hours_ratio=behavior.off_hours_ratio,
            burst_episodes=behavior.burst_episodes,
            avg_trade_size=behavior.avg_trade_size,
            max_trade_size=behavior.max_trade_size,

            # Performance
            sharpe_ratio=behavior.sharpe_ratio,
            early_exit_ratio=behavior.early_exit_ratio,

            # Scores
            preliminary_insider_score=insider_score,
            negative_signal_score=0.0,  # Computed below
        )

        # Compute negative signal score
        profile.negative_signal_score = self.negative_signal_detector.compute_negative_signal_score(profile)

        return profile

    def build_all_profiles(
        self,
        wallets: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, WalletProfile]:
        """
        Build profiles for all wallets.

        Args:
            wallets: List of wallet addresses (None = use cached analyses)
            progress_callback: Optional callback(current, total)

        Returns:
            Dict mapping wallet address to WalletProfile
        """
        # Ensure analyses are computed
        if self._freshness_profiles is None:
            self.compute_all_analyses(wallets)

        # Get wallet list from cached analyses
        if wallets is None:
            wallets = list(self._freshness_profiles.keys())

        logger.info(f"Building profiles for {len(wallets)} wallets")

        profiles = {}
        for i, wallet in enumerate(wallets):
            if (i + 1) % 100 == 0:
                logger.info(f"Built {i + 1}/{len(wallets)} profiles")
                if progress_callback:
                    progress_callback(i + 1, len(wallets))

            try:
                profiles[wallet] = self.build_profile(wallet)
            except Exception as e:
                logger.error(f"Error building profile for {wallet}: {e}")
                continue

        logger.info(f"Built {len(profiles)} complete wallet profiles")
        return profiles

    def get_top_insider_candidates(
        self,
        profiles: Dict[str, WalletProfile],
        n: int = 100,
        min_volume: float = 0,
        exclude_high_negative: bool = True,
        use_strict_filtering: bool = True
    ) -> List[WalletProfile]:
        """
        Get wallets most likely to be insiders with comprehensive false positive filtering.

        Args:
            profiles: Dict of WalletProfile objects
            n: Number of candidates to return
            min_volume: Minimum volume filter
            exclude_high_negative: Exclude wallets with high negative signal score
            use_strict_filtering: Apply comprehensive false positive prevention

        Returns:
            List of WalletProfile sorted by insider score descending
        """
        candidates = []
        filtered_count = 0
        filter_reasons = {}

        for profile in profiles.values():
            # Volume filter
            if profile.total_volume < min_volume:
                filtered_count += 1
                continue

            # Use comprehensive false positive filtering
            if use_strict_filtering:
                should_filter, reasons = self.negative_signal_detector.should_filter_as_false_positive(
                    profile,
                    insider_score_threshold=0.5,
                    negative_score_threshold=0.4
                )
                if should_filter:
                    filtered_count += 1
                    # Track filter reasons for debugging
                    primary_reason = reasons[0] if reasons else "Unknown"
                    filter_reasons[primary_reason] = filter_reasons.get(primary_reason, 0) + 1
                    continue
            elif exclude_high_negative and profile.negative_signal_score > 0.5:
                # Legacy simple filtering
                filtered_count += 1
                continue

            candidates.append(profile)

        # Log filtering statistics
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} wallets as likely false positives")
            for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1])[:5]:
                logger.debug(f"  - {reason}: {count} wallets")

        # Sort by insider score (adjusted by negative signals)
        def adjusted_score(p: WalletProfile) -> float:
            # More aggressive negative signal adjustment
            # High negative score significantly reduces final score
            base_score = p.preliminary_insider_score
            negative_penalty = p.negative_signal_score * 0.6
            return base_score * (1 - negative_penalty)

        candidates.sort(key=adjusted_score, reverse=True)

        logger.info(f"Returning {min(n, len(candidates))} top insider candidates from {len(candidates)} remaining")
        return candidates[:n]

    def get_candidates_with_explanations(
        self,
        profiles: Dict[str, WalletProfile],
        n: int = 20
    ) -> List[Dict]:
        """
        Get top insider candidates with detailed explanations for each.

        Useful for manual review and understanding why wallets are flagged.

        Returns:
            List of dicts with profile and explanation fields
        """
        candidates = self.get_top_insider_candidates(profiles, n=n * 2, use_strict_filtering=True)

        results = []
        for profile in candidates[:n]:
            # Get behavior profile for additional context
            behavior = self._behavior_profiles.get(profile.address) if self._behavior_profiles else None

            # Generate explanation
            explanation = self._generate_insider_explanation(profile, behavior)

            # Get any false positive indicators (for transparency)
            fp_reasons = self.negative_signal_detector.get_false_positive_reasons(profile, behavior)

            results.append({
                "profile": profile,
                "adjusted_score": profile.preliminary_insider_score * (1 - profile.negative_signal_score * 0.6),
                "explanation": explanation,
                "false_positive_indicators": fp_reasons,
                "confidence": self._compute_confidence(profile, behavior)
            })

        return results

    def _generate_insider_explanation(
        self,
        profile: WalletProfile,
        behavior: Optional[BehaviorProfile] = None
    ) -> List[str]:
        """Generate human-readable explanation for why wallet is flagged."""
        reasons = []

        if profile.freshness_score > 0.7:
            reasons.append(f"Very new wallet (freshness: {profile.freshness_score:.2f})")
        elif profile.freshness_score > 0.5:
            reasons.append(f"Relatively new wallet (freshness: {profile.freshness_score:.2f})")

        if profile.funding_risk_score > 0.6:
            reasons.append(f"High-risk funding source ({profile.primary_funding_source})")

        if profile.sophistication_score > 0.7:
            reasons.append(f"Sophisticated trading patterns (score: {profile.sophistication_score:.2f})")

        if profile.max_trade_size > 10000:
            reasons.append(f"Large trade sizes (max: ${profile.max_trade_size:,.2f})")

        if profile.retail_likelihood < 0.3:
            reasons.append(f"Low retail likelihood ({profile.retail_likelihood:.2f})")

        if profile.burst_episodes > 2:
            reasons.append(f"Multiple burst trading episodes ({profile.burst_episodes})")

        if profile.off_hours_ratio > 0.6:
            reasons.append(f"High off-hours trading ({profile.off_hours_ratio:.0%})")

        if behavior and behavior.market_concentration > 0.8:
            reasons.append(f"Highly concentrated in few markets (HHI: {behavior.market_concentration:.2f})")

        return reasons

    def _compute_confidence(
        self,
        profile: WalletProfile,
        behavior: Optional[BehaviorProfile] = None
    ) -> str:
        """Compute confidence level for the insider prediction."""
        score = profile.preliminary_insider_score
        negative = profile.negative_signal_score
        trades = profile.total_trades

        # Low confidence if few trades (unreliable patterns)
        if trades < 5:
            return "low"

        # High confidence requires strong signal and low negative
        if score > 0.7 and negative < 0.2 and trades >= 10:
            return "high"

        # Medium confidence
        if score > 0.5 and negative < 0.4:
            return "medium"

        return "low"

    def to_dataframe(self, profiles: Dict[str, WalletProfile]) -> pd.DataFrame:
        """Convert profiles dict to DataFrame for storage."""
        if not profiles:
            return pd.DataFrame()

        records = []
        for wallet, profile in profiles.items():
            record = asdict(profile)
            # Convert lists to strings for parquet compatibility
            record['cluster_members'] = ','.join(profile.cluster_members)
            record['from_tracked_wallets'] = ','.join(profile.from_tracked_wallets)
            records.append(record)

        return pd.DataFrame(records)
