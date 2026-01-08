"""
Negative signal filter.

Filters out obvious non-insiders based on negative indicators.
"""

from typing import Dict, List, Set

from src.analysis import WalletProfile


class NegativeSignalFilter:
    """
    Filters signals based on negative indicators.

    Negative signals (DECREASE insider likelihood):
    - Long trading history (>180 days)
    - Known entity
    - Retail-like behavior
    - Frequent early exits
    - High behavioral variance
    """

    # Thresholds
    LONG_HISTORY_DAYS = 180
    HIGH_RETAIL_LIKELIHOOD = 0.7
    FREQUENT_EARLY_EXIT = 0.5

    # Known entities (market makers, public traders, etc.)
    # This would typically be loaded from a config file
    KNOWN_ENTITIES: Set[str] = set()

    def __init__(
        self,
        wallet_profiles: Dict[str, WalletProfile],
        known_entities: List[str] = None,
    ):
        """
        Initialize filter with wallet profiles.

        Args:
            wallet_profiles: Dict mapping wallet address to WalletProfile
            known_entities: Optional list of known non-insider wallets
        """
        self.profiles = wallet_profiles
        if known_entities:
            self.KNOWN_ENTITIES = set(known_entities)

    def compute_negative_score(self, wallet_profile: WalletProfile) -> float:
        """
        Compute negative signal score (0-1).

        Higher = MORE likely to be non-insider (less suspicious).

        Args:
            wallet_profile: Profile of the wallet

        Returns:
            Score 0-1, higher means less likely to be insider
        """
        score = 0.0

        # Long history: >6 months on platform
        if wallet_profile.days_on_polymarket > self.LONG_HISTORY_DAYS:
            score += 0.3

        # High retail likelihood
        if wallet_profile.retail_likelihood > self.HIGH_RETAIL_LIKELIHOOD:
            score += 0.3

        # Frequent early exits (non-insider behavior)
        # Insiders typically hold to resolution
        if (
            hasattr(wallet_profile, "holds_to_resolution_ratio")
            and wallet_profile.holds_to_resolution_ratio < 0.5
        ):
            score += 0.2

        # Known entity (from whitelist)
        if self._is_known_entity(wallet_profile.address):
            score += 0.2

        # High sophistication but low freshness = likely professional trader, not insider
        if (
            wallet_profile.sophistication_score > 0.7
            and wallet_profile.freshness_score < 0.3
        ):
            score += 0.1

        # Many trades across diverse markets = retail/professional, not one-off insider
        if wallet_profile.total_trades > 50 and wallet_profile.unique_markets > 10:
            score += 0.1

        return min(score, 1.0)

    def should_filter(
        self,
        wallet_profile: WalletProfile,
        threshold: float = 0.7,
    ) -> bool:
        """
        Check if wallet should be filtered out.

        Args:
            wallet_profile: Profile to check
            threshold: Score threshold for filtering

        Returns:
            True if wallet should be filtered (likely not insider)
        """
        return self.compute_negative_score(wallet_profile) > threshold

    def _is_known_entity(self, wallet: str) -> bool:
        """
        Check against known entities list.

        Args:
            wallet: Wallet address to check

        Returns:
            True if wallet is a known entity
        """
        return wallet.lower() in {e.lower() for e in self.KNOWN_ENTITIES}

    def add_known_entity(self, wallet: str) -> None:
        """Add a wallet to the known entities list."""
        self.KNOWN_ENTITIES.add(wallet.lower())

    def get_filter_reasons(self, wallet_profile: WalletProfile) -> List[str]:
        """
        Get list of reasons why a wallet might be filtered.

        Args:
            wallet_profile: Profile to analyze

        Returns:
            List of reason strings
        """
        reasons = []

        if wallet_profile.days_on_polymarket > self.LONG_HISTORY_DAYS:
            reasons.append(f"Long history ({wallet_profile.days_on_polymarket} days)")

        if wallet_profile.retail_likelihood > self.HIGH_RETAIL_LIKELIHOOD:
            reasons.append(f"High retail likelihood ({wallet_profile.retail_likelihood:.2f})")

        if (
            hasattr(wallet_profile, "holds_to_resolution_ratio")
            and wallet_profile.holds_to_resolution_ratio < 0.5
        ):
            reasons.append("Frequent early exits")

        if self._is_known_entity(wallet_profile.address):
            reasons.append("Known entity")

        if wallet_profile.total_trades > 50:
            reasons.append(f"High trade count ({wallet_profile.total_trades})")

        return reasons
