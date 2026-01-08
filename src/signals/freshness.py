"""
Freshness signal detector.

Detects trades from suspiciously new or fresh wallets.
"""

from typing import Dict, List

from src.analysis import WalletProfile

from .types import (
    FreshnessSignalType,
    Signal,
    SignalCategory,
    Trade,
)


class FreshnessSignalDetector:
    """
    Detects freshness-based insider signals.

    Signals:
    - ZERO_HISTORY: First-ever trade on Polymarket
    - NEW_WALLET: Wallet created recently (< 7 days)
    - RECENT_FUNDING: Funded shortly before trade
    - LOW_ACTIVITY: Very few prior trades (< 5)
    """

    # Thresholds
    NEW_WALLET_DAYS = 7
    LOW_ACTIVITY_TRADES = 5

    def __init__(self, wallet_profiles: Dict[str, WalletProfile]):
        """
        Initialize detector with wallet profiles.

        Args:
            wallet_profiles: Dict mapping wallet address to WalletProfile
        """
        self.profiles = wallet_profiles

    def detect(
        self,
        trade: Trade,
        wallet_profile: WalletProfile,
    ) -> List[Signal]:
        """
        Detect freshness signals for a trade.

        Args:
            trade: The trade to analyze
            wallet_profile: Profile of the wallet making the trade

        Returns:
            List of detected signals with confidence scores
        """
        signals = []

        # Zero history: First trade ever
        if wallet_profile.total_trades == 1:
            signals.append(Signal(
                type=FreshnessSignalType.ZERO_HISTORY,
                category=SignalCategory.FRESHNESS,
                confidence=1.0,
                details={"first_trade": True}
            ))

        # New wallet: Less than 7 days on platform
        if wallet_profile.days_on_polymarket < self.NEW_WALLET_DAYS:
            # Confidence decreases as days increase
            confidence = 1.0 - (wallet_profile.days_on_polymarket / self.NEW_WALLET_DAYS)
            signals.append(Signal(
                type=FreshnessSignalType.NEW_WALLET,
                category=SignalCategory.FRESHNESS,
                confidence=confidence,
                details={"days_active": wallet_profile.days_on_polymarket}
            ))

        # Recent funding: Funded within 24 hours of trade
        if wallet_profile.is_recently_funded:
            signals.append(Signal(
                type=FreshnessSignalType.RECENT_FUNDING,
                category=SignalCategory.FRESHNESS,
                confidence=0.8,
                details={"recently_funded": True}
            ))

        # Low activity: Less than 5 prior trades
        if wallet_profile.total_trades < self.LOW_ACTIVITY_TRADES:
            # Confidence decreases as trade count increases
            confidence = 1.0 - (wallet_profile.total_trades / self.LOW_ACTIVITY_TRADES)
            signals.append(Signal(
                type=FreshnessSignalType.LOW_ACTIVITY,
                category=SignalCategory.FRESHNESS,
                confidence=confidence * 0.7,  # Lower weight for this signal
                details={"total_trades": wallet_profile.total_trades}
            ))

        return signals
