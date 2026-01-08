"""
Funding signal detector.

Detects trades from wallets with suspicious funding patterns.
"""

from typing import Dict, List

from src.analysis import FundingSource, WalletProfile

from .types import (
    FundingSignalType,
    Signal,
    SignalCategory,
    Trade,
)


class FundingSignalDetector:
    """
    Detects funding-based insider signals.

    Signals based on funding source priority:
    - PRIVACY_FUNDING: Funded via privacy tools (CRITICAL)
    - TRACKED_WALLET_FUNDING: Funded from another tracked wallet
    - BRIDGE_FUNDING: Funded via cross-chain bridge
    - CEX_FUNDING: Funded from centralized exchange
    """

    # Signal confidence by funding type
    FUNDING_CONFIDENCE = {
        "privacy_tool": 1.0,      # Critical
        "tracked_wallet": 0.8,   # High
        "cross_chain": 0.6,      # Medium-High
        "cex": 0.4,              # Medium
    }

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
        Detect funding signals for a trade.

        Args:
            trade: The trade to analyze
            wallet_profile: Profile of the wallet

        Returns:
            List of detected signals
        """
        signals = []

        # Get primary funding source from profile
        source = wallet_profile.primary_funding_source

        if source == "privacy_tool" or wallet_profile.has_privacy_funding:
            signals.append(Signal(
                type=FundingSignalType.PRIVACY_FUNDING,
                category=SignalCategory.FUNDING,
                confidence=1.0,
                details={"funding_source": "privacy_tool"}
            ))

        elif source == "tracked_wallet":
            signals.append(Signal(
                type=FundingSignalType.TRACKED_WALLET_FUNDING,
                category=SignalCategory.FUNDING,
                confidence=0.8,
                details={
                    "funding_source": "tracked_wallet",
                    "source_wallets": wallet_profile.from_tracked_wallets[:5],  # Limit for display
                }
            ))

        elif source == "cross_chain":
            signals.append(Signal(
                type=FundingSignalType.BRIDGE_FUNDING,
                category=SignalCategory.FUNDING,
                confidence=0.6,
                details={"funding_source": "bridge"}
            ))

        elif source == "cex":
            signals.append(Signal(
                type=FundingSignalType.CEX_FUNDING,
                category=SignalCategory.FUNDING,
                confidence=0.4,
                details={"funding_source": "cex"}
            ))

        # Also check funding risk score from profile
        if wallet_profile.funding_risk_score > 0.7 and not signals:
            # High funding risk but no specific signal detected
            # Use CEX as default with adjusted confidence
            signals.append(Signal(
                type=FundingSignalType.CEX_FUNDING,
                category=SignalCategory.FUNDING,
                confidence=wallet_profile.funding_risk_score * 0.5,
                details={
                    "funding_source": "unknown_high_risk",
                    "risk_score": wallet_profile.funding_risk_score,
                }
            ))

        return signals
